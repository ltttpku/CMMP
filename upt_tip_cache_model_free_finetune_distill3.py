"""
Unary-pairwise transformer for human-object interaction detection

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""


from builtins import Exception
import os
import torch
import torch.distributed as dist


from torch import nn, Tensor
from typing import Optional, List
from torchvision.ops.boxes import batched_nms, box_iou

from ops import binary_focal_loss_with_logits

import sys
from hico_list import hico_verb_object_list,hico_verbs,hico_verbs_sentence,hico_verbs_sentence_2
from vcoco_list import vcoco_verbs_sentence
sys.path.append('detr')
# print(sys.path)
from detr.models import build_model
# from d_detr.models import build_model as build_model_d_detr
from util import box_ops
from util.misc import nested_tensor_from_tensor_list
import pdb
# from CLIP_models import CLIP_ResNet, tokenize
# from CLIP_models_adapter_prior2 import CLIP_ResNet, tokenize, CLIP
import CLIP_models_adapter_prior2
import torchvision
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F
from transformer_module import TransformerDecoderLayer
from CLIP.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import clip 
from ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
import pickle, random
from tqdm import tqdm
from torch.cuda import amp
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
from hico_text_label import hico_unseen_index
import hico_text_label
from HICO_utils import HOI_IDX_TO_ACT_IDX

_tokenizer = _Tokenizer()
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class Weight_Pred(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        self.linear1 = MLP(input_dim=input_dim, hidden_dim=512, output_dim=128, num_layers=2)
        self.drop1 = nn.Dropout()
        self.linear2 = MLP(input_dim=128, hidden_dim=32, output_dim=3, num_layers=2)
    
    def forward(self, x):
        x = self.drop1(self.linear1(x))
        x = self.linear2(x)
        return F.sigmoid(x)

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.token_embedding = clip_model.token_embedding
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, prior=None):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, prior)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class PromptLearner(nn.Module):
    def __init__(self, args, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = args.N_CTX # cfg.TRAINER.COOP.N_CTX
        ctx_init = args.CTX_INIT ## cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        # clip_imsize = clip_model.visual.input_resolution
        # cfg_imsize = cfg.INPUT.SIZE[0]
        # assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            if args.CSC: # cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        
        # classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        self.dtype = dtype
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = args.CLASS_TOKEN_POSITION  # cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts

class CustomCLIP(nn.Module):
    def __init__(self, args, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(args, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.classnames = classnames
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        raise ValueError
        return logits


class UPT(nn.Module):
    """
    Unary-pairwise transformer

    Parameters:
    -----------
    detector: nn.Module
        Object detector (DETR)
    postprocessor: nn.Module 
        Postprocessor for the object detector
    interaction_head: nn.Module
        Interaction head of the network
    human_idx: int
        Index of the human class
    num_classes: int
        Number of action/interaction classes
    alpha: float
        Hyper-parameter in the focal loss
    gamma: float
        Hyper-parameter in the focal loss
    box_score_thresh: float
        Threshold used to eliminate low-confidence objects
    fg_iou_thresh: float
        Threshold used to associate detections with ground truth
    min_instances: float
        Minimum number of instances (human or object) to sample
    max_instances: float
        Maximum number of instances (human or object) to sample
    """
    def __init__(self,
        args,
        detector: nn.Module,
        postprocessor: nn.Module,
        model: nn.Module,
        origin_text_embeddings: torch.tensor,
        object_embedding: torch.tensor,
        human_idx: int, num_classes: int,
        alpha: float = 0.5, gamma: float = 2.0,
        box_score_thresh: float = 0.2, fg_iou_thresh: float = 0.5,
        min_instances: int = 3, max_instances: int = 15,
        object_class_to_target_class: List[list] = None,
        object_n_verb_to_interaction: List[list] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.detector = detector
        self.postprocessor = postprocessor
        self.clip_head = model
        self.origin_text_embeddings = origin_text_embeddings
        self.object_embedding = object_embedding
        self.visual_output_dim = model.image_encoder.output_dim
        self.object_n_verb_to_interaction = np.asarray(
                                object_n_verb_to_interaction, dtype=float
                            )

        self.human_idx = human_idx
        self.num_classes = num_classes

        self.alpha = alpha
        self.gamma = gamma

        self.box_score_thresh = box_score_thresh
        self.fg_iou_thresh = fg_iou_thresh

        self.min_instances = min_instances
        self.max_instances = max_instances
        self.object_class_to_target_class = object_class_to_target_class
        self.num_anno = kwargs["num_anno"]

        self.use_distill = args.use_distill
        self.use_consistloss = args.use_consistloss

        self.num_classes = num_classes
        self.use_multi_hot = args.use_multi_hot
        self.obj_affordance = args.obj_affordance

        self.feature = []
        if 'HO' in args.logits_type:
            self.feature.append('hum_obj')
        if 'U' in args.logits_type or 'T' in args.logits_type:
            self.feature.append('uni')
        self.feature = '_'.join(self.feature)
        self.logits_type = args.logits_type

        num_shot = args.num_shot
        file1 = args.file1
        # self.annotation_clip = pickle.load(open(file1,'rb'))

        self.use_kmeans = False # true
        if args.zs:
            self.zs_type = args.zs_type
            self.filtered_hoi_idx = hico_unseen_index[self.zs_type]
        else:
            self.filtered_hoi_idx = []
            self.zs_type = None

        self.seen_verb_idxs = list(set([HOI_IDX_TO_ACT_IDX[idx] for idx in range(600) if idx not in self.filtered_hoi_idx]))
        self.label_choice = args.label_choice
        if 'HO' in self.logits_type:
            self.cache_model_HO, self.one_hots_HO, self.sample_lens_HO = self.load_cache_model(file1=file1, feature='hum_obj',num_classes=self.num_classes, num_shot=num_shot, filtered_hoi_idx = self.filtered_hoi_idx, use_multi_hot=self.use_multi_hot, label_choice=self.label_choice, num_anno=self.num_anno)
            self.cache_model_HO, self.one_hots_HO, self.sample_lens_HO  = self.cache_model_HO.cuda().float(), self.one_hots_HO.cuda().float(), self.sample_lens_HO.cuda().float()
        if 'U' in self.logits_type:
            self.cache_model_U, self.one_hots_U, self.sample_lens_U = self.load_cache_model(file1=file1, feature='uni',num_classes=self.num_classes, num_shot=num_shot, filtered_hoi_idx = self.filtered_hoi_idx, use_multi_hot=self.use_multi_hot, label_choice=self.label_choice, num_anno=self.num_anno)
            self.cache_model_U, self.one_hots_U, self.sample_lens_U = self.cache_model_U.cuda().float(), self.one_hots_U.cuda().float(), self.sample_lens_U.cuda().float()

        if self.num_classes == 117:
            self.unseen_verb_idxs = [i for i in range(self.num_classes) if i not in self.seen_verb_idxs]
        elif self.num_classes == 600:
            self.seen_hoi_idxs = [i for i in range(self.num_classes) if i not in self.filtered_hoi_idx]
        
        self.individual_norm = True
        self.logits_type = args.logits_type #
        self.consist = True
        self.evaluate_type = 'detr' # gt, detr
        
        self.use_type = 'crop'
        self.beta_cache = torch.tensor(10)
        self.alpha_cache = torch.tensor(1.0)

        self.prior_type = args.prior_type
        self.finetune_adapter = True
        if self.prior_type == 'cbe':
            self.priors_initial_dim = self.visual_output_dim+5
        elif self.prior_type == 'cb':
            self.priors_initial_dim = 5
        elif self.prior_type == 'ce':
            self.priors_initial_dim = self.visual_output_dim+1
        elif self.prior_type == 'be':
            self.priors_initial_dim = self.visual_output_dim+4
        elif self.prior_type == 'c':
            self.priors_initial_dim = 1
        elif self.prior_type == 'b':
            self.priors_initial_dim = 4
        elif self.prior_type == 'e':
            self.priors_initial_dim = self.visual_output_dim
        else:
            raise NotImplementedError

        self.use_weight_pred = args.use_weight_pred
        self.prompt_learning = args.prompt_learning
        if self.finetune_adapter:
            if 'HO' in self.logits_type:
                self.adapter_HO_weight = nn.Parameter(self.cache_model_HO.clone().detach())
                self.adapter_HO_bias = nn.Parameter(-torch.ones(self.cache_model_HO.shape[0]))
                self.label_HO = nn.Parameter(self.one_hots_HO, requires_grad=False)
                if not self.use_weight_pred:
                    self.logit_scale_HO = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)) 

            if 'U' in self.logits_type:
                self.adapter_U_weight = nn.Parameter(self.cache_model_U.clone().detach())
                self.adapter_U_bias = nn.Parameter(-torch.ones(self.cache_model_U.shape[0]))
                self.label_U = nn.Parameter(self.one_hots_U, requires_grad=False)
                if not self.use_weight_pred:
                    self.logit_scale_U = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)) 

            if 'T' in self.logits_type:
                if not self.prompt_learning:
                    self.adapter_union_weight = nn.Parameter(self.origin_text_embeddings.clone().detach())
                if not self.use_weight_pred:
                    self.logit_scale_text = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)) 
            
            if 'H' in self.logits_type:
                # self.adapter_H_weight = nn.Parameter(torch.empty(self.num_classes, self.visual_output_dim))
                self.adapter_H_weight = nn.Parameter(torch.empty(self.visual_output_dim, self.visual_output_dim))
                nn.init.xavier_normal_(self.adapter_H_weight)
                self.adapter_H_bias = nn.Parameter(-torch.ones(self.num_classes))
                if not self.use_weight_pred:
                    self.logit_scale_H = nn.Parameter(torch.ones([]) * np.log(1 / 0.7)) 
            
            if 'O' in self.logits_type:
                # self.adapter_O_weight = nn.Parameter(torch.empty(self.num_classes, self.visual_output_dim))
                self.adapter_O_weight = nn.Parameter(torch.empty(self.visual_output_dim, self.visual_output_dim))
                nn.init.xavier_normal_(self.adapter_O_weight)
                self.adapter_O_bias = nn.Parameter(-torch.ones(self.num_classes))
                if not self.use_weight_pred:
                    self.logit_scale_O = nn.Parameter(torch.ones([]) * np.log(1 / 0.7)) 

        if args.use_insadapter:
            if args.prior_method == 0:
                self.priors_downproj = MLP(self.priors_initial_dim, 128, args.bottleneck, 3) # old 512+5   
            elif args.prior_method == 1:
                self.priors_downproj = MLP(self.priors_initial_dim * 2, 128, args.bottleneck, 3) # old 512+5   
            elif args.prior_method == 2:
                self.learnable_prior = nn.Parameter(torch.empty(args.vis_prompt_num, args.bottleneck))
                nn.init.xavier_normal_(self.learnable_prior)
            elif args.prior_method == 3:
                self.priors_downproj = MLP(512, 128, args.bottleneck, 1)
            elif args.prior_method == 5:
                self.priors_downproj = MLP(36, 64, args.bottleneck, 3)
                self.global_prior = pickle.load(open(f'./pairwise_prior/setting={args.zs_type}_method=2_coord=1_k={args.n_gsp}.pkl','rb'))
                self.global_prior = nn.Parameter(torch.tensor(self.global_prior).float())
            elif args.prior_method == 6:
                self.priors_downproj = MLP(self.priors_initial_dim, 128, args.bottleneck, 3) # old 512+5
                self.global_prior = pickle.load(open(f'./pairwise_prior/setting={args.zs_type}_method=2_coord=1_k={args.n_gsp}.pkl','rb'))
                self.global_prior = nn.Parameter(torch.tensor(self.global_prior).float())
                self.priors_downproj_global = MLP(36, 64, args.bottleneck, 3)
                self.fusion_attn = TransformerDecoderLayer(d_model=args.bottleneck, nhead=1, dim_feedforward=args.bottleneck, dropout=0.1)
                # self.fusion_attn = nn.MultiheadAttention(embed_dim=64, num_heads=1, dropout=0.3, batch_first=False)
        if args.prior_method == 4: #CoCoOp
            self.priors_downproj = MLP(512, 512, 512, 1)

        self.no_interaction_indexes = [9, 23, 30, 45, 53, 64, 75, 85, 91, 95, 106, 110, 128, 145, 159, 169, 173, 185, 193, 197, 207, 213, 223, 231, 234, 238, 242, 246, 251, 256, 263, 272, 282, 289, 294, 304, 312, 324, 329, 335, 341, 347, 351, 355, 362, 367, 375, 382, 388, 392, 396, 406, 413, 417, 428, 433, 437, 444, 448, 452, 462, 473, 482, 487, 501, 505, 515, 527, 532, 537, 545, 549, 557, 561, 566, 575, 583, 587, 594, 599]
        self.HOI_IDX_TO_OBJ_IDX = [
                4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 14,
                14, 14, 14, 14, 14, 14, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 39,
                39, 39, 39, 39, 39, 39, 39, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2, 2, 2, 2, 2,
                2, 2, 2, 2, 2, 2, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 56, 56, 56, 56,
                56, 56, 57, 57, 57, 57, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 60, 60,
                60, 60, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
                16, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 3,
                3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 58,
                58, 58, 58, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 6, 6, 6, 6, 6,
                6, 6, 6, 62, 62, 62, 62, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 24, 24,
                24, 24, 24, 24, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 34, 34, 34, 34, 34,
                34, 34, 34, 35, 35, 35, 21, 21, 21, 21, 59, 59, 59, 59, 13, 13, 13, 13, 73,
                73, 73, 73, 73, 45, 45, 45, 45, 45, 50, 50, 50, 50, 50, 50, 50, 55, 55, 55,
                55, 55, 55, 55, 55, 55, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 67, 67, 67,
                67, 67, 67, 67, 74, 74, 74, 74, 74, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41,
                54, 54, 54, 54, 54, 54, 54, 54, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
                20, 10, 10, 10, 10, 10, 42, 42, 42, 42, 42, 42, 29, 29, 29, 29, 29, 29, 23,
                23, 23, 23, 23, 23, 78, 78, 78, 78, 26, 26, 26, 26, 52, 52, 52, 52, 52, 52,
                52, 66, 66, 66, 66, 66, 33, 33, 33, 33, 33, 33, 33, 33, 43, 43, 43, 43, 43,
                43, 43, 63, 63, 63, 63, 63, 63, 68, 68, 68, 68, 64, 64, 64, 64, 49, 49, 49,
                49, 49, 49, 49, 49, 49, 49, 69, 69, 69, 69, 69, 69, 69, 12, 12, 12, 12, 53,
                53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 72, 72, 72, 72, 72, 65, 65, 65, 65,
                48, 48, 48, 48, 48, 48, 48, 76, 76, 76, 76, 71, 71, 71, 71, 36, 36, 36, 36,
                36, 36, 36, 36, 36, 36, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 31, 31,
                31, 31, 31, 31, 31, 31, 31, 44, 44, 44, 44, 44, 32, 32, 32, 32, 32, 32, 32,
                32, 32, 32, 32, 32, 32, 32, 11, 11, 11, 11, 28, 28, 28, 28, 28, 28, 28, 28,
                28, 28, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 77, 77, 77, 77, 77,
                38, 38, 38, 38, 38, 27, 27, 27, 27, 27, 27, 27, 27, 70, 70, 70, 70, 61, 61,
                61, 61, 61, 61, 61, 61, 79, 79, 79, 79, 9, 9, 9, 9, 9, 7, 7, 7, 7, 7, 7, 7,
                7, 7, 25, 25, 25, 25, 25, 25, 25, 25, 75, 75, 75, 75, 40, 40, 40, 40, 40,
                40, 40, 22, 22, 22, 22, 22
            ]
        self.obj_to_no_interaction = torch.as_tensor([169, 23, 75, 159, 9, 64, 193, 575, 45, 566, 329, 505, 417, 246,
                                                        30,  85, 128, 145, 185, 106, 324, 238, 599, 347, 213, 583, 355, 545,
                                                        515, 341, 473, 482, 501, 375, 231, 234, 462, 527, 537,  53, 594, 304,
                                                        335, 382, 487, 256, 223, 207, 444, 406, 263, 282, 362, 428, 312, 272,
                                                        91,  95, 173, 242, 110, 557, 197, 388, 396, 437, 367, 289, 392, 413,
                                                        549, 452, 433, 251, 294, 587, 448, 532, 351, 561])

        self.epoch = 0
        # self.use_deformable_attn = args.use_deformable_attn
        self.COCO_CLASSES = ['N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', \
                    'fire hydrant','N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',\
                    'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', \
                    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', \
                    'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', \
                    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', \
                    'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', \
                    'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        self.reserve_indices = [idx for (idx, name) in enumerate(self.COCO_CLASSES) if name != 'N/A']
        self.reserve_indices = self.reserve_indices + [91]
        self.reserve_indices = torch.as_tensor(self.reserve_indices)
        self.dataset = args.dataset
        self.hyper_lambda = args.hyper_lambda
        self.pseudo_label = args.pseudo_label
        self.tpt = args.tpt
        self.featmap_dropout = nn.Dropout(kwargs["featmap_dropout_rate"])
        self.feat_mask_type = args.feat_mask_type
        self.language_aware = args.LA 
        self.use_insadapter = args.use_insadapter
        self.prior_method = args.prior_method
        self.LA_weight = args.LA_weight
        self.box_proj = args.box_proj
        if self.box_proj:
            self.box_proj_mlp = MLP(8, 128, self.visual_output_dim, num_layers=3)
        if self.use_weight_pred:
            num_branch = len(self.logits_type.split('+'))
            self.weight_pred = Weight_Pred(input_dim=self.visual_output_dim*3, output_dim=num_branch)
        if self.obj_affordance:
            self.obj_affordance_query = nn.Parameter(torch.empty(1, self.visual_output_dim, dtype=self.clip_head.dtype))  # to be optimized
            self.obj_affordance_learner = nn.MultiheadAttention(embed_dim=512*1, num_heads=1, dropout=0.3, batch_first=True)

        self.use_mlp_proj = kwargs["use_mlp_proj"]
        if self.use_mlp_proj:
            self.mlp_proj = MLP(512, 512, 512, 3)
        self.vision_regularize = args.vision_regularize
        if self.vision_regularize:
            clip_state_dict = torch.load("checkpoints/pretrained_clip/ViT-B-16.pt", map_location="cpu").state_dict()
            self.origin_clip_head = CLIP_models_adapter_prior2.build_model(state_dict=clip_state_dict, use_adapter=False)
        self.use_text_adapter = args.use_text_adapter
    
    @torch.no_grad()
    def refresh_unseen_verb_cache_mem(self, ):
        assert self.num_classes == 117
        seen_idxs = self.seen_verb_idxs
        unseen_idxs = self.unseen_verb_idxs
        text_embedding = self.origin_text_embeddings
        for i in unseen_idxs:
            cur_logits = (text_embedding[i] @ text_embedding[torch.as_tensor(seen_idxs)].T)
            cur_logits = F.softmax(cur_logits)
            if 'H' in self.logits_type:
                self.adapter_H_weight[i] = cur_logits @ self.adapter_H_weight[torch.as_tensor(seen_idxs)]
            if 'O' in self.logits_type:
                self.adapter_O_weight[i] = cur_logits @ self.adapter_O_weight[torch.as_tensor(seen_idxs)]

    def load_cache_model(self,file1, feature='union',num_classes=117, num_shot=10, filtered_hoi_idx=[], use_multi_hot=False, label_choice='random', num_anno=None):  ## √
        annotation = pickle.load(open(file1,'rb'))
        categories = num_classes
        union_embeddings = [[] for i in range(categories)]
        obj_embeddings = [[] for i in range(categories)]
        hum_embeddings = [[] for i in range(categories)]
        real_verbs = [[] for i in range(categories)]
        filenames = list(annotation.keys())
        verbs_iou = [[] for i in range(categories)] # contain 600hois or 117 verbs
        # hois_iou = [[] for i in range(len(hois))]
        each_filenames = [[] for i in range(categories)]
        for file_n in filenames:
            anno = annotation[file_n]
            # dict_keys (['boxes_h', 'boxes_o', 'verbs', 'union_boxes', 'union_features', 'huamn_features''object_features''objects', 'global_feature'])
            if categories == 117 or categories == 24: verbs = anno['verbs']
            else: verbs = (self.object_n_verb_to_interaction[anno['objects'], anno['verbs']]).astype(int)
            
            num_ho_pair = len(anno['boxes_h'])
            anno['real_verbs'] = np.zeros(shape=(num_ho_pair, categories))
            for i in range(num_ho_pair):
                anno['real_verbs'][i][verbs[i]] = 1

            if use_multi_hot:
                boxes_h_iou = torchvision.ops.box_iou(torch.as_tensor(anno['boxes_h']), torch.as_tensor(anno['boxes_h']))
                boxes_o_iou = torchvision.ops.box_iou(torch.as_tensor(anno['boxes_o']), torch.as_tensor(anno['boxes_o']))
                for i in range(num_ho_pair):
                    idx_h = boxes_h_iou[i] > 0.6
                    idx_o = torch.logical_and(boxes_o_iou[i] > 0.6, torch.as_tensor(anno['objects']) == anno['objects'][i])
                    idx_ho = torch.logical_and(idx_h, idx_o)
                    anno['real_verbs'][i] = torch.sum(torch.as_tensor(anno['real_verbs'])[idx_ho], dim=0)
                
                anno['real_verbs'][anno['real_verbs']>1] = 1

            ious = torch.diag(box_iou(torch.as_tensor(anno['boxes_h']), torch.as_tensor(anno['boxes_o'])))
            if len(verbs) == 0:
                print(file_n)

            for i, v in enumerate(verbs):
                if 'hico' in file1: ## TODO ??? why vcoco list idx out of range
                    if num_classes == 117:
                        if anno['verbs'][i] not in self.object_class_to_target_class[anno['objects'][i]]:
                            continue
                    elif num_classes == 600:
                        if v in filtered_hoi_idx:
                            continue
                union_embeddings[v].append(anno['union_features'][i] / np.linalg.norm(anno['union_features'][i]))
                obj_embeddings[v].append(anno['object_features'][i] / np.linalg.norm(anno['object_features'][i]))
                hum_embeddings[v].append(anno['huamn_features'][i] / np.linalg.norm(anno['huamn_features'][i]))
                each_filenames[v].append(file_n)
                real_verbs[v].append(anno['real_verbs'][i])
                # add iou
                verbs_iou[v].append(ious[i])

        if num_classes == 117:
            for i in range(categories):
                if len(union_embeddings[i]) == 0:
                    self.unseen_verb_idxs.append(i)
            print('[INFO]: missing idxs of verbs:', self.unseen_verb_idxs)
            for i in self.unseen_verb_idxs:
                for z in range(num_shot):
                    union_embeddings[i].append(np.random.randn(self.visual_output_dim))
                    obj_embeddings[i].append(np.random.randn(self.visual_output_dim))
                    hum_embeddings[i].append(np.random.randn(self.visual_output_dim))
        elif num_classes == 600:
            for i in filtered_hoi_idx:
                for z in range(num_shot):
                    union_embeddings[i].append(np.random.randn(self.visual_output_dim))
                    obj_embeddings[i].append(np.random.randn(self.visual_output_dim))
                    hum_embeddings[i].append(np.random.randn(self.visual_output_dim))
        ## re-implement cachemodel construction
        cache_models_lst, each_lens_lst = [], []
        real_verbs_lst = []
        if feature == 'hum_obj':
            indexes = np.arange(len(union_embeddings))
            for i, hum_emb, obj_emb, embeddings, real_v in tqdm(zip(indexes, hum_embeddings, obj_embeddings, union_embeddings, real_verbs)):
                hum_emb =  torch.as_tensor(np.array(hum_emb)).float()   
                obj_emb = torch.as_tensor(np.array(obj_emb)).float()
                real_v = torch.as_tensor(np.array(real_v))
                new_embeddings = torch.cat([hum_emb, obj_emb], dim=-1)
                new_embeddings = new_embeddings.cuda().float()

                num_to_select = min(hum_emb.shape[0], num_shot)
                
                if num_to_select < hum_emb.shape[0]:
                    if label_choice == 'random':
                        topk_idx = torch.randperm(new_embeddings.shape[0])[:num_to_select] 
                    elif label_choice == 'multi_first':
                        v_, topk_idx = torch.topk(torch.sum(real_v, dim=-1), k=num_to_select)
                    elif label_choice == 'single_first':
                        v_, topk_idx = torch.topk(torch.sum(real_v, dim=-1), k=num_to_select, largest=False)
                    elif label_choice == 'single+multi':
                        v_, topk_idx1 = torch.topk(torch.sum(real_v, dim=-1), k=num_to_select//2, largest=True)
                        v_, topk_idx2 = torch.topk(torch.sum(real_v, dim=-1), k=num_to_select//2, largest=False)
                        topk_idx = torch.cat((topk_idx1, topk_idx2))
                    elif label_choice == 'rare_first':
                        real_freq = real_v @ num_anno.to(torch.float64)
                        v_, topk_idx = torch.topk(real_freq, k=num_to_select, largest=False)
                    elif label_choice == 'non_rare_first':
                        real_freq = real_v @ num_anno.to(torch.float64)
                        v_, topk_idx = torch.topk(real_freq, k=num_to_select, largest=True)
                    elif label_choice == 'rare+non_rare':
                        real_freq = real_v @ num_anno.to(torch.float64)
                        v_, topk_idx1 = torch.topk(real_freq, k=num_to_select//2, largest=True)
                        v_, topk_idx2 = torch.topk(real_freq, k=num_to_select//2, largest=False)
                        topk_idx = torch.cat((topk_idx1, topk_idx2))
                    new_embeddings = new_embeddings[topk_idx]
                    real_v = real_v[topk_idx]
                
                cache_models_lst.append(new_embeddings)
                each_lens_lst.append(num_to_select)
                real_verbs_lst.append(real_v)
        elif feature == 'uni':
            indexes = np.arange(len(union_embeddings))
            for i, hum_emb, obj_emb, embeddings, real_v in tqdm(zip(indexes, hum_embeddings, obj_embeddings, union_embeddings, real_verbs)):
                uni_emb = torch.as_tensor(np.array(embeddings)).float()
                real_v = torch.as_tensor(np.array(real_v))
                new_embeddings = uni_emb
                new_embeddings = new_embeddings.cuda().float()

                num_to_select = min(uni_emb.shape[0], num_shot)

                if num_to_select < uni_emb.shape[0]:
                    if label_choice == 'random':
                        topk_idx = torch.randperm(new_embeddings.shape[0])[:num_to_select] 
                    elif label_choice == 'multi_first':
                        v_, topk_idx = torch.topk(torch.sum(real_v, dim=-1), k=num_to_select)
                    elif label_choice == 'single_first':
                        v_, topk_idx = torch.topk(torch.sum(real_v, dim=-1), k=num_to_select, largest=False)
                    elif label_choice == 'single+multi':
                        v_, topk_idx1 = torch.topk(torch.sum(real_v, dim=-1), k=num_to_select//2, largest=True)
                        v_, topk_idx2 = torch.topk(torch.sum(real_v, dim=-1), k=num_to_select//2, largest=False)
                        topk_idx = torch.cat((topk_idx1, topk_idx2))
                    elif label_choice == 'rare_first':
                        real_freq = real_v @ num_anno.to(torch.float64)
                        v_, topk_idx = torch.topk(real_freq, k=num_to_select, largest=False)
                    elif label_choice == 'non_rare_first':
                        real_freq = real_v @ num_anno.to(torch.float64)
                        v_, topk_idx = torch.topk(real_freq, k=num_to_select, largest=True)
                    elif label_choice == 'rare+non_rare':
                        real_freq = real_v @ num_anno.to(torch.float64)
                        v_, topk_idx1 = torch.topk(real_freq, k=num_to_select//2, largest=True)
                        v_, topk_idx2 = torch.topk(real_freq, k=num_to_select//2, largest=False)
                        topk_idx = torch.cat((topk_idx1, topk_idx2))
                    new_embeddings = new_embeddings[topk_idx]
                    real_v = real_v[topk_idx]
                cache_models_lst.append(new_embeddings)
                each_lens_lst.append(num_to_select)
                real_verbs_lst.append(real_v)
        else:
            raise NotImplementedError
        
        cache_models = torch.cat(cache_models_lst, dim=0)
        labels = torch.cat(real_verbs_lst, dim=0)
        return cache_models, labels, torch.sum(labels, dim=0)

    def get_clip_feature(self,image):  ## xxx
        x = self.clip_model.visual.conv1(image.type(self.clip_model.dtype))
        
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip_model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip_model.visual.positional_embedding.to(x.dtype)
        x = self.clip_model.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        
        local_feat = self.clip_model.visual.transformer.resblocks[:11]((x,None))[0]
        # x = self.clip_model.visual.transformer(x)
        # x = x.permute(1, 0, 2)  # LND -> NLD
        return local_feat
    
    def _reset_parameters(self):  ## xxx
        for p in self.context_aware.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.layer_norm.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def compute_prior_scores(self, x: Tensor, y: Tensor, scores: Tensor, object_class: Tensor):
        
        prior_h = torch.zeros(len(x), self.num_classes, device=scores.device)
        prior_o = torch.zeros_like(prior_h)
        
        # Raise the power of object detection scores during inference
        p = 1.0 if self.training else self.hyper_lambda
        s_h = scores[x].pow(p)
        s_o = scores[y].pow(p)
        # Map object class index to target class index
        # Object class index to target class index is a one-to-many mapping 
        target_cls_idx = [self.object_class_to_target_class[obj.item()]
            for obj in object_class[y]]
        # Duplicate box pair indices for each target class
        pair_idx = [i for i, tar in enumerate(target_cls_idx) for _ in tar]
        # Flatten mapped target indices
        flat_target_idx = [t for tar in target_cls_idx for t in tar]

        prior_h[pair_idx, flat_target_idx] = s_h[pair_idx]
        prior_o[pair_idx, flat_target_idx] = s_o[pair_idx]

        return torch.stack([prior_h, prior_o])

    @torch.no_grad()
    def init_adapter_union_weight(self, device):
        if self.prompt_learning and self.use_text_adapter:
            classnames = self.clip_head.classnames
            tokenized_prompts = torch.cat([clip.tokenize(p) for p in classnames]).to(device)
            with torch.no_grad():
                embedding = self.clip_head.text_encoder.token_embedding(tokenized_prompts).type(self.clip_head.text_encoder.dtype)
            ctx = self.clip_head.prompt_learner.ctx[:, :64] ## quick hack: the prior-dim should be 64
            if ctx.dim() == 2:
                ctx = ctx.unsqueeze(0).expand(self.clip_head.prompt_learner.n_cls, -1, -1)
            prior_mask = None
            prior = (ctx, prior_mask)
            self.adapter_union_weight = self.clip_head.text_encoder(embedding, tokenized_prompts, prior)
        else:
            prompts = self.clip_head.prompt_learner()
            tokenized_prompts = self.clip_head.tokenized_prompts
            self.adapter_union_weight = self.clip_head.text_encoder(prompts, tokenized_prompts)
            self.adapter_union_weight = self.adapter_union_weight.to(device)

    def compute_roi_embeddings(self, features: OrderedDict, image_size: Tensor, region_props: List[dict], feat_global):
        device = features.device
        boxes_h_collated = []; boxes_o_collated = []
        prior_collated = []; object_class_collated = []
        # pairwise_tokens_collated = []
        attn_maps_collated = []
        all_logits = []

        img_h, img_w = image_size.unbind(-1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)

        gt_feats_collated = []
        pair_feats_collated = []
        gt_all_logits = []
        pair_logits = []
        pair_prior = []
        gt_labels = []
        if self.prompt_learning and self.training and self.use_text_adapter:
            classnames = self.clip_head.classnames
            tokenized_prompts = torch.cat([clip.tokenize(p) for p in classnames]).to(device)
            with torch.no_grad():
                embedding = self.clip_head.text_encoder.token_embedding(tokenized_prompts).type(self.clip_head.text_encoder.dtype)
            ctx = self.clip_head.prompt_learner.ctx[:, :64] ## quick hack: the prior-dim should be 64
            if ctx.dim() == 2:
                ctx = ctx.unsqueeze(0).expand(self.clip_head.prompt_learner.n_cls, -1, -1)
            prior_mask = None
            prior = (ctx, prior_mask)
            self.adapter_union_weight = self.clip_head.text_encoder(embedding, tokenized_prompts, prior)
        elif self.prompt_learning and self.training and self.prior_method != 4:
            prompts = self.clip_head.prompt_learner()
            tokenized_prompts = self.clip_head.tokenized_prompts
            self.adapter_union_weight = self.clip_head.text_encoder(prompts, tokenized_prompts)

        for b_idx, props in enumerate(region_props):
            local_features = features[b_idx]
            boxes = props['boxes']
            scores = props['scores']
            labels = props['labels']

            is_human = labels == self.human_idx
            n_h = torch.sum(is_human); n = len(boxes)
            # Permute human instances to the top
            if not torch.all(labels[:n_h]==self.human_idx):
                h_idx = torch.nonzero(is_human).squeeze(1)
                o_idx = torch.nonzero(is_human == 0).squeeze(1)
                perm = torch.cat([h_idx, o_idx])
                boxes = boxes[perm]; scores = scores[perm]
                labels = labels[perm]
            # Skip image when there are no valid human-object pairs
            if n_h == 0 or n <= 1:
                boxes_h_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                boxes_o_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                object_class_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                prior_collated.append(torch.zeros(2, 0, self.num_classes, device=device))
                continue

            # Get the pairwise indices
            x, y = torch.meshgrid(
                torch.arange(n, device=device),
                torch.arange(n, device=device)
            )
            # Valid human-object pairs
            x_keep, y_keep = torch.nonzero(torch.logical_and(x != y, x < n_h)).unbind(1)
            if len(x_keep) == 0:
                # Should never happen, just to be safe
                raise ValueError("There are no valid human-object pairs")
            x = x.flatten(); y = y.flatten()
            
            # extract single roi features
            sub_boxes = boxes[x_keep]
            obj_boxes = boxes[y_keep]
            lt = torch.min(sub_boxes[..., :2], obj_boxes[..., :2]) # left point
            rb = torch.max(sub_boxes[..., 2:], obj_boxes[..., 2:]) # right point
            union_boxes = torch.cat([lt,rb],dim=-1)
            
            spatial_scale = 1 / (image_size[0,0]/local_features.shape[1])
            # union_features = torchvision.ops.roi_align(local_features.unsqueeze(0),[union_boxes],output_size=(1, 1),spatial_scale=spatial_scale,aligned=True).flatten(2).mean(-1)
            # single_features = torchvision.ops.roi_align(local_features.unsqueeze(0),[boxes],output_size=(1, 1),spatial_scale=spatial_scale,aligned=True).flatten(2).mean(-1)
            union_features = torchvision.ops.roi_align(local_features.unsqueeze(0),[union_boxes],output_size=(7, 7),spatial_scale=spatial_scale,aligned=True)
            single_features = torchvision.ops.roi_align(local_features.unsqueeze(0),[boxes],output_size=(7, 7),spatial_scale=spatial_scale,aligned=True)

            if self.feat_mask_type == 0:
                union_features = self.featmap_dropout(union_features).flatten(2).mean(-1)
                single_features = self.featmap_dropout(single_features).flatten(2).mean(-1)
            elif self.feat_mask_type == 1:
                union_features = union_features.flatten(2).mean(-1)
                single_features = single_features.flatten(2).mean(-1)
            # 
            # if self.box_proj == 1:
            #     box_feat = self.box_proj_mlp(torch.cat((sub_boxes, obj_boxes), dim=-1))
            
            human_features = single_features[x_keep]
            object_features = single_features[y_keep]
            if self.individual_norm: ## todo should use norm during finetuning?? 
                concat_feat_original = torch.cat([human_features,object_features, union_features],dim=-1)
                human_features = human_features / human_features.norm(dim=-1, keepdim=True)
                object_features = object_features / object_features.norm(dim=-1, keepdim=True)
                union_features = union_features / union_features.norm(dim=-1, keepdim=True)
                if self.feature == 'hum_obj_uni':
                    concat_feat = torch.cat([human_features, object_features, union_features],dim=-1) 
                elif self.feature == 'hum_obj':
                    concat_feat = torch.cat([human_features, object_features], dim=-1)
                elif self.feature == 'uni':
                    concat_feat = union_features
            else:
                concat_feat = torch.cat([human_features,object_features, union_features],dim=-1) 
                concat_feat = concat_feat/concat_feat.norm(dim=-1,keepdim=True) 

            if self.prompt_learning and self.prior_method == 4: # CoCoOp
                self.clip_head.prompt_learner.ctx = nn.Parameter(self.clip_head.prompt_learner.ctx + self.priors_downproj(feat_global[b_idx]))
                prompts = self.clip_head.prompt_learner()
                tokenized_prompts = self.clip_head.tokenized_prompts
                self.adapter_union_weight = self.clip_head.text_encoder(prompts, tokenized_prompts)
                
            if self.logits_type == 'HO+U+T':
                phi_union_HO = torch.cat([human_features, object_features], dim=-1) @ self.adapter_HO_weight.T + self.adapter_HO_bias
                phi_union_U = union_features @ self.adapter_U_weight.T + self.adapter_U_bias
                logits_cache_HO = ((phi_union_HO @ self.label_HO) / self.sample_lens_HO) / 2
                logits_cache_U = (phi_union_U @ self.label_U) / self.sample_lens_U
                logits_text = union_features @ self.adapter_union_weight.T
                
                if self.use_weight_pred:
                    logits_weights = self.weight_pred(torch.cat([human_features,object_features, union_features], dim=-1))
                    logits = logits_cache_HO * logits_weights[:, 0:1] + logits_cache_U * logits_weights[:, 1:2] + logits_text * logits_weights[:, 2:3]
                else:
                    logits = logits_cache_HO * self.logit_scale_HO + logits_cache_U * self.logit_scale_U + logits_text * self.logit_scale_text

            elif self.logits_type == 'HO+T':
                phi_union_HO = torch.cat([human_features, object_features], dim=-1) @ self.adapter_HO_weight.T + self.adapter_HO_bias
                logits_cache_HO = ((phi_union_HO @ self.label_HO) / self.sample_lens_HO) / 2
                logits_text = union_features @ self.adapter_union_weight.T
                if self.use_weight_pred:
                    logits_weights = self.weight_pred(torch.cat([human_features,object_features, union_features], dim=-1))
                    logits = logits_cache_HO * logits_weights[:, 0:1] + logits_text * logits_weights[:, 1:2]
                else:
                    logits = logits_cache_HO * self.logit_scale_HO + logits_text * self.logit_scale_text

            elif self.logits_type == 'HO+U':
                phi_union_HO = torch.cat([human_features, object_features], dim=-1) @ self.adapter_HO_weight.T + self.adapter_HO_bias
                phi_union_U = union_features @ self.adapter_U_weight.T + self.adapter_U_bias
                logits_cache_HO = ((phi_union_HO @ self.label_HO) / self.sample_lens_HO) / 2
                logits_cache_U = (phi_union_U @ self.label_U) / self.sample_lens_U
                if self.use_weight_pred:
                    logits_weights = self.weight_pred(torch.cat([human_features,object_features, union_features], dim=-1))
                    logits = logits_cache_HO * logits_weights[:, 0:1] + logits_cache_U * logits_weights[:, 1:2]
                else:
                    logits = logits_cache_HO * self.logit_scale_HO + logits_cache_U * self.logit_scale_U
            
            elif self.logits_type == 'HO':
                phi_union_HO = torch.cat([human_features, object_features], dim=-1) @ self.adapter_HO_weight.T + self.adapter_HO_bias
                logits_cache_HO = ((phi_union_HO @ self.label_HO) / self.sample_lens_HO) / 2
                if self.use_weight_pred:
                    logits_weights = self.weight_pred(torch.cat([human_features,object_features, union_features], dim=-1))
                    logits = logits_cache_HO * logits_weights[:, 0:1]
                else:
                    logits = logits_cache_HO * self.logit_scale_HO
            
            elif self.logits_type == 'U':
                phi_union_U = union_features @ self.adapter_U_weight.T + self.adapter_U_bias
                logits_cache_U = (phi_union_U @ self.label_U) / self.sample_lens_U
                if self.use_weight_pred:
                    logits_weights = self.weight_pred(torch.cat([human_features,object_features, union_features], dim=-1))
                    logits = logits_cache_U * logits_weights[:, 0:1]
                else:
                    logits = logits_cache_U * self.logit_scale_U
                
            elif self.logits_type == 'T':
                logits_text = union_features @ self.adapter_union_weight.T
                if self.use_weight_pred:
                    logits_weights = self.weight_pred(torch.cat([human_features,object_features, union_features], dim=-1))
                    logits = logits_text * logits_weights[:, 0:1]
                else:
                    logits = logits_text * self.logit_scale_text
            
            elif self.logits_type == 'H+T':
                logits_H = human_features @ self.adapter_H_weight.T + self.adapter_H_bias
                logits_text = union_features @ self.adapter_union_weight.T
                if self.use_weight_pred:
                    logits_weights = self.weight_pred(torch.cat([human_features,object_features, union_features], dim=-1))
                    logits = logits_H * logits_weights[:, 0:1] + logits_text * logits_weights[:, 1:2]
                else:
                    logits = logits_H * self.logit_scale_H + logits_text * self.logit_scale_text
            
            elif self.logits_type == 'H+O+T':
                # logits_H = human_features @ self.adapter_H_weight.T + self.adapter_H_bias
                # logits_O = object_features @ self.adapter_O_weight.T + self.adapter_O_bias
                logits_H = human_features @ self.adapter_H_weight @ self.adapter_union_weight.T
                logits_O = object_features @ self.adapter_O_weight @ self.adapter_union_weight.T
                logits_text = union_features @ self.adapter_union_weight.T
                if self.use_weight_pred:
                    logits_weights = self.weight_pred(torch.cat([human_features,object_features, union_features], dim=-1))
                    logits = logits_H * logits_weights[:, 0:1] + logits_O * logits_weights[:, 1:2] + logits_text * logits_weights[:, 2:]
                else:
                    logits = logits_H * self.logit_scale_H + logits_O * self.logit_scale_O + logits_text * self.logit_scale_text

            boxes_h_collated.append(x_keep)
            boxes_o_collated.append(y_keep)
            object_class_collated.append(labels[y_keep])
            prior_collated.append(self.compute_prior_scores(
                x_keep, y_keep, scores, labels)
            )
            all_logits.append(logits)
        
        if self.use_consistloss:
            pair_prior = torch.cat(pair_prior, dim=1).prod(0)
            x_, y_ = torch.nonzero(pair_prior).unbind(1)
            return all_logits, prior_collated, boxes_h_collated, boxes_o_collated, object_class_collated, gt_feats_collated, pair_feats_collated, gt_all_logits
        else:
            return all_logits, prior_collated, boxes_h_collated, boxes_o_collated, object_class_collated, gt_feats_collated, pair_feats_collated
          
    def recover_boxes(self, boxes, size):  
        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        h, w = size
        scale_fct = torch.stack([w, h, w, h])
        boxes = boxes * scale_fct
        return boxes

    def associate_with_ground_truth(self, boxes_h, boxes_o, targets): ## for training
        n = boxes_h.shape[0]
        labels = torch.zeros(n, self.num_classes, device=boxes_h.device)

        gt_bx_h = self.recover_boxes(targets['boxes_h'], targets['size'])
        gt_bx_o = self.recover_boxes(targets['boxes_o'], targets['size'])
        
        x, y = torch.nonzero(torch.min(
            box_iou(boxes_h, gt_bx_h),
            box_iou(boxes_o, gt_bx_o)
        ) >= self.fg_iou_thresh).unbind(1)

        # IndexError: tensors used as indices must be long, byte or bool tensors
        if self.num_classes == 117 or self.num_classes == 24 or self.num_classes == 407:
            labels[x, targets['labels'][y]] = 1  ## target['labels']: verb/action
        else:
            labels[x, targets['hoi'][y]] = 1

        return labels

    def compute_interaction_loss(self, boxes, bh, bo, logits, prior, targets, gt_feats, pair_feats,): ### loss
        ## bx, bo: indices of boxes
        labels = torch.cat([
            self.associate_with_ground_truth(bx[h], bx[o], target)
            for bx, h, o, target in zip(boxes, bh, bo, targets)
        ])

        if self.pseudo_label and self.zs_type =='unseen_verb':
            W = (self.text_embedding[torch.as_tensor(self.seen_verb_idxs)] @ self.text_embedding[torch.as_tensor(self.unseen_verb_idxs)].T).to(labels.device)
            W = W.T
            W /= W.norm(dim=1, keepdim=True) ## 20 * 97
            labels[:, torch.as_tensor(self.unseen_verb_idxs).to(labels.device)] = labels[:, torch.as_tensor(self.seen_verb_idxs).to(labels.device)] @ W.T
        
        prior = torch.cat(prior, dim=1).prod(0)
        x, y = torch.nonzero(prior).unbind(1)
        num_one_label = torch.sum(labels)
        logits = torch.cat(logits) 
        logits = logits[x, y]; prior = prior[x, y]; labels = labels[x, y]
        
        n_p = len(torch.nonzero(labels))
        # print("#(labels==1)", n_p)
        if n_p == 0:
            n_p = 1e9
            print("!!!!!!!!!!!!! QUICK HACK!!!!!!!!!!!!!!!!!")

        if dist.is_initialized():
            world_size = dist.get_world_size()
            n_p = torch.as_tensor([n_p], device='cuda')
            # n_p_distll = torch.as_tensor([n_p_distll], device='cuda')
            dist.barrier() 
            dist.all_reduce(n_p)
            n_p = (n_p / world_size).item()

            # dist.all_reduce(n_p_distll)
            # n_p_distll = (n_p_distll / world_size).item()
            # n_p = (n_p.true_divide(world_size)).item()
        
        loss = binary_focal_loss_with_logits(
            torch.log(
                prior / (1 + torch.exp(-logits) - prior) + 1e-8
            ), labels, reduction='sum',
            alpha=self.alpha, gamma=self.gamma
            )
        
        if self.use_distill:
            raise NotImplementedError
            # loss_feat = F.l1_loss(pair_feats, gt_feats,reduction='sum')/gt_feats.shape[1]
            loss_feat = torch.sum(3.0 - torch.diag(pair_feats @ gt_feats.t())) 
            return loss  / n_p + max((1-self.epoch * 0.05), 0) * loss_feat / n_p_distll
        else:
            return loss / n_p

    def prepare_region_proposals(self, results): ## √ detr extracts the human-object pairs
        region_props = []
        for res in results:
            sc, lb, bx = res.values()

            keep = batched_nms(bx, sc, lb, 0.5)
            sc = sc[keep].view(-1)
            lb = lb[keep].view(-1)
            bx = bx[keep].view(-1, 4)
            
            keep = torch.nonzero(sc >= self.box_score_thresh).squeeze(1)

            is_human = lb == self.human_idx
            hum = torch.nonzero(is_human).squeeze(1)
            obj = torch.nonzero(is_human == 0).squeeze(1)
            n_human = is_human[keep].sum(); n_object = len(keep) - n_human
            # Keep the number of human and object instances in a specified interval
            if n_human < self.min_instances:
                keep_h = sc[hum].argsort(descending=True)[:self.min_instances]
                keep_h = hum[keep_h]
            elif n_human > self.max_instances:
                keep_h = sc[hum].argsort(descending=True)[:self.max_instances]
                keep_h = hum[keep_h]
            else:
                keep_h = torch.nonzero(is_human[keep]).squeeze(1)
                keep_h = keep[keep_h]

            if n_object < self.min_instances:
                keep_o = sc[obj].argsort(descending=True)[:self.min_instances]
                keep_o = obj[keep_o]
            elif n_object > self.max_instances:
                keep_o = sc[obj].argsort(descending=True)[:self.max_instances]
                keep_o = obj[keep_o]
            else:
                keep_o = torch.nonzero(is_human[keep] == 0).squeeze(1)
                keep_o = keep[keep_o]

            keep = torch.cat([keep_h, keep_o])

            region_props.append(dict(
                boxes=bx[keep],
                scores=sc[keep],
                labels=lb[keep],
            ))

        return region_props

    def postprocessing(self, boxes, bh, bo, logits, prior, objects, image_sizes): ### √
        n = [len(b) for b in bh]
        logits = torch.cat(logits)
        logits = logits.split(n)

        detections = []
        for bx, h, o, lg, pr, obj, size,  in zip(
            boxes, bh, bo, logits, prior, objects, image_sizes,
        ):
            pr = pr.prod(0)
            x, y = torch.nonzero(pr).unbind(1)
            scores = torch.sigmoid(lg[x, y])
            
            detections.append(dict(
                boxes=bx, pairing=torch.stack([h[x], o[x]]),
                scores=scores * pr[x, y], labels=y,
                objects=obj[x], size=size
            ))

        return detections

    def get_obj_affordances(self, obj_labels, device):
        assert len(self.origin_text_embeddings) == 24 or len(self.origin_text_embeddings) == 117, "dim not match"
        bs = len(obj_labels); dim = self.origin_text_embeddings.shape[-1]
        verb_idxs = [self.object_class_to_target_class[obj] for obj in obj_labels]
        max_len = max([len(v) for v in verb_idxs])
        mask = torch.ones((bs, max_len),dtype=torch.bool,device=device)
        
        key = torch.zeros(bs, max_len, dim,device=device)
        for i in range(bs):
            cur_len = len(verb_idxs[i])
            key[i, :cur_len, :] = self.origin_text_embeddings[torch.as_tensor(verb_idxs[i])]
            mask[i, :cur_len] = False
        query = self.obj_affordance_query.unsqueeze(0).repeat(bs, 1, 1)
        obj_affordances, _ = self.obj_affordance_learner(query, key, key, key_padding_mask=mask)
        return obj_affordances

    def get_prior(self, region_props, image_size, prior_method): ##  for adapter module training
        if prior_method == 5:
            priors = self.priors_downproj(self.global_prior)
            mask = torch.zeros((len(region_props), priors.shape[0]), dtype=torch.bool, device=region_props[0]['boxes'].device)
            return priors.unsqueeze(0).repeat(len(region_props), 1, 1), mask
        max_feat = self.priors_initial_dim
        max_length = max(rep['boxes'].shape[0] for rep in region_props)
        mask = torch.ones((len(region_props),max_length),dtype=torch.bool,device=region_props[0]['boxes'].device)
        priors = torch.zeros((len(region_props),max_length, max_feat), dtype=torch.float32, device=region_props[0]['boxes'].device)
        img_h, img_w = image_size.unbind(-1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        
        for b_idx, props in enumerate(region_props):
            boxes = props['boxes'] / scale_fct[b_idx][None,:]
            scores = props['scores']
            labels = props['labels']
            is_human = labels == self.human_idx
            n_h = torch.sum(is_human); n = len(boxes)
            if n_h == 0 or n <= 1:
                print(n_h,n)
                # sys.exit()
            
            object_embs = self.object_embedding.to(labels.device)[labels]
            if self.obj_affordance:
                affordance_embs = self.get_obj_affordances(labels, region_props[0]['boxes'].device)
                object_embs = affordance_embs.squeeze(1)

            mask[b_idx,:n] = False
            
            if self.prior_type == 'cbe':
                priors[b_idx,:n,:5] = torch.cat((scores.unsqueeze(-1),boxes),dim=-1)
                priors[b_idx,:n,5:self.visual_output_dim+5] = object_embs
                # priors[b_idx,:n,512+5:] = unary_tokens
            elif self.prior_type == 'cb':
                priors[b_idx,:n,:5] = torch.cat((scores.unsqueeze(-1),boxes),dim=-1)
            elif self.prior_type == 'ce':
                priors[b_idx,:n,:1] = scores.unsqueeze(-1)
                priors[b_idx,:n,1:self.visual_output_dim+1] = object_embs
            elif self.prior_type == 'be':
                priors[b_idx,:n,:4] = boxes
                priors[b_idx,:n,4:self.visual_output_dim+4] = object_embs
            elif self.prior_type == 'c':
                priors[b_idx,:n,:1] = scores.unsqueeze(-1)
            elif self.prior_type == 'b':
                priors[b_idx,:n,:4] = boxes
            elif self.prior_type == 'e':
                priors[b_idx,:n,:self.visual_output_dim] = object_embs
            else:
                raise NotImplementedError

        if prior_method == 0:
            priors = self.priors_downproj(priors)
        elif prior_method == 1:
            pair_wise_priors = []
            for b_idx, props in enumerate(region_props):
                boxes = props['boxes'] / scale_fct[b_idx][None,:]
                scores = props['scores']
                labels = props['labels']
                is_human = labels == self.human_idx
                n_h = torch.sum(is_human); n = len(boxes)
                if n_h == 0 or n <= 1:
                    pair_wise_priors.append(torch.zeros(0, 0), )
                    print(n_h,n)
                    continue
                instance_wise_prior = priors[b_idx, :n]
                # Get the pairwise indices
                x, y = torch.meshgrid(
                    torch.arange(n, device=instance_wise_prior.device),
                    torch.arange(n, device=instance_wise_prior.device)
                )
                # Valid human-object pairs
                x_keep, y_keep = torch.nonzero(torch.logical_and(x != y, x < n_h)).unbind(1)
                if len(x_keep) == 0:
                    # Should never happen, just to be safe
                    raise ValueError("There are no valid human-object pairs")
                
                # extract single roi features
                sub_prior = instance_wise_prior[x_keep]
                obj_prior = instance_wise_prior[y_keep]
                
                pair_wise_priors.append(torch.cat((sub_prior, obj_prior), dim=-1))
            
            max_length = max(p.shape[0] for p in pair_wise_priors)
            mask = torch.ones((len(region_props),max_length),dtype=torch.bool,device=region_props[0]['boxes'].device)
            priors = torch.zeros((len(region_props),max_length, max_feat*2), dtype=torch.float32, device=region_props[0]['boxes'].device)
            for b_idx, props in enumerate(region_props):
                num_pair = pair_wise_priors[b_idx].shape[0]
                if num_pair > 0:
                    mask[b_idx, :num_pair] = False
                    priors[b_idx, :num_pair] = pair_wise_priors[b_idx]
            priors = self.priors_downproj(priors)   
        elif prior_method == 2:
            priors = self.learnable_prior.unsqueeze(0).repeat(len(region_props), 1, 1)
            mask = torch.zeros((priors.shape[0], priors.shape[1]), dtype=torch.bool,device=region_props[0]['boxes'].device)
        elif prior_method == 3:
            assert self.prompt_learning
            priors = self.priors_downproj(self.clip_head.prompt_learner.ctx).unsqueeze(0).repeat(len(region_props), 1, 1)
            mask = torch.zeros((priors.shape[0], priors.shape[1]), dtype=torch.bool,device=region_props[0]['boxes'].device)
        elif prior_method == 6:
            priors = self.priors_downproj(priors)
        return (priors, mask)
    
    def prepare_target_hois(self, targets, device):
        unique_hois, cnt = {}, 0
        tgt_ids = []
        for t in targets:
            for hoi in t["hoi"]:
                hoi_id = hoi.item()
                if self.training:
                    # Only consider the texts within each mini-batch
                    if hoi_id not in unique_hois:
                        unique_hois[hoi_id] = cnt
                        cnt += 1
                    tgt_ids.append(unique_hois[hoi_id])
                else:
                    # Consider all hois in the dataset
                    tgt_ids.append(hoi_id)
        tgt_ids = torch.as_tensor(tgt_ids, dtype=torch.int64, device=device)
        return unique_hois
    
    def forward(self,
        images: List[Tensor],
        targets: Optional[List[dict]] = None
    ) -> List[dict]:
        """
        Parameters:
        -----------
        images: List[Tensor]
            Input images in format (C, H, W)
        targets: List[dict], optional
            Human-object interaction targets

        Returns:
        --------
        results: List[dict]
            Detected human-object interactions. Each dict has the following keys:
            `boxes`: torch.Tensor
                (N, 4) Bounding boxes for detected human and object instances
            `pairing`: torch.Tensor
                (2, M) Pairing indices, with human instance preceding the object instance
            `scores`: torch.Tensor
                (M,) Interaction score for each pair
            `labels`: torch.Tensor
                (M,) Predicted action class for each pair
            `objects`: torch.Tensor
                (M,) Predicted object class for each pair
            `attn_maps`: list
                Attention weights in the cooperative and competitive layers
            `size`: torch.Tensor
                (2,) Image height and width
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        batch_size = len(images)
        images_orig = [im[0].float() for im in images]
        images_clip = [im[1] for im in images]
        device = images_clip[0].device
        image_sizes = torch.as_tensor([
            im.size()[-2:] for im in images_clip
        ], device=device)
        image_sizes_orig = torch.as_tensor([
            im.size()[-2:] for im in images_orig
            ], device=device)
        
        if isinstance(images_orig, (list, torch.Tensor)):
            images_orig = nested_tensor_from_tensor_list(images_orig)
        features, pos = self.detector.backbone(images_orig)
        src, mask = features[-1].decompose()
        # assert mask is not None2
        hs, detr_memory = self.detector.transformer(self.detector.input_proj(src), mask, self.detector.query_embed.weight, pos[-1])
        outputs_class = self.detector.class_embed(hs) # 6x8x100x81 or 6x8x100x92
        outputs_coord = self.detector.bbox_embed(hs).sigmoid() # 6x8x100x4 
        if self.dataset == 'vcoco' and outputs_class.shape[-1] == 92:
            outputs_class = outputs_class[:, :, :, self.reserve_indices]
            assert outputs_class.shape[-1] == 81, 'reserved shape NOT match 81'
        
        results = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        results = self.postprocessor(results, image_sizes)

        region_props = self.prepare_region_proposals(results)
        priors = None
        if self.use_insadapter:
            priors = self.get_prior(region_props,image_sizes, self.prior_method) ## priors: (prior_feat, mask): (batch_size*14*64, batch_size*14)
        
        if self.prior_method == 6: # update priors with the global prior
            global_prior = self.priors_downproj_global(self.global_prior)
            global_prior = global_prior.unsqueeze(0).repeat(batch_size, 1, 1)
            prior_content = self.fusion_attn(tgt=priors[0].transpose(0, 1), memory=global_prior.transpose(0, 1))
            # prior_content, _ = self.fusion_attn(priors[0].transpose(0, 1), global_prior.transpose(0, 1), global_prior.transpose(0, 1))
            prior_content = prior_content.transpose(0, 1)
            priors = (prior_content, priors[1])
        # with amp.autocast(enabled=True):
        images_clip = nested_tensor_from_tensor_list(images_clip)
        # 8x512, 8x512x14x14
        feat_global, feat_local = self.clip_head.image_encoder(images_clip.decompose()[0], priors)
        if self.use_mlp_proj:
            feat_local = feat_local.permute(0,2,3,1) # 8x14x14x512
            feat_local = self.mlp_proj(feat_local)
            feat_local = feat_local.permute(0,3,1,2)
        if self.tpt:
            interaction_loss = self.compute_loss_tpt(feat_local, image_sizes, region_props, targets)
            loss_dict = dict(
                interaction_loss=interaction_loss
            )
            return loss_dict
        
        if self.use_consistloss:
            logits, prior, bh, bo, objects, gt_feats, pair_feats, gt_all_logits = self.compute_roi_embeddings(feat_local, image_sizes, region_props)
        else:
            logits, prior, bh, bo, objects, gt_feats, pair_feats = self.compute_roi_embeddings(feat_local, image_sizes, region_props, feat_global)
            gt_all_logits = None
        boxes = [r['boxes'] for r in region_props] 
        
        if self.training:
            interaction_loss = self.compute_interaction_loss(boxes, bh, bo, logits, prior, targets, gt_feats, pair_feats)
            if self.vision_regularize:
                origin_feat_global, origin_feat_local = self.origin_clip_head.encode_image(images_clip.decompose()[0])
                vis_regulazizer_loss = nn.MSELoss(reduce=True, size_average=True)(feat_local, origin_feat_local)
                interaction_loss = interaction_loss + vis_regulazizer_loss
            loss_dict = dict(
                interaction_loss=interaction_loss
            )
            # if interaction_loss.isnan():
            if self.language_aware:
                self.origin_text_embeddings = self.origin_text_embeddings.to(self.adapter_union_weight.device)
                # language_aware_loss = (1 - torch.diagonal((self.adapter_union_weight / self.adapter_union_weight.norm(dim=-1, keepdim=True)) @ self.origin_text_embeddings.T)).mean()
                sim_matrix = (self.adapter_union_weight / self.adapter_union_weight.norm(dim=-1, keepdim=True)) @ self.origin_text_embeddings.T
                language_aware_loss = nn.CrossEntropyLoss()(sim_matrix, torch.arange(sim_matrix.shape[0]).to(self.adapter_union_weight.device))
                loss_dict['language_aware_loss'] = self.LA_weight * language_aware_loss
            return loss_dict
        if len(logits) == 0:
            print(targets)
            return None
        detections = self.postprocessing(boxes, bh, bo, logits, prior, objects, image_sizes)
        return detections


def get_multi_prompts(classnames):   ## https://github.com/openai/CLIP/blob/main/data/prompts.md, 
    templates = ['a photo of a person {}.',
                'a video of a person {}.',
                'a example of a person {}.',
                'a demonstration of a person {}.',
                'a photo of the person {}.',
                'a video of the person {}.',
                'a example of the person {}.', 
                'a demonstration of the person {}.',
                ]
    hico_texts = [' '.join(name.split(' ')[5:]) for name in classnames]
    all_texts_input = []
    for temp in templates:
        texts_input = torch.cat([clip.tokenize(temp.format(text)) for text in hico_texts ])
        all_texts_input.append(texts_input)
    all_texts_input = torch.stack(all_texts_input,dim=0)
    return all_texts_input

@torch.no_grad()
def get_origin_text_emb(args, clip_model, tgt_class_names, obj_class_names):
    use_templates = args.use_templates
    if use_templates == False:
        text_inputs = torch.cat([clip.tokenize(classname) for classname in tgt_class_names])
    elif use_templates:
        text_inputs = get_multi_prompts(tgt_class_names)
        bs_t, nums, c = text_inputs.shape
        text_inputs = text_inputs.view(-1, c)

    with torch.no_grad():
        origin_text_embedding = clip_model.encode_text(text_inputs)
    if use_templates:
        origin_text_embedding = origin_text_embedding.view(bs_t, nums, -1).mean(0)

    origin_text_embedding = origin_text_embedding / origin_text_embedding.norm(dim=-1, keepdim=True) # text embeddings of hoi 117*512 or 600*512

    obj_text_inputs = torch.cat([clip.tokenize(obj_text) for obj_text in obj_class_names])
    with torch.no_grad():
        obj_text_embedding = clip_model.encode_text(obj_text_inputs)
        object_embedding = obj_text_embedding
        # obj_text_embedding = obj_text_embedding[hoi_obj_list,:]
    return origin_text_embedding, object_embedding


def build_detector(args, class_corr, object_n_verb_to_interaction, clip_model_path, num_anno, verb2interaction=None):
    detr, _, postprocessors = build_model(args)
    if os.path.exists(args.pretrained):
        if dist.get_rank() == 0:
            print(f"Load weights for the object detector from {args.pretrained}")
        if 'e632da11' in args.pretrained:
            detr.load_state_dict(torch.load(args.pretrained, map_location='cpu')['model']) 
        else:
            detr.load_state_dict(torch.load(args.pretrained, map_location='cpu')['model_state_dict'])
    
    clip_state_dict = torch.load(clip_model_path, map_location="cpu").state_dict()
    clip_model = CLIP_models_adapter_prior2.build_model(state_dict=clip_state_dict, use_adapter=args.use_insadapter, 
                                        adapter_pos=args.adapter_pos, adapter_num_layers=args.adapter_num_layers,
                                        use_text_adapter=args.use_text_adapter, bottleneck=args.bottleneck,)
    if args.num_classes == 117:
        classnames = hico_verbs_sentence
    elif args.num_classes == 24:
        classnames = vcoco_verbs_sentence
    elif args.num_classes == 600:
        classnames = list(hico_text_label.hico_text_label.values())
    else:
        raise NotImplementedError
    model = CustomCLIP(args, classnames=classnames, clip_model=clip_model)

    obj_class_names = [obj[1] for obj in hico_text_label.hico_obj_text_label]

    origin_text_embeddings, object_embedding = get_origin_text_emb(args, clip_model=clip_model, tgt_class_names=classnames, obj_class_names=obj_class_names)
    origin_text_embeddings = origin_text_embeddings.clone().detach()
    object_embedding = object_embedding.clone().detach()
    
    detector = UPT(args,
        detr, postprocessors['bbox'], model, origin_text_embeddings, object_embedding,
        human_idx=args.human_idx, num_classes=args.num_classes,
        alpha=args.alpha, gamma=args.gamma,
        box_score_thresh=args.box_score_thresh,
        fg_iou_thresh=args.fg_iou_thresh,
        min_instances=args.min_instances,
        max_instances=args.max_instances,
        object_class_to_target_class=class_corr,
        object_n_verb_to_interaction=object_n_verb_to_interaction,
        num_anno = num_anno,
        # verb2interaction = verb2interaction,
        use_mlp_proj = args.use_mlp_proj,
        featmap_dropout_rate=args.featmap_dropout_rate,
    )
    return detector

