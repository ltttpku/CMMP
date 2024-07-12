# train
python main_tip_finetune.py --world-size 2 --pretrained checkpoints/detr-r50-hicodet.pth --output-dir checkpoints/camera_UO_1 --use_insadapter --num_classes 117 --file1 hicodet_pkl_files/union_embeddings_cachemodel_crop_padding_zeros_vitb16.p --clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt --prompt_learning --LA --zs --zs_type unseen_object

# test
python main_tip_finetune.py --world-size 1 --pretrained checkpoints/detr-r50-hicodet.pth --output-dir checkpoints/camera_UO_1 --use_insadapter --num_classes 117 --file1 hicodet_pkl_files/union_embeddings_cachemodel_crop_padding_zeros_vitb16.p --clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt --prompt_learning --LA --zs --zs_type unseen_object --eval --resume checkpoints/camera_UO_1/ckpt_29205_15.pt


# train w/ scaled-up version
python main_tip_finetune.py --world-size 2 --pretrained checkpoints/detr-r50-hicodet.pth --output-dir [OUTPUT_DIR] --use_insadapter --num_classes 117 --file1 hicodet_pkl_files/hicodet_union_embeddings_cachemodel_crop_padding_zeros_vit336.p --clip_dir_vit checkpoints/pretrained_clip/ViT-L-14-336px.pt --prompt_learning --LA --zs --zs_type unseen_object

# test w/ scaled-up version
python main_tip_finetune.py --world-size 1 --pretrained checkpoints/detr-r50-hicodet.pth --output-dir [OUTPUT_DIR] --use_insadapter --num_classes 117 --file1 hicodet_pkl_files/hicodet_union_embeddings_cachemodel_crop_padding_zeros_vit336.p --clip_dir_vit checkpoints/pretrained_clip/ViT-L-14-336px.pt --prompt_learning --LA --zs --zs_type unseen_object --eval --resume checkpoints/camera_UO_1_L/ckpt_29205_15.pt
