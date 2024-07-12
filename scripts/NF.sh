# train
python main_tip_finetune.py --world-size 2 --pretrained checkpoints/detr-r50-hicodet.pth --output-dir checkpoints/camera_NF_1 --use_insadapter --num_classes 117 --file1 hicodet_pkl_files/union_embeddings_cachemodel_crop_padding_zeros_vitb16.p --clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt --prompt_learning --LA --zs --zs_type non_rare_first

# test
python main_tip_finetune.py --world-size 1 --pretrained checkpoints/detr-r50-hicodet.pth --output-dir checkpoints/camera_NF_1 --use_insadapter --num_classes 117 --file1 hicodet_pkl_files/union_embeddings_cachemodel_crop_padding_zeros_vitb16.p --clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt --prompt_learning --LA --zs --zs_type non_rare_first --eval --resume checkpoints/camera_NF_1/ckpt_15585_15.pt


# train w/ scaled-up version
python main_tip_finetune.py --world-size 2 --pretrained checkpoints/detr-r50-hicodet.pth --output-dir [OUTPUT_DIR] --use_insadapter --num_classes 117 --file1 hicodet_pkl_files/hicodet_union_embeddings_cachemodel_crop_padding_zeros_vit336.p --clip_dir_vit checkpoints/pretrained_clip/ViT-L-14-336px.pt --prompt_learning --LA --zs --zs_type non_rare_first

# test w/ scaled-up version
python main_tip_finetune.py --world-size 1 --pretrained checkpoints/detr-r50-hicodet.pth --output-dir [OUTPUT_DIR] --use_insadapter --num_classes 117 --file1 hicodet_pkl_files/hicodet_union_embeddings_cachemodel_crop_padding_zeros_vit336.p --clip_dir_vit checkpoints/pretrained_clip/ViT-L-14-336px.pt --prompt_learning --LA --zs --zs_type non_rare_first --eval --resume checkpoints/camera_NF_1_L/ckpt_15585_15.pt