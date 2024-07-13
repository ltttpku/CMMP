# train 
python main_tip_finetune.py --world-size 2 --dataset vcoco --data-root vcoco/ --partitions trainval test --pretrained checkpoints/detr-r50-vcoco.pth --output-dir checkpoints/camera_vcoco --use_insadapter --num_classes 24 --file1 vcoco_pkl_files/vcoco_union_embeddings_cachemodel_crop_padding_zeros_vit16.p --clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt

# cache results
python main_tip_finetune.py --world-size 1 --dataset vcoco --data-root vcoco/ --partitions trainval test --pretrained checkpoints/detr-r50-vcoco.pth --output-dir checkpoints/camera_vcoco --use_insadapter --num_classes 24 --file1 vcoco_pkl_files/vcoco_union_embeddings_cachemodel_crop_padding_zeros_vit16.p --clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt --cache --resume [CKPT]

# scaled-up train
python main_tip_finetune.py --world-size 2 --dataset vcoco --data-root vcoco/ --partitions trainval test --pretrained checkpoints/detr-r50-vcoco.pth --output-dir checkpoints/camera_vcoco_L --use_insadapter --num_classes 24 --file1 vcoco_pkl_files/vcoco_union_embeddings_cachemodel_crop_padding_zeros_vit336.p --clip_dir_vit checkpoints/pretrained_clip/ViT-L-14-336px.pt

# cache 
python main_tip_finetune.py --world-size 1 --dataset vcoco --data-root vcoco/ --partitions trainval test --pretrained checkpoints/detr-r50-vcoco.pth --output-dir checkpoints/camera_vcoco_L --use_insadapter --num_classes 24 --file1 vcoco_pkl_files/vcoco_union_embeddings_cachemodel_crop_padding_zeros_vit336.p --clip_dir_vit checkpoints/pretrained_clip/ViT-L-14-336px.pt --cache --resume [CKPT]