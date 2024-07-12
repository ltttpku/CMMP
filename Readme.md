## [ECCV 2024] Exploring Conditional Multi-Modal Prompts for Zero-shot HOI Detection

### Dataset 
Follow the process of [UPT](https://github.com/fredzzhang/upt).

The downloaded files should be placed as follows. Otherwise, please replace the default path to your custom locations.
```
|- CMMP
|   |- hicodet
|   |   |- hico_20160224_det
|   |       |- annotations
|   |       |- images
|   |- vcoco
|   |   |- mscoco2014
|   |       |- train2014
|   |       |-val2014
:   :      
```

### Dependencies
1. Follow the environment setup in [UPT](https://github.com/fredzzhang/upt).

2. Our code is built upon [CLIP](https://github.com/openai/CLIP). Install the local package of CLIP:
```
cd CLIP && python setup.py develop && cd ..
```

3. Download the CLIP weights to `checkpoints/pretrained_clip`.
```
|- CMMP
|   |- checkpoints
|   |   |- pretrained_clip
|   |       |- ViT-B-16.pt
|   |       |- ViT-L-14-336px.pt
:   :      
```

4. Download the weights of DETR and put them in `checkpoints/`.


| Dataset | DETR weights |
| --- | --- |
| HICO-DET | [weights](https://drive.google.com/file/d/1BQ-0tbSH7UC6QMIMMgdbNpRw2NcO8yAD/view?usp=sharing)  |
| V-COCO | [weights](https://drive.google.com/file/d/1AIqc2LBkucBAAb_ebK9RjyNS5WmnA4HV/view?usp=sharing) |


```
|- CMMP
|   |- checkpoints
|   |   |- detr-r50-hicodet.pth
|   |   |- detr-r50-vcoco.pth
:   :   :
```

### Pre-extracted Features
Download the pre-extracted features from [HERE](https://drive.google.com/file/d/1lUnUQD3XcWyQdwDHMi74oXBcivibGIWN/view?usp=sharing) and the pre-extracted bboxes from [HERE](https://drive.google.com/file/d/19Mo1d4J6xX9jDNvDJHEWDpaiPKxQHQsT/view?usp=sharing). The downloaded files have to be placed as follows.

```
|- CMMP
|   |- hicodet_pkl_files
|   |   |- union_embeddings_cachemodel_crop_padding_zeros_vitb16.p
|   |   |- hicodet_union_embeddings_cachemodel_crop_padding_zeros_vit336.p
|   |- vcoco_pkl_files
|   |   |- vcoco_union_embeddings_cachemodel_crop_padding_zeros_vit16.p
|   |   |- vcoco_union_embeddings_cachemodel_crop_padding_zeros_vit336.p
:   :      
```

### Train/Test

Please follow the commands in ```./scripts```.



### Model Zoo

| Dataset |  Backbone  | mAP | Rare | Non-rare | Weights |
| ---- |  ----  | ----  | ----  | ----  | ----  |
| HICO-DET | ResNet-50+ViT-B  | 33.80 | 31.72 | 34.42 | [weights](https://drive.google.com/file/d/1utTPqQkDIvlNhDzAs8mhoSN7FMQjBToH/view?usp=sharing) |
| HICO-DET |ResNet-50+ViT-L  | 38.40 | 37.52 | 38.66 | [weights](https://drive.google.com/file/d/1JqX61ZSDXmDuLz4DPavK3aa1ISG7W8Dj/view?usp=sharing) |


| Dataset |  Backbone  | Scenario 1 | Scenario 2 | Weights |
| ---- |  ----  | ----  | ----  | ----  |
|V-COCO| ResNet-50+ViT-B  | 56.12 | 61.45 | [weights](https://drive.google.com/file/d/13WiXzP08MKSMD-jZrtIpWcyFa7zYXnRE/view?usp=sharing) |
|V-COCO| ResNet-50+ViT-L  | 58.57 | 63.97 | [weights](https://drive.google.com/file/d/1amqgWOPjC8mlHMrmoZj6YzxCFBPLUeww/view?usp=sharing) |


## Citation
If you find our paper and/or code helpful, please consider citing:
```
@article{ting2024CMMP,
  title={Exploring Conditional Multi-Modal Prompts for Zero-shot HOI Detection},
  author={Ting Lei and Shaofeng Yin and Yuxin Peng and Yang Liu},
  year={2024},
  booktitle={ECCV},
  organization={IEEE},
}
```


