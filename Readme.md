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

| Method          | Type  | Unseen↑ | Seen↑ | Full↑ | HM↑   |
|-----------------|-------|---------|-------|-------|-------|
<!-- | CMMP (Ours)     | UC    | 29.60   | 32.39 | 31.84 | 30.93 |
| CMMP† (Ours)    | UC    | 34.46   | 37.15 | 36.56 | 35.75 | -->
| CMMP (Ours)     | RF-UC | 29.45   | 32.87 | 32.18 | 31.07 |
| CMMP† (Ours)    | RF-UC | 35.98   | 37.42 | 37.13 | 36.69 |
| CMMP (Ours)     | NF-UC | 32.09   | 29.71 | 30.18 | 30.85 |
| CMMP† (Ours)    | NF-UC | 33.52   | 35.53 | 35.13 | 34.50 |
| CMMP (Ours)     | UO    | 33.76   | 31.15 | 31.59 | 32.40 |
| CMMP† (Ours)    | UO    | 39.67   | 36.15 | 36.74 | 37.83 |
| CMMP (Ours)     | UV    | 26.23   | 32.75 | 31.84 | 29.13 |
| CMMP† (Ours)    | UV    | 30.84   | 37.28 | 36.38 | 33.75 |


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


