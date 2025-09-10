# CLIPer: Hierarchically Improving Spatial Representation of CLIP for Open-Vocabulary Semantic Segmentation (ICCV 2025)

This repo is the official pytorch implementation of the  [CLIPer](https://arxiv.org/abs/2411.13836).

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/sunlin449/CLIPer)
[![Leaderboard](https://img.shields.io/badge/ModelScope-Space%20with%20GPU-orange)](https://modelscope.cn/studios/sunlin449/CLIPer/)
[![Leaderboard](https://img.shields.io/badge/ModelScope-Model-green)](https://modelscope.cn/models/sunlin449/CLIPer)

## :fire: News
- We have incorporated a sliding window method into the code.
- A multi-GPU version has been released (by  running ```sh_ovs_mgp.sh```), along with new results on several widely adopted benchmark datasets.
- We are pleased to announce that this work has been accepted by ICCV 2025.
- We have released a gradio demo, and you can find it in [huggingface](https://huggingface.co/spaces/sunlin449/CLIPer) with CPU 
and [ModelScape](https://modelscope.cn/studios/sunlin449/CLIPer/) with GPU or in you device by running ```app.py```.
- We have released the sourse code of CLIPer (all the models needed will be downloaded automatically when running the code).

## Introduction
<img width="100%" src="assets/00framework.png">

- We introduce the early-layer fusion, including patch embeddings and attention maps of early layer to the final layer of CLIP for improving spatial representation.
- We further introduce  the fine-grained compensation, utilizing the detailed spacial information in the self-attention maps of Stable Diffusion to obtain the precise segmentation results.
- The proposed CLIPer achieve the state-of-the-art performance on multiple datasets in open-vocabulary semantic segmentation tasks. our proposed CLIPer obtains mIoU scores of 69.8% on VOC and 43.3% on Object when using ViT-L/14 backbone.

For further details, please check out our [paper](https://arxiv.org/abs/2411.13836).

## Installation
Please follow the code bellow to create the environment
```
conda create -n CLIPer python=3.9
conda activate CLIPer
pip install -r requirements.txt
```
## Data Preparation
Please struct the datasets as follows
```none
datasets
├── ADEchallengeData2016
│   ├── images
│   │   ├── training
│   │   │   ├── ADE_train_00000001.jpg
│   │   ├── validation
│   │   │   ├── ADE_val_00000001.jpg
│   ├── annotations
│   │   ├── training
│   │   │   ├── ADE_train_00000001.png
│   │   ├── validation
│   │   │   ├── ADE_val_00000001.png
├── CityScapes
│   ├── gtFine
│   │   ├── train
│   │   ├── test
│   │   ├── val
│   │   │   ├── frankfurt
│   │   │   │   ├── frankfurt_000000_000294_gtFine_labelTrainIds.png
│   │   │   │   ├── ...
│   │   │   ├── lindau
│   │   │   ├── munster
├── ├── leftImg8bit
│   │   ├── train
│   │   ├── test
│   │   ├── val
│   │   │   ├── frankfurt
│   │   │   │   ├── frankfurt_000000_000294_leftImg8bit.png
│   │   │   │   ├── ...
│   │   │   ├── lindau
│   │   │   ├── munster
├── coco2014
│   ├── train2014
│   │   ├── COCO_train2014_000000000009.jpg
│   ├── val2014
│   │   ├── COCO_val2014_000000000042.jpg
│   ├── coco_seg_anno
│   │   ├── 000000000009.png
├── coco2017
│   ├── train2017
│   │   ├── 000000000009.jpg
│   ├── val2017
│   │   ├── 000000000139.jpg
│   ├── stuff_anno164
│   │   ├── train2017
│   │   │   ├── 000000000009.png
│   │   ├── val2017
│   │   │   ├── 000000000139.png
├── VOCdevkit
│   ├── VOC2010
│   │   ├── JPEGImages
│   │   │   ├── 2007_000027.jpg
│   │   ├── SegmentationClassContext
│   │   │   ├──2008_000002.png
│   ├── VOC2012
│   │   ├── JPEGImages
│   │   │   ├── 2007_000027.jpg
│   │   ├── SegmentationClassAug
│   │   │   ├──2007_000032.png
```
## Evaluation
To evaluate our CLIPer, please enter the scripts folder and run the code
```
# select the config file to evaluate the code
# evaluate voc dataset with background
sh sh_ovs.sh ../scripts/config/vit-l-14/ovs_voc21.yaml
# evaluate voc dataset without background
sh sh_ovs.sh ../scripts/config/vit-l-14/ovs_voc20.yaml
# ...
```
## Results
Run the code in this repo, you should get similar results (* means using sliding widow, and values in parentheses indicate results obtained without applying FGC) in the following table:

| Encoder  | VOC  |Context|Object |VOC20 |Contex59 |Stuff |ADE |CITY |
|  :----:  |  :----:  |  :----:  |  :----:  |  :----:  |  :----:  |  :----:  |  :----:  |  :----:  |
| ViT-B/16 | 66.4(60.7) |37.5(34.7) | 39.2(36.3) | 85.5(84.3) | 41.6(38.4) | 27.6(25.4) | 21.5(19.8) | 37.0(35.9) |
| ViT-B/16* | 66.5(62.1) |38.3(35.7) | 40.0(37.5) | 86.0(85.0) | 42.4(39.6) | 28.6(26.4) | 22.0(20.6) | 38.7(38.3) |
| ViT-B/16 | 71.6(62.5) |39.0(34.8) | 44.4(40.4) | 90.3(88.7) | 44.0(39.6) | 29.7(26.3) | 24.5(22.1) | 41.6(37.9) |
| ViT-B/16* | 72.2(64.0) |39.5(35.7) | 44.7(41.4) | 89.8(89.0) | 44.6(40.9) | 30.4(27.1) | 25.0(22.9) | 42.5(39.4) |

## Visualization
<img width="100%" src="assets/visualization.png">

## Citation
```
@misc{Sun_2024_CLIPer,
      title={CLIPer: Hierarchically Improving Spatial Representation of CLIP for Open-Vocabulary Semantic Segmentation}, 
      author={Lin Sun and Jiale Cao and Jin Xie and Xiaoheng Jiang and Yanwei Pang},
      year={2024},
      eprint={2411.13836},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.13836}, 
}
```