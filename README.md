# Beluga Whale Detection from Satellite Imagery with Point Labels

---

<h4 align="center">    
    <em>
        <a href="https://voyagerxvoyagerx.github.io/" target="_blank">Yijie Zheng</a>, &nbsp; &nbsp; 
        Jinxuan Yang, &nbsp; &nbsp; 
        Yu Chen, &nbsp; &nbsp; 
        Yaxuan Wang,
    </em></h4>

<h4 align="center"><em>Yihang Lu, &nbsp; &nbsp; Guoqing Li✉</em></h4>

<h4 align="center"><em>University of Chinese Academy of Sciences</em></h4>

<p align="center">
<img src="https://github.com/user-attachments/assets/aec9b438-ef34-473e-bc4f-bb3d4d334292" alt="institute" style="width:500px; display: block; margin: 0 auto;">
</p>

[**News**]: Our paper ([**arXiv**](https://arxiv.org/abs/2505.12066)) is accepted for oral presentation at IGARSS 2025! Welcome to meet us at ([session](https://www.2025.ieeeigarss.org/view_paper.php?PaperNum=2430&SessionID=1426)) ☺️

---
- [Beluga Whale Detection from Satellite Imagery with Point Labels](#beluga-whale-detection-from-satellite-imagery-with-point-labels)
  - [Installation](#installation)
  - [Model Checkpoints](#model-checkpoints)
  - [Getting Start](#getting-start)
  - [Preprocessing](#preprocessing)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Acknowledgements](#acknowledgements)

Our work detects beluga whales in the Arctic Region from very high resolution satellite imagery with manually created point labels.
<p align="center">
<img src="https://github.com/user-attachments/assets/da736f8a-86b5-4549-a209-2c2f8f8602fd" alt="annotation_process" style="width:800px; display: block; margin: 0 auto;">
</p>

We create bounding box labels from point labels using SAM, significantly improved annotaion efficiency.
<p align="center">
<img src="https://github.com/user-attachments/assets/616a7d01-c044-4561-a171-70a10c5e3a2a" alt="annotation_process" style="width:500px; display: block; margin: 0 auto;">
</p>

Based on the high-quality automated annotation process, the YOLOv8s model trained on SAM annotated data (YOLO-SAM) produces bbounding boxes that fits the actual whales shape.
<p align="center">
<img src="https://github.com/user-attachments/assets/1ae52be7-bc7b-4b57-8893-a3e00b31d2c3" alt="results" style="width:800px; display: block; margin: 0 auto;">
</p>

YOLO-SAM significantly surpasses the model trained on point labels (YOLO-Buffer) in terms of precision and $F_1$ score, and almost matches the model trained on human-refined bounding box labels (YOLO-Box).
<p align="center">
<img src="https://github.com/user-attachments/assets/343da7b7-336d-4672-863c-eadcd900461e" alt="results" style="width:800px; display: block; margin: 0 auto;">
</p>

Our YOLO-SAM could distinguish whales from seals in high accuracy.
<p align="center">
<img src="https://github.com/user-attachments/assets/5f70ed21-156e-48c3-b55b-dae6a1b7d63a" alt="results" style="width:800px; display: block; margin: 0 auto;">
</p>

## Installation
Please install [PyTorch](https://pytorch.org/get-started/previous-versions/) first. `torch<2.1.0` is recomended for compatibility with other dependencies.
This repo is developed using `python 3.10` and `torch 2.0.0+cu118`
```
conda create -n belugaSeeker python=3.10
conda activate belugaSeeker
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```
For training and inference, please install the mmyolo codebase.
```
pip install openmim
mim install "mmengine>=0.6.0" "mmcv>=2.0.0rc4,<2.1.0" "mmdet>=3.0.0rc6,<3.1.0" "mmyolo"
```

The following depencdencies are required for data preprocess and annotation.
```
pip install rasterio geopandas
pip install git+https://github.com/facebookresearch/segment-anything.git
```

## Model Checkpoints
The checkpoints for [SAM](https://github.com/facebookresearch/segment-anything#:~:text=for%20more%20details.-,Model%20Checkpoints,-Three%20model%20versions) can be downloaded via [SAM-ViT-H](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth).

Click the links below to download the checkpoints for beluga whale detector:

- [YOLO-SAM](https://github.com/VoyagerXvoyagerx/beluga-seeker/releases/download/detect-from-sat-img/YOLO-SAM.pth)
- [YOLO-Buffer](https://github.com/VoyagerXvoyagerx/beluga-seeker/releases/download/detect-from-sat-img/YOLO-Buffer.pth)

## Getting Start
It's easy to get started with inference code. We provide [inference_demo.ipynb](/inference_demo.ipynb) to help you get started.

## Preprocessing
Please refer to [crop.ipynb](/annotation_tools/crop.ipynb) for cropping the images and [create_whale_masks.ipynb](/annotation_tools/create_whale_masks.ipynb) for generating whale masks and bounding boxes.

## Training
The file structure follows the mmyolo convention. Use the command below to specify the GPU devices. Automatic mixed precision training is enabled to accelerate training while maintaining the performance.
```
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs_beluga/yolov8_s_b24-100e.py --amp
```
## Evaluation
Please refer to [evaluate.ipynb](/evaluate.ipynb) for evaluating on all the metrics.

## Acknowledgements
This work is supported by the [4th IEEE GRSS Student Challenge](https://www.grss-ieee.org/community/groups-initiatives/ieee-grss-student-grand-challenge).
The authors gratefully acknowledge Amou for the assistance with ArcGIS Pro, and Zori for dilivering the data.

This project uses the following open source libraries:
- [MMYOLO](https://github.com/open-mmlab/mmyolo)
- [segment-anything](https://github.com/facebookresearch/segment-anything)
