# Efficient Segmentation Learning

## Introduction

- Efficient Segmentation Learning is an open source, PyTorch-based segmentation method for 3D medical image.

## Installation

#### Environment

- Ubuntu 16.04.12
- Python 3.6+
- Pytorch 1.5.0+
- CUDA 10.0+

1.Git clone

```
git clone https://github.com/Shanghai-Aitrox-Technology/EfficientSegLearning.git
```

2.Install Nvidia Apex

- Perform the following command

```shell
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir ./
```

3.Install dependencies

```shell
pip install -r requirements.txt
```

## Training
Please refer to 'FlareSemiSeg/README.md'.

## Inference

Run inference pipeline.

```shell
sh predict.sh
```
