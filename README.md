# POSEC3D_v2
This repository is intended for conducting experiments related to action recognition problems based on PoseC3D. The main focus of this research is to investigate the effectiveness of PoseC3D in recognizing human actions and movements.

## Supported Algorithms
- [x] [PoseConv3D (CVPR 2022 Oral)](https://arxiv.org/abs/2104.13586)

## Supported Skeleton Datasets
- [x] [NTURGB+D (CVPR 2016)](https://arxiv.org/abs/1604.02808) and [NTURGB+D 120 (TPAMI 2019)](https://arxiv.org/abs/1905.04757)

## Installation
```shell
git clone https://github.com/CMC-ATI-IOT/ACTION_HOANGG.git
# Please first install pytorch according to instructions on the official website: https://pytorch.org/get-started/locally/. Please use pytorch with version smaller than 1.11.0 and larger (or equal) than 1.5.0
pip install -r requirements.txt
pip install -e .
```

## Training and validating
```shell
python train.py --config-file
#The configuration file can be found in the "configs" folder.
#
```
Feel free to make adjustments to the parameters and hyperparameters in the configuration files located in the "configs" folder.
## Logging
To achieve optimal performance of the model, we can fine-tune certain parameters in the `train.py` file. These parameters can be logged using Wandb, which helps us keep track of their values and monitor their impact on the model's performance.

