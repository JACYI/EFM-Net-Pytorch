# EFM-Net: An Essential Feature Mining Network for Target Fine-Grained Classification in Optical Remote Sensing Images

By Yonghao Yi (*Student Member, IEEE*) , Yanan You (*Member, IEEE*) , Chao Li, and Wenli Zhou.

Special thanks to [Haopeng Zhang](https://orcid.org/0000-0003-1981-8307) for providing the dataset source file for [FGSCR-42](https://www.mdpi.com/2072-4292/13/4/747).

## Introduction

This code provides an reference version for the implementation of the *IEEE-TGRS* paper “EFM-Net: An Essential Feature Mining Network for Target Fine-Grained Classification in Optical Remote Sensing Images”. The projects are still under construction.



## How to run

### Prepare the datasets

**If you want to use the training and testing sets divided in this paper, please click on the following link to download.**

Download the public benchmark datasets  and unzip them in your own path.

- [FGSC-23](https://drive.google.com/file/d/1DFir2wvzVnMYqleqgvxLoN760hYZe3TW/view?usp=sharing)

- [FGSCR-42](https://drive.google.com/file/d/1o8QzGA3wEhobGFZ-Hbey0GCgCNIdnEmf/view?usp=sharing)

- [Aircraft-16](https://drive.google.com/file/d/1n0aoB0FJIvrA5xpC8AfeXKZKqCFu2Gca/view?usp=sharing)

Move the directory with `train` and `test` sub-directories to the `./datasets/` directory and rename it with database name, such as

```
mv FGSC23 /path/to/EFM-Net-Pytorch/datasets/FGSC-23
```

Make sure your datasets are correctly split into the training set and test set. The training set should be placed in the directory named “train”  while test set named “test”.

**The source files download links of those public datasets can be found in https://github.com/JACYI/Dataset-for-Remote-Sensing.** 



### Download the pre-trained model

The feature extraction model is based on ConvNeXt-Base. Please download the pre-trained parameters file:

"convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",

and move the pre-trained model to `./pretrained` directory.



### Build the running environment

1. **Create the virtual environment:**

```shell
conda create -n efmnet python=3.8
source activate efmnet
```

2. **requirements**

- python=3.8
- pytorch>=1.8.1, torchvision>=0.9.1
- timm=0.3.2
- opencv-python
- scikit-learn
- tensorboardX
- termcolor



### For Training

1. **Run on a single GPU**

```shell
python main.py --exp_name test --attentions 16 --epochs 100 --dataset FGSC-23 --output_dir logs/test --novel_loss 0.5
```

2. **Run on multiple GPUs** (2 GPUs for example)

```shell
python -m torch.distributed.launch --master_port 12345 \
 --nproc_per_node=2 main.py \
 --exp_name test --attentions 16 \
 --epochs 120 --dataset FGSC-23 \
 --output_dir logs/test --novel_loss 0.5
```



## Citation

More details need to be added.

```
@ARTICLE{10097708,
  author={Yi, Yonghao and You, Yanan and Li, Chao and Zhou, Wenli},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={EFM-Net: An Essential Feature Mining Network for Target Fine-Grained Classification in Optical Remote Sensing Images}, 
  year={2023},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TGRS.2023.3265669}
}
```



## Corresponding author information

Yanan You, Beijing, China,

Phone: (+86) 15201148169

Email: youyanan@bupt.edu.cn



## To do

1. Support more feature extraction models;
2. Provide more interfaces for modifying parameters.
