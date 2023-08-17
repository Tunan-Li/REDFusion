# REDFusion
This is the implementation of REDFusion (AAAI 2024 under review).
## Preparation
# Datasets
1. Download [MVSA_Single](https://www.kaggle.com/datasets/vincemarcs/mvsasingle) and put it in REDFusion/datasets/MVSA_Single/.
2. Download [bert-base-uncased](https://huggingface.co/bert-base-uncased) and put it in REDFusion/.
3. Note: All datasets will be released if the paper is accepted.
# Enviroment
```
torch==1.10.1
torchvision==0.11.2
sklearn
pytorch-pretrained-bert
tqdm
```
## Command for trianing
```
bash shells/batch_train_early_wd.sh
```
