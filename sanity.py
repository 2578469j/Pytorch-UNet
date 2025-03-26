import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from evaluate import evaluate
from unet import UNet
from unet.loss import WeightedBinaryCrossEntropyLoss, WeightedBinaryCrossEntropyLossGlobal, compute_local_variance
from utils.data_loading import BasicDataset, CarvanaDataset, GemsyDataset
from utils.dice_score import dice_loss

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Gemsy model has:
# defected pixels: 6813187 
# Non-defectd pixels: 1088686589 
# Defected rate: 0.006258171147545935


#model = UNet(3, 3, bilinear=False)

#inp = torch.ones((5,3,600,600))

#res1 = compute_local_variance(inp, 3)
#print(res1)
#res = model.forward(inp)

dir_img = Path("C:\\Users\\Admin\\Desktop\\Gemsy\\Data\\processed\\.subsplit\\features\\")
dir_mask = Path("C:\\Users\\Admin\\Desktop\\Gemsy\\Data\\processed\\.subsplit\\masks\\")
loader_args = dict(batch_size=16, num_workers=0, pin_memory=True)

training_augmentation = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
          #  A.RandomBrightnessContrast(p=0.2),
            A.ElasticTransform(p=0.2),
          #  A.GaussNoise(p=0.2),
           # A.Normalize(),
            ToTensorV2()
        ])

img_scale = 1
val_percent = 0.0
dataset = GemsyDataset(dir_img, dir_mask, img_scale, transform=None)#training_augmentation)
# 2. Split into train / validation partitions
n_val = int(len(dataset) * val_percent)
n_train = len(dataset) - n_val

train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

train_loader = DataLoader(train_set, shuffle=True, **loader_args)

positive = 0
negative = 0
with tqdm(total=n_train, desc=f'Epoch 1', unit='img') as pbar:
    for batch in train_loader:
        images, true_masks = batch['image'], batch['mask']
        positive += true_masks.sum().item()
        negative += (torch.logical_not(true_masks)).sum().item()
        print(positive, negative, positive/negative)
        pbar.update(images.shape[0])

print("final:")
print(positive, negative, positive/negative)

