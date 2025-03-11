import os
import sys
sys.path.append('.')
import time
import shutil
import wandb
import logging
import numpy as np
from PIL import Image
import math
from tqdm import tqdm

import torch
import torchvision
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from util import util
from UNet3D.config import (
    TRAINING_EPOCH, NUM_CLASSES, IN_CHANNELS, BCE_WEIGHTS, BACKGROUND_AS_CLASS, TRAIN_CUDA
)
from UNet3D.unet3d import UNet3D
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from models.networks import arch_parameters
from transforms import fake_transform
from util.util import zero_division


def dice_score(pred_logits, target, epsilon=1e-6):
    # Convert logits to predicted mask (argmax over channel dimension)
    pred = torch.argmax(pred_logits, dim=1)

    # Convert tensors to float for Dice calculation
    pred = pred.float()
    target = target.float()

    # Compute Dice Score
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    dice = (2. * intersection + epsilon) / (union + epsilon)
    
    return dice.item()

def dice_loss(y_true, y_pred, smooth=1):
    return (1 - dice_score(y_true, y_pred)) * smooth

def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    JC_index = 0

    # iterate over the validation set
    with torch.no_grad():
        # for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        for i, batch in enumerate(dataloader):
            image, mask_true = batch['B'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32)
            # mask_true = mask_true.to(device=device, dtype=torch.long)
            mask_true = mask_true.to(device=device, dtype=torch.long).squeeze(0)
            
            # predict the mask
            mask_pred = net(image)
            # compute the Dice score
            JC_index += dice_score(mask_pred, mask_true)

    net.train()
    return JC_index / max(num_val_batches, 1)


opt = TrainOptions().parse()   # get training options
# config = get_config(opt)
device = torch.device('cuda:0')

net = UNet3D(in_channels=IN_CHANNELS , num_classes= NUM_CLASSES+1)
net = net.to(device=device)

print(f'########## Inference on Test Set ##########')
opt.phase = 'val'
test_loader = create_dataset(opt)
net.load_state_dict(torch.load(f'/data/li/Pix2PixNIfTI/checkpoint_e2e/end2end-liver-98-20250228-221033/unet.pkl'))
net.eval()
test_dice = 0.0
for data in tqdm(test_loader, leave=False):
    image, ground_truth = data['B'], data['mask']
    image = image.to(device=device, dtype=torch.float32)
    ground_truth = ground_truth.to(device=device, dtype=torch.long).squeeze(0)
    target = net(image)
    test_dice += dice_score(target, ground_truth.long())
print(f'Test Dice: {test_dice / len(test_loader)}')