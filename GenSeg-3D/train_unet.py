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
sys.exit()

save_path = './checkpoint_e2e/'+'UNet-liver-98-'+time.strftime("%Y%m%d-%H%M%S")
if not os.path.exists(save_path):
    os.mkdir(save_path) 
unet_save_path = save_path+'/unet.pkl'  

##### Initialize logging #####
# logger = wandb.init(project='end2end-unet-ISIC', name="unet-200", resume='allow', anonymous='must')
logger = wandb.init(project='unet-lung', name="unet-liver-98", 
                    resume='allow', anonymous='must', mode='disabled')
logger.config.update(vars(opt))
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

##### define optimizer for unet #####
optimizer_unet = optim.RMSprop(net.parameters(), lr=1e-4, weight_decay=1e-8, momentum=0.9, foreach=True)

##### prepare dataloader #####
data_loader = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
dataset = data_loader.dataset
total_size = len(dataset)
split_1 = 78
split_2 = 20
split_3 = total_size - split_1 - split_2

# Perform the split
subset1, subset2, _ = random_split(dataset, [split_1, split_2, split_3])

# Create new DataLoaders
train_loader = DataLoader(subset1, batch_size=opt.batch_size, num_workers=int(opt.num_threads), shuffle=not opt.serial_batches)
val_loader = DataLoader(subset2, batch_size=opt.batch_size, num_workers=int(opt.num_threads), shuffle=not opt.serial_batches)
logging.info('The number of training images = %d' % len(train_loader))
logging.info('The number of validate images = %d' % len(val_loader))

criterion = CrossEntropyLoss(weight=torch.Tensor(BCE_WEIGHTS)).cuda() if torch.cuda.is_available() and TRAIN_CUDA else CrossEntropyLoss(weight=torch.Tensor(BCE_WEIGHTS))

optimizer = Adam(params=net.parameters())
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-9)

min_valid_loss = math.inf
max_valid_dice = 0.0


for epoch in range(TRAINING_EPOCH):
    
    train_loss = 0.0
    net.train()
    for data in tqdm(train_loader, leave=False):
        image, ground_truth = data['B'], data['mask']
        image = image.to(device=device, dtype=torch.float32)
        ground_truth = ground_truth.to(device=device, dtype=torch.long).squeeze(0)

        optimizer.zero_grad()
        target = net(image)
        loss = criterion(target, ground_truth.long())
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    scheduler.step()
    valid_loss = 0.0
    dice_result = 0.0
    net.eval()
    for data in tqdm(val_loader, leave=False):
        image, ground_truth = data['B'], data['mask']
        image = image.to(device=device, dtype=torch.float32)
        ground_truth = ground_truth.to(device=device, dtype=torch.long).squeeze(0)
        
        target = net(image)
        loss = criterion(target, ground_truth.long())
        valid_loss += loss.item()
        
        dice_result += dice_score(target, ground_truth[0].long())
    
    print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / len(train_loader)} \t\t Validation Loss: {valid_loss / len(val_loader)} \t\t Validation Dice: {dice_result / len(val_loader)}')
    
    # if min_valid_loss > valid_loss:
    #     print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
    #     min_valid_loss = valid_loss
    #     # Saving State Dict
    #     torch.save(model.state_dict(), f'checkpoints/epoch{epoch}_valLoss{min_valid_loss}.pth')
    if max_valid_dice < (dice_result / len(val_loader)):
        print(f'Validation Dice Increased({max_valid_dice:.6f}--->{dice_result / len(val_loader)}) \t Saving The Model')
        max_valid_dice = (dice_result / len(val_loader))
        # Saving State Dict
        os.makedirs(save_path, exist_ok=True)
        torch.save(net.state_dict(), f'{save_path}/best.pth')

# inference
print(f'########## Inference on Test Set ##########')
opt.phase = 'val'
test_loader = create_dataset(opt)
net.load_state_dict(torch.load(f'{save_path}/best.pth'))
net.eval()
test_dice = 0.0
for data in tqdm(test_loader, leave=False):
    image, ground_truth = data['B'], data['mask']
    image = image.to(device=device, dtype=torch.float32)
    ground_truth = ground_truth.to(device=device, dtype=torch.long).squeeze(0)
    target = net(image)
    test_dice += dice_score(target, ground_truth.long())
print(f'Test Dice: {test_dice / len(test_loader)}')
