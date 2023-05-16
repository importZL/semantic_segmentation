import os
import sys
import time
import shutil
import wandb
import logging
import imgaug as ia
import numpy as np
from PIL import Image
from imgaug import augmenters as iaa

import torch
import torchvision
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset

from util import util
# from util.data_loading import BasicDataset
from util.omnipose_data import OmniposeDataset as BasicDataset
from util.dice_score import dice_loss
from options.train_options import TrainOptions

from models_pix2pix import create_model
from models_pix2pix import networks
from unet import UNet
from deeplab import *
from unet.evaluate import evaluate

from betty.engine import Engine
from betty.configs import Config, EngineConfig
from betty.problems import ImplicitProblem

opt = TrainOptions().parse()   # get training options
assert opt.cuda_index == int(opt.gpu_ids[0]), 'gpu types should be same'
device = torch.device('cuda:0' if opt.cuda_index == 0 else 'cuda:1')
save_path = './checkpoint_omni/'+'UNet-omni-30-'+time.strftime("%Y%m%d-%H%M%S")
if not os.path.exists(save_path):
    os.mkdir(save_path) 
unet_save_path = save_path+'/unet.pkl'  

##### Initialize logging #####
# logger = wandb.init(project='end2end-unet-ISIC', name="unet-200", resume='allow', anonymous='must')
logger = wandb.init(project='vanilla-unet', name="unet-omni-30", resume='allow', anonymous='must')
logger.config.update(vars(opt))
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

if opt.seg_model == 'unet':
    net = UNet(n_channels=opt.output_nc, n_classes=opt.classes, bilinear=opt.bilinear)
elif opt.seg_model == 'deeplab':
    net = DeepLabV3(num_classes=opt.classes)
net = net.to(device=device)

##### define optimizer for unet #####
optimizer_unet = optim.RMSprop(net.parameters(), lr=opt.unet_learning_rate, 
                                weight_decay=1e-8, momentum=0.9, foreach=True)
scheduler_unet = optim.lr_scheduler.ReduceLROnPlateau(optimizer_unet, 'max', patience=5)  # goal: maximize Dice score
# scheduler_unet = optim.lr_scheduler.CosineAnnealingLR(optimizer_unet, T_max=500, eta_min=1e-9)
grad_scaler = torch.cuda.amp.GradScaler(enabled=opt.amp)

##### prepare dataloader #####
dataset = BasicDataset('/home/li/workspace/data/omnipose-worm/worm_train', train=True)

n_train = 8 # 165, 35, 9
n_val = 2

indices = list(range(len(dataset)))
train_set = Subset(dataset, indices[:n_train])
val_set = Subset(dataset, indices[n_train:n_train+n_val])

loader_args = dict(batch_size=opt.batch_size, num_workers=4, pin_memory=True)
train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_args)
val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)


##### Training of vanilla UNet #####
total_iters = 0                # the total number of training iterations
val_best_score = 0.0          # the best score of unet
criterion = nn.CrossEntropyLoss() if opt.classes > 1 else nn.BCEWithLogitsLoss()

while total_iters < opt.n_epochs:
    for i, batch in enumerate(train_loader):
        images = batch['image'].to(device=device, dtype=torch.float32)
        true_masks = batch['mask'].to(device=device, dtype=torch.long).squeeze(0)
        total_iters += opt.batch_size

        masks_pred = net(images)
        loss = criterion(masks_pred.squeeze(1), true_masks.float())
        loss += dice_loss(torch.sigmoid(masks_pred.squeeze()), true_masks.float().squeeze())

        optimizer_unet.zero_grad(set_to_none=True)
        grad_scaler.scale(loss).backward()
        grad_scaler.step(optimizer_unet)
        grad_scaler.update()

        if total_iters % opt.display_freq == 0:
            val_score = evaluate(net, val_loader, device, opt.amp)
            if val_score > val_best_score:
                val_best_score = val_score
                torch.save(net.state_dict(), unet_save_path)
            message = 'Iters: %s, Performance of UNet: ' % (total_iters)
            message += '%s: %.5f, %s: %.5f' % ('loss', loss, 'val', val_score)
            logging.info(message)
            logger.log({'val_score': val_score})
    
    scheduler_unet.step(val_best_score)
torch.save(net.state_dict(), save_path+'/final.pkl')
    
