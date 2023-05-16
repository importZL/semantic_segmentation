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
import torchvision.transforms.functional as func
from torch.utils.data import DataLoader, random_split, Subset

from util import util
from util.JSRT_loader import CarvanaDataset as BasicDataset
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
save_path = './checkpoint_ablation_JSRT/'+str(opt.seg_model)+'-JSRT-175-'+str(opt.aug_type)+time.strftime("%Y%m%d-%H%M%S")
if not os.path.exists(save_path):
    os.mkdir(save_path) 
unet_save_path = save_path+'/unet.pkl'  

##### Initialize logging #####
logger = wandb.init(project='JSRT_ablation', name="JSRT-175"+str(opt.aug_type), resume='allow', anonymous='must')
logger.config.update(vars(opt))
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

if opt.seg_model == 'unet':
    net = UNet(n_channels=opt.output_nc, n_classes=opt.classes, bilinear=opt.bilinear)
elif opt.seg_model == 'deeplab':
    net = DeepLabV3(num_classes=opt.classes)
net = net.to(device=device)

model = create_model(opt)      # create a model given opt.model and other options
model.setup(opt)               # regular setup: load and print networks; create schedulers
# load pre-trained model
model_path = './pix2pix_JSRT_model/20230320-214246-pix2pix-JSRT-175'  # './pix2pix_JSRT_model/20230320-213518-pix2pix-JSRT-9' './pix2pix_JSRT_model/20230320-214049-pix2pix-JSRT-35'
model.load_model(model_path + '/pix2pix_discriminator.pkl', model_path + '/pix2pix_generator.pkl')

##### define optimizer for unet #####
optimizer_unet = optim.RMSprop(net.parameters(), lr=opt.unet_learning_rate, 
                                weight_decay=1e-8, momentum=0.9, foreach=True)
scheduler_unet = optim.lr_scheduler.ReduceLROnPlateau(optimizer_unet, 'max', patience=5)  # goal: maximize Dice score
# scheduler_unet = optim.lr_scheduler.CosineAnnealingLR(optimizer_unet, T_max=500, eta_min=1e-9)
grad_scaler = torch.cuda.amp.GradScaler(enabled=opt.amp)

##### prepare dataloader #####
dataset = BasicDataset(opt.dataroot+'/Images', opt.dataroot+'/Masks', 1.0, '')

n_test = 72
n_train = 140 # 165, 35, 9
n_val = 35
n_extra = len(dataset) - n_train - n_test - n_val
indices = list(range(len(dataset)))
train_set = Subset(dataset, indices[:n_train])
val_set = Subset(dataset, indices[n_train:n_train+n_val])

loader_args = dict(batch_size=opt.batch_size, num_workers=4, pin_memory=True)
train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_args)
val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

# Image Augmenter
seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
    # iaa.Flipud(0.25), # vertical flips

    iaa.CropAndPad(percent=(0, 0.1)), # random crops and pad

    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.        
    iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},),
    iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},),
    iaa.Affine(rotate=(-15, 15),),
    iaa.Affine(shear=(-8, 8),)
], random_order=True) # apply augmenters in random order

fake_trans = transforms.Compose([
    transforms.RandomEqualize(p=0.5),  # histogram equalization
    transforms.RandomPosterize(4, p=1.0),  # Posterization
    transforms.RandomAdjustSharpness(0.3, p=0.5),  # Sharpness adjust
    transforms.RandomAutocontrast(p=0.5),  # contrast adjust
    transforms.ColorJitter(saturation=0.5),  # saturation adjust
])

##### Training of vanilla UNet #####
total_iters = 0                # the total number of training iterations
val_best_score = 0.0          # the best score of unet
criterion = nn.CrossEntropyLoss() if opt.classes > 1 else nn.BCEWithLogitsLoss()

while total_iters < opt.n_epochs:
    for i, batch in enumerate(train_loader):
        images = batch['image'].to(device=device, dtype=torch.float32)
        true_masks = batch['mask'].to(device=device, dtype=torch.long).squeeze(0)
        total_iters += 1

        masks_pred = net(images)
        loss = criterion(masks_pred.squeeze(0), true_masks.float())
        loss += dice_loss(torch.sigmoid(masks_pred.squeeze()), true_masks.float().squeeze())

        if opt.aug_type == 'rotate':
            fake_mask, fake_image = func.rotate(true_masks, 8), func.rotate(images, 8)
        elif opt.aug_type == 'translate': 
            fake_mask = func.affine(true_masks, angle=0, translate=(0.1, 0.1), scale=1, shear=0)
            fake_image = func.affine(images, angle=0, translate=(0.1, 0.1), scale=1, shear=0)
        elif opt.aug_type == 'flip':
            fake_mask, fake_image = func.hflip(true_masks), func.hflip(images)
        elif opt.aug_type == 'combine':
            fake_mask = func.rotate(func.hflip(func.affine(true_masks, angle=0, translate=(0.1, 0.1), scale=1, shear=0)), 8)
            fake_image = func.rotate(func.hflip(func.affine(images, angle=0, translate=(0.1, 0.1), scale=1, shear=0)), 8)
        elif opt.aug_type == 'separate':
            fake_mask = true_masks.to('cpu', torch.float).numpy()
            if opt.batch_size > 1:
                for i in range(opt.batch_size):
                    fake_mask[i] = seq(images=fake_mask[i])
            fake_mask = torch.tensor(fake_mask).type(torch.cuda.FloatTensor).to(device=device)
            zero = torch.zeros_like(fake_mask)
            one = torch.ones_like(fake_mask)
            fake_mask = torch.where(fake_mask > 0.1, one, zero)
            fake_image = model.netG(fake_mask)

            fake_mask = fake_mask.squeeze(0).detach()
            # fake_image = fake_image.mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8)
            # fake_image = transforms.functional.adjust_gamma(transforms.functional.equalize(fake_image), 0.5) / 255.0

        masks_pred = net(fake_image)
        loss_fake = criterion(masks_pred.squeeze(0), fake_mask.float())
        loss_fake += dice_loss(torch.sigmoid(masks_pred.squeeze()), fake_mask.float().squeeze())

        loss = loss + 1.0 * loss_fake

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

            ims_dict = {}
            show_image = images[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu').numpy()
            show_mask = true_masks[0].mul(255).permute(1, 2, 0).to('cpu').numpy()
            show_fake_image = fake_image[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu').detach().numpy()
            show_fake_mask = fake_mask[0].mul(255).permute(1, 2, 0).to('cpu').numpy()
            wandb_image = wandb.Image(show_image)
            ims_dict['show_image'] = wandb_image
            wandb_image = wandb.Image(show_mask)
            ims_dict['show_mask'] = wandb_image
            wandb_image = wandb.Image(show_fake_image)
            ims_dict['show_fake_image'] = wandb_image
            wandb_image = wandb.Image(show_fake_mask)
            ims_dict['show_fake_mask'] = wandb_image
            logger.log(ims_dict)
    
    scheduler_unet.step(val_best_score)
torch.save(net.state_dict(), save_path+'/final.pkl')
    
