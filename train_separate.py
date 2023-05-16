import os
import sys
import time
import shutil
import wandb
import logging
import imgaug as ia
import numpy as np
from PIL import Image
from pathlib import Path
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
from util.data_loading import CarvanaDataset, BasicDataset
from util.dice_score import dice_loss
from util import OmniposeDataset, B_subtilisDataset
from options.train_options import TrainOptions

from models_pix2pix import create_model
from models_pix2pix import networks
from unet import UNet
from unet.evaluate import evaluate

from betty.engine import Engine
from betty.configs import Config, EngineConfig
from betty.problems import ImplicitProblem


opt = TrainOptions().parse()   # get training options
assert opt.cuda_index == int(opt.gpu_ids[0]), 'gpu types should be same'
device = torch.device('cuda:0' if opt.cuda_index == 0 else 'cuda:1')
save_path = './checkpoint/'+time.strftime("%Y%m%d-%H%M%S"+'-separate-'+str(opt.datasource))
if not os.path.exists(save_path):
    os.mkdir(save_path) 
unet_save_path = save_path+'/unet.pkl'  

##### Initialize logging #####
logger = wandb.init(project='end2end_JSRT', name="Train-9-00", resume='allow', anonymous='must')
logger.config.update(vars(opt))
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

##### create models: pix2pix, unet #####
model = create_model(opt)      # create a model given opt.model and other options
model.setup(opt)               # regular setup: load and print networks; create schedulers

net = UNet(n_channels=opt.input_nc, n_classes=opt.classes, bilinear=opt.bilinear)
net = net.to(device=device)

##### define optimizer for pix2pix model #####
'''have defined in the pix2pix_model.py file'''

##### define optimizer for unet #####
optimizer_unet = optim.RMSprop(net.parameters(), lr=opt.unet_learning_rate, 
                                weight_decay=1e-8, momentum=0.9, foreach=True)
scheduler_unet = optim.lr_scheduler.ReduceLROnPlateau(optimizer_unet, 'max', patience=2)  # goal: maximize Dice score
grad_scaler = torch.cuda.amp.GradScaler(enabled=opt.amp)

##### prepare dataloader #####
'''
dir_img = Path('/home/li/workspace/semantic_segmentation/data/ImagesTr_NoZeros')
dir_mask = Path('/home/li/workspace/semantic_segmentation/data/MasksTr_NoZeros')
dir_img_te = Path('/home/li/workspace/semantic_segmentation/data/ImagesTe_NoZeros')
dir_mask_te = Path('/home/li/workspace/semantic_segmentation/data/MasksTe_NoZeros')
try:
    dataset = CarvanaDataset(dir_img, dir_mask, 1)
except (AssertionError, RuntimeError, IndexError):
    dataset = BasicDataset(dir_img, dir_mask, 1)

# 2. Split into train / validation partitions
n_val = int(len(dataset) * 0.2)
n_train = len(dataset) - n_val
indices = list(range(len(dataset)))
train_set = Subset(dataset, indices[:n_train])
val_set = Subset(dataset, indices[n_train:])
try:
    test_set = CarvanaDataset(dir_img_te, dir_mask_te, 1)
except (AssertionError, RuntimeError, IndexError):
    test_set = BasicDataset(dir_img_te, dir_mask_te, 1)
'''

if opt.datasource == "omnipose":
    train_datadir = '/home/li/workspace/data/omnipose-bact/bact_phase_train/train_sorted_website'
    test_datadir = '/home/li/workspace/data/omnipose-bact/bact_phase_test/test_sorted_website'
    train_set_all = OmniposeDataset(train_datadir, train=True)
    test_set = OmniposeDataset(test_datadir, train=False)
elif opt.datasource == "B_subtilis":
    train_datadir = '/home/li/workspace/data/DeepBacs/training'
    test_datadir = '/home/li/workspace/data/DeepBacs/test'
    train_set_all = B_subtilisDataset(train_datadir, train=True)
    test_set = B_subtilisDataset(test_datadir, train=False)
    
n_train_all = len(train_set_all)
n_val = int(n_train_all * 0.2)
n_train = n_train_all - n_val
n_test = len(test_set)

indices = list(range(n_train_all))
train_set = Subset(train_set_all, indices[:n_train])
val_set = Subset(train_set_all, indices[n_train:])


# 3. Create data loaders
loader_args = dict(batch_size=opt.batch_size, num_workers=2, pin_memory=True)
train_loader = DataLoader(train_set, shuffle=False, **loader_args)
val_loader = DataLoader(val_set, shuffle=True, **loader_args)
test_loader = DataLoader(test_set, shuffle=False, drop_last=True, **loader_args)

##### Pre-Training process of Pix2Pix model #####
total_iters = 0
logging.info('##### Start of Pix2Pix model Train #####')

for epoch in range(opt.n_epochs):
    model.update_learning_rate()    # update learning rates in the beginning of every epoch.
    epoch_iter = 0
    for i, data in enumerate(train_loader):  # inner loop within one epoch
        val_data = next(iter(val_loader))

        train_data = data['image']
        train_mask = data['mask'].unsqueeze(0)
        valid_data = val_data['image']
        valid_mask = val_data['mask'].unsqueeze(0)

        total_iters += opt.batch_size
        epoch_iter += opt.batch_size
        model.set_input(train_data, train_mask)         # unpack data from dataset and apply preprocessing
        model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
        model.optimize_architect(valid_data, valid_mask)
        
        # display images on wandb
        if total_iters % opt.display_freq == 0:
            model.compute_visuals()
            visuals = model.get_current_visuals()
            ims_dict = {}
            for label, image in visuals.items():
                image_numpy = image[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                wandb_image = wandb.Image(image_numpy)
                ims_dict[label] = wandb_image
            logger.log(ims_dict)
        
        # print training losses
        if total_iters % opt.print_freq == 0: 
            losses = model.get_current_losses()
            message = '(epoch: %d, iters: %d ) ' % (epoch, epoch_iter)
            for k, v in losses.items():
                message += '%s: %.3f ' % (k, v)
            logging.info(message)
        # save our latest model
        if total_iters % opt.save_latest_freq == 0: 
            logging.info('saving the model')
            model.save_model(save_path)

model.save_model(save_path)
logging.info('##### End of Pix2Pix model Train #####')

# re-create the pix2pix model to cut off any possible gradient path from pre-training
model = create_model(opt)
model.setup(opt)
model.load_model(save_path+'/pix2pix_discriminator.pkl', save_path+'/pix2pix_generator.pkl')

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


logging.info('The number of training images = %d' % n_train)
logging.info('The number of validate images = %d' % n_val)
logging.info('The number of test images = %d' % len(test_set))

##### Training process of UNet #####
##### Datasets: train_loader, val_loader, test_loader, NLM_loader, SZ_loader, all_train_loader #####
n_epochs = 100
train_iters = int(len(train_set) * n_epochs)
total_iters = 0                # the total number of training iterations
unet_best_score = 0.0          # the best score of unet

criterion = nn.CrossEntropyLoss() if opt.classes > 1 else nn.BCEWithLogitsLoss()

for epoch in range(n_epochs):
    epoch_iters = 0
    for i, batch in enumerate(train_loader):
        total_iters += opt.batch_size
        epoch_iters += opt.batch_size

        images = batch['image'].to(device=device, dtype=torch.float32)
        true_masks = batch['mask'].to(device=device, dtype=torch.long).unsqueeze(0)

        fake_mask = next(iter(train_loader))['mask'].to('cpu', torch.float).numpy()
        fake_mask = seq(images=fake_mask)
        fake_mask = torch.tensor(fake_mask).unsqueeze(0).type(torch.cuda.FloatTensor).to(device=device)
        zero = torch.zeros_like(fake_mask)
        one = torch.ones_like(fake_mask)
        fake_mask = torch.where(fake_mask > 0.1, one, zero)
        fake_image = model.netG(fake_mask)
        fake_mask = fake_mask
        with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=opt.amp):
            masks_pred = net(images)
            fake_pred = net(fake_image)
            loss = criterion(masks_pred.squeeze(0), true_masks.float().squeeze(0))
            loss += dice_loss(torch.sigmoid(masks_pred.squeeze(0)), true_masks.float().squeeze(0), multiclass=False)
            fake_loss = criterion(fake_pred.squeeze(0), fake_mask.float().squeeze(0))
            fake_loss += dice_loss(torch.sigmoid(fake_pred.squeeze(0)), fake_mask.float().squeeze(0), multiclass=False)
            unet_loss = loss + 0.5 * fake_loss
        
        optimizer_unet.zero_grad(set_to_none=True)
        grad_scaler.scale(unet_loss).backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        grad_scaler.step(optimizer_unet)
        grad_scaler.update()

        if total_iters % opt.display_freq == 0:
            test_score = evaluate(net, test_loader, device, opt.amp)
            if test_score > unet_best_score:
                unet_best_score = test_score
                torch.save(net.state_dict(), unet_save_path)
            message = f'Epoch {epoch}/ Iters {epoch_iters}: Performance of UNet: '
            message += '%s: %.5f ' % ('unet_test_score', test_score)
            message += '%s: %.5f ' % ('unet_best_score', unet_best_score)
            logging.info(message)
            logger.log({'unet_test_score': test_score})

    scheduler_unet.step(unet_best_score)







