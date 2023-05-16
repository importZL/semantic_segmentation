import os
import sys
sys.path.append('.')
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
from util.JSRT_loader import CarvanaDataset as BasicDataset
from util.dice_score import dice_loss
from options.train_options import TrainOptions

from models_pix2pix import create_model
from models_pix2pix import networks
from unet import UNet
from unet.evaluate import evaluate
from deeplab import *
from deeplabv2 import DeepLabV2 as deeplab


opt = TrainOptions().parse()   # get training options
assert opt.cuda_index == int(opt.gpu_ids[0]), 'gpu types should be same'
device = torch.device('cuda:0' if opt.cuda_index == 0 else 'cuda:1')

##### create models: unet #####
model = create_model(opt)      # create a model given opt.model and other options
model.setup(opt)               # regular setup: load and print networks; create schedulers

if opt.seg_model == 'unet': 
    net = UNet(n_channels=opt.output_nc, n_classes=opt.classes, bilinear=opt.bilinear)
elif opt.seg_model == 'deeplab':
    net = DeepLabV3(num_classes=1)
    # net = deeplab(num_classes=1)
net = net.to(device=device)

##### prepare dataloader #####
dataset = BasicDataset(opt.dataroot+'/Images', opt.dataroot+'/Masks', 1.0, '')
NLM_dataset = BasicDataset('../data/NLM/Images', '../data/NLM/Masks', 1.0, '') # use as the out-domain dataset
SZ_dataset = BasicDataset('../data/SZ/Images', '../data/SZ/Masks', 1.0, '_mask') # use as the extra dataset

n_test = 72
indices = list(range(len(dataset)))
test_set = Subset(dataset, indices[-n_test:])

loader_args = dict(batch_size=opt.batch_size, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_set, shuffle=False, drop_last=True, **loader_args)
# out-domain test dataloader    
NLM_loader = DataLoader(NLM_dataset, shuffle=False, **loader_args)
SZ_loader = DataLoader(SZ_dataset, shuffle=False, **loader_args)

##### Evaluation of UNet #####
net.load_state_dict(torch.load(opt.model_dir)) # load trained model

unet_score = evaluate(net, test_loader, device, opt.amp)
NLM_score = evaluate(net, NLM_loader, device, opt.amp)
SZ_score = evaluate(net, SZ_loader, device, opt.amp)
print('UNet score: '+str(unet_score))
print('NLM score: '+str(NLM_score))
print('SZ score: '+str(SZ_score))

    
