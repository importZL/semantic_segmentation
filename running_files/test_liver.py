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
from util.liver_data import CarvanaDataset as BasicDataset
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
if opt.seg_model == 'unet': 
    net = UNet(n_channels=opt.output_nc, n_classes=opt.classes, bilinear=opt.bilinear)
elif opt.seg_model == 'deeplab':
    net = DeepLabV3(num_classes=opt.classes)
    # net = deeplab(num_classes=1)
net = net.to(device=device)

##### prepare dataloader #####
dataset = BasicDataset(opt.dataroot+'/Images', opt.dataroot+'/Masks', 1.0, '')

loader_args = dict(batch_size=opt.batch_size, num_workers=4, pin_memory=True)
test_loader = DataLoader(dataset, shuffle=False, drop_last=True, **loader_args)

##### Evaluation of UNet #####
net.load_state_dict(torch.load(opt.model_dir)) # load trained model

unet_score = evaluate(net, test_loader, device, opt.amp)
print('UNet score: '+str(unet_score))

    
