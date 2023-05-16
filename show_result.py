import os
import sys
import time
import shutil
import wandb
import logging
import imgaug as ia
import numpy as np
from PIL import Image, ImageOps
from imgaug import augmenters as iaa

import torch
import torchvision
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
import torchvision.transforms.functional as func
import cv2

from util import util
#from util.data_loading import BasicDataset
from util.JSRT_loader import CarvanaDataset as BasicDataset
from util.ISIC_loader import CarvanaDataset as BasicDataset
from util.dice_score import dice_loss
from options.train_options import TrainOptions

from models_pix2pix import create_model
from models_pix2pix import networks
from unet import UNet
from unet.evaluate import evaluate
from deeplab import *

from betty.engine import Engine
from betty.configs import Config, EngineConfig
from betty.problems import ImplicitProblem

device = torch.device('cuda:0')

# net = UNet(n_channels=3, n_classes=1, bilinear=False)
net = DeepLabV3(num_classes=1)

net.load_state_dict(torch.load(
    '/home/li/workspace/semantic_segmentation/checkpoint_kvasir/end2end-kvasir-400-deeplab-1.0-20230425-161239-7769/deeplab.pkl',
))
net = net.to(device=device)

# generate qualitative examples
'''
##### Lung segmentaiton tasks
image1 = np.asarray(transforms.functional.adjust_gamma(ImageOps.equalize(Image.open('../data/JSRT/Images/JPCLN016.png')), 0.5).convert('L').resize((256, 256), resample=Image.NEAREST))
image2 = np.asarray(transforms.functional.adjust_gamma(ImageOps.equalize(Image.open('../data/NLM/Images/MCUCXR_0060_0.png')), 0.5).convert('L').resize((256, 256), resample=Image.NEAREST))
image3 = np.asarray(transforms.functional.adjust_gamma(ImageOps.equalize(Image.open('../data/SZ/Images/CHNCXR_0073_0.png')), 0.5).convert('L').resize((256, 256), resample=Image.NEAREST))

##### Skin segmentation examples 
image1 = np.asarray(Image.open('/home/li/workspace/data/ISIC2018/Images/ISIC_0016070.jpg').resize((256, 256), resample=Image.NEAREST))
image2 = np.asarray(Image.open('/home/li/workspace/data/PH2/Images/IMD435.jpg').resize((256, 256), resample=Image.NEAREST))
image3 = np.asarray(Image.open('/home/li/workspace/data/DermIS/Images/5.jpg').resize((256, 256), resample=Image.NEAREST))


##### Breast cancer segmentation examples
image1 = np.asarray(transforms.functional.adjust_gamma(ImageOps.equalize(Image.open('../data/breast/Images/malignant (195).png').convert('RGB')), 0.5).convert('L').resize((256, 256), resample=Image.NEAREST))

##### Fetoscopy vessel segmentation examples
image1 = np.asarray(Image.open('/home/li/workspace/data/fetoscopy/test/Images/anon012_33421.png').resize((256, 256), resample=Image.NEAREST))
'''

##### Gastrointestinal disease segmentation examples
image1 = np.asarray(Image.open('/home/li/workspace/data/Kvasir-SEG/Images/ck2bxw18mmz1k0725litqq2mc.jpg').resize((256, 256), resample=Image.NEAREST))

image1 =  image1/ 255
image1 = torch.as_tensor(image1.transpose((2, 0, 1)).copy()).float().contiguous().unsqueeze(0).to(device)

image1 = torch.concatenate((image1, image1), axis=0)
print(image1.size())

image1 = net(image1)[0]
zero = torch.zeros_like(image1)
one = torch.ones_like(image1)
image1 = torch.where(image1 > 0.1, one, zero).squeeze()

image1 = image1.to('cpu').numpy()
im_base = np.zeros((256, 256, 3))
im_base[image1 == 1] = [255, 102, 102]
im_base = torch.as_tensor(im_base)

cv2.imwrite('./test.png', im_base.to('cpu').numpy())
print('OK')

'''
image1 = np.asarray(Image.open('/home/li/workspace/data/Kvasir-SEG/Masks/ck2bxw18mmz1k0725litqq2mc.jpg').convert('L').resize((256, 256), resample=Image.NEAREST))
im_base = np.zeros((256, 256, 3))
im_base[image1 == 255] = [255, 102, 102]
cv2.imwrite('./test.png', im_base)
'''
