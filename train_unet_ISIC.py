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
from torch.utils.data import DataLoader, random_split

from util import util
from util.ISIC_loader import CarvanaDataset as BasicDataset
from util.dice_score import dice_loss
from options.train_options import TrainOptions

from models_pix2pix import create_model
from models_pix2pix import networks
from unet import UNet
from models_dcgan.dcgan_darts import DCGAN_MODEL
from model_wgan.wgan_gradient_penalty import WGAN_GP
from deeplabv2 import DeepLabV2_ResNet101_MSC
from deeplabv2.deeplabv2 import *
from deeplabv2.deeplabv3 import *

from betty.engine import Engine
from betty.configs import Config, EngineConfig
from betty.problems import ImplicitProblem

def jaccard_index(y_true, y_pred, smooth=1e-6):
    if y_pred.dim() != 2:
        jac = 0.0
        for i in range(y_pred.size()[0]):
            intersection = torch.abs(y_true[i] * y_pred[i]).sum(dim=(-1, -2))
            sum_ = torch.sum(torch.abs(y_true[i]) + torch.abs(y_pred[i]), dim=(-1, -2))
            jac += (intersection + smooth) / (sum_ - intersection + smooth)
        jac = jac / y_pred.size()[0]
    else:
        intersection = torch.abs(y_true * y_pred).sum(dim=(-1, -2))
        sum_ = torch.sum(torch.abs(y_true) + torch.abs(y_pred), dim=(-1, -2))
        jac = (intersection + smooth) / (sum_ - intersection + smooth)
    #return (1 - jac) * smooth
    return jac

def jaccard_index_loss(y_true, y_pred, smooth=1):
    return (1 - jaccard_index(y_true, y_pred))

def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    JC_index = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        # for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        for i, batch in enumerate(dataloader):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            # mask_true = mask_true.to(device=device, dtype=torch.long)
            mask_true = mask_true.to(device=device, dtype=torch.long).squeeze(0)
            
            # predict the mask
            mask_pred = net(image)

            assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
            mask_pred = (torch.sigmoid(mask_pred) > 0.5).float()
            # compute the Dice score
            JC_index += jaccard_index(mask_pred.squeeze(), mask_true.squeeze())

    net.train()
    return JC_index / max(num_val_batches, 1)

def resize_labels(labels, size):
    """
    Downsample labels for 0.5x and 0.75x logits by nearest interpolation.
    Other nearest methods result in misaligned labels.
    -> F.interpolate(labels, shape, mode='nearest')
    -> cv2.resize(labels, shape, interpolation=cv2.INTER_NEAREST)
    """
    new_labels = []
    for label in labels:
        label = label.float().cpu().numpy()
        label = Image.fromarray(label).resize(size, resample=Image.NEAREST)
        new_labels.append(np.asarray(label))
    new_labels = torch.tensor(np.array(new_labels))
    return new_labels


opt = TrainOptions().parse()   # get training options
assert opt.cuda_index == int(opt.gpu_ids[0]), 'gpu types should be same'
device = torch.device('cuda:0' if opt.cuda_index == 0 else 'cuda:1')

##### create models: pix2pix, unet #####
model = create_model(opt)      # create a model given opt.model and other options
model.setup(opt)               # regular setup: load and print networks; create schedulers
model.load_model('./checkpoint/20230127-000618-end2end_ISIC_test/pix2pix_discriminator.pkl', './checkpoint/20230127-000618-end2end_ISIC_test/pix2pix_generator.pkl')

# net = DeepLabV2(n_classes=1, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24])
# net = DeepLabV2(n_classes=1, n_blocks=[3, 4, 4, 3], atrous_rates=[3, 4, 4, 3])
net = DeepLabV3(n_classes=1, n_blocks=[3, 4, 15, 3], atrous_rates=[4, 8, 12], multi_grids=[1, 2, 4], output_stride=8,)
# net = DeepLabV2_ResNet101_MSC(n_classes=1)

# net = UNet(n_channels=opt.output_nc, n_classes=opt.classes, bilinear=opt.bilinear)
net = net.to(device=device)

##### define optimizer for pix2pix model #####
'''have defined in the pix2pix_model.py file'''

##### define optimizer for unet #####
optimizer_unet = optim.RMSprop(net.parameters(), lr=1e-4, 
                                weight_decay=1e-8, momentum=0.9, foreach=True)
scheduler_unet = optim.lr_scheduler.ReduceLROnPlateau(optimizer_unet, 'max', patience=2)  # goal: maximize Dice score
grad_scaler = torch.cuda.amp.GradScaler(enabled=opt.amp)

##### prepare dataloader #####
dataset = BasicDataset(opt.dataroot+'/Images', opt.dataroot+'/Masks', 1.0, '_segmentation')
PH2_dataset = BasicDataset('../data/PH2/Images', '../data/PH2/Masks', 1.0, '_lesion') # use as the out-domain dataset
dermIS_dataset = BasicDataset('../data/DermIS/Images', '../data/DermIS/Masks', 1.0) # use as the extra dataset
dermQuest_dataset = BasicDataset('../data/DermQuest/Images', '../data/DermQuest/Masks', 1.0) # use as the out-domain dataset

n_test = 594
n_train = 40 # 165, 35, 9
n_val = len(dataset) - n_train - n_test
train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(0))

len_extra = 0
extra_dataset, _ = random_split(dermQuest_dataset, [len_extra, len(dermQuest_dataset)-len_extra], generator=torch.Generator().manual_seed(0))
train_set = torch.utils.data.ConcatDataset([train_set, extra_dataset])

loader_args = dict(batch_size=opt.batch_size, num_workers=4, pin_memory=True)
train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_args)
val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
test_loader = DataLoader(test_set, shuffle=False, drop_last=True, **loader_args)
# out-domain test dataloader    
PH2_loader = DataLoader(PH2_dataset, shuffle=False, **loader_args)
dermIS_loader = DataLoader(dermIS_dataset, shuffle=False, **loader_args)

# Generating Fake images
pix2pix_set = torch.utils.data.ConcatDataset([train_set, val_set])
all_train_loader = DataLoader(pix2pix_set, batch_size=1, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
sometimes = lambda aug: iaa.Sometimes(0.5, aug)
seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
    iaa.Flipud(0.5), # vertical flips

    iaa.CropAndPad(percent=(0, 0.1)), # random crops and pad

    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.        
    iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},),
    iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},),
    iaa.Affine(rotate=(-25, 25),),
    iaa.Affine(shear=(-25, 25),)
], random_order=True) # apply augmenters in random order

n_fake = 100
generate_path = './generate_data2'

if not os.path.exists(generate_path):
    os.makedirs(generate_path)
shutil.rmtree(generate_path+'/Images', ignore_errors=True)
shutil.rmtree(generate_path+'/Masks', ignore_errors=True)
os.mkdir(generate_path+'/Images')
os.mkdir(generate_path+'/Masks')
generate_index = 0
for i in range(n_fake):
    # if n_val > 0:
    #     data = next(iter(val_loader))
    # else:
    data = next(iter(all_train_loader))
    
    # fake_mapping = data['mask'].to('cpu', torch.float).numpy()
    fake_mapping = data['mask_pix2pix'].to('cpu', torch.float).squeeze(0).numpy()
    fake_mapping = seq(images=fake_mapping)
    fake_mapping = torch.tensor(fake_mapping).unsqueeze(0).type(torch.cuda.FloatTensor).to(device=device)

    zero = torch.zeros_like(fake_mapping)
    one = torch.ones_like(fake_mapping)
    fake_mapping = torch.where(fake_mapping > 0.1, one, zero)

    fake_images = model.netG(fake_mapping)
    
    save_mask = fake_mapping.squeeze()
    save_image = fake_images.squeeze()

    save_mask = Image.fromarray(save_mask.mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8).numpy())
    save_image = Image.fromarray(save_image.mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8).numpy().transpose(1, 2, 0))
    save_mask.convert('L').save(generate_path+'/Masks/%s.png' % generate_index)
    save_image.save(generate_path+'/Images/%s.jpg' % generate_index)
    print('saved: %s' % generate_index)
    generate_index = generate_index + 1

fake_dataset = BasicDataset(generate_path+'/Images', generate_path+'/Masks', 1.0)
train_set = torch.utils.data.ConcatDataset([train_set, fake_dataset, extra_dataset])
train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_args)

print('The number of training images = %d' % n_train)
print('The number of validate images = %d' % n_val)
print('The number of test images = %d' % n_test)
print('The number of overall training images = %d' % len(train_set))
print('The number of PH2 images = %d' % len(PH2_dataset))
print('The number of DermIS images = %d' % len(dermIS_dataset))
print('The number of DermQuest images = %d' % len(dermQuest_dataset))

##### Training process of UNet #####
##### Datasets: train_loader, val_loader, test_loader, PH2_loader, DermIS_loader, all_train_loader #####
n_epochs = 100
train_iters = int(len(train_set) * n_epochs)
total_iters = 0                # the total number of training iterations
unet_best_score = 0.0          # the best score of unet
PH2_best_score = 0.0           # the best score of PH2 dataset
DermIS_best_score = 0.0        # the best score of DermIS dataset

criterion = nn.CrossEntropyLoss() if opt.classes > 1 else nn.BCEWithLogitsLoss()
criterionGAN = networks.GANLoss(opt.gan_mode).to(device)
criterionL1 = torch.nn.L1Loss()

for epoch in range(n_epochs):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
    epoch_start_time = time.time()  # timer for entire epoch
    epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch        
    
    for i, data in enumerate(train_loader):  # inner loop within one epoch
        net.train()
        total_iters += opt.batch_size
        epoch_iter += opt.batch_size
        
        # update parameters of unet
        images = data['image'].to(device=device, dtype=torch.float32)
        # true_masks = data['mask'].to(device=device, dtype=torch.long)
        true_masks = data['mask'].to(device=device, dtype=torch.long).squeeze(0)
        with torch.cuda.amp.autocast(enabled=opt.amp):
        # with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=opt.amp):

            masks_pred = net(images)
            
            loss = criterion(masks_pred, true_masks.float())
            loss += jaccard_index_loss(torch.sigmoid(masks_pred.squeeze()), true_masks.float().squeeze())
            '''
            loss = 0.0
            for logit in masks_pred:
                # Resize labels for {100%, 75%, 50%, Max} logits
                _, _, H, W = logit.shape
                labels_ = resize_labels(true_masks, size=(H, W))
                loss += criterion(logit.squeeze(1), labels_.float().to(device))
            loss += jaccard_index_loss(torch.sigmoid(masks_pred[0].squeeze()), true_masks.float().squeeze())
            '''
            unet_loss = loss
            optimizer_unet.zero_grad(set_to_none=True)
            grad_scaler.scale(unet_loss).backward()
            # torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            grad_scaler.step(optimizer_unet)
            grad_scaler.update()
        
        # Evaluation round
        if total_iters % opt.display_freq == 0:
            test_score = evaluate(net, test_loader, device, opt.amp)                
            if test_score > unet_best_score:
                unet_best_score = test_score
            # scheduler_unet.step(test_score)

            message = 'Performance of UNet: '
            message += '%s: %.5f ' % ('unet_train_loss', unet_loss)
            message += '%s: %.5f ' % ('unet_test_score', test_score)
            message += '%s: %.5f ' % ('unet_best_score', unet_best_score)
            print(message)

    PH2_score = evaluate(net, PH2_loader, device, opt.amp)
    DermIS_score = evaluate(net, dermIS_loader, device, opt.amp)
    if PH2_score > PH2_best_score:
        PH2_best_score = PH2_score
    if DermIS_score > DermIS_best_score:
        DermIS_best_score = DermIS_score
    message = 'Performance on out-domain dataset: '
    message += '%s: %.5f ' % ('PH2_score', PH2_score)
    message += '%s: %.5f ' % ('PH2_best_score', PH2_best_score)
    message += '%s: %.5f ' % ('DermIS_score', DermIS_score)
    message += '%s: %.5f ' % ('DermIS_best_score', DermIS_best_score)
    print(message)

    scheduler_unet.step(unet_best_score)
        

    print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))


