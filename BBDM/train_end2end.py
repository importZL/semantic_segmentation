"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import sys
from collections import OrderedDict

from torch.utils.data import DataLoader, Subset

from betty.engine import Engine
from betty.configs import Config, EngineConfig
from betty.problems import ImplicitProblem
from unet import UNet
import torch
from torch import optim
from torch import Tensor
from imgaug import augmenters as iaa
import wandb
import logging
import time
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from model.BrownianBridge.base.modules.diffusionmodules import openaimodel
import cv2
from PIL import Image

import yaml
from Register import Registers
import argparse
from utils import dict2namespace, get_runner, namespace2dict
from runners.utils import make_save_dirs, make_dir, get_dataset, get_dataset1, remove_file


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('-c', '--config', type=str, default='BB_base.yml', help='Path to the config file')
    parser.add_argument('-s', '--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('-r', '--result_path', type=str, default='results', help="The directory to save results")

    parser.add_argument('-t', '--train', action='store_true', default=False, help='train the model')
    parser.add_argument('--sample_to_eval', action='store_true', default=False, help='sample for evaluation')
    parser.add_argument('--sample_at_start', action='store_true', default=False, help='sample at start(for debug)')
    parser.add_argument('--save_top', action='store_true', default=False, help="save top loss checkpoint")

    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids, 0,1,2,3 cpu=-1')
    parser.add_argument('--port', type=str, default='12355', help='DDP master port')

    parser.add_argument('--resume_model', type=str, default=None, help='model checkpoint')
    parser.add_argument('--resume_optim', type=str, default=None, help='optimizer checkpoint')

    parser.add_argument('--max_epoch', type=int, default=None, help='optimizer checkpoint')
    parser.add_argument('--max_steps', type=int, default=None, help='optimizer checkpoint')

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        dict_config = yaml.load(f, Loader=yaml.FullLoader)

    namespace_config = dict2namespace(dict_config)
    namespace_config.args = args

    if args.resume_model is not None:
        namespace_config.model.model_load_path = args.resume_model
    if args.resume_optim is not None:
        namespace_config.model.optim_sche_load_path = args.resume_optim
    if args.max_epoch is not None:
        namespace_config.training.n_epochs = args.max_epoch
    if args.max_steps is not None:
        namespace_config.training.n_steps = args.max_steps

    dict_config = namespace2dict(namespace_config)

    return namespace_config, dict_config
            

# def jaccard_index(y_true, y_pred, smooth=1):
#     if y_pred.dim() != 2:
#         jac = 0.0
#         for i in range(y_pred.size()[0]):
#             intersection = torch.abs(y_true[i] * y_pred[i]).sum(dim=(-1, -2))
#             sum_ = torch.sum(torch.abs(y_true[i]) + torch.abs(y_pred[i]), dim=(-1, -2))
#             jac += (intersection + smooth) / (sum_ - intersection + smooth)
#         jac = jac / y_pred.size()[0]
#     else:
#         intersection = torch.abs(y_true * y_pred).sum(dim=(-1, -2))
#         sum_ = torch.sum(torch.abs(y_true) + torch.abs(y_pred), dim=(-1, -2))
#         jac = (intersection + smooth) / (sum_ - intersection + smooth)
#     #return (1 - jac) * smooth
#     return jac

# def jaccard_index_loss(y_true, y_pred, smooth=1):
#     return (1 - jaccard_index(y_true, y_pred)) * smooth


def jaccard_index(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return jaccard_index(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def jaccard_index_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else jaccard_index
    return 1 - fn(input, target, reduce_batch_first=True)

def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    JC_index = 0

    # iterate over the validation set
    with torch.no_grad():
        # for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        for i, batch in enumerate(dataloader):
            (image, x_name), (x_cond, mask_true) = batch

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            # mask_true = mask_true.to(device=device, dtype=torch.long)
            mask_true = mask_true.to(device=device, dtype=torch.long)
            
            # predict the mask
            mask_pred = net(image)

            assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
            mask_pred = (torch.sigmoid(mask_pred) > 0.5).float()
            # compute the Dice score
            JC_index += jaccard_index(mask_pred.squeeze(), mask_true.squeeze())

    net.train()
    return JC_index / max(num_val_batches, 1)


# parse options
config, dconfig = parse_args_and_config()
args = config.args
gpu_ids = args.gpu_ids
gpu_list = gpu_ids.split(",")
config.training.use_DDP = False
config.training.device = [torch.device(f"cuda:{gpu_list[0]}")]

device = torch.device(f"cuda:{gpu_list[0]}")
save_path = './checkpoint/'+'end2end-CVC-40-'+'unet'+'-'+time.strftime("%Y%m%d-%H%M%S")
if not os.path.exists(save_path):
    os.mkdir(save_path) 
unet_save_path = save_path+'/unet.pkl'
logger = wandb.init(project='end2end-ISIC', name="unet-BBDM-40", resume='allow', anonymous='must', mode='disabled')
logger.config.update(vars(args))
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
# dataloader = data.create_dataloader(opt)

train_dataset_all, test_dataset = get_dataset1(config.data)
indices = list(range(len(train_dataset_all)))
num_train = int(len(train_dataset_all)*0.8)
train_set = Subset(train_dataset_all, indices[:num_train])
val_set = Subset(train_dataset_all, indices[num_train:])
        

train_loader = DataLoader(train_set, num_workers=8, batch_size=2, shuffle=True)
val_loader = DataLoader(val_set, num_workers=8, batch_size=2, shuffle=True)

net = UNet(n_channels=3, n_classes=1, bilinear=False)
net = net.to(device=device)

# create trainer for our model
runner = Registers.runners[config.runner](config)
##### define optimizer for unet #####
optimizer_unet = optim.RMSprop(net.parameters(), lr=1e-4, 
                                weight_decay=1e-8, momentum=0.9, foreach=True)
scheduler_unet = optim.lr_scheduler.ReduceLROnPlateau(optimizer_unet, 'max', patience=5)  # goal: maximize Dice score
# scheduler_unet = optim.lr_scheduler.CosineAnnealingLR(optimizer_unet, T_max=500, eta_min=1e-9)
grad_scaler = torch.cuda.amp.GradScaler(enabled=True)

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
    transforms.ToTensor()
])

criterion = nn.BCEWithLogitsLoss()
val_best_score = 0.0           # the best val score of unet

class Generator(ImplicitProblem):
    def training_step(self, batch):
        
        loss = runner.loss_fn(net=runner.net, batch=batch, epoch=0, step=0, opt_idx=0, stage='train', write=False)
        return loss
    
show_index = 0
class Unet(ImplicitProblem):
    def training_step(self, batch):
        (x, x_name), (x_cond, x_cond_name) = batch        
        images = x.to(device=device, dtype=torch.float32)
        true_masks = x_cond_name.to(device=device)
        
        masks_pred = net(images)
        loss = criterion(masks_pred.squeeze(), true_masks.float().squeeze())
        loss += jaccard_index_loss(torch.sigmoid(masks_pred.squeeze()), true_masks.float().squeeze())

        # fake images and masks
        real_mask = true_masks.to('cpu', torch.float).numpy()
        fake_masks = []
        
        for i in range(real_mask.shape[0]):
            real_mask[i] = seq(images=real_mask[i])
            real_mask[i] = (real_mask[i] > 0.1) * 255.0
            fake_mask = Image.fromarray(real_mask[i]).convert("RGB")
            fake_mask = fake_trans(fake_mask)
            fake_mask = (fake_mask - 0.5) * 2.
            fake_mask.clamp_(-1., 1.)
            fake_masks.append(fake_mask.to(device=device))
        
        fake_masks = torch.stack(fake_masks, 0)
        fake_image = runner.net.sample(fake_masks, clip_denoised=config.testing.clip_denoised)

        fake_mask = fake_masks.detach()
        fake_image = ((fake_image-fake_image.min()) / (fake_image.max()-fake_image.min())).detach()
        # fake_image = fake_trans(fake_image.mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8)) / 255.0
        
        fake_pred = net(fake_image)
        fake_loss = criterion(fake_pred.squeeze(), (fake_mask[:,0,:,:]/2+0.5).float().squeeze())
        fake_loss += jaccard_index_loss(torch.sigmoid(fake_pred.squeeze()), (fake_mask[:,0,:,:]/2+0.5).float().squeeze())
        
        global show_index
        show_index += 1
        if show_index % int(len(train_set)/10) == 0:
            ims_dict = {}
            show_image = images[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu').numpy()
            show_mask = true_masks[0].mul(255).to('cpu').numpy()
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
            show_index = 0
        
        unet_loss = loss + 0.0 * fake_loss
        return unet_loss


class Arch(ImplicitProblem):
    def training_step(self, batch):
        (x, x_name), (x_cond, x_cond_name) = batch        
        image_valid = x.type(torch.cuda.FloatTensor).to(device)
        mask_valid = x_cond_name.type(torch.cuda.FloatTensor).to(device)
        mask_pred = self.unet(image_valid)
        loss_arch = criterion(mask_pred.squeeze(), mask_valid.float().squeeze())
        loss_arch += jaccard_index_loss(torch.sigmoid(mask_pred.squeeze()), mask_valid.float().squeeze())
        return loss_arch


class SSEngine(Engine):

    @torch.no_grad()
    def validation(self):
        global val_best_score
        val_score = evaluate(self.unet.module, val_loader, device, True)
        
        message = 'Performance of UNet: '
        message += '%s: %.5f ' % ('unet_val_score', val_score)
        logging.info(message)
        logger.log({'val_score': val_score})
        if val_score > val_best_score:
            val_best_score = val_score
            torch.save(net.state_dict(), unet_save_path)
        
        if self.global_step % 10 == 0 and self.global_step:
            scheduler_unet.step(val_best_score)            


outer_config = Config(retain_graph=True)
inner_config = Config(type="darts", unroll_steps=1)
engine_config = EngineConfig(
    valid_step=10,
    train_iters=5000,
    roll_back=True,
)

netG = Generator(
    name='netG',
    module=runner.net.denoise_fn,
    optimizer=runner.optimizer[0],
    train_data_loader=train_loader,
    config=inner_config,
    device=device,
)


unet = Unet(
    name='unet',
    module=net,
    optimizer=optimizer_unet,
    train_data_loader=train_loader,
    config=inner_config,
    device=device,
)

optimizer_arch = torch.optim.Adam(openaimodel.conv_arch_parameters(), lr=1e-6, betas=(0.5, 0.999), weight_decay=1e-5)
arch = Arch(
    name='arch',
    module=net,
    optimizer=optimizer_arch,
    train_data_loader=val_loader,
    config=outer_config,
    device=device,
)

problems = [netG, unet, arch]
l2u = {netG: [unet], unet: [arch]}
u2l = {arch: [netG]}
dependencies = {"l2u": l2u, "u2l": u2l}

engine = SSEngine(config=engine_config, problems=problems, dependencies=dependencies)
runner.net = runner.net.to(device=device)
net = net.to(device=device)
engine.run()
torch.save(net.state_dict(), save_path+'/unet_final.pkl')
torch.save(runner.net.state_dict(), save_path+'/bbdm_final.pkl')
torch.save(openaimodel.conv_arch_parameters(),  save_path+'/arch_final.pkl')
