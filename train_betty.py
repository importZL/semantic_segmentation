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
from util.JSRT_loader import CarvanaDataset as BasicDataset
from util.dice_score import dice_loss
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
save_path = './checkpoint/'+time.strftime("%Y%m%d-%H%M%S"+'-pix2pix-unet-9-00')
if not os.path.exists(save_path):
    os.mkdir(save_path) 
unet_save_path = save_path+'/unet_9.pkl'  

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
dataset = BasicDataset(opt.dataroot+'/Images', opt.dataroot+'/Masks', 1.0)
SZ_dataset = BasicDataset('../data/SZ/Images', '../data/SZ/Masks', 1.0, '_mask') # use as the out-domain dataset
NIH_dataset = BasicDataset('../data/NIH/Images', '../data/NIH/Masks', 1.0, '_mask') # use as the extra dataset
NLM_dataset = BasicDataset('../data/NLM/Images', '../data/NLM/Masks', 1.0) # use as the out-domain dataset

n_test = int(len(dataset)*(opt.test_percent/100))
n_train = 9 # 165, 35, 9
n_val = len(dataset) - n_train - n_test
train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(0))

len_extra = int(len(NIH_dataset) * 0.0)
extra_dataset, _ = random_split(NIH_dataset, [len_extra, len(NIH_dataset)-len_extra], generator=torch.Generator().manual_seed(0))
train_set = torch.utils.data.ConcatDataset([train_set, extra_dataset])

loader_args = dict(batch_size=opt.batch_size, num_workers=4, pin_memory=True)
train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_args)
val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
test_loader = DataLoader(test_set, shuffle=False, drop_last=True, **loader_args)
# NLM: out-domain dataset    
NLM_loader = DataLoader(NLM_dataset, shuffle=False, **loader_args)
# NIH_loader = DataLoader(NIH_dataset, shuffle=False, **loader_args)
SZ_loader = DataLoader(SZ_dataset, shuffle=False, **loader_args)


##### Pre-Training process of Pix2Pix model #####
total_iters = 0
pix2pix_set = torch.utils.data.ConcatDataset([train_set, val_set])
all_train_loader = DataLoader(pix2pix_set, shuffle=True, drop_last=True, **loader_args)
logging.info('##### Start of Pix2Pix model Train #####')

for epoch in range(opt.epoch_count, opt.n_epochs):
    model.update_learning_rate()    # update learning rates in the beginning of every epoch.
    epoch_iter = 0
    for i, data in enumerate(all_train_loader):  # inner loop within one epoch
        total_iters += opt.batch_size
        epoch_iter += opt.batch_size
        model.set_input(data)         # unpack data from dataset and apply preprocessing
        model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
        
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
logging.info('The number of test images = %d' % n_test)
logging.info('The number of overall training images = %d' % len(train_set))
logging.info('The number of NLM images = %d' % len(NLM_dataset))
logging.info('The number of NIH images = %d' % len(NIH_dataset))
logging.info('The number of SZ images = %d' % len(SZ_dataset))

##### Training process of UNet #####
##### Datasets: train_loader, val_loader, test_loader, NLM_loader, SZ_loader, all_train_loader #####
n_epochs = 100
train_iters = int(len(train_set) * n_epochs)
total_iters = 0                # the total number of training iterations
unet_best_score = 0.0          # the best score of unet
NLM_best_score = 0.0           # the best score of NLM dataset
SZ_best_score = 0.0            # the best score of SZ dataset

criterion = nn.CrossEntropyLoss() if opt.classes > 1 else nn.BCEWithLogitsLoss()
criterionGAN = networks.GANLoss(opt.gan_mode).to(device)
criterionL1 = torch.nn.L1Loss()
class Generator(ImplicitProblem):
    def training_step(self, batch):
        real_mask = batch['mask'].type(torch.cuda.FloatTensor).to(device)
        real_image = batch['image'].type(torch.cuda.FloatTensor).to(device)
        fake_image = self.module(real_mask)

        fake_mask_image = torch.cat((real_mask, fake_image), 1)
        pred_fake = self.netD(fake_mask_image)
        loss_G_GAN = criterionGAN(pred_fake, True)
        # Second, G(A) = B
        loss_G_L1 = criterionL1(fake_image, real_image) * opt.lambda_L1
        # combine loss and calculate gradients
        loss_G = loss_G_GAN + loss_G_L1
        return loss_G


class Discriminator(ImplicitProblem):
    def training_step(self, batch):
        real_mask = batch['mask'].type(torch.cuda.FloatTensor).to(device)
        real_image = batch['image'].type(torch.cuda.FloatTensor).to(device)
        fake_image = self.netG(real_mask)

        fake_mask_image = torch.cat((real_mask, fake_image), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.module(fake_mask_image.detach())
        loss_D_fake = criterionGAN(pred_fake, False)
        # Real
        real_mask_image = torch.cat((real_mask, real_image), 1)
        pred_real = self.module(real_mask_image)
        loss_D_real = criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        return loss_D


class Unet(ImplicitProblem):
    def training_step(self, batch):
        images = batch['image'].to(device=device, dtype=torch.float32)
        true_masks = batch['mask'].to(device=device, dtype=torch.long).squeeze(0)

        fake_mask = next(iter(all_train_loader))['mask'].to('cpu', torch.float).squeeze(0).numpy()
        fake_mask = seq(images=fake_mask)
        fake_mask = torch.tensor(fake_mask).unsqueeze(0).type(torch.cuda.FloatTensor).to(device=device)
        zero = torch.zeros_like(fake_mask)
        one = torch.ones_like(fake_mask)
        fake_mask = torch.where(fake_mask > 0.1, one, zero)
        fake_image = model.netG(fake_mask)
        fake_mask = fake_mask.squeeze(0)

        masks_pred = net(images)
        fake_pred = net(fake_image)
        loss = criterion(masks_pred.squeeze(1), true_masks.float())
        loss += dice_loss(torch.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
        fake_loss = criterion(fake_pred.squeeze(1), fake_mask.float())
        fake_loss += dice_loss(torch.sigmoid(fake_pred.squeeze(1)), fake_mask.float(), multiclass=False)
        unet_loss = loss + 0.5 * fake_loss
        return unet_loss


class Arch(ImplicitProblem):
    def training_step(self, batch):
        mask_valid = batch['mask'].type(torch.cuda.FloatTensor).to(device).squeeze(0)
        image_valid = batch['image'].type(torch.cuda.FloatTensor).to(device)
        mask_pred = self.unet(image_valid)
        loss_arch = criterion(mask_pred.squeeze(0), mask_valid.float())
        loss_arch += dice_loss(torch.sigmoid(mask_pred.squeeze(0)), mask_valid.float(), multiclass=False)
        return loss_arch


class SSEngine(Engine):

    @torch.no_grad()
    def validation(self):
        global unet_best_score, NLM_best_score, SZ_best_score
        test_score = evaluate(self.unet.module, test_loader, device, opt.amp)
        if test_score > unet_best_score:
            unet_best_score = test_score
            torch.save(net.state_dict(), unet_save_path)
        message = 'Performance of UNet: '
        message += '%s: %.5f ' % ('unet_test_score', test_score)
        message += '%s: %.5f ' % ('unet_best_score', unet_best_score)
        logging.info(message)
        logger.log({'unet_test_score': test_score})
        if self.global_step % len(train_set) == 0:
            NLM_score = evaluate(self.unet.module, NLM_loader, device, opt.amp)
            SZ_score = evaluate(self.unet.module, SZ_loader, device, opt.amp)
            if NLM_score > NLM_best_score:
                NLM_best_score = NLM_score
            if SZ_score > SZ_best_score:
                SZ_best_score = SZ_score
            message = 'Performance on out-domain dataset: '
            message += '%s: %.5f ' % ('NLM_score', NLM_score)
            message += '%s: %.5f ' % ('NLM_best_score', NLM_best_score)
            message += '%s: %.5f ' % ('SZ_score', SZ_score)
            message += '%s: %.5f ' % ('SZ_best_score', SZ_best_score)
            logging.info(message)
            logger.log({'NLM_score': NLM_score,
                        'SZ_score': SZ_score,})
            scheduler_unet.step(unet_best_score)


outer_config = Config(retain_graph=True)
inner_config = Config(type="darts", unroll_steps=opt.unroll_steps)
engine_config = EngineConfig(
    valid_step=opt.display_freq * opt.unroll_steps,
    train_iters=train_iters,
    roll_back=True,
)

netG = Generator(
    name='netG',
    module=model.netG,
    optimizer=model.optimizer_G,
    train_data_loader=train_loader,
    config=inner_config,
    device=device,
)

netD = Discriminator(
    name='netD',
    module=model.netD,
    optimizer=model.optimizer_D,
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

optimizer_arch = torch.optim.Adam(networks.arch_parameters(), lr=1e-4, betas=(0.5, 0.999), weight_decay=1e-3)
arch = Arch(
    name='arch',
    module=net,
    optimizer=optimizer_arch,
    train_data_loader=val_loader,
    config=outer_config,
    device=device,
)

problems = [netG, netD, unet, arch]
l2u = {netG: [unet], netG: [arch]}
u2l = {arch: [netG]}
dependencies = {"l2u": l2u, "u2l": u2l}

engine = SSEngine(config=engine_config, problems=problems, dependencies=dependencies)
engine.run()


