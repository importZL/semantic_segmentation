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
from util.ISIC_loader import CarvanaDataset as BasicDataset
from util.dice_score import dice_loss
from util.config import get_config
from options.train_options import TrainOptions

from models_pix2pix import create_model
from models_pix2pix import networks
from unet import UNet
from deeplab import *
from swin_unet.vision_transformer import SwinUnet as ViT_seg

from betty.engine import Engine
from betty.configs import Config, EngineConfig
from betty.problems import ImplicitProblem


def jaccard_index(y_true, y_pred, smooth=1):
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
    return (1 - jaccard_index(y_true, y_pred)) * smooth

def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    JC_index = 0

    # iterate over the validation set
    with torch.no_grad():
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


opt = TrainOptions().parse()   # get training options
config = get_config(opt)
assert opt.cuda_index == int(opt.gpu_ids[0]), 'gpu types should be same'
device = torch.device('cuda:0' if opt.cuda_index == 0 else 'cuda:1')
save_path = './checkpoint_skin/'+'end2end-ISIC-40-'+str(opt.seg_model)+'-'+str(opt.loss_lambda)+'-'+time.strftime("%Y%m%d-%H%M%S")
if not os.path.exists(save_path):
    os.mkdir(save_path) 
unet_save_path = save_path+'/'+str(opt.seg_model)+'.pkl'  

##### Initialize logging #####
# logger = wandb.init(project='end2end-unet-ISIC', name="unet-200", resume='allow', anonymous='must')
logger = wandb.init(project='end2end-ISIC', name=str(opt.seg_model)+"-40-"+str(opt.loss_lambda), resume='allow', anonymous='must', mode='disabled')
logger.config.update(vars(opt))
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

##### create models: pix2pix, unet #####
model = create_model(opt)      # create a model given opt.model and other options
model.setup(opt)               # regular setup: load and print networks; create schedulers
# load pre-trained model
model_path =  '/data/li/semantic_segmentation1/pix2pix_model/20250207-145805-pix2pix-isic-40' # './pix2pix_model/20230315-055526-pix2pix-UNet-200' './pix2pix_model/20230320-194227-pix2pix-UNet-100' # 
model.load_model(model_path + '/pix2pix_discriminator.pkl', model_path + '/pix2pix_generator.pkl')

'''
img = Image.open('/home/li/workspace/data/ISIC2018/labels/ISIC_0000019.png').convert('L').resize((256, 256), resample=Image.NEAREST)
img_ndarray = np.asarray(img)
img_ndarray = np.expand_dims(img_ndarray, axis=0)
img_tensor = torch.as_tensor((img_ndarray / 255.0).copy()).contiguous().to('cuda:1', torch.float).unsqueeze(0)
fake_image = model.netG(img_tensor)

show_image = np.uint8(fake_image[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu').detach().numpy())
img = Image.fromarray(show_image)
img.save('test.png')
# cv2.imwrite('test.png', show_image)
sys.exit()
'''


if opt.seg_model == 'unet': 
    net = UNet(n_channels=opt.output_nc, n_classes=opt.classes, bilinear=opt.bilinear)
elif opt.seg_model == 'deeplab':
    net = DeepLabV3(num_classes=1)
elif opt.seg_model == 'vit':
    net = ViT_seg(config, img_size=opt.patch_size, num_classes=opt.classes).cuda()
    net.load_from(config)
# net = DeepLab(n_classes=1, n_blocks=[3, 4, 15, 3], atrous_rates=[4, 8, 12], multi_grids=[1, 2, 4], output_stride=8,)
net = net.to(device=device)


##### define optimizer for unet #####
optimizer_unet = optim.RMSprop(net.parameters(), lr=opt.unet_learning_rate, 
                                weight_decay=1e-8, momentum=0.9, foreach=True)
scheduler_unet = optim.lr_scheduler.ReduceLROnPlateau(optimizer_unet, 'max', patience=5)  # goal: maximize Dice score
# scheduler_unet = optim.lr_scheduler.CosineAnnealingLR(optimizer_unet, T_max=500, eta_min=1e-9)
grad_scaler = torch.cuda.amp.GradScaler(enabled=opt.amp)

##### prepare dataloader #####
dataset = BasicDataset(opt.dataroot+'/Images', opt.dataroot+'/Masks', 1.0, '_segmentation')
PH2_dataset = BasicDataset('/data2/li/workspace/data/PH2/Images', '/data2/li/workspace/data/PH2/Masks', 1.0, '_lesion') # use as the out-domain dataset
dermIS_dataset = BasicDataset('/data2/li/workspace/data/DermIS/Images', '/data2/li/workspace/data/DermIS/Masks', 1.0) # use as the extra dataset

n_test = 594
n_train = 32 # 165, 35, 9
n_val = 8
n_extra = len(dataset) - n_train - n_test - n_val
indices = list(range(len(dataset)))
train_set = Subset(dataset, indices[:n_train])
val_set = Subset(dataset, indices[n_train:n_train+n_val])
test_set = Subset(dataset, indices[-n_test:])

loader_args = dict(batch_size=opt.batch_size, num_workers=4, pin_memory=True)
train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_args)
val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
test_loader = DataLoader(test_set, shuffle=False, drop_last=True, **loader_args)
# out-domain test dataloader    
PH2_loader = DataLoader(PH2_dataset, shuffle=False, **loader_args)
dermIS_loader = DataLoader(dermIS_dataset, shuffle=False, **loader_args)

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

logging.info('The number of training images = %d' % n_train)
logging.info('The number of validate images = %d' % n_val)
logging.info('The number of test images = %d' % n_test)
logging.info('The number of overall training images = %d' % len(train_set))
logging.info('The number of PH2 images = %d' % len(PH2_dataset))
logging.info('The number of DermIS images = %d' % len(dermIS_dataset))

##### Datasets: train_loader, val_loader, test_loader, PH2_loader, DermIS_loader #####
# n_epochs = opt.n_epochs
train_iters = opt.n_epochs     # training iterations
total_iters = 0.0              # the total number of training iterations
val_best_score = 0.0           # the best val score of unet
unet_best_score = 0.0          # the best score of unet
PH2_best_score = 0.0           # the best score of PH2 dataset
DermIS_best_score = 0.0        # the best score of DermIS dataset

criterion = nn.CrossEntropyLoss() if opt.classes > 1 else nn.BCEWithLogitsLoss()
criterionGAN = networks.GANLoss(opt.gan_mode).to(device)
criterionL1 = torch.nn.L1Loss()


class Generator(ImplicitProblem):
    def training_step(self, batch):
        real_mask = batch['mask_pix2pix'].type(torch.cuda.FloatTensor).to(device)
        real_image = batch['image_pix2pix'].type(torch.cuda.FloatTensor).to(device)
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
        real_mask = batch['mask_pix2pix'].type(torch.cuda.FloatTensor).to(device)
        real_image = batch['image_pix2pix'].type(torch.cuda.FloatTensor).to(device)
        fake_image = self.netG(real_mask)

        fake_mask_image = torch.cat((real_mask, fake_image), 1) 
        pred_fake = self.module(fake_mask_image.detach())
        loss_D_fake = criterionGAN(pred_fake, False)
        # Real
        real_mask_image = torch.cat((real_mask, real_image), 1)
        pred_real = self.module(real_mask_image)
        loss_D_real = criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        return loss_D


show_index = 0
class Unet(ImplicitProblem):
    def training_step(self, batch):
        images = batch['image'].to(device=device, dtype=torch.float32)
        true_masks = batch['mask'].to(device=device, dtype=torch.long).squeeze(0)
        
        masks_pred = net(images)
        loss = criterion(masks_pred, true_masks.float())
        loss += jaccard_index_loss(torch.sigmoid(masks_pred.squeeze()), true_masks.float().squeeze())

        # fake images and masks
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
        fake_image = ((fake_image-fake_image.min()) / (fake_image.max()-fake_image.min())).detach()
        fake_image = fake_trans(fake_image.mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8)) / 255.0

        fake_pred = net(fake_image)
        fake_loss = criterion(fake_pred, fake_mask.float())
        fake_loss += jaccard_index_loss(torch.sigmoid(fake_pred.squeeze()), fake_mask.float().squeeze())
        
        global show_index
        show_index += 1
        if show_index % int(len(train_set)/10) == 0:
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
            show_index = 0
        
        unet_loss = loss + opt.loss_lambda * fake_loss
        return unet_loss


class Arch(ImplicitProblem):
    def training_step(self, batch):
        mask_valid = batch['mask'].type(torch.cuda.FloatTensor).to(device).squeeze(0)
        image_valid = batch['image'].type(torch.cuda.FloatTensor).to(device)
        mask_pred = self.unet(image_valid)
        loss_arch = criterion(mask_pred.squeeze(0), mask_valid.float())
        loss_arch += jaccard_index_loss(torch.sigmoid(mask_pred.squeeze()), mask_valid.float().squeeze())
        return loss_arch


class SSEngine(Engine):

    @torch.no_grad()
    def validation(self):
        global val_best_score
        val_score = evaluate(self.unet.module, val_loader, device, opt.amp)
        
        message = 'Performance of UNet: '
        message += '%s: %.5f ' % ('unet_val_score', val_score)
        logging.info(message)
        logger.log({'val_score': val_score})
        if val_score > val_best_score:
            val_best_score = val_score
            torch.save(net.state_dict(), unet_save_path)
        
        if self.global_step % len(train_set) == 0 and self.global_step:
            scheduler_unet.step(val_best_score)
        '''
        if self.global_step > 4000:
            torch.save(net.state_dict(), './model_soups_200_1/model_'+str(self.global_step)+'.pt')
        
            global unet_best_score, PH2_best_score, DermIS_best_score
            test_score = evaluate(self.unet.module, test_loader, device, opt.amp)
            if test_score > unet_best_score:
                unet_best_score = test_score
                torch.save(net.state_dict(), unet_save_path)
            message = 'Performance of UNet: '
            message += '%s: %.5f ' % ('unet_test_score', test_score)
            message += '%s: %.5f ' % ('unet_best_score', unet_best_score)
            logging.info(message)
            logger.log({'unet_test_score': test_score})
                
            PH2_score = evaluate(self.unet.module, PH2_loader, device, opt.amp)
            DermIS_score = evaluate(self.unet.module, dermIS_loader, device, opt.amp)
            if PH2_score > PH2_best_score:
                PH2_best_score = PH2_score
            if DermIS_score > DermIS_best_score:
                DermIS_best_score = DermIS_score
            message = 'Performance on out-domain dataset: '
            message += '%s: %.5f ' % ('PH2_score', PH2_score)
            message += '%s: %.5f ' % ('PH2_best_score', PH2_best_score)
            message += '%s: %.5f ' % ('DermIS_score', DermIS_score)
            message += '%s: %.5f ' % ('DermIS_best_score', DermIS_best_score)
            logging.info(message)
            logger.log({'PH2_score': PH2_score,
                        'DermIS_score': DermIS_score,})
        '''
            


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

optimizer_arch = torch.optim.Adam(networks.arch_parameters(), lr=opt.arch_lr, betas=(0.5, 0.999), weight_decay=1e-5)
arch = Arch(
    name='arch',
    module=net,
    optimizer=optimizer_arch,
    train_data_loader=val_loader,
    config=outer_config,
    device=device,
)

problems = [netG, netD, unet, arch]
l2u = {netG: [unet], unet: [arch]}
u2l = {arch: [netG]}
# l2u = {}
# u2l = {}
dependencies = {"l2u": l2u, "u2l": u2l}

engine = SSEngine(config=engine_config, problems=problems, dependencies=dependencies)
engine.run()
torch.save(net.state_dict(), save_path+'/unet_final.pkl')


