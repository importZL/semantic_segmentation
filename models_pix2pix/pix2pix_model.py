import torch
from .base_model import BaseModel
from . import networks
import sys
import os
import torchvision.transforms as transforms

class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_mask', 'fake_image', 'real_image']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        self.trans = transforms.Compose([
            transforms.Normalize((0.5, ), (0.5, )),
        ])
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        networks.conv_arch = networks.conv_arch.cuda(opt.cuda_index)
        networks.upconv_arch = networks.upconv_arch.cuda(opt.cuda_index)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_arch_upconv = torch.optim.Adam(networks.upconv_arch_parameters(), lr=opt.arch_lr, betas=(0.5, 0.999), weight_decay=1e-3)
            self.optimizer_arch_conv = torch.optim.Adam(networks.conv_arch_parameters(), lr=opt.arch_lr, betas=(0.5, 0.999), weight_decay=1e-3)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_arch_upconv)
            self.optimizers.append(self.optimizer_arch_conv)

    def set_input(self, image, mask):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.real_mask = mask.type(torch.cuda.FloatTensor).to(self.device)
        self.real_image = image.type(torch.cuda.FloatTensor).to(self.device)
        # self.real_mask = self.trans(self.real_mask)
        # self.real_image = self.trans(self.real_image)
        # self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def set_input_1(self, input):
        self.real_mask = input['mask'].type(torch.cuda.FloatTensor).to(self.device)
        self.real_image = input['image'].type(torch.cuda.FloatTensor).to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_image = self.netG(self.real_mask)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_mask_image = torch.cat((self.real_mask, self.fake_image), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_mask_image.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_mask_image = torch.cat((self.real_mask, self.real_image), 1)
        pred_real = self.netD(real_mask_image)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_mask_image = torch.cat((self.real_mask, self.fake_image), 1)
        pred_fake = self.netD(fake_mask_image)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_image, self.real_image) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

    def optimize_architect(self, image, mask):
        real_mask = mask.type(torch.cuda.FloatTensor).to(self.device)
        real_image = image.type(torch.cuda.FloatTensor).to(self.device)
        # real_mask = self.trans(real_mask)
        # real_image = self.trans(real_image)

        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_arch_upconv.zero_grad()     # set arch's gradienets to zero
        self.optimizer_arch_conv.zero_grad()
        
        fake_image = self.netG(real_mask)                   # compute fake images: G(A)
        fake_mask_image = torch.cat((real_mask, fake_image), 1)
        pred_fake = self.netD(fake_mask_image)
        loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        loss_G_L1 = self.criterionL1(fake_image, real_image) * self.opt.lambda_L1
        # combine loss and calculate gradients
        loss_G = loss_G_GAN + loss_G_L1
        loss_G.backward()
             
        self.optimizer_arch_upconv.step()          # update arch's weights
        self.optimizer_arch_conv.step()

    def save_model(self, save_path):
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        torch.save(self.netD.state_dict(), save_path+'/pix2pix_discriminator.pkl')
        torch.save(self.netG.state_dict(), save_path+'/pix2pix_generator.pkl')

    def load_model(self, D_model_filename, G_model_filename):
        D_model_path = os.path.join(os.getcwd(), D_model_filename)
        G_model_path = os.path.join(os.getcwd(), G_model_filename)
        self.netD.load_state_dict(torch.load(D_model_path, map_location={'cuda:0':'cuda:0'}))
        self.netG.load_state_dict(torch.load(G_model_path, map_location={'cuda:0':'cuda:0'}))
