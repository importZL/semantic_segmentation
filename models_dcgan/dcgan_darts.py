import torch
import torch.nn as nn
from torch.autograd import Variable
import time as t
import os
import sys
from itertools import chain
from torchvision import utils
import logging
import wandb
import numpy as np
import torchvision.transforms as transforms

from models_dcgan.model_search_ss import Network
from models_dcgan.inception_score import get_inception_score

class Generator(torch.nn.Module):
    def __init__(self, channels, cuda_index):
        super().__init__()
        # Filters [1024, 512, 256]
        # Input_dim = 100
        # Output_dim = C (number of channels)
        self.pre_process = nn.Sequential(
            # Z latent vector 100
            nn.ConvTranspose2d(in_channels=100, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(True),
        )

        channel_list = [1024, 512, 256, 128, 64, 32]
        layers = 5
        self.main_module = Network(layers=layers, channel_list=channel_list, cuda_index=cuda_index)
        
        self.output = nn.Sequential(
            # State (32x128x128)
            nn.ConvTranspose2d(in_channels=32, out_channels=channels, kernel_size=4, stride=2, padding=1),
            # output of main module --> Image (Cx256x256)
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.pre_process(x)
        x = self.main_module(x)
        return self.output(x)


class Discriminator(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx64x64)
        # Output_dim = 1
        
        self.classifier = nn.Sequential(
            # Image (Cx256x256)
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # State (64x64x64)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # State (128x32x32)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x16x16)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x8x8)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.output = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0),
            # Output 1
            nn.Sigmoid(),
        )


    def forward(self, x):
        score = self.classifier(x)
        return self.output(score)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384 features
        x = self.main_module(x)
        return x.view(-1, 1024*4*4)

class DCGAN_MODEL(object):
    def __init__(self, args):
        logging.info("DCGAN model initalization.")
        self.args = args
        self.G = Generator(args.channels, args.cuda_index)
        self.D = Discriminator(args)
        self.C = args.channels
        self.trans = transforms.Compose([
            transforms.Normalize((0.5, ), (0.5, )),
        ])
        # binary cross entropy loss and optimizer
        self.loss = nn.BCELoss()

        self.cuda = False
        self.cuda_index = args.cuda_index
        # check if cuda is available
        self.check_cuda(args.cuda)

        # Using lower learning rate than suggested by (ADAM authors) lr=0.0002  and Beta_1 = 0.5 instead od 0.9 works better [Radford2015]
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=args.lr_dcgan, betas=(0.5, 0.999))
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=args.lr_d_dcgan, betas=(0.5, 0.999))
        self.arch_optimizer = torch.optim.Adam(self.G.main_module.arch_parameters(),
            lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

        # self.d_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.d_optimizer, float(args.epochs), eta_min=2e-4)
        # self.g_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.g_optimizer, float(args.epochs), eta_min=2e-4)
        self.d_scheduler = torch.optim.lr_scheduler.LambdaLR(self.d_optimizer, lr_lambda=self.lambda_rule)
        self.g_scheduler = torch.optim.lr_scheduler.LambdaLR(self.g_optimizer, lr_lambda=self.lambda_rule)

        self.epochs = args.n_epochs
        self.batch_size = args.batch_size

        # Set the logger
        self.number_of_images = 10

    def lambda_rule(self, epoch):
        lr_l = 1.0 - max(0, epoch + self.args.epoch_count - self.args.n_epochs) / float(self.args.n_epochs_decay + 1)
        return lr_l

    # cuda support
    def check_cuda(self, cuda_flag=False):
        if cuda_flag:
            self.cuda = True
            self.D.cuda(self.cuda_index)
            self.G.cuda(self.cuda_index)
            self.loss = nn.BCELoss().cuda(self.cuda_index)
            logging.info(f"Cuda enabled flag: {self.cuda}")

    def train_architect(self):
        # Train architecture
        if self.cuda:
            fake_labels = Variable(torch.zeros(self.batch_size)).cuda(self.cuda_index)
            z_valid = Variable(torch.randn(self.batch_size, 100, 1, 1)).cuda(self.cuda_index)
        else:
            z_valid = Variable(torch.randn(self.batch_size, 100, 1, 1))
            fake_labels = Variable(torch.zeros(self.batch_size))
        fake_images_valid = self.G(z_valid)
        outputs = self.D(fake_images_valid)
        d_loss_fake_valid = self.loss(outputs.flatten(), fake_labels)

        d_loss_valid = d_loss_fake_valid
        self.arch_optimizer.zero_grad()
        d_loss_valid.backward()
        self.arch_optimizer.step()

    def train_parameters(self, train_data):
            
        images = train_data['mask'].type(torch.cuda.FloatTensor).cuda(self.cuda_index)
        images = self.trans(images)
        z = torch.rand((self.batch_size, 100, 1, 1))
        real_labels = torch.ones(self.batch_size)
        fake_labels = torch.zeros(self.batch_size)

        if self.cuda:
            images, z = Variable(images).cuda(self.cuda_index), Variable(z).cuda(self.cuda_index)
            real_labels, fake_labels = Variable(real_labels).cuda(self.cuda_index), Variable(fake_labels).cuda(self.cuda_index)
        else:
            images, z = Variable(images), Variable(z)
            real_labels, fake_labels = Variable(real_labels), Variable(fake_labels)                

        # self.train_architect()
        # Train discriminator
        # Compute BCE_Loss using real images
        outputs = self.D(images)
        d_loss_real = self.loss(outputs.flatten(), real_labels)
        real_score = outputs

        # Compute BCE Loss using fake images
        if self.cuda:
            z = Variable(torch.randn(self.batch_size, 100, 1, 1)).cuda(self.cuda_index)
        else:
            z = Variable(torch.randn(self.batch_size, 100, 1, 1))
        fake_images = self.G(z)
        outputs = self.D(fake_images)
        d_loss_fake = self.loss(outputs.flatten(), fake_labels)
        fake_score = outputs

        # Optimize discriminator
        d_loss = d_loss_real + d_loss_fake
        self.D.zero_grad()
        d_loss.backward()
        self.d_optimizer.step()

        # Train generator
        # Compute loss with fake images
        if self.cuda:
            z = Variable(torch.randn(self.batch_size, 100, 1, 1)).cuda(self.cuda_index)
        else:
            z = Variable(torch.randn(self.batch_size, 100, 1, 1))
        fake_images = self.G(z)
        outputs = self.D(fake_images)
        g_loss = self.loss(outputs.flatten(), real_labels)

        # Optimize generator
        self.D.zero_grad()
        self.G.zero_grad()
        g_loss.backward()
        self.g_optimizer.step()

        # TensorBoard logging
        # Log the scalar values
        info_loss = {
            'dcgan_d_loss': d_loss.data,
            'dcgan_g_loss': g_loss.data
        }
        # Log the images while training
        info_visual = {
            'dcgan_real_mask': self.real_images(images, self.number_of_images),
            'dcgan_generated_mask': self.generate_img(z, self.number_of_images)
        }
        return info_loss, info_visual

    def evaluate(self, test_loader, D_model_path, G_model_path):
        self.load_model(D_model_path, G_model_path)
        z = Variable(torch.randn(self.batch_size, 100, 1, 1)).cuda(self.cuda_index)
        samples = self.G(z)
        samples = samples.mul(0.5).add(0.5)
        samples = samples.data.cpu()
        grid = utils.make_grid(samples)
        logging.info(f"Grid of 8x8 images saved to 'dgan_model_image.png'.")
        utils.save_image(grid, 'dgan_model_image.png')

    def real_images(self, images, number_of_images):
        if (self.C == 3):
            images = images.view(-1, self.C, 256, 256)[:self.number_of_images]
        else:
            images = images.view(-1, 256, 256)[:self.number_of_images]
            images = images.unsqueeze(1)
        images = utils.make_grid(images, nrow = self.number_of_images, padding = 0)
        images = images.mul(0.5).add(0.5)
        return self.to_np(images[0,:,:])

    def generate_img(self, z, number_of_images):
        samples = self.G(z)[:number_of_images]
        images = utils.make_grid(samples, nrow = self.number_of_images, padding=0)
        images = images.mul(0.5).add(0.5)
        return self.to_np(images[0,:,:])

    def to_np(self, x):
        return x.data.cpu().numpy()

    def save_model(self, save_path):
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        torch.save(self.G.state_dict(), save_path+'/dcgan_generator.pkl')
        torch.save(self.D.state_dict(), save_path+'/dcgan_discriminator.pkl')

    def load_model(self, D_model_filename, G_model_filename):
        D_model_path = os.path.join(os.getcwd(), D_model_filename)
        G_model_path = os.path.join(os.getcwd(), G_model_filename)
        self.D.load_state_dict(torch.load(D_model_path))
        self.G.load_state_dict(torch.load(G_model_path))
        # logging.info(f'Generator model loaded from {G_model_path}')
        # logging.info(f'Discriminator model loaded from {D_model_path}')

    def generate_latent_walk(self, number):
        if not os.path.exists('interpolated_images/'):
            os.makedirs('interpolated_images/')

        # Interpolate between twe noise(z1, z2) with number_int steps between
        number_int = 10
        z_intp = torch.FloatTensor(1, 100, 1, 1)
        z1 = torch.randn(1, 100, 1, 1)
        z2 = torch.randn(1, 100, 1, 1)
        if self.cuda:
            z_intp = z_intp.cuda()
            z1 = z1.cuda()
            z2 = z2.cuda()

        z_intp = Variable(z_intp)
        images = []
        alpha = 1.0 / float(number_int + 1)
        logging.info(f'alpha: {alpha}')
        for i in range(1, number_int + 1):
            z_intp.data = z1*alpha + z2*(1.0 - alpha)
            alpha += alpha
            fake_im = self.G(z_intp)
            fake_im = fake_im.mul(0.5).add(0.5) #denormalize
            images.append(fake_im.view(self.C,32,32).data.cpu())

        grid = utils.make_grid(images, nrow=number_int )
        utils.save_image(grid, 'interpolated_images/interpolated_{}.png'.format(str(number).zfill(3)))
        logging.info(f"Saved interpolated images to interpolated_images/interpolated_{str(number).zfill(3)}.")
