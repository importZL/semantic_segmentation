import numpy as np
import warnings
warnings.filterwarnings("ignore")
import torch
from torch import nn, optim
import torchvision
from datasets import *
from models import *
from Utilities import *
from loss import *
import  os
#import itertools
import time
#import pdb
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from functools import partial
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@hydra.main(version_base=None,config_path="config", config_name="train")
def train(cfg):
	# Random seeds (optional)
	torch.manual_seed(1); np.random.seed(1)
	torch.backends.cudnn.deterministic = True

	# save_results_path = os.path.join(os.path.join(cfg.paths.root_dir, "image_results"),cfg.experiment_name)
	# os.makedirs(save_pth_path, exist_ok=True)
	save_results_path = os.path.join(cfg.paths.root_dir, "BCGAN_results")
	os.makedirs(save_results_path, exist_ok=True)
	save_pth_path = os.path.join(cfg.paths.checkpoints_dir,cfg.experiment_name)
	os.makedirs(save_pth_path, exist_ok=True)

	#OmegaConf.resolve(cfg)
	#print(OmegaConf.to_yaml(cfg))
	model = instantiate(cfg.model.init)
	writer = SummaryWriter(cfg.experiment_path)
	# Create the dataset
	train_dataset = instantiate(cfg.datas.train)
	train_loader = torch.utils.data.DataLoader(
		train_dataset, batch_size=cfg.params.batch_size, 
		drop_last=True, shuffle=True)

	val_dataset = instantiate(cfg.datas.val)
	val_loader = torch.utils.data.DataLoader(
		val_dataset, batch_size=cfg.params.test_batch_size,
		drop_last=True, shuffle=False)

	generator = model.generator.to(device)
	encoder = model.encoder.to(device)
	D_VAE = model.D_VAE.to(device)
	D_LR = model.D_LR.to(device)

	optimizer_E = instantiate(cfg.optimizers.encoder)(encoder.parameters())
	optimizer_G = instantiate(cfg.optimizers.decoder)(generator.parameters())
	optimizer_D_VAE = instantiate(cfg.optimizers.DVAE)(D_VAE.parameters())
	optimizer_D_LR = instantiate(cfg.optimizers.DLR)(D_LR.parameters())

	scheduler_E = instantiate(cfg.schedulers.encoder)(optimizer_E)
	scheduler_G = instantiate(cfg.schedulers.decoder)(optimizer_G)
	scheduler_D_VAE = instantiate(cfg.schedulers.DVAE)(optimizer_D_VAE)
	scheduler_D_LR = instantiate(cfg.schedulers.DLR)(optimizer_D_LR)
	
	# For adversarial loss
	Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

	# For adversarial loss (optional to use)
	#criterion_GAN = torch.nn.MSELoss(reduction='sum').to(device)
	criterion_GAN = partial(compute_GANloss, loss_func=torch.nn.MSELoss().to(device))
	criterion_pixel = torch.nn.L1Loss().to(device)
	if cfg.params.recon_loss_type == 'vgg':
		criterion_latent = VGGLoss()
	else:
		criterion_latent = torch.nn.L1Loss().to(device)
	criterion_kl = compute_KLloss

	#criterion_kl = torch.nn.KLDivLoss().to(device)
	fixed_z = var(
			torch.randn(cfg.params.test_batch_size,
			cfg.params.test_img_num,
			cfg.model.names.latent_dim)).to(device)

	# Initialize a counter for the total number of iterations
	global_step = 0

	for e in range(cfg.params.num_epochs):
		start = time.time()
		for idx, data in enumerate(train_loader):
			# Increment the global step counter
			global_step += 1

			########## Process Inputs ##########
			edge_tensor, rgb_tensor = data
			edge_tensor, rgb_tensor = Normalize(edge_tensor).to(device), Normalize(rgb_tensor).to(device)
			real_A = edge_tensor;real_B = rgb_tensor

			half_batch = real_A.size(0)//2
			vae_real_A = real_A[:half_batch, ...]
			vae_real_B = real_B[:half_batch, ...]
			lr_real_A = real_A[half_batch:, ...]
			lr_real_B = real_B[half_batch:, ...]

			noise = Variable(Tensor(np.random.normal(0, 1, (half_batch, cfg.model.names.latent_dim))))

			#-------------------------------
			#  Train Generator and Encoder
			#------------------------------
			encoder.train(); generator.train()
			set_requires_grad(D_VAE)
			set_requires_grad(D_LR)
			set_requires_grad(encoder, True)
			set_requires_grad(generator, True)
			optimizer_E.zero_grad(); optimizer_G.zero_grad()

			mean, log_var = encoder(vae_real_B) # VAE encode
			z_encoded = reparameterization(mean, log_var)
			# KL loss
			# kl_loss = criterion_kl(z_encoded,noise)
			kl_loss = criterion_kl(mean, log_var).to(device)

			#generator loss for VAE-GAN
			loss_VAE_GAN, fake_B_VAE = loss_generator(generator, vae_real_A, z_encoded, D_VAE, criterion_GAN)

			#generator loss for LR-GAN
			loss_LR_GAN, fake_B_LR = loss_generator(generator, lr_real_A, noise, D_LR, criterion_GAN)


			#l1 loss between generated image and real image (VAE-GAN)
			l1_image = loss_image(vae_real_B, fake_B_VAE, criterion_pixel)

			loss_GE = loss_VAE_GAN + loss_LR_GAN + cfg.params.lambda_pixel*l1_image + cfg.params.lambda_kl*kl_loss

			loss_GE.backward(retain_graph=True)
			# Update E
			optimizer_E.step()

			#latent loss between encoded z and noise
			l1_latent = loss_latent(fake_B_LR, encoder, noise, criterion_latent)

			loss_G = cfg.params.lambda_latent*l1_latent

			loss_G.backward()
			# Update G
			optimizer_G.step()
			

			#----------------------------------
			#  Train Discriminator (cVAE-GAN)
			#----------------------------------
			set_requires_grad(D_VAE, True)
			set_requires_grad(D_LR, True)
			set_requires_grad(encoder)
			set_requires_grad(generator)
			D_VAE.train()
			optimizer_D_VAE.zero_grad()
			#loss for D_VAE
			loss_D_VAE = loss_discriminator(D_VAE, fake_B_VAE, vae_real_B, criterion_GAN)
			loss_D_VAE.backward()
			optimizer_D_VAE.step()

			#---------------------------------
			#  Train Discriminator (cLR-GAN)
			#---------------------------------
			D_LR.train()
			optimizer_D_LR.zero_grad()
			#loss for D_LR
			loss_D_LR = loss_discriminator(D_LR, fake_B_LR, lr_real_B, criterion_GAN)
			loss_D_LR.backward()
			optimizer_D_LR.step()

			# Log losses to TensorBoard
			writer.add_scalar('Loss/G', loss_G.item(), global_step=global_step)
			writer.add_scalar('Loss/D_VAE', loss_D_VAE.item(), global_step=global_step)
			writer.add_scalar('Loss/D_LR', loss_D_LR.item(), global_step=global_step)

			""" Optional TODO: 
				1. You may want to visualize results during training for debugging purpose
				2. Save your model every few iterations
			"""
			if idx % 1000 == 0:  # visualize every 1000 batches
				result_img = make_img(
									val_loader, generator, fixed_z, 
									cfg.params.test_img_num, vae_real_A.size(2))
				
				save_img_path = os.path.join(
									save_results_path, 
									f'gen_epoch{e}_batch{idx}.png')
				
				torchvision.utils.save_image(
									result_img, save_img_path, 
									nrow=cfg.params.test_img_num + 1)


			if idx % 500 == 0:  # save every 500 batches
				torch.save(generator.state_dict(), os.path.join(save_pth_path, f'generator_epoch{e}_batch{idx}.pth'))
				torch.save(encoder.state_dict(), os.path.join(save_pth_path, f'encoder_epoch{e}_batch{idx}.pth'))
				torch.save(D_VAE.state_dict(), os.path.join(save_pth_path, f'D_VAE_epoch{e}_batch{idx}.pth'))
				torch.save(D_LR.state_dict(), os.path.join(save_pth_path, f'D_LR_epoch{e}_batch{idx}.pth'))

		print(f'Epoch [{e+1}/{cfg.params.num_epochs}], Step [{global_step}], Loss G: {loss_G.item()}, Loss D_VAE: {loss_D_VAE.item()}, Loss D_LR: {loss_D_LR.item()}')

		scheduler_E.step()
		scheduler_G.step()
		scheduler_D_VAE.step()
		scheduler_D_LR.step()

	end = time.time()
	print(f'Training time: {end-start}')
		
	return 


if __name__ == '__main__':
	train()



		