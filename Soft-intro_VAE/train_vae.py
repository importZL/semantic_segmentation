import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision

# standard
import os
import time
import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import matplotlib
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from datasets import *
from Utilities import *
from models import *
from loss import *
import torch
torch.cuda.empty_cache()

matplotlib.use('Agg')
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@hydra.main(version_base=None,config_path="config", config_name="train_vae")
def train(cfg):
    if cfg.params.seed != -1:
        random.seed(cfg.params.seed)
        np.random.seed(cfg.params.seed)
        torch.manual_seed(cfg.params.seed)
        torch.cuda.manual_seed(cfg.params.seed)
        torch.backends.cudnn.deterministic = True
        print("random seed: ", cfg.params.seed)
    
    save_results_path = os.path.join(cfg.paths.root_dir, "VAE_val_results")
    os.makedirs(save_results_path, exist_ok=True)
    save_pth_path = os.path.join(cfg.paths.checkpoints_dir,cfg.experiment_name)
    os.makedirs(save_pth_path, exist_ok=True)
    model = instantiate(cfg.model.init).to(device)
    # torch.save(model, './model.pth')
    # exit()

    writer = SummaryWriter(cfg.experiment_path)
    # Create the dataset
    train_dataset = instantiate(cfg.datas.train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.params.batch_size)

    val_dataset = instantiate(cfg.datas.val)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.params.test_batch_size, shuffle=False)
    
    fixed_z = torch.randn(
                cfg.params.test_batch_size,
                cfg.params.test_img_num,
                cfg.model.names.latent_dim).to(device)
    
    if cfg.params.recon_loss_type == 'vgg':
        vgg_loss = VGGLoss().to(device)
    
    if cfg.pretrained:
        # Load the trained model weights
        checkpoint_path = os.path.join(
                save_pth_path,
                # specify your weight path
                'generator_epoch5_batch0.pth')
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])

    optimizer_e = instantiate(
                cfg.optimizers.encoder,
                )(params=model.encoder.parameters())
    optimizer_d = instantiate(
                cfg.optimizers.decoder,
                )(params=model.decoder.parameters())
    
    e_scheduler = instantiate(cfg.schedulers.encoder)(optimizer=optimizer_e)
    d_scheduler = instantiate(cfg.schedulers.decoder)(optimizer=optimizer_d)

    start_time = time.time()
    # normalizing factor, 's' in the paper
    scale = 1 / (
        cfg.model.init.img_shape[0] *
        cfg.model.init.img_shape[1] *
        cfg.model.init.img_shape[2])

    cur_iter = 0
    kls_real = []
    kls_fake = []
    kls_rec = []
    rec_errs = []

    for epoch in range(cfg.params.start_epoch, cfg.params.num_epochs):
        # save models
        if epoch % cfg.params.save_interval == 0 and epoch > 0:
            save_epoch = (epoch // cfg.params.save_interval) * \
                        cfg.params.save_interval
            
            save_checkpoint(model, save_epoch, cur_iter, save_pth_path)

        model.train()
        batch_kls_real = []
        batch_kls_fake = []
        batch_kls_rec = []
        batch_rec_errs = []

        for it, batch in enumerate(train_loader):
            edge_tensor, rgb_tensor = batch
            edge_tensor, rgb_tensor = Normalize(edge_tensor).to(device), Normalize(rgb_tensor).to(device)
            real_A = edge_tensor; real_B = rgb_tensor

            # --------------train----------------
            if it < cfg.params.num_vae:
                # vanilla VAE training, optimizeing the ELBO for both encoder and decoder
                # =========== Update E, D ================
                real_mu, real_logvar, z, rec = model(real_A ,real_B)
                if cfg.params.recon_loss_type != "vgg":
                    loss_rec = calc_reconstruction_loss(
                        real_B, rec, loss_type=cfg.params.recon_loss_type,
                        reduction="mean")
                else:
                    loss_rec = vgg_loss(real_B, rec)
                
                loss_kl = calc_kl(real_logvar, real_mu, reduce="mean")
                loss = cfg.params.cfg.params.beta_rec * loss_rec + \
                    cfg.params.beta_rec_kl * loss_kl

                optimizer_e.zero_grad()
                optimizer_d.zero_grad()
                loss.backward()
                optimizer_e.step()
                optimizer_d.step()

                if it % cfg.params.cfg.params.test_iters == 0:
                    info = "\nEpoch[{}]({}/{}): time: {:4.4f}: ".format(epoch, it, len(train_loader),
                                                                        time.time() - start_time)
                    info += 'Rec: {:.4f}, KL: {:.4f}, '.format(loss_rec.data.cpu(), loss_kl.data.cpu())
                    print(info)
            else:
                # soft-intro-vae training
                # generate random noise to produce 'fake' later
                noise_batch = torch.randn(
                    size=(
                        real_A.shape[0],
                        cfg.model.names.latent_dim)).to(device)

                # =========== Update E ================
                for param in model.encoder.parameters():
                    param.requires_grad = True
                for param in model.decoder.parameters():
                    param.requires_grad = False
                # print(real_A.shape,noise_batch.shape)
                # generate 'fake' data
                fake = model.decode(real_A, noise_batch)
                # optimize for real data
                real_mu, real_logvar = model.encode(real_B)
                z = reparameterization(real_mu, real_logvar)
                rec = model.decode(real_A,z)  # reconstruction
                if cfg.params.recon_loss_type != "vgg":
                    loss_rec = calc_reconstruction_loss(
                        real_B, rec, loss_type=cfg.params.recon_loss_type,
                        reduction="mean")
                else:
                    loss_rec = vgg_loss(real_B, rec)
                # KLD loss for the real data
                lossE_real_kl = calc_kl(real_logvar, real_mu, reduce="mean")

                # prepare the fake data for the expELBO
                fake_mu, fake_logvar, z_fake, rec_fake = model(real_A,fake.detach())
                # we also consider the reconstructions as 'fake' data, as they are output of the decoder
                rec_mu, rec_logvar, z_rec, rec_rec = model(real_A,rec.detach())

                # KLD loss for the fake data
                fake_kl_e = calc_kl(fake_logvar, fake_mu, reduce="none")
                rec_kl_e = calc_kl(rec_logvar, rec_mu, reduce="none")

                # reconstruction loss for the fake data
                if cfg.params.recon_loss_type != "vgg":
                    loss_fake_rec = calc_reconstruction_loss(
                                        fake, rec_fake,
                                        loss_type=cfg.params.recon_loss_type,
                                        reduction="none")
                    loss_rec_rec = calc_reconstruction_loss(
                                        rec, rec_rec,
                                        loss_type=cfg.params.recon_loss_type,
                                        reduction="none")
                else:
                    loss_fake_rec = vgg_loss(fake, rec_fake)
                    loss_rec_rec = vgg_loss(rec, rec_rec)

                # expELBO
                exp_elbo_fake = (
                    -2 * scale * (cfg.params.beta_rec * loss_fake_rec +
                                  cfg.params.beta_neg * fake_kl_e)
                                ).exp().mean()
                exp_elbo_rec = (
                    -2 * scale * (cfg.params.beta_rec * loss_rec_rec +
                                  cfg.params.beta_neg * rec_kl_e)
                                ).exp().mean()

                # total loss
                lossE = scale * (cfg.params.beta_rec * loss_rec +
                                 cfg.params.beta_kl * lossE_real_kl) + \
                                (exp_elbo_fake + exp_elbo_rec) * 0.25
                
                # backprop
                optimizer_e.zero_grad()
                lossE.backward()
                optimizer_e.step()

                # ========= Update D ==================
                for param in model.encoder.parameters():
                    param.requires_grad = False
                for param in model.decoder.parameters():
                    param.requires_grad = True

                # generate fake
                fake = model.decode(real_A, noise_batch)
                rec = model.decode(real_A,z.detach())
                # ELBO loss for real -- just the reconstruction, KLD for real doesn't affect the decoder
                if cfg.params.recon_loss_type != "vgg":
                    loss_rec = calc_reconstruction_loss(
                        real_B, rec, loss_type=cfg.params.recon_loss_type,
                        reduction="mean")
                else:
                    loss_rec = vgg_loss(real_B, rec)

                # prepare fake data for ELBO
                rec_mu, rec_logvar = model.encode(rec)
                z_rec = reparameterization(rec_mu, rec_logvar)

                fake_mu, fake_logvar = model.encode(fake)
                z_fake = reparameterization(fake_mu, fake_logvar)

                rec_rec = model.decode(real_A,z_rec.detach())
                rec_fake = model.decode(real_A,z_fake.detach())
                if cfg.params.recon_loss_type != "vgg":
                    loss_rec_rec = calc_reconstruction_loss(
                        rec.detach(), rec_rec, 
                        loss_type=cfg.params.recon_loss_type, 
                        reduction="mean")
                    
                    loss_rec_fake = calc_reconstruction_loss(
                        fake.detach(), rec_fake, 
                        loss_type=cfg.params.recon_loss_type,
                        reduction="mean")
                else:
                    loss_rec_rec = vgg_loss(rec.detach(), rec_rec)
                    loss_rec_fake = vgg_loss(fake.detach(), rec_fake)

                fake_kl = calc_kl(fake_logvar, fake_mu, reduce="mean")
                rec_kl = calc_kl(rec_logvar, rec_mu, reduce="mean")

                lossD = cfg.params.beta_rec * loss_rec + 0.5 * \
                    cfg.params.beta_kl * (fake_kl + rec_kl) + \
                    cfg.params.gamma_r * 0.5 * cfg.params.beta_rec * \
                    (loss_rec_rec + loss_rec_fake)

                lossD = scale * lossD

                optimizer_d.zero_grad()
                lossD.backward()
                optimizer_d.step()
                # Log losses to TensorBoard
                writer.add_scalar('Loss/D', lossD.item(), global_step=cur_iter)
                writer.add_scalar('Loss/E', lossE.item(), global_step=cur_iter)

                writer.add_scalar('SubLoss/Rec', loss_rec.item(), global_step=cur_iter)
                writer.add_scalar('SubLoss/Kl_E', lossE_real_kl.item(), global_step=cur_iter)
                writer.add_scalar('SubLoss/KL_F', rec_kl.item(), global_step=cur_iter)
                writer.add_scalar('SubLoss/KL_R', fake_kl.item(), global_step=cur_iter)
                writer.add_scalar('SubLoss/expELBO_R', exp_elbo_rec.item(), global_step=cur_iter)
                writer.add_scalar('SubLoss/expELBO_F', exp_elbo_fake.item(), global_step=cur_iter)
                writer.add_scalar('SubLoss/DIFF_Kl_F', (fake_kl - lossE_real_kl).item(), global_step=cur_iter)

                if cur_iter % cfg.params.test_iters == 0:
                    info = "\nEpoch[{}]({}/{}): time: {:4.4f}: ".format(epoch, it, len(train_loader),
                                                                        time.time() - start_time)
                    info += 'Rec: {:.4f}, '.format(loss_rec.data.cpu())
                    info += 'Kl_E: {:.4f}, expELBO_R: {:.4e}, expELBO_F: {:.4e}, '.format(lossE_real_kl.data.cpu(),
                                                                                    exp_elbo_rec.data.cpu(),
                                                                                    exp_elbo_fake.cpu())
                    info += 'Kl_F: {:.4f}, KL_R: {:.4f}'.format(rec_kl.data.cpu(), fake_kl.data.cpu())
                    info += ' DIFF_Kl_F: {:.4f}'.format(-lossE_real_kl.data.cpu() + fake_kl.data.cpu())
                    print(info)

                if torch.isnan(lossE) or torch.isnan(lossD):
                    plt.close('all')
                    raise SystemError("loss is NaN.")
            cur_iter += 1

        if epoch % cfg.params.visualize_epoch == 0:
            result_img = make_img(
                                val_loader, model.decoder, fixed_z,
                                cfg.params.test_img_num, real_A.size(2))

            save_img_path = os.path.join(
                                save_results_path,
                                f'gen_epoch{epoch}_batch{it}.png')

            torchvision.utils.save_image(
                                result_img, save_img_path,
                                nrow=cfg.params.test_img_num + 1)

        e_scheduler.step()
        d_scheduler.step()

        if epoch > cfg.params.num_vae - 1:
            kls_real.append(np.mean(batch_kls_real))
            kls_fake.append(np.mean(batch_kls_fake))
            kls_rec.append(np.mean(batch_kls_rec))
            rec_errs.append(np.mean(batch_rec_errs))

    return model


if __name__ == '__main__':
    # hyperparameters
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = train()

