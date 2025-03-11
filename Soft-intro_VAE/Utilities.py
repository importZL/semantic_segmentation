import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import functools
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
from loss import VGGLoss

# Normalize image tensor
def Normalize(image):
	return (image/255.0-0.5)*2.0

# Denormalize image tensor
def Denormalize(tensor):
	return ((tensor+1.0)/2.0)*255.0

def visualize_inference(denorm_tensor, prefix, style_num, image_num, save_dir,title):
    """
    Visualizes the inference by saving side-by-side images of input and output.

    Args:
        denorm_tensor (Tensor): The denormalized generated image tensor.
        prefix (str): Prefix for saving the file.
        style_num (int): The style number to include in the filename.
        image_num (int): The image number to include in the filename.
        save_dir (str): The directory to save the images.
    """
    os.makedirs(save_dir, exist_ok=True)
    image_path = os.path.join(save_dir, f"{prefix}_style{style_num}_image{image_num}.png")
    # Convert the tensor to PIL image for saving
    image = denorm_tensor.permute(1, 2, 0).numpy()
    image = np.clip(image, 0, 255).astype(np.uint8)
    # pil_image = F.to_pil_image(denorm_tensor)
    # pil_image.save(image_path)
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    
    # Save the figure
    plt.savefig(image_path)
    plt.close()  # Close the figure to avoid display

def visualize_images(image, title, epoch, idx, save_path):
    os.makedirs(save_path, exist_ok=True)
    # Create a grid with 2 images per row for each pair of real and generated images
    grid_img = torchvision.utils.make_grid(image, nrow=2)
    # Convert to a numpy image
    np_grid_img = grid_img.permute(1, 2, 0).numpy()
    np_grid_img = np.clip(np_grid_img, 0, 255).astype(np.uint8)

    # Plot and save the images
    plt.figure(figsize=(16, 12))
    plt.imshow(np_grid_img)
    plt.title(title)
    plt.axis('off')
    
    # Save the figure
    file_name = f"{title}_epoch{epoch}_batch{idx}.png"
    save_full_path = os.path.join(save_path, file_name)
    plt.savefig(save_full_path)
    plt.close()  # Close the figure to avoid display



'''
    < var >
    Convert tensor to Variable
'''
def var(tensor, requires_grad=True):
    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    var = Variable(tensor.type(dtype), requires_grad=requires_grad)

    return var

'''
    < make_img >
    Generate images

    * Parameters
    dloader : Data loader for test data set
    G : Generator
    z : random_z(size = (N, img_num, z_dim))
        N : test img number / img_num : Number of images that you want to generate with one test img / z_dim : 8
    img_num : Number of images that you want to generate with one test img
'''
def make_img(dloader, G, z, img_num=5, img_size=128):
    G.eval()
    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    dloader = iter(dloader)
    img, _ = next(dloader)

    N = img.size(0)
    img = Normalize(var(img.type(dtype)))

    result_img = torch.FloatTensor(N * (img_num + 1), 3, img_size, img_size).type(dtype)

    for i in range(N):
        # original image to the leftmost
        result_img[i * (img_num + 1)] = img[i]

        # Insert generated images to the next of the original image
        for j in range(img_num):
            img_ = img[i].unsqueeze(dim=0)
            z_ = z[i, j, :].unsqueeze(dim=0)

            out_img = G(img_, z_)

            result_img[i * (img_num + 1) + j + 1] = (Denormalize(out_img)/255.0)


    # [-1, 1] -> [0, 1]
    #result_img = result_img / 2 + 0.5
    return result_img


def plot_distances(dist_list, save_path):
    """
    Plots the distances between the generated images.

    Args:
        dist_list (list): List of distances between the generated images.
        title (str): Title of the plot.
        save_path (str): Path to save the plot.
    """
    os.makedirs(save_path, exist_ok=True)
    fig_path = os.path.join(save_path, "distances.png")

    plt.figure(figsize=(16, 12))
    plt.plot(np.arange(len(dist_list)), np.array(dist_list), 'o-')
    plt.title("Distances between generated images")
    plt.xlabel("Image number")
    plt.ylabel("Distance")
    plt.savefig(fig_path)
    plt.close()  # Close the figure to avoid display


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_non_linearity(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=True)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(
            nn.LeakyReLU, negative_slope=0.2, inplace=True)
    elif layer_type == 'elu':
        nl_layer = functools.partial(nn.ELU, inplace=True)
    else:
        raise NotImplementedError(
            'nonlinearity activitation [%s] is not found' % layer_type)
    return nl_layer

def reparameterization(mean, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)#Returns a tensor with the same size as std that is filled with random numbers from a normal distribution with mean 0 and variance 1
    z = mean + eps * std

    return z

def loss_image(real_B, fake, criterion_pixel):
    loss_pixel = criterion_pixel(fake, real_B)
    return loss_pixel

def loss_latent(fake, E, noise, criterion_latent):
    encoded_z_tuple = E(fake)
    encoded_z = encoded_z_tuple[0]
    encoded_z = encoded_z.squeeze() # squeeze the dimension of 1
    loss_latent = criterion_latent(encoded_z, noise)
    return loss_latent

def loss_discriminator(D, fake, real_B, criterion):
    '''
    1. Forward real images into the discriminator
    2. Compute loss between Valid_label and dicriminator output on real images
    4. Forward fake images to the discriminator
    5. Compute loss between Fake_label and discriminator output on fake images
    6. sum real loss and fake loss as the loss_D
    7. we also need to output fake images generate by G(noise) for loss_generator computation
    '''
    real_output = D(real_B)
    real_output = real_output.squeeze()
    valid_label = torch.ones_like(real_output, requires_grad=False)
    real_loss = criterion(real_output, valid_label)

    fake_output = D(fake.detach())
    fake_output = fake_output.squeeze()
    fake_label = torch.zeros_like(fake_output, requires_grad=False)
    fake_loss = criterion(fake_output, fake_label)
    loss_D = real_loss + fake_loss
    # return total loss_D
    return loss_D

def loss_generator(G, real, z, D, criterion_GAN):
    '''
    loss_generator function is applied to compute loss for both generator G_AB and G_BA:
    For example, we want to compute the loss for G_AB.
    real2G will be the real image in domain A, then we map real2G into domain B to get fake B,
    then we compute the loss between D_B(fake_B) and valid, which is 1.
    The fake_B image will also be one of the outputs, since we want to use it in the loss_cycle_consis.
    '''

    # Generate fake images by forwarding real images from domain A/B through the generator G
    fake = G(real, z)

    # Forward the generated fake images through the corresponding discriminator D
    fake_pred = D(fake)
    valid = torch.ones_like(fake_pred, requires_grad=False)
    # Compute loss between the discriminator's predictions on the fake images and valid labels (which are all 1s)
    loss_G = criterion_GAN(fake_pred, valid)

    return loss_G, fake


# Helper function for intro_VAE
def load_model(model, pretrained):
    # weights = torch.load(pretrained)
    # pretrained_dict = weights['model']
    # model.load_state_dict(pretrained_dict)
    # model_dict = model.state_dict()
    weights = torch.load(pretrained, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    pretrained_dict = weights['model']
    
    # Filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model.state_dict()}
    
    # Handle missing keys if necessary
    model_dict = model.state_dict()
    model_dict.update(pretrained_dict)
    
    model.load_state_dict(model_dict)
    return model


def save_checkpoint(model, epoch, iteration, prefix=""):
    model_out_path = os.path.join(prefix , "model_epoch_{}_iter_{}.pth".format(epoch, iteration))
    state = {"epoch": epoch, "model": model.state_dict()}
    torch.save(state, model_out_path)

    print("model checkpoint saved @ {}".format(model_out_path))


def setup_grid(range_lim=4, n_pts=1000, device=torch.device("cpu")):
    x = torch.linspace(-range_lim, range_lim, n_pts)
    xx, yy = torch.meshgrid((x, x))
    zz = torch.stack((xx.flatten(), yy.flatten()), dim=1)
    return xx, yy, zz.to(device)


def format_ax(ax, range_lim):
    ax.set_xlim(-range_lim, range_lim)
    ax.set_ylim(-range_lim, range_lim)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.invert_yaxis()


def plot_vae_density(model, ax, test_grid, n_pts, batch_size, colorbar=False, beta_kl=1.0,
                     beta_recon=1.0, set_title=True, device=torch.device('cpu')):
    """ plots square grid and vae density """
    model.eval()
    xx, yy, zz = test_grid
    # compute posterior approx density
    # p(x) = E_{z~p(z)}[q(z|x)]
    zzk = []
    with torch.no_grad():
        for zz_i in zz.split(batch_size, dim=0):
            zz_i = zz_i.to(device)
            mu, logvar, _, rec = model(zz_i, deterministic=True)
            recon_error = calc_reconstruction_loss(zz_i, rec, loss_type='mse', reduction='none')
            while len(recon_error.shape) > 1:
                recon_error = recon_error.sum(-1)
            kl = calc_kl(logvar=logvar, mu=mu, reduce="none")
            zzk_i = -1.0 * (beta_kl * kl + beta_recon * recon_error)
            zzk += [zzk_i.exp()]
    p_x = torch.cat(zzk, 0)
    # plot
    cmesh = ax.pcolormesh(xx.data.cpu().numpy(), yy.data.cpu().numpy(), p_x.view(n_pts, n_pts).data.cpu().numpy(),
                          cmap=plt.cm.jet)
    ax.set_facecolor(plt.cm.jet(0.))
    if set_title:
        ax.set_title('VAE density')
    if colorbar:
        plt.colorbar(cmesh)





def plot_samples_density(dataset, model, scale, device):
    """
    Plot real data from dataset, generated samples from model and density estimation
    """
    model.eval()
    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(1, 3, 1)
    plot_batch = dataset.next_batch(batch_size=1024, device=device)
    plot_batch = plot_batch.data.cpu().numpy()
    ax1.scatter(plot_batch[:, 0], plot_batch[:, 1], s=8, label="true dist")
    ax1.set_xlim((-scale * 2, scale * 2))
    ax1.set_ylim((-scale * 2, scale * 2))
    ax1.set_axis_off()
    ax1.set_title('Real Data')

    ax2 = fig.add_subplot(1, 3, 2)
    noise_batch = torch.randn(size=(1024, model.zdim)).to(device)
    plot_fake_batch = model.sample(noise_batch)
    plot_fake_batch = plot_fake_batch.data.cpu().numpy()
    ax2.scatter(plot_fake_batch[:, 0], plot_fake_batch[:, 1], s=8, c='g', label="fake")
    ax2.set_xlim((-scale * 2, scale * 2))
    ax2.set_ylim((-scale * 2, scale * 2))
    ax2.set_axis_off()
    ax2.set_title('Fake Samples')

    ax3 = fig.add_subplot(1, 3, 3)
    test_grid = setup_grid(range_lim=scale * 2, n_pts=1024, device=torch.device('cpu'))
    plot_vae_density(model, ax3, test_grid, n_pts=1024, batch_size=256, colorbar=False,
                     beta_kl=1.0, beta_recon=1.0, set_title=False, device=device)
    ax3.set_axis_off()
    ax3.set_title("Density Estimation")
    return fig


def calculate_elbo_with_grid(model, evalset, test_grid, beta_kl=1.0, beta_recon=1.0, batch_size=512, num_iter=100,
                             device=torch.device("cpu")):
    model.eval()
    xx, yy, zz = test_grid
    zzk = []
    elbos = []
    with torch.no_grad():
        for zz_i in zz.split(batch_size, dim=0):
            zz_i = zz_i.to(device)
            mu, logvar, _, rec = model(zz_i, deterministic=True)
            recon_error = calc_reconstruction_loss(zz_i, rec, loss_type='mse', reduction='none')
            while len(recon_error.shape) > 1:
                recon_error = recon_error.sum(-1)
            kl = calc_kl(logvar=logvar, mu=mu, reduce="none")
            zzk_i = 1.0 * (beta_kl * kl + beta_recon * recon_error)
            zzk += [zzk_i]
        elbos_grid = torch.cat(zzk, 0)
        for i in range(num_iter):
            batch = evalset.next_batch(batch_size=batch_size, device=device)
            mu, logvar, _, rec = model(batch, deterministic=True)
            recon_error = calc_reconstruction_loss(batch, rec, loss_type='mse', reduction='none')
            while len(recon_error.shape) > 1:
                recon_error = recon_error.sum(-1)
            kl = calc_kl(logvar=logvar, mu=mu, reduce="none")
            elbos += [1.0 * (beta_kl * kl + beta_recon * recon_error)]
    elbos = torch.cat(elbos, dim=0)
    normalizing_factor = torch.cat([elbos_grid, elbos], dim=0).sum()
    elbos = elbos / normalizing_factor
    return elbos.mean().data.cpu().item()


def calculate_sample_kl(model, evalset, num_samples=5000, device=torch.device("cpu"), hist_bins=100, use_jsd=False,
                        xy_range=(-2, 2)):
    hist_range = [[xy_range[0], xy_range[1]], [xy_range[0], xy_range[1]]]
    real_samples = evalset.next_batch(batch_size=num_samples, device=device).data.cpu().numpy()
    real_hist, _, _ = np.histogram2d(real_samples[:, 0], real_samples[:, 1], bins=hist_bins, density=True,
                                     range=hist_range)
    real_hist = torch.tensor(real_hist).to(device)
    fake_samples = model.sample_with_noise(num_samples=num_samples, device=device).data.cpu().numpy()
    fake_hist, _, _ = np.histogram2d(fake_samples[:, 0], fake_samples[:, 1], bins=hist_bins, density=True,
                                     range=hist_range)
    fake_hist = torch.tensor(fake_hist).to(device)
    if use_jsd:
        kl_1 = F.kl_div(torch.log(real_hist + 1e-14), 0.5 * (fake_hist + real_hist), reduction='batchmean')
        kl_2 = F.kl_div(torch.log(fake_hist + 1e-14), 0.5 * (fake_hist + real_hist), reduction='batchmean')
        jsd = 0.5 * (kl_1 + kl_2)
        return jsd.data.cpu().item()
    else:
        kl = F.kl_div(torch.log(fake_hist + 1e-14), real_hist, reduction='batchmean')
        return kl.data.cpu().item()