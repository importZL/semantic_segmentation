from torchvision.models import resnet18

import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
#import pdb
from UNet import *
from Utilities import *
from Discriminator import *
print(torch.__version__)

##### CONV Block with differentiable architecture #####
cell_index = 0
PRIMITIVES_conv = [
    'conv_311',
    'conv_512',
    'conv_713',
]
OPS = {
	'conv_311' : lambda C_in, C_out: conv(C_in, C_out, 3, 1, 1),
	'conv_512' : lambda C_in, C_out: conv(C_in, C_out, 5, 1, 2),
	'conv_713' : lambda C_in, C_out: conv(C_in, C_out, 7, 1, 3),
}

num_ops_conv = len(PRIMITIVES_conv)
conv_arch = Variable(1e-3*torch.randn(10, num_ops_conv), requires_grad=True)
_conv_arch_parameters = [conv_arch,]
def conv_arch_parameters():
    return _conv_arch_parameters

class conv(nn.Module):

	def __init__(self, C_in, C_out, kernel_size, stride, padding):
		super(conv, self).__init__()
		self.op = nn.Conv2d(in_channels=C_in, out_channels=C_out, 
                      kernel_size=kernel_size, stride=stride, padding=padding)

	def forward(self, x):
		return self.op(x)

class MixedOp_conv(nn.Module):

    def __init__(self, C_in, C_out):
        super(MixedOp_conv, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES_conv:
            op = OPS[primitive](C_in, C_out)
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell_conv(nn.Module):

    def __init__(self, C_in, C_out, layer_index):
        super(Cell_conv, self).__init__()
        self._layer_index = layer_index
        self._ops =  MixedOp_conv(C_in, C_out)

    def forward(self, input):
        weight = conv_arch[self._layer_index, :]
        weight = nn.functional.softmax(weight, dim=-1)
        states = self._ops(input, weight)

        return states

def _set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)
    
##############################
#        Encoder 
##############################
class Encoder(nn.Module):
    def __init__(self, channels, latent_dim):
        super(Encoder, self).__init__()
        """ The encoder used in both cVAE-GAN and cLR-GAN, which encode image B or B_hat to latent vector
            This encoder uses resnet-18 to extract features, and further encode them into a distribution
            similar to VAE encoder. 

            Note: You may either add "reparametrization trick" and "KL divergence" or in the train.py file
            
            Args in constructor: 
                latent_dim: latent dimension for z 
  
            Args in forward function: 
                img: image input (from domain B)

            Returns: 
                mu: mean of the latent code 
                logvar: sigma of the latent code 
        """
        global cell_index
        # Extracts features at the last fully-connected
        resnet18_model = resnet18(pretrained=False) 
        resnet18_model.conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
        self.feature_extractor = nn.Sequential(*list(resnet18_model.children())[:-3])
        for name, layer in self.feature_extractor.named_modules():
            if isinstance(layer, nn.Conv2d) and layer.kernel_size[0]==3 and layer.stride[0]==1 and layer.padding[0]==1:
                new_layer = Cell_conv(layer.in_channels, layer.out_channels, cell_index)
                _set_module(self.feature_extractor, name, new_layer)
                cell_index += 1
       
        self.pooling = nn.AvgPool2d(kernel_size=8, stride=8, padding=0)

        # Output is mu and log(var) for reparameterization trick used in VAEs
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, img):
        out = self.feature_extractor(img)
        out = self.pooling(out)
        out = out.view(out.size(0), -1)
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        return mu, logvar


##############################
#        Generator 
##############################
    """ The generator used in both cVAE-GAN and cLR-GAN, which transform A to B
        
        Args in constructor: 
            latent_dim: latent dimension for z 
            image_shape: (channel, h, w), you may need this to specify the output dimension (optional)
        
        Args in forward function: 
            x: image input (from domain A)
            z: latent vector (encoded B)

        Returns: 
            fake_B: generated image in domain B
    """
def Generator(latent_dim, img_shape,output_nc, ngf, netG='unet_128', norm='batch', nl='relu',
             use_dropout=False, init_type='xavier', init_gain=0.02, where_add='input', upsample='bilinear'):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    nl_layer = get_non_linearity(layer_type=nl)
    channels, h, w  =img_shape
    input_nc = channels
    if latent_dim == 0:
        where_add = 'input'

    if netG == 'unet_128' and where_add == 'input':
        net = G_Unet_add_input(input_nc, output_nc, latent_dim, 7, ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                               use_dropout=use_dropout, upsample=upsample)
    elif netG == 'unet_256' and where_add == 'input':
        net = G_Unet_add_input(input_nc, output_nc, latent_dim, 8, ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                               use_dropout=use_dropout, upsample=upsample)
    elif netG == 'unet_128' and where_add == 'all':
        net = G_Unet_add_all(input_nc, output_nc, latent_dim, 7, ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                             use_dropout=use_dropout, upsample=upsample)
    elif netG == 'unet_256' and where_add == 'all':
        net = G_Unet_add_all(input_nc, output_nc, latent_dim, 8, ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                             use_dropout=use_dropout, upsample=upsample)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % net)

    init_weights(net, init_type, init_gain)
    return net
    

##############################
#        Discriminator
##############################
def Discriminator(img_shape, ndf, netD, norm='batch', nl='lrelu', init_type='xavier', init_gain=0.02, num_Ds=1):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    nl = 'lrelu'  # use leaky relu for D
    nl_layer = get_non_linearity(layer_type=nl)
    channels, h, w  =img_shape
    input_nc = channels

    if netD == 'basic_128':
        net = D_NLayers(input_nc, ndf, n_layers=2, norm_layer=norm_layer)  #, nl_layer=nl_layer
    elif netD == 'basic_256':
        net = D_NLayers(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'basic_128_multi':
        net = D_NLayersMulti(input_nc=input_nc, ndf=ndf, n_layers=2, norm_layer=norm_layer, num_D=num_Ds)
    elif netD == 'basic_256_multi':
        net = D_NLayersMulti(input_nc=input_nc, ndf=ndf, n_layers=3, norm_layer=norm_layer, num_D=num_Ds)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    init_weights(net, init_type, init_gain)
    return net

##############################


##############################
class BicycleGAN(nn.Module):
    def __init__(self, latent_dim, img_shape,output_nc, ngf, netG='unet_128', norm='batch', nl='relu',
             use_dropout=False, init_type='xavier', init_gain=0.02, where_add='input', upsample='bilinear',ndf=64, netD='basic_128'):
        super(BicycleGAN, self).__init__()


        self.generator = Generator(latent_dim, img_shape,output_nc, ngf, netG, norm, nl,
             use_dropout, init_type, init_gain, where_add, upsample)

        self.D_VAE = Discriminator(img_shape, ndf, netD, norm, nl, init_type, init_gain, num_Ds=1)

        self.D_LR = Discriminator(img_shape, ndf, netD, norm, nl, init_type, init_gain, num_Ds=1)
        
        self.encoder = Encoder(3, latent_dim)

# SoftIntroVAE model
class SoftIntroVAESimple(nn.Module):
    def __init__(self, latent_dim, img_shape,output_nc, ngf, netG='unet_128', norm='batch', nl='relu',
             use_dropout=False, init_type='xavier', init_gain=0.02, where_add='input', upsample='bilinear'):
        super(SoftIntroVAESimple, self).__init__()
        self.encoder = Encoder(3, latent_dim)
        self.latent_dim = latent_dim
        self.decoder = Generator(latent_dim, img_shape,output_nc, ngf, netG, norm, nl,
             use_dropout, init_type, init_gain, where_add, upsample)

    def forward(self, A, B, deterministic=False):
        mu, logvar = self.encode(B)
        if deterministic:
            z = mu
        else:
            z = reparameterization(mu, logvar)
        y = self.decode(A, z)
        return mu, logvar, z, y

    # def sample(self, A, z):
    #     y = self.decode(A, z)
    #     return y

    def sample_with_noise(self, A, num_samples=1, device=torch.device("cpu")):
        z = torch.randn(num_samples, self.latent_dim).to(device)
        return self.decode(A, z)

    def encode(self, B):
        mu, logvar = self.encoder(B)
        return mu, logvar

    def decode(self, A, z):
        y = self.decoder(A, z)
        return y

