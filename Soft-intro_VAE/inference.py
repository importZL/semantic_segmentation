import torch
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
from PIL import Image
import os
from datasets import *
from models import *
from Utilities import *
from torch.autograd import Variable
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from metrics import *
import itertools
import math

# Device configuration
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
@hydra.main(version_base=None, config_path="config", config_name="eval")
def inference(cfg):
    save_infer_path = os.path.join(
        cfg.paths.inference_dir, cfg.experiment_name)
    out_path = os.path.join(save_infer_path, 'inference')
    subdirectories = ['FID_real', 'FID_fake']
    os.makedirs(save_infer_path, exist_ok=True)

    for subdirectory in subdirectories:
        os.makedirs(os.path.join(save_infer_path, subdirectory), exist_ok=True)
        
    model = instantiate(cfg.model.init)

    best_model_path = os.path.join(cfg.paths.checkpoints_dir,
                                   cfg.experiment_name,
                                   'model_epoch_24_iter_37392.pth')

    if cfg.type == "vae":
        generator = model.decoder.to(device)
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
    else:
        generator = model.generator.to(device)
        checkpoint = torch.load(best_model_path, map_location=device)
        generator.load_state_dict(checkpoint)
    generator.eval()
    val_dataset = instantiate(cfg.datas.val)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.params.batch_size, shuffle=False)

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    PerceptualLoss = instantiate(cfg.metrics.init)

    dist_list = []
    for i, data in enumerate(val_loader):
        real_A, real_B = data
        save_image(real_B.squeeze(), os.path.join(save_infer_path, 'FID_real/real') + str(i) + '.png', normalize=True)
        real_A = Normalize(real_A).to(device)

        out_styles = []
        for k in range(cfg.params.num_styles):
            # Generate random noise
            noise = Variable(
                    Tensor(
                        np.random.normal(0, 1, (
                                                real_A.size(0),
                                                cfg.model.names.latent_dim))))

            # Generate images
            with torch.no_grad():
                fake_Bs = generator(real_A, noise)
                fake = fake_Bs.squeeze()
                out_styles.append(fake)

            # Denormalize and save the generated images
            visualize_inference(
                    Denormalize(fake.detach()).cpu(),
                    'inference', k, i,
                    out_path, title=f"style{k}_image{i}"
            )
        save_image(fake, os.path.join(save_infer_path, 'FID_fake/fake') + str(i) + '.png', normalize=True)

        # Compute the perceptual loss
        dist = 0
        for imgs_pair in itertools.combinations(range(len(out_styles)), 2):
            index1, index2 = imgs_pair
            img1, img2 = out_styles[index1], out_styles[index2]
            perceptual_loss = PerceptualLoss(img1, img2)
            dist += perceptual_loss.item()
        dist /= (
            math.factorial(len(out_styles)) //
            (2 * math.factorial(len(out_styles) - 2))
            )
        dist_list.append(dist)
        print(f"Perceptual loss for image {i}: {dist}")
    
    print(f"Average perceptual loss: {sum(dist_list) / len(dist_list)}")
    save_infer_path = os.path.join(save_infer_path, 'metrics')
    plot_distances(dist_list, save_infer_path)

    return

if __name__ == '__main__':
    inference()