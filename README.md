# semantic_segmentation


## To Ask

- Train Pix2Pix only on Train dataset of JSRT
- Generate fake images from mask of training dataset and not val dataset
- What is our contribution?
  - Using less images, but then how did it beat SemanticGAN, maybe fake data makes it difficult to overfit
- How are we reporting SZ best score? Based on the best UNet performance on JSRT Test?
- Do we need test_loader? Changed it to val_loader

## Ideas to implement

- Neural Architecture Search
- Diffusion model

### Diffusion model

 - Train a diffusion based generator to generate XRay images and then an encoder to encode the mask into the input of Generator
- Do we use extra data for Diffusion model (pretraining)
- Check if this has been implemented before

### Run skin segmentation experiments

```bash
python train_betty_ISIC.py --model pix2pix  --is_train True --cuda True --gpu_ids 0 --cuda_index 0 --dataroot ../data/ISIC2018 --amp --loss_lambda 1.0 --n_epochs 25 --unet_epoch 20 --lr 2e-5 --lr_dcgan 4e-5 --lr_d_dcgan 0 --display_freq 20 --classes 1 --output_nc 3 --input_nc 1

```
