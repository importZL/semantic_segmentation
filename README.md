# Data Augmentation with NAS-GAN: An End-to-End framework to train semantic segmentation model

## Requirements
- Python 3.7 + and Pytorch 1.12.1 + are recommended.
- This code is tested with CUDA 11.3 toolkit and Wandb.
- Please  check the python package requirement from [requirements.txt](./requirements.txt), and install using
```bash
pip install -r requirements.txt
```

## Datasets download
we use [JSRT](http://db.jsrt.or.jp/eng.php) as the in-domain dataset, which is used to train and evaluate the model. Further, we use two additional datasets, [NLM(MC)](https://drive.google.com/file/d/1cBKYYtlNIsOjxaeo9eQoCr9RL13CdAma/view?usp=share_link) and [NLM(SZ)](https://drive.google.com/drive/folders/1TewVvRjoZ1Ynm9AVsVzauGmlQYjA1QDH?usp=share_link), as the out-of-domain datasets, which are only used to evaluate. (*For some image examples, their lung segmentation masks are divided into the right and the left lung mask. For these, the masks need to be combined first.*)

Dataset tree structure example:
```
data/JSRT
├── Images
│   ├── JPCLN001.png
│   ├── JPCLN002.png
│   ├── ...
├── Masks
│   ├── JPCLN001.gif
│   ├── JPCLN002.gif
│   ├── ...
project code
├── ...
```

## Training process

```bash
python train_betty.py \--model pix2pix  --is_train True --cuda True --gpu_ids [cuda_index] --cuda_index [cuda_index] --dataroot [dataroot] --amp --loss_lambda 1.0 --n_epochs [pre-train epoch]  --display_freq [display_req] --classes [number of class] --batch_size [batch_size]

```

## how to run the training

## how to test the model

## checkpoints

## reference



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
