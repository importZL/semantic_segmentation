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
We pre-train the augmentation model on the training and validation sets first, and then, we train and test our method on an in-domain dataset and test on several out-of-domain datasets without training. Training process can be run by:

```bash
python train_betty.py \
--model pix2pix \
--is_train True \
--cuda True \
--gpu_ids [cuda_index] \
--cuda_index [cuda_index] \
--dataroot [dataroot] \
--amp \
--loss_lambda 1.0 \
--n_epochs [pre-train epoch] \
--display_freq [display_req] \
--classes [number of class] \
--batch_size [batch_size]
```

## Pre-trained model

A pre-trained model is available for the JSRT dataset (*trained with 9 labeled data examples*). It can be download from: 

## Code dedendency
[Pix2Pix model](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/master/models) | [U-Net](https://github.com/milesial/Pytorch-UNet) | [Betty framework](https://github.com/leopard-ai/betty)


