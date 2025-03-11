CUDA_VISIBLE_DEVICES=0 python train_unet.py \
    --dataroot /data/li/Pix2PixNIfTI/data_liver \
    --dataset_mode nifti \
    --model pix2pix3d \
    --name liver \