CUDA_VISIBLE_DEVICES=1 python train_end2end.py \
    --dataroot /data/li/Pix2PixNIfTI/data_liver \
    --dataset_mode nifti \
    --model pix2pix3d \
    --name liver-98 \