python running_files/train_pix2pix_lung.py \
    --model pix2pix  \
    --is_train True \
    --cuda True \
    --gpu_ids 1 \
    --cuda_index 1 \
    --dataroot ../data/JSRT \
    --amp \
    --loss_lambda 1.0 \
    --n_epochs 10000 \
    --unet_epoch 20 \
    --lr 2e-5 \
    --arch_lr 1e-5 \
    --lr_d_dcgan 0 \
    --display_freq 10 \
    --save_latest_freq 50 \
    --classes 1 \
    --output_nc 1 \
    --input_nc 1 \
    --batch_size 2
