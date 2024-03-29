python running_files/train_separate.py \
    --datasource omnipose \
    --model pix2pix  \
    --is_train True \
    --cuda True \
    --gpu_ids 0 \
    --cuda_index 0 \
    --dataroot ../data/JSRT \
    --amp \
    --loss_lambda 1.0 \
    --n_epochs 50 \
    --unet_epoch 20 \
    --lr 2e-5 \
    --lr_dcgan 4e-5 \
    --lr_d_dcgan 0 \
    --display_freq 20 \
    --classes 1 \
    --batch_size 1
