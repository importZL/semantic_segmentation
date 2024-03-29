python running_files/test_opti.py \
    --model pix2pix  \
    --is_train True \
    --cuda True \
    --gpu_ids 1 \
    --cuda_index 1 \
    --dataroot ../data/JSRT \
    --amp \
    --loss_lambda 1.0 \
    --n_epochs 5000 \
    --unet_epoch 20 \
    --lr 2e-7 \
    --arch_lr 1e-7 \
    --display_freq 10 \
    --classes 6 \
    --output_nc 3 \
    --input_nc 1 \
    --batch_size 2 \
    --seg_model unet \
    --model_dir /home/li/workspace/semantic_segmentation/checkpoint_opti/UNet_JSRT-100-20230323-214654/unet.pkl