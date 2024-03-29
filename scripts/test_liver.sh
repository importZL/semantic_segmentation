python running_files/test_liver.py \
    --model pix2pix  \
    --is_train True \
    --cuda True \
    --gpu_ids 1 \
    --cuda_index 1 \
    --dataroot ../data/liver-2D/liver/val \
    --amp \
    --loss_lambda 1.0 \
    --n_epochs 5000 \
    --unet_epoch 20 \
    --lr 2e-7 \
    --arch_lr 1e-7 \
    --display_freq 10 \
    --classes 1 \
    --output_nc 1 \
    --input_nc 1 \
    --batch_size 2 \
    --seg_model deeplab \
    --model_dir /home/li/workspace/semantic_segmentation/checkpoint_liver/deeplab_liver2-20230518-042554/unet.pkl