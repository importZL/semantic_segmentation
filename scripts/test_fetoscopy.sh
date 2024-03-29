python running_files/test_fetoscopy.py \
    --model pix2pix  \
    --is_train True \
    --cuda True \
    --gpu_ids 0 \
    --cuda_index 0 \
    --dataroot ../data/fetoscopy/test \
    --amp \
    --loss_lambda 1.0 \
    --n_epochs 5000 \
    --unet_epoch 20 \
    --lr 2e-7 \
    --arch_lr 1e-7 \
    --display_freq 10 \
    --classes 1 \
    --output_nc 3 \
    --input_nc 1 \
    --batch_size 2 \
    --seg_model unet \
    --model_dir /home/li/workspace/semantic_segmentation/checkpoint_ablation/unet-fetoscopy-100-separate20230502-210000/unet.pkl