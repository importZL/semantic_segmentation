python running_files/test_breast.py \
    --model pix2pix  \
    --is_train True \
    --cuda True \
    --gpu_ids 0 \
    --cuda_index 0 \
    --dataroot ../data/breast \
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
    --seg_model unet \
    --model_dir /home/li/workspace/semantic_segmentation/checkpoint_ablation/unet-breast-100-separate20230502-190204/unet.pkl