# Further improve GenSeg by using VAE-based mask-to-image model

## Datasets

The dataset used in this task is the same as the main experiments, like lung segmentation data or skin lesion segmentation data.


## Training and Testing

We pre-train the Soft-intro VAE Model first, then, train the UNet model in GenSeg framework.

To train the models from scratch, use the following command (Related configurations of model path should be changed mutually):

```
# Pre-train the augmentation model
python train_vae.py

# Train the segmentation based on our framework
bash scripts/train_end2end.sh

# Inference the trained segmentation model
The inference process is the same as what used in the main experiment.

```

## Citation
If you find this project useful in your research, please consider citing:
```bash
@article{zhang2024generative,
  title={Generative AI Enables Medical Image Segmentation in Ultra Low-Data Regimes},
  author={Zhang, Li and Jindal, Basu and Alaa, Ahmed and Weinreb, Robert and Wilson, David and Segal, Eran and Zou, James and Xie, Pengtao},
  journal={medRxiv},
  pages={2024--08},
  year={2024},
  publisher={Cold Spring Harbor Laboratory Press}
}
```

## Code Dependencies

Our code is based on the following repositories: [Soft-intro VAE](https://github.com/eddie0509tw/Image-to-Image-Translation/tree/main)

## License

GenSeg is licensed under the [Apache 2.0 License](LICENSE).
