# Extension of GenSeg for 3D medical image segmentation

## Datasets

We use [Hippocampus](./data_hippo/) from [MSD challenge](http://medicaldecathlon.com/) as the dataset to train and evaluate the model.


## Training and Testing

We pre-train the Pix2PixNIfTI model first, then, train the 3D UNet model in GenSeg framework.

To train the models from scratch, use the following command (Related configurations of model path should be changed mutually):

```
# Pre-train the augmentation model
bash train.sh

# Train the segmentation based on our framework
bash scripts/train_end2end.sh

# Inference the trained segmentation model
bash scripts/test.sh

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

Our code is based on the following repositories: [Pix2PixNIfTI model](https://github.com/giuliabaldini/Pix2PixNIfTI) | [3D UNet](https://github.com/aghdamamir/3D-UNet/tree/main) | [Betty framework](https://github.com/leopard-ai/betty)

## License

GenSeg is licensed under the [Apache 2.0 License](LICENSE).
