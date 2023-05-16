import os
import pickle
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from util.transforms import RandomResizedCrop


class DeePiCtDataset(Dataset):
    def __init__(self, file, img_size=256, train=True) -> None:
        super().__init__()
        self.root = file
        self.train = train
        self.imgs = []
        self.masks = []

        with open(file, "rb") as f:
            d = pickle.load(f)

        if train:
            self.imgs = torch.from_numpy(d["train_features"]).permute(0, 3, 1, 2)
            self.masks = torch.from_numpy(d["train_labels"]).permute(0, 3, 1, 2)
        else:
            self.imgs = torch.from_numpy(d["test_features"]).permute(0, 3, 1, 2)
            self.masks = torch.from_numpy(d["test_labels"]).permute(0, 3, 1, 2)

        if train:
            self.transform = RandomResizedCrop(img_size, scale=(0.7, 1))
        else:
            self.transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
            ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = self.imgs[index]
        mask = self.masks[index]

        if self.train:
            img, mask = self.transform(img, mask)
            # flip
            if np.random.randint(2) == 1:
                img = torch.flip(img, dims=[2])
                mask = torch.flip(mask, dims=[2])
            # rotation
            k = np.random.randint(4)  # 0, 90, 180, 270 degree
            if k > 0:
                img = torch.rot90(img, k, dims=[1, 2])
                mask = torch.rot90(mask, k, dims=[1, 2])
        else:
            img = self.transform(img)
            mask = self.transform(mask)
        
        mask = mask.squeeze(0)
        mask = mask.long()

        return {
            'image': img,
            'mask': mask
        }

