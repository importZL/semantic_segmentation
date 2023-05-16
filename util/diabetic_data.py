import os
import sys
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from util.transforms import RandomResizedCrop


class DiabeticDataset(Dataset):
    def __init__(self, root, img_size=512, train=True) -> None:
        super().__init__()
        self.root = root
        self.train = train
        self.imgs = []
        self.masks = []

        for file in os.listdir(root):
            if file.endswith("_img.jpg"):
                im = Image.open(os.path.join(root, file))
                arr = np.array(im).transpose((2, 0, 1))
                self.imgs.append(torch.from_numpy(arr.astype(float)))
                
                mask_file = file.replace("_img.jpg", ".png")
                im = Image.open(os.path.join(root, mask_file))
                arr = np.array(im) # > 0.01  # the original mask is float, not binary
                self.masks.append(torch.from_numpy(arr.astype(float)).unsqueeze(0))
                
        imgs = torch.cat([img.flatten() for img in self.imgs])
        mean = imgs.mean()
        std = imgs.std()
        print(f"{train=} img mean {mean}, std {std}")
        # self.imgs = [(img - mean) / (std + 1e-5) for img in self.imgs]
        self.imgs = [img/255.0 for img in self.imgs]

        if train:
            self.transform = RandomResizedCrop(img_size, scale=(0.7, 1))
        else:
            self.transform = transforms.Compose([
                transforms.Resize(img_size),
                # transforms.CenterCrop(img_size),
            ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = self.imgs[index]
        mask = self.masks[index]

        if self.train:
            img, mask = self.transform(img, mask)
        else:
            img = self.transform(img)
            mask = self.transform(mask)
        
        mask = mask.squeeze(0)
        mask = (mask > 0.1).long()

        return {
            'image': img,
            'mask': mask
        }

