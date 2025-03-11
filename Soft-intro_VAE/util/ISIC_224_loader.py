import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import sys

class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, )),
        ])

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))

            img_ndarray = img_ndarray / 255
        else:
            img_ndarray = np.expand_dims(img_ndarray, axis=0)
            img_ndarray = img_ndarray / 255
        return img_ndarray

    @staticmethod
    def load(filename):
        ext = splitext(filename)[1]
        if ext == '.npy':
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            im = Image.open(filename)
            if im.mode != 'RBG':
                im = im.convert('RGB')
            # return transforms.functional.adjust_gamma(ImageOps.equalize(im), 0.5)
            return Image.open(filename)

    @staticmethod
    def load_pix2pix(filename):
        ext = splitext(filename)[1]
        if ext == '.npy':
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = self.load(mask_file[0]).convert('L').resize((224, 224), resample=Image.NEAREST)
        # mask = mask.resize((33, 33), resample=Image.NEAREST)
        img = self.load(img_file[0]).resize((224, 224))
        
        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        mask_pix2pix = self.load_pix2pix(mask_file[0]).convert('L').resize((256, 256))
        img_pix2pix = self.load_pix2pix(img_file[0]).resize((256, 256))
        assert img_pix2pix.size == mask_pix2pix.size, \
            f'Image and mask {name} should be the same size, but are {img_pix2pix.size} and {mask_pix2pix.size}'

        img_pix2pix = self.preprocess(img_pix2pix, self.scale, is_mask=False)
        mask_pix2pix = self.preprocess(mask_pix2pix, self.scale, is_mask=True)
        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous(),
            'image_pix2pix': torch.as_tensor(img_pix2pix.copy()).float().contiguous(),
            'mask_pix2pix': torch.as_tensor(mask_pix2pix.copy()).long().contiguous(),
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, scale=1, mask_suffix=''):
        super().__init__(images_dir, masks_dir, scale, mask_suffix=mask_suffix)
