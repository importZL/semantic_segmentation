"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod
import torchio.transforms as t_transforms


class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess == 'resize_and_crop' or opt.preprocess == 'take_center_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w

    center_x = (w - opt.load_size) // 2
    center_y = (h - opt.load_size) // 2

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    padding_x = opt.load_size - w
    padding_y = opt.load_size - h
    add_1 = padding_x % 2
    add_2 = padding_y % 2
    div_1 = int(padding_x / 2)
    div_2 = int(padding_y / 2)

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'center_pos': (center_x, center_y),
            'padding_vals': (div_1 + add_1, div_2 + add_2, div_1, div_2), 'flip': flip}


def get_params_3d(opt, size):
    _, w, h, d = size
    new_h = h
    new_w = w
    new_d = d
    if opt.preprocess == 'resize_and_crop' or opt.preprocess == 'take_center_and_crop':
        new_h = new_w = new_d = opt.load_size
    elif opt.preprocess == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))
    z = random.randint(0, np.maximum(0, new_d - opt.crop_size))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y, z), 'flip': flip}


def get_transform(opt, params=None, grayscale=False, method=Image.BICUBIC, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if 'resize' in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, method))
    elif 'pad' in opt.preprocess:
        transform_list.append(transforms.Pad(params['padding_vals']))
    elif 'scale_width' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, opt.crop_size, method)))
    elif 'take_center' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['center_pos'], opt.load_size)))

    if 'crop' in opt.preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))

    if opt.preprocess == 'none':
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if not opt.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def get_transform_torchio(opt, params=None, convert=True):
    transform_list = []

    if 'resize' in opt.preprocess or 'pad' in opt.preprocess or 'take_center' in opt.preprocess:
        transform_list.append(t_transforms.CropOrPad(opt.load_size))

    if 'crop' in opt.preprocess and params is not None:
        transform_list.append(t_transforms.Lambda(lambda img: __crop3d(img, params['crop_pos'], opt.crop_size)))

    if not opt.no_flip and opt.phase != 'val':
        # transform_list.append(t_transforms.RandomFlip(axes=('LR', 'ap'), flip_probability=params['flip']))
        if params is None:
            transform_list.append(t_transforms.RandomFlip(axes=('LR', 'ap',)))
        else:
            transform_list.append(t_transforms.RandomFlip(axes=('LR', 'ap',), flip_probability=params['flip']))

    if convert:
        transform_list += [t_transforms.RescaleIntensity(out_min_max=(0, 1))]
        transform_list += [t_transforms.Lambda(lambda img: __normalize(img, 0.5, 0.5))]
    return t_transforms.Compose(transform_list)


def __normalize(tensor, mean, std):
    return (tensor - mean) / std


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale_width(img, target_size, crop_size, method=Image.BICUBIC):
    ow, oh = img.size
    if ow == target_size and oh >= crop_size:
        return img
    w = target_size
    h = int(max(target_size * oh / ow, crop_size))
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if ow > tw or oh > th:
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __crop3d(img, pos, size):
    _, ow, oh, od = img.shape
    x1, y1, z1 = pos
    tw = th = td = size
    if ow > tw or oh > th or od > td:
        # return img.crop((x1, y1, z1, x1 + tw, y1 + th, z1 + td))
        return img[:, x1:x1 + tw, y1:y1 + th, z1:z1 + td]
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True
