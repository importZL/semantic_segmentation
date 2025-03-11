"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import nibabel as nib
import datetime
import matplotlib.pyplot as plt
import time
import cv2 as cv

OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'


def print_timestamped(string):
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%d-%m-%Y_%H:%M:%S')
    print(st + ": " + string)


def zero_division(n, d):
    return n / d if d else n * 0


def error(string):
    print(f"{FAIL}" + string + f"{ENDC}")
    exit(-1)


def info(string):
    print(f"{OKBLUE}" + string + f"{ENDC}")


def warning(string):
    print(f"{WARNING}" + string + f"{ENDC}")


def postprocess_images(visuals, opt, original_shape):
    np_dict = {}
    # Transform images
    for label, image in visuals.items():
        np_dict[label] = image.detach().cpu().numpy().reshape(image.shape[2:])

    # Add the brain mask to the new image
    zero_brain_mask = np.where(np_dict['real_A'] == np_dict['real_A'].min())
    np_dict['fake_B'][zero_brain_mask] = np_dict['fake_B'].min()

    # Apply the median filter to fakeB
    np_dict['fake_B_smoothed'] = filter_blur(np_dict['fake_B'], opt.smoothing)

    for label, image in np_dict.items():
        # They could potentially have different sizes, but not tested
        if all(i >= j for i, j in zip(image.shape, original_shape)):
            np_dict[label] = crop_center(image, original_shape)
    return np_dict


def plot_2d(image, filename):
    fig = plt.figure(figsize=(5, 5), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    image = np.rot90(image, k=-1)
    image = np.flip(image, axis=1)
    ax.set_yticks([])
    ax.set_xticks([])

    ax.imshow(image, cmap="gray")
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)


def filter_blur(mapped, mode="median", k_size=3):
    if mode == "average":
        kernel = np.ones((k_size, k_size), np.float32) / (k_size * k_size)
        dst = cv.filter2D(mapped, -1, kernel)
    elif mode == "median":
        dst = cv.medianBlur(np.float32(mapped), ksize=k_size)
    elif mode == "blur":
        dst = cv.blur(mapped, ksize=(k_size, k_size))
    elif mode == "gblur":
        dst = cv.GaussianBlur(mapped, k_size=(k_size, k_size), sigmaX=0)
    else:
        error("Smoothing mode not recognized.")

    return dst


def nifti_to_np(image_path, sliced, chosen_slice):
    nifti = nib.load(image_path)
    affine = nifti.affine

    nifti_data = nifti.get_fdata()
    if sliced:
        nifti_data = nifti_data[:, :, chosen_slice]
    return nifti_data, affine


def normalize_with_opt(arr, opt):
    # print(opt, "[", arr.min(), arr.max(), "]", end=" - ")
    if opt == 0:
        return (arr - arr.min()) / (arr.max() - arr.min())
    elif opt == 1:
        return (arr - np.mean(arr[arr > arr.min()])) / np.std(arr[arr > arr.min()])
    # print("[", arr.min(), arr.max(), "]")
    return arr


def plot_nifti(image, filename, affine=None):
    if affine is None:
        affine = np.array([[-1., 0., 0., -0.],
                           [0., -1., 0., 239.],
                           [0., 0., 1., 0.],
                           [0., 0., 0., 1.]])
    new_nifti = nib.Nifti1Image(image, affine=affine)
    nib.save(new_nifti, filename)
    print_timestamped("Saved in " + str(filename))


def crop_center(img, target_shape):
    if len(img.shape) == 3:
        x, y, z = img.shape
        cropx, cropy, cropz = target_shape
        startx = x // 2 - cropx // 2
        starty = y // 2 - cropy // 2
        startz = z // 2 - cropz // 2
        return img[starty:starty + cropy, startx:startx + cropx, startz:startz + cropz]
    elif len(img.shape) == 2:
        x, y = img.shape
        cropx, cropy = target_shape
        startx = x // 2 - cropx // 2
        starty = y // 2 - cropy // 2
        return img[starty:starty + cropy, startx:startx + cropx]
    else:
        pass  # Not supported
    return img


def rotate_flip(data):
    data = np.rot90(data, k=3)
    # data = np.flip(data, axis=1)
    return data


def np_to_pil(data):
    nifti_data = (data * 255).astype(np.uint8)
    data = Image.fromarray(nifti_data, "L")
    return data


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
