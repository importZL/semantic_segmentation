"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
from data.base_dataset import BaseDataset, get_params_3d, get_params, get_transform, get_transform_torchio
import os
from util.util import error, warning, nifti_to_np, np_to_pil, normalize_with_opt
import torchio
import torch

class NIfTIDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--chosen_slice', type=int, default=76, help='the slice to choose (in case of 2d)')
        parser.add_argument('--mapping_source', type=str, default="t1", const="t1", nargs='?',
                            choices=['t1', 't2', 't1ce', 'flair'],
                            help='the source sequencing for the mapping')
        parser.add_argument('--mapping_target', type=str, default="t2", const="t2", nargs='?',
                            choices=['t1', 't2', 't1ce', 'flair'],
                            help='the source sequencing for the mapping')
        parser.add_argument('--excel', action='store_true',
                            help='choose to print an excel file with useful information (1) or not (0)')
        parser.add_argument('--smoothing', type=str, default="median", const="median", nargs='?',
                            choices=['average', 'median'],
                            help='the kind of smoothing to apply to the image after mapping')
        parser.add_argument('--show_plots', action='store_true',
                            help='choose to show the final plots for the fake images while testing')
        parser.add_argument('--truth_folder', type=str, default="truth",
                            help='folder where the truth files are saved (if exists).')
        parser.add_argument('--postprocess', type=int, default=-1, const=-1, nargs='?',
                            choices=[-1, 0, 1],
                            help='the kind of post-processing to apply to the images. -1 means no postprocessing, '
                                 '0 means normalize in range [0, 1], '
                                 '1 means normalize with unit variance and mean 0.')
        parser.set_defaults(input_nc=1, output_nc=1)  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        # get the image paths of your dataset;
        rootpathA = os.path.join(opt.dataroot, opt.phase, opt.mapping_source)
        rootpathB = os.path.join(opt.dataroot, opt.phase, opt.mapping_target)
        self.truthpath = os.path.join(opt.dataroot, opt.phase, opt.truth_folder)
        filesA = sorted(os.listdir(rootpathA))
        filesB = sorted(os.listdir(rootpathB))
        if opt.model == "pix2pix3d":
            self.sliced = False
        elif opt.model == "pix2pix":
            self.sliced = True
        else:
            warning("The model " + opt.model + " has not been tested and might produce unexpected results.")
        self.affine = None
        self.original_shape = None
        self.chosen_slice = opt.chosen_slice
        self.image_pathsA = [os.path.join(rootpathA, f) for f in filesA]
        self.image_pathsB = [os.path.join(rootpathB, f) for f in filesB]
        if len(self.image_pathsA) != len(self.image_pathsB):
            error("The length of the image paths does not correspond, please check if the images are the same.")

        # You can call sorted(make_dataset(self.root, opt.max_dataset_size))
        # to get all the image paths under the directory self.root
        # define the default transform function. You can use <base_dataset.get_transform>;
        # You can also define your custom transform function
        # self.transform = get_transform(opt, grayscale=True)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        chosen_imgA = self.image_pathsA[index]
        chosen_imgB = self.image_pathsB[index]

        if os.path.basename(chosen_imgA) != os.path.basename(chosen_imgB):
            error("The chosen images are different. Please check the folder for correctness.")

        truth = None
        current_truthpath = os.path.join(self.truthpath, os.path.basename(chosen_imgA))

        if self.sliced:
            A, affine = nifti_to_np(chosen_imgA, self.sliced, self.chosen_slice)
            B, affine = nifti_to_np(chosen_imgB, self.sliced, self.chosen_slice)
            self.original_shape = A.shape
            A = normalize_with_opt(A, 0)
            B = normalize_with_opt(B, 0)
            A = np_to_pil(A)
            B = np_to_pil(B)
            if os.path.exists(current_truthpath):
                truth, _ = nifti_to_np(current_truthpath, self.sliced, self.chosen_slice)
                truth = (truth != truth.min())
                truth = np_to_pil(truth)
            transform_params = get_params(self.opt, A.size)
            c_transform = get_transform(self.opt, transform_params, grayscale=True)
        else:
            A = torchio.Image(chosen_imgA, torchio.INTENSITY)
            B = torchio.Image(chosen_imgB, torchio.INTENSITY)
            if os.path.exists(current_truthpath):
                truth = torchio.LabelMap(current_truthpath)
                truth.data[truth.data > 1] = 1
            self.original_shape = A.shape[1:]
            affine = A.affine
            transform_params = get_params_3d(self.opt, A.shape)
            c_transform = get_transform_torchio(self.opt, transform_params)

        self.affine = affine
        A_torch = c_transform(A)
        B_torch = c_transform(B)
        
        truth_torch = None
        if truth is not None:
            truth = c_transform(truth)
            truth_torch = (truth.data != truth.data.min())
        else:
            truth_torch = torch.zeros(B_torch.data.shape, dtype=torch.bool)
        A_mask = (A_torch.data != A_torch.data.min())
        return {'A': A_torch.data, 'B': B_torch.data,
                'mask': A_mask, 'truth': truth_torch,
                'A_paths': chosen_imgA, 'B_paths': chosen_imgB}

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_pathsA)
