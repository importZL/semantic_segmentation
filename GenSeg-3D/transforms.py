import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
import numpy as np

# 1. Add Channel (Equivalent to `AddChanneld`)
class AddChannel:
    """Adds a channel dimension to 3D tensors (D, H, W) → (1, D, H, W)."""
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if isinstance(image, np.ndarray):
            image = torch.tensor(image, dtype=torch.float32)
        if isinstance(label, np.ndarray):
            label = torch.tensor(label, dtype=torch.float32)
        if image.ndimension() == 3:  # Assuming shape (D, H, W)
            image = image.unsqueeze(0)
        if label.ndimension() == 3:
            label = label.unsqueeze(0)
        return {'image': image, 'label': label}

# 2. Normalize Intensity (Equivalent to `NormalizeIntensityd`)
class NormalizeIntensity:
    """Normalizes intensity values while ignoring zeros (nonzero=True)."""
    def __call__(self, sample):
        image = sample['image']
        mask = image != 0
        mean = image[mask].mean()
        std = image[mask].std()
        image = (image - mean) / (std + 1e-5)
        return {'image': image, 'label': sample['label']}

# 3. Random Flip (Equivalent to `RandFlipd`)
class RandomFlip:
    """Randomly flips along the specified axis."""
    def __init__(self, prob=0.5, spatial_axis=0):
        self.prob = prob
        self.axis = spatial_axis

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if torch.rand(1).item() < self.prob:
            image = torch.flip(image, dims=[self.axis])
            label = torch.flip(label, dims=[self.axis])
        return {'image': image, 'label': label}

# 4. Pad to Divisible (Equivalent to `DivisiblePadd`)
class PadToDivisible:
    """Pads image and label so that dimensions are divisible by k."""
    def __init__(self, k=16):
        self.k = k

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        _, d, h, w = image.shape  # Assuming (C, D, H, W)
        pad_d = (self.k - d % self.k) % self.k
        pad_h = (self.k - h % self.k) % self.k
        pad_w = (self.k - w % self.k) % self.k
        image = F.pad(image, (0, pad_w, 0, pad_h, 0, pad_d))
        label = F.pad(label, (0, pad_w, 0, pad_h, 0, pad_d))
        return {'image': image, 'label': label}

# 5. Move tensors to CUDA (for GPU training)
class ToTensorCuda:
    """Moves image and label tensors to CUDA."""
    def __call__(self, sample):
        return {'image': sample['image'].cuda(), 'label': sample['label'].cuda()}

# Training Transform (CPU)
train_transform = Compose([
    AddChannel(),
    RandomFlip(prob=0.5, spatial_axis=0),
    RandomFlip(prob=0.5, spatial_axis=1),
    RandomFlip(prob=0.5, spatial_axis=2),
    NormalizeIntensity(),
    PadToDivisible(k=16)
])

# Training Transform (CUDA)
train_transform_cuda = Compose([
    AddChannel(),
    RandomFlip(prob=0.5, spatial_axis=0),
    RandomFlip(prob=0.5, spatial_axis=1),
    RandomFlip(prob=0.5, spatial_axis=2),
    NormalizeIntensity(),
    PadToDivisible(k=16),
    ToTensorCuda()
])

# Validation Transform (CPU)
val_transform = Compose([
    AddChannel(),
    NormalizeIntensity(),
    PadToDivisible(k=16)
])

# Validation Transform (CUDA)
val_transform_cuda = Compose([
    AddChannel(),
    NormalizeIntensity(),
    PadToDivisible(k=16),
    ToTensorCuda()
])

fake_transform = Compose([
    RandomFlip(prob=0.5, spatial_axis=0),
    RandomFlip(prob=0.5, spatial_axis=1),
    RandomFlip(prob=0.5, spatial_axis=2),
])

if __name__ == '__main__':
    # Example Usage
    sample = {
        "image": torch.randn(37, 36, 57),  # Example 3D volume
        "label": torch.randint(0, 2, (37, 36, 57))  # Example segmentation mask
    }

    # Apply transforms
    transformed_sample = train_transform(sample)
    transformed_sample_cuda = train_transform_cuda(sample)

    # Print shape info
    print("CPU Transformed Image Shape:", transformed_sample["image"].shape)
    print("CUDA Transformed Image Shape:", transformed_sample_cuda["image"].shape if torch.cuda.is_available() else "CUDA not available")

# from monai.transforms import (
#     Compose,
#     ToTensord,
#     RandFlipd,
#     Spacingd,
#     RandScaleIntensityd,
#     RandShiftIntensityd,
#     NormalizeIntensityd,
#     AddChanneld,
#     DivisiblePadd
# )


# #Transforms to be applied on training instances
# train_transform = Compose(
#     [   
#         AddChanneld(keys=["image", "label"]),
#         Spacingd(keys=['image', 'label'], pixdim=(1., 1., 1.), mode=("bilinear", "nearest")),
#         RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
#         RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=1),
#         RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=2),
#         NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
#         RandScaleIntensityd(keys='image', factors=0.1, prob=1.0),
#         RandShiftIntensityd(keys='image', offsets=0.1, prob=1.0),
#         DivisiblePadd(k=16, keys=["image", "label"]),
#         ToTensord(keys=['image', 'label'])
#     ]
# )

# #Cuda version of "train_transform"
# train_transform_cuda = Compose(
#     [   
#         AddChanneld(keys=["image", "label"]),
#         Spacingd(keys=['image', 'label'], pixdim=(1., 1., 1.), mode=("bilinear", "nearest")),
#         RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
#         RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=1),
#         RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=2),
#         NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
#         RandScaleIntensityd(keys='image', factors=0.1, prob=1.0),
#         RandShiftIntensityd(keys='image', offsets=0.1, prob=1.0),
#         DivisiblePadd(k=16, keys=["image", "label"]),
#         ToTensord(keys=['image', 'label'], device='cuda')
#     ]
# )

# #Transforms to be applied on validation instances
# val_transform = Compose(
#     [   
#         AddChanneld(keys=["image", "label"]),
#         NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
#         DivisiblePadd(k=16, keys=["image", "label"]),
#         ToTensord(keys=['image', 'label'])
#     ]
# )

# #Cuda version of "val_transform"
# val_transform_cuda = Compose(
#     [   
#         AddChanneld(keys=["image", "label"]),
#         NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
#         DivisiblePadd(k=16, keys=["image", "label"]),
#         ToTensord(keys=['image', 'label'], device='cuda')
#     ]
# )