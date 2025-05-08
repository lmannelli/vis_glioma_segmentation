"""
Data augmentations
"""
import torch
from torch import nn
from random import random, uniform
from monai.transforms.spatial.array import Zoom
from monai.transforms.intensity.array import RandGaussianNoise, GaussianSharpen, AdjustContrast
from monai.transforms import RandAffined, RandAxisFlipd

# credit CKD-TransBTS
from monai.transforms import (
    Compose,
    NormalizeIntensityd,
    RandAffined,
    RandZoomd,
    RandFlipd,
    Rand3DElastic,
    RandBiasFieldd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandAdjustContrastd,
)
import torch.nn as nn

class DataAugmenter(nn.Module):
    def __init__(self, roi_size=(128, 128, 128)):
        super().__init__()
        self.roi_size = roi_size  # (Z, Y, X)

        self.transforms = Compose([
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandAffined(keys=["image", "label"], prob=0.5,
                       rotate_range=(0.26, 0.26, 0.26),
                       translate_range=(0.1, 0.1, 0.1),
                       scale_range=(0.1, 0.1, 0.1),
                       padding_mode='border',
                       mode=('bilinear', 'nearest')),
            Rand3DElastic(keys=["image", "label"], prob=0.3,
                          sigma_range=(10, 20), magnitude_range=(0.2, 0.4),
                          padding_mode='border',
                          mode=('bilinear', 'nearest')),
            RandZoomd(keys=["image", "label"], prob=0.3,
                     min_zoom=0.7, max_zoom=1.4,
                     padding_mode='constant', keep_size=True,
                     mode=('bilinear', 'nearest')),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandBiasFieldd(keys="image", prob=0.3, degree=3, coeff_range=(0.3, 0.7)),
            RandGaussianNoised(keys="image", prob=0.3, std=(0, 0.15)),
            RandGaussianSmoothd(keys="image", prob=0.3,
                                sigma_x=(0.5, 1.5),
                                sigma_y=(0.5, 1.5),
                                sigma_z=(0.5, 1.5)),
            RandAdjustContrastd(keys="image", prob=0.3, gamma=(0.7, 1.5)),
        ])

    def forward(self, images, labels):
        """
        images: Tensor[B, C_img, Z, Y, X]
        labels: Tensor[B, C_lbl, Z, Y, X]
        returns: aug_images, aug_labels with shape [B, C_*, roi_z, roi_y, roi_x]
        """
        batch_size, _, Z, Y, X = images.shape
        out_imgs = torch.zeros((batch_size, images.shape[1], *self.roi_size), device=images.device)
        out_lbls = torch.zeros((batch_size, labels.shape[1], *self.roi_size), device=labels.device)

        for b in range(batch_size):
            img = images[b]
            lbl = labels[b]

            # Crop aleatorio
            dz, dy, dx = Z - self.roi_size[0], Y - self.roi_size[1], X - self.roi_size[2]
            z0 = torch.randint(0, dz + 1, ()).item() if dz > 0 else 0
            y0 = torch.randint(0, dy + 1, ()).item() if dy > 0 else 0
            x0 = torch.randint(0, dx + 1, ()).item() if dx > 0 else 0
            img = img[:, z0 : z0 + self.roi_size[0], y0 : y0 + self.roi_size[1], x0 : x0 + self.roi_size[2]]
            lbl = lbl[:, z0 : z0 + self.roi_size[0], y0 : y0 + self.roi_size[1], x0 : x0 + self.roi_size[2]]

            # Transform
            data = {"image": img, "label": lbl}
            data = self.transforms(data)

            out_imgs[b] = data["image"]
            out_lbls[b] = data["label"]

        return out_imgs, out_lbls
class AttnUnetAugmentation(nn.Module):
    def __init__(self):
      super(AttnUnetAugmentation, self).__init__()
      self.axial_prob = uniform(0.1, 0.6)
      self.affine_prob = uniform(0.1, 0.5)
      self.crop_prob = uniform(0.1, 0.5)
      self.axial_flips = RandAxisFlipd(keys=["image", "label"], prob=self.axial_prob)
      self.affine = RandAffined(
          keys=["image", "label"],
          mode=("bilinear", "nearest"),
          prob=self.affine_prob,
          shear_range=(-0.1, 0.1, -0.1, 0.1, -0.1, 0.1),
          padding_mode="border",
      )

    def forward(self, data):
      with torch.no_grad():
        data = self.affine(data)
        data = self.axial_flips(data)
        return data