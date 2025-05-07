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
    Compose, RandZoomd, RandFlipd, RandRotate90d, RandAffined, RandElasticd,
    RandBiasFieldd, RandGaussianNoised, RandRicianNoised, RandMotionBlurd,
    RandAdjustContrastd, RandScaleIntensityd, RandShiftIntensityd, NormalizeIntensityd
)

class DataAugmenter(nn.Module):
    def __init__(self):
        super().__init__()
        self.augmentations = Compose([
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandFlipd(keys=["image","label"], prob=0.5, spatial_axis=[0,1,2]),
            RandRotate90d(keys=["image","label"], prob=0.5, max_k=3),
            RandAffined(keys=["image","label"], prob=0.3, rotate_range=(0.1,0.1,0.1),
                        translate_range=(10,10,10), scale_range=(0.1,0.1,0.1), mode=["trilinear","nearest"]),
            RandBiasFieldd(keys=["image"], prob=0.3, coeff_range=(0.1,0.5)),
            RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=(0.0,0.05)),
            RandRicianNoised(keys=["image"], prob=0.2, mean=0.0, std=(0.0,0.05)),
            RandAdjustContrastd(keys=["image"], prob=0.2, gamma=(0.7,1.5)),
            RandScaleIntensityd(keys=["image"], prob=0.2, factors=(0.9,1.1)),
            RandShiftIntensityd(keys=["image"], prob=0.2, offsets=(-0.1,0.1)),
        ])

    def forward(self, images, labels):
        d = {"image": images, "label": labels}
        augmented = self.augmentations(d)
        return augmented["image"], augmented["label"]

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