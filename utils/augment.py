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
    RandZoomd,
    RandFlipd,
    RandAffined,
    RandBiasFieldd,
    RandGaussianNoised,
    RandRicianNoised,
    RandAdjustContrastd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Orientationd,
    Spacingd,
    CropForegroundd,
)
from monai.data import decollate_batch
class DataAugmenter(nn.Module):
    def __init__(self, prob=0.5):
        super().__init__()
        self.augmentations = Compose(
            [
                # Preprocesamiento común (podrías tener esto fuera del augmenter si es determinístico)
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
                # CropForegroundd(keys=["image", "label"], source_key="image", k_divisible=[16, 16, 16]),

                RandAffined(
                    keys=["image", "label"],
                    mode=("bilinear", "nearest"),
                    prob=prob,
                    rotate_range=(0, 0, (-0.1, 0.1)), # Rotación sutil en el plano axial
                    shear_range=None,
                    translate_range=None,
                    scale_range=None,
                    padding_mode="border",
                ),
                RandFlipd(keys=["image", "label"], prob=prob/2, spatial_axis=[0]), # Flip en el eje X
                RandFlipd(keys=["image", "label"], prob=prob/2, spatial_axis=[1]), # Flip en el eje Y
                RandZoomd(keys=["image", "label"], prob=prob/3, min_zoom=0.9, max_zoom=1.1, mode=("trilinear", "nearest")),
                RandGaussianNoised(keys=["image"], prob=prob/3, mean=0.0, std=0.1),
                RandAdjustContrastd(keys=["image"], prob=prob/4, gamma=(0.8, 1.2)),
                # Augmentaciones más específicas (usar con precaución y menor probabilidad)
                # RandElasticd(keys=["image", "label"], prob=prob/5, sigma_range=(0.1, 0.5), magnitude_range=(20, 50), spatial_dims=3, mode=("bilinear", "nearest"), padding_mode="border"),
                # RandBiasFieldd(keys=["image"], prob=prob/6, degree=3),
                # RandMotionBlurd(keys=["image"], prob=prob/7, max_kernel_size=3, angle_range=(0, 0)),
                RandRicianNoised(keys=["image"], prob=prob/8, mean=0.0, std=(0.0, 0.05)),
            ]
        )

    def forward(self, data):
        with torch.no_grad():
            return self.augmentations(data)

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