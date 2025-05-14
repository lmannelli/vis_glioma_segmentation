"""
Data augmentations
"""
import torch
from torch import nn
from random import random, uniform
from monai.transforms.spatial.array import Zoom
from monai.transforms.intensity.array import RandGaussianNoise, GaussianSharpen, AdjustContrast
from monai.transforms import RandAffined, RandAxisFlipd
from monai.transforms import Rand3DElasticd,RandShiftIntensity, RandBiasField, RandRotated, RandCoarseDropout,NormalizeIntensity

# credit CKD-TransBTS
class DataAugmenter(nn.Module):
    """
    Data augmentation unificado (sin fases):
      - flips
      - zoom ligero
      - ruido gaussiano suave
      - ajuste de contraste suave
      - rotaciones 3D leves
      - bias field sutil
    """
    def __init__(self):
        super(DataAugmenter, self).__init__()
        self.normalizer = NormalizeIntensity(nonzero=True, channel_wise=True)

    def forward(self, images: torch.Tensor, labels: torch.Tensor):
        """Recibe batch [B, 1, H, W, D] y aplica augmentaciones."""
        with torch.no_grad():
            for b in range(images.shape[0]):
                img = images[b].squeeze(0)
                lbl = labels[b].squeeze(0)

                img = self.normalizer(img)

                # Zoom ligero (p=0.15)
                if random() < 0.15:
                    z = uniform(0.8, 1.1)
                    img = Zoom(zoom=z, mode="trilinear", padding_mode="constant")(img)
                    lbl = Zoom(zoom=z, mode="nearest", padding_mode="constant")(lbl)

                # Flips en 3 ejes (p=0.5 cada uno)
                if random() < 0.5:
                    img = torch.flip(img, dims=(1,)); lbl = torch.flip(lbl, dims=(1,))
                if random() < 0.5:
                    img = torch.flip(img, dims=(2,)); lbl = torch.flip(lbl, dims=(2,))
                if random() < 0.5:
                    img = torch.flip(img, dims=(3,)); lbl = torch.flip(lbl, dims=(3,))

                # Ruido gaussiano suave (p=0.1)
                if random() < 0.1:
                    img = RandShiftIntensity(nonzero = True, channel_wise = True)(img)

                # Contraste suave (p=0.1)
                if random() < 0.1:
                    img = AdjustContrast(gamma=uniform(0.8, 1.2))(img)

                # Bias field muy sutil (p=0.1)
                if random() < 0.1:
                    img = RandBiasField(prob=1.0, coeff_range=(0.02, 0.05))(img)

                # Rotaciones 3D leves (p=0.2)
                if random() < 0.2:
                    rot = RandRotated(
                        keys=["img", "lbl"],
                        range_x=(-2, 2),
                        range_y=(-2, 2),
                        range_z=(-2, 2),
                        prob=1.0,
                        mode=("bilinear", "nearest"),
                        padding_mode="zeros"
                    )
                    out = rot({"img": img, "lbl": lbl})
                    img, lbl = out["img"], out["lbl"]

                images[b] = img.unsqueeze(0)
                labels[b] = lbl.unsqueeze(0)

        return images, labels
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