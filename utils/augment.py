"""
Data augmentations
"""
import torch
from torch import nn
from random import random, uniform
from monai.transforms.spatial.array import Zoom
from monai.transforms.intensity.array import RandGaussianNoise, GaussianSharpen, AdjustContrast
from monai.transforms import RandAffined, RandAxisFlipd
from monai.transforms import Rand3DElasticd, RandBiasField, RandRotated, RandCoarseDropout,NormalizeIntensity

# credit CKD-TransBTS
class DataAugmenter(nn.Module):
    """
    Data augmentation en fases de 50 épocas:
      - fase 0 (épocas 0–49): flips, zoom ligero, ruido y contraste suave
      - fase 1 (épocas 50–99): + blur más intenso, ruido aumentado
      - fase 2 (épocas 100–149): + deformación elástica y bias field
    """
    def __init__(self, phase_epochs: int = 50):
        super(DataAugmenter, self).__init__()
        self.phase_epochs = phase_epochs
        self.phase = 0
        self.normalizer = NormalizeIntensity(nonzero=True, channel_wise=True)

    def update_phase(self, epoch: int):
        # Calcula fase actual según la época
        self.phase = min(epoch // self.phase_epochs, 3)

    def forward(self, images: torch.Tensor, labels: torch.Tensor):
        """Recibe batch [B, 1, H, W, D] y aplica augment según la fase."""
        with torch.no_grad():
            for b in range(images.shape[0]):
                img = images[b].squeeze(0)
                lbl = labels[b].squeeze(0)
                img = self.normalizer(img)
                # --- Fase 0: flips y zoom ligero ---
                # Zoom ligero (p=0.15)
                if random() < 0.15:
                    z = uniform(0.7, 1.0)
                    img = Zoom(zoom=z, mode="trilinear", padding_mode="constant")(img)
                    lbl = Zoom(zoom=z, mode="nearest", padding_mode="constant")(lbl)
                # Flips en 3 ejes (p=0.5 cada uno)
                if random() < 0.5:
                    img = torch.flip(img, dims=(1,)); lbl = torch.flip(lbl, dims=(1,))
                if random() < 0.5:
                    img = torch.flip(img, dims=(2,)); lbl = torch.flip(lbl, dims=(2,))
                if random() < 0.5:
                    img = torch.flip(img, dims=(3,)); lbl = torch.flip(lbl, dims=(3,))
                # Ruido y contraste suave
                if random() < 0.10:
                    img = RandGaussianNoise(prob=1.0, mean=0.0, std=uniform(0.0, 0.2))(img)
                if random() < 0.10:
                    img = AdjustContrast(gamma=uniform(0.8, 1.2))(img)

                # # --- Fase 1: blur más intenso y ruido aumentado ---
                # if self.phase >= 1:
                #     if random() < 0.20:
                #         img = RandGaussianNoise(prob=1.0, mean=0.0, std=uniform(0.2, 0.4))(img)
                #     if random() < 0.20:
                #         img = GaussianSharpen(sigma1=uniform(0.5,1.5), sigma2=uniform(0.5,1.5))(img)

                # --- Fase 2: deformación elástica y bias field ---
                if self.phase >= 1:
                    # Elastic deformation (Monai)
                    deform = Rand3DElasticd(
                        keys=["img","lbl"],                   # aplicar a ambos
                        sigma_range=(2.0, 3.5),               # rango de sigma
                        magnitude_range=(2.0, 5.0),           # desplazamientos en voxels
                        prob=1.0,                             # ya hicimos la probabilidad externa
                        mode=("bilinear","nearest"),          # trilinear para img y nearest para lbl
                        padding_mode="zeros"                  # background=0 fuera de volúmen
                    )
                    if random() < 0.30:
                        data = {"img": img, "lbl": lbl}
                        out = deform(data)
                        img, lbl = out["img"], out["lbl"]
                    # Bias field
                    if random() < 0.20:
                        img = RandBiasField(prob=1.0, coeff_range=(0.1, 0.5))(img)

                # Fase 3: rotaciones 3D y coarse dropout
                if self.phase >= 2:
                    if random() < 0.30:
                        rot = RandRotated(                           # o RandRotateD(...)
                        keys=["img", "lbl"],                     # ambos reciben la misma rotación
                        range_x=(-10,10),                        # rotación aleatoria ±10° en X
                        range_y=(-10,10),                        # rotación aleatoria ±10° en Y
                        range_z=(-10,10),                        # rotación aleatoria ±10° en Z
                        prob=1.0,                                # ya controlas pexterna ≈0.30
                        mode=("bilinear", "nearest"),            # bilinear para imagen, nearest para mask
                        padding_mode="zeros",                    # fondo=0 en bordes
                    )
                        data = {"img": img, "lbl": lbl}
                        out = rot(data)                             # devuelve dict con llaves transformadas
                        img, lbl = out["img"], out["lbl"]
                    # if random() < 0.20:
                    #     img = RandCoarseDropout(holes=8, spatial_size=(16,16,16), prob=1.0)(img)
                # --- Fase 4: usa todo el pipeline (no cambios extra, está cubierto) ---
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