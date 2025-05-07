"""
==========================
Data loading and processing
==========================

credit: https://github.com/faizan1234567/CKD-TransBTS/blob/main/BraTS.py
"""

import torch
import os
from torch.utils.data.dataset import Dataset
from utils.all_utils import (
    pad_or_crop_image, minmax, load_nii,
    pad_image_and_label, listdir, get_brats_folder
)
import numpy as np
from monai.transforms import (
    apply_transform,
    MapTransform,
)
from monai.inferers import sliding_window_inference
# --------------------------------------------------------------------------------
# 1) Primero definimos el transform que convierte la etiqueta en 3 canales:
# --------------------------------------------------------------------------------
class ConvertToMultiChannelBratsLabelsd(MapTransform):
    """
    Convierte las etiquetas BRATS en 3 canales binarios:
      - canal 0: Enhancing Tumor (ET, label == 3)
      - canal 1: Tumor Core (TC = NCR ∪ NET ∪ ET; labels 1,4,3)
      - canal 2: Whole Tumor (WT = TC ∪ ED; labels 1,4,3,2)
    """
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            lbl = d[key]
            # máscaras elementales
            et  = lbl == 3
            ncr = lbl == 1
            net = lbl == 4
            ed  = lbl == 2

            # Tumor Core = NCR ∪ NET ∪ ET
            tc = torch.logical_or(ncr, net)
            tc = torch.logical_or(tc, et)

            # Whole Tumor = TC ∪ ED
            wt = torch.logical_or(tc, ed)

            # stack en orden [ET, TC, WT]
            d[key] = torch.stack([et, tc, wt], dim=0).float()
        return d



# --------------------------------------------------------------------------------
# 3) Un Dataset de ejemplo usando MONAI Dataset
# --------------------------------------------------------------------------------
class CustomDataset(Dataset):
    def __init__(self, root_dir, section="train", transform=None):
        """
        root_dir/
         ├ train/
         │   ├ images/
         │   │  ├ <patient_id>/
         │   │  │   ├ <patient_id>-t1n.nii.gz
         │   │  │   ├ <patient_id>-t1c.nii.gz
         │   │  │   ├ <patient_id>-t2w.nii.gz
         │   │  │   └ <patient_id>-t2f.nii.gz
         │   └ masks/
         │       └ <patient_id>.seg.nii.gz
         └ val/...
        """
        self.transform  = transform
        self.section    = section
        base            = os.path.join(root_dir, section)
        images_base     = os.path.join(base, "images")
        masks_base      = os.path.join(base, "masks")

        # Listado de pacientes: solo carpetas
        self.patient_ids = [
            d for d in sorted(os.listdir(images_base))
            if os.path.isdir(os.path.join(images_base, d))
        ]

        # Construyo listas paralelas de paths
        self.image_files = []
        self.label_files = []
        for pid in self.patient_ids:
            folder = os.path.join(images_base, pid)
            # cuatro modalidades en el orden que quieras
            modalities = [
                f"{pid}-t1n.nii.gz",
                f"{pid}-t1c.nii.gz",
                f"{pid}-t2w.nii.gz",
                f"{pid}-t2f.nii.gz",
            ]
            img_paths = [os.path.join(folder, fn) for fn in modalities]
            seg_path  = os.path.join(masks_base, f"{pid}-seg.nii.gz")

            # sanity check
            if not all(os.path.exists(p) for p in img_paths):
                missing = [p for p in img_paths if not os.path.exists(p)]
                raise FileNotFoundError(f"Faltan imágenes para {pid}: {missing}")
            if not os.path.exists(seg_path):
                raise FileNotFoundError(f"No encontré la máscara para {pid}: {seg_path}")

            self.image_files.append(img_paths)
            self.label_files.append(seg_path)

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        data = {
            "image": self.image_files[idx],  # rutas, MONAI LoadImaged leerá
            "label": self.label_files[idx],
        }
        try:
            if self.transform:
                data = apply_transform(self.transform, data)
                return data
        except Exception as e:
            print(f"[ERROR] Falló el archivo: {data}")
            raise e


