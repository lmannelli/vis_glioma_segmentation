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
class BraTS(Dataset):
    def __init__(self, patients_dir, patient_ids, mode,
                 target_size=(128,128,128), version="brats2023",
                 transform=None):                        # <-- añadir aquí
        super().__init__()
        self.patients_dir = patients_dir
        self.patients_ids = patient_ids
        self.mode         = mode
        self.target_size  = target_size
        self.version      = version
        self.transform    = transform  

        # Define modality suffixes for each version
        if version in ["brats2023", "brats2024"]:
            modality_suffixes = {
                "t1": "-t1n",
                "t1ce": "-t1c",
                "t2": "-t2w",
                "flair": "-t2f"
            }
        elif version in ["brats2019", "brats2020"]:
            modality_suffixes = {
                "t1": "_t1",
                "t1ce": "_t1ce",
                "t2": "_t2",
                "flair": "_flair"
            }
        else:
            raise ValueError(f"Unsupported version: {version}")

        self.modalities = list(modality_suffixes.keys())

        for patient_id in patient_ids:
            # Create file paths for each modality
            image_paths = {
                modality: f"{patient_id}{suffix}.nii.gz" if version == "brats2023" else f"{patient_id}{suffix}.nii.gz"
                for modality, suffix in modality_suffixes.items()
            }

            if mode in ["train", "val", "test", "visualize"]:
                seg_filename = f"{patient_id}.nii.gz" if version == "brats2023" else f"{patient_id}-seg.nii.gz"
            else:
                seg_filename = None

            patient = dict(id=patient_id, **image_paths, seg=seg_filename)
            self.datas.append(patient)

    def __getitem__(self, idx):
        patient = self.datas[idx]
        # Cargo volúmenes en numpy con load_nii (shape D×H×W)…
        imgs = {mod: load_nii(os.path.join(self.patients_dir, "images", patient["id"], patient[mod]))
                for mod in self.modalities}

        # cargo máscara si aplica…
        if self.mode in ["train","val","test","visualize"]:
            seg = load_nii(os.path.join(self.patients_dir, "masks", patient["seg"])).astype("int8")
        else:
            seg = None

        # construyo diccionario para MONAI
        data = {
            "image": np.stack([imgs[mod] for mod in self.modalities], axis=0),  # C×D×H×W
            "label": seg                                           #   D×H×W
        }

        # aplico transform si está definido
        if self.transform:
            data = self.transform(data)

        # convierto a torch.Tensor
        image = torch.as_tensor(data["image"], dtype=torch.float32)
        label = torch.as_tensor(data["seg_mask"], dtype=torch.int8) if data.get("label") is not None else None

        return {
            "patient_id": patient["id"],
            "image": image,
            "label": label
        }

    def __len__(self):
        return len(self.datas)


def get_datasets(dataset_folder, mode, target_size=(128, 128, 128), version="brats2024"):
    dataset_folder = get_brats_folder(dataset_folder, mode, version=version)
    assert os.path.exists(dataset_folder), f"Dataset Folder Does Not Exist: {dataset_folder}"
    # Obtener los IDs de los pacientes desde el directorio images/
    images_dir = os.path.join(dataset_folder, "images")
    patients_ids = [x for x in listdir(images_dir)]  # Lista de subdirectorios en images/
    return BraTS(dataset_folder, patients_ids, mode, target_size=target_size, version=version)
