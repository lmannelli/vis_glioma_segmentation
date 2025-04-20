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

class BraTS(Dataset):
    def __init__(self, patients_dir, patient_ids, mode, target_size=(128, 128, 128), version="brats2023"):
        super().__init__()
        self.patients_dir = patients_dir
        self.patients_ids = patient_ids
        self.mode = mode
        self.target_size = target_size
        self.version = version
        self.datas = []

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
                seg_filename = f"{patient_id}.nii.gz" if version == "brats2023" else f"{patient_id}_seg.nii.gz"
            else:
                seg_filename = None

            patient = dict(id=patient_id, **image_paths, seg=seg_filename)
            self.datas.append(patient)

    def __getitem__(self, idx):
        patient = self.datas[idx]
        crop_list = []
        pad_list = []

        # Cargar imágenes para todas las modalidades (corregir la ruta)
        patient_image = {
            modality: torch.tensor(load_nii(
                os.path.join(  # Incluir el ID del paciente en la ruta
                    self.patients_dir, "images", patient["id"], patient[modality]
                )
            ))
            for modality in self.modalities
        }

        # Load segmentation if available
        if self.mode in ["train", "val", "test", "visualize"]:
            patient_label = torch.tensor(
                load_nii(os.path.join(self.patients_dir, "masks", patient["seg"])).astype("int8")
            )
        else:
            patient_label = torch.zeros_like(next(iter(patient_image.values())), dtype=torch.int8)

        # Stack channels into single tensor
        patient_image = torch.stack([minmax(patient_image[mod]) for mod in self.modalities])

        # Crop black borders
        nonzero_index = torch.nonzero(torch.sum(patient_image, dim=0) != 0)
        z_indexes, y_indexes, x_indexes = nonzero_index[:, 0], nonzero_index[:, 1], nonzero_index[:, 2]
        zmin, ymin, xmin = [max(0, int(torch.min(idxs) - 1)) for idxs in (z_indexes, y_indexes, x_indexes)]
        zmax, ymax, xmax = [int(torch.max(idxs) + 1) for idxs in (z_indexes, y_indexes, x_indexes)]

        patient_image = patient_image[:, zmin:zmax, ymin:ymax, xmin:xmax].float()
        patient_label = patient_label[zmin:zmax, ymin:ymax, xmin:xmax]

        # One-hot style conversion to 3 channels: ET, TC, WT
        if self.mode in ["train", "val", "test"]:
            ed_label = 2
            ncr_label = 1
            et_label = 3 if self.version in ["brats2023", "brats2024"] else 4
            et = patient_label == et_label
            tc = torch.logical_or(patient_label == ncr_label, et)
            wt = torch.logical_or(tc, patient_label == ed_label)
            patient_label = torch.stack([et, tc, wt])

        # Apply padding/cropping
        if self.mode in ["train", "val", "test"]:
            patient_image, patient_label, pad_list, crop_list = pad_or_crop_image(
                patient_image, patient_label, target_size=self.target_size
            )
        elif self.mode == "test_pad":
            d, h, w = patient_image.shape[1:]
            pad_d = max(0, 128 - d)
            pad_h = max(0, 128 - h)
            pad_w = max(0, 128 - w)
            patient_image, patient_label, pad_list = pad_image_and_label(
                patient_image, patient_label, target_size=(d + pad_d, h + pad_h, w + pad_w)
            )

        return {
            "patient_id": patient["id"],
            "image": patient_image.to(dtype=torch.float32),
            "label": patient_label.to(dtype=torch.float32),
            "nonzero_indexes": ((zmin, zmax), (ymin, ymax), (xmin, xmax)),
            "box_slice": crop_list,
            "pad_list": pad_list
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
