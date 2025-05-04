"""Some helper functions"""

import torch
import torch.nn as nn
import monai
import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path
import sys
import random

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parents[0]
if ROOT not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
# from config.configs import*

def save_checkpoint(model, optimizer, scheduler, epoch, best_acc,
                    filename: str = "checkpoint.pth", save_dir: str = ".",
                    scaler=None, rng_state=None):
    """
    Guarda el estado completo del entrenamiento, incluyendo:
      - modelo
      - optimizador
      - scheduler
      - grad scaler (si existe)
      - estado de RNG de Python, Torch y CUDA (si existe)
      - época actual
      - mejor accuracy
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'best_acc': best_acc,
    }
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    if rng_state is not None:
        checkpoint['rng_state'] = rng_state
    path = f"{save_dir}/{filename}"
    torch.save(checkpoint, path)

def resume_training(model, optimizer, scheduler, ckpt_path, device, scaler=None):
    """
    Restaura todo el estado guardado en el checkpoint:
      - modelo
      - optimizador
      - scheduler
      - grad scaler (si existe)
      - estado de RNG de Python, Torch y CUDA (si existe)
    Devuelve (start_epoch, best_acc).
    """
    # En PyTorch >=2.6, forzar carga completa con weights_only=False
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])

    # Restaurar GradScaler
    if scaler is not None and 'scaler_state_dict' in ckpt:
        scaler.load_state_dict(ckpt['scaler_state_dict'])

    # Restaurar estados de RNG
    if 'rng_state' in ckpt:
        rstate = ckpt['rng_state']
        random.setstate(rstate['python'])
        torch.set_rng_state(rstate['torch'])
        torch.cuda.set_rng_state_all(rstate['cuda'])

    start_epoch = ckpt['epoch']
    best_acc    = ckpt['best_acc']
    return start_epoch, best_acc


def load_pretrained_model(model: nn.Module,
                          state_path: str,
                          device: torch.device = torch.device("cpu"),
                          strict: bool = True) -> nn.Module:
    """
    Carga pesos pre-entrenados en un modelo, opcionalmente sin hacer match exacto de
    nombres de parámetros (strict=False).

    Parameters
    ----------
    model: nn.Module
    state_path: str            Path al .pth o .pt que contiene {"state_dict": ...}
    device: torch.device       Dispositivo de mapeo (cpu o cuda)
    strict: bool               Si True, exige coincidencia exacta de keys

    Returns
    -------
    model: nn.Module           Modelo con pesos cargados
    """
    checkpoint = torch.load(state_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    # En algunos casos tu checkpoint podría tener prefijos distintos (p.ej. "module.")
    # Aquí puedes limpiar prefijos si fuera necesario:
    new_state = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "")  # quita 'module.' si existe
        new_state[name] = v
    model.load_state_dict(new_state, strict=strict)
    print(f"=> Pretrained weights loaded from {state_path} (strict={strict})")
    return model


def visualize_data_sample(case, id,  slice=78, modality= "flair"):
    """visualize a modality along with the segmentation label in a subplot
    
    Parmeters
    ---------
    case: str
    slice: int
    modality: str"""
    # img_add = os.path.join(data_dir, f"TrainingData/BraTS2021_00006/BraTS2021_00006_{modality}.nii.gz")
    # label_add = os.path.join(data_dir, "TrainingData/BraTS2021_00006/BraTS2021_00006_g.nii.gz")
    test_image = case + f"/{id}_{modality}.nii.gz"
    label = case + f"/{id}_seg.nii.gz"
    img = nib.load(test_image).get_fdata()
    label = nib.load(label).get_fdata()
    print(f"image shape: {img.shape}, label shape: {label.shape}")
    IMAGES = [img, label]
    TITLES =["Image", "Label"]
    fig, axes = plt.subplots(1, 2, figsize = (18, 6))
    for i in range(len(IMAGES)):
        if i == 0:
            axes[i].imshow(IMAGES[i][:, :, slice], cmap = 'gray')
        else:
            axes[i].imshow(IMAGES[i][:, :, slice])

        axes[i].set_title(TITLES[i])
        axes[i].set_axis_off()
    plt.show()




def seed_everything(seed: int):
    """generate a random seed
    
    Parameters:
    ----------
    seed: int
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def plot_train_histroy(data):
    """plot training history of the model
    
    Parameters
    ----------
    data_df: dict
    """
    NAMES = ["training_loss", "WT", "ET", "TC", "mean_dice", "epochs"]
    data_lists = []
    for name in NAMES:
        data_list = data[name]
        data_lists.append(data_list)
    with plt.style.context("seaborn-dark-palette"):
            fig, axes = plt.subplots(3, 1, figsize=(8, 10))
            for i, ax in enumerate(axes):
                ax.plot(data[i]['val'], c=colors[0], label="val")
                ax.plot(data[i]['train'], c=colors[-1], label="train")
                ax.set_title(labels[i])
                ax.legend(loc="upper right")
                
            plt.tight_layout()
            plt.show()
    


if __name__ == "__main__":
    print()
    print('Loading the patient case for visualization')
    visualize_data_sample(Config.newGlobalConfigs.full_patient_path, Config.newGlobalConfigs.a_test_patient)