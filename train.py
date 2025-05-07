import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import time
import gc
import nibabel as nib
import tqdm as tqdm
import wandb
import psutil
from utils.meter import AverageMeter
from utils.general import save_checkpoint, load_pretrained_model, resume_training
import hydra
from brats import get_datasets
from omegaconf import OmegaConf, DictConfig
from monai.data import decollate_batch, DataLoader
import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.amp import autocast, GradScaler
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.utils.enums import MetricReduction
from networks.models.ResUNetpp.model import ResUnetPlusPlus
from monai.losses import DiceLoss, DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    Activations,
)
from monai.transforms import (
    Compose,
    RandZoomd,
    RandFlipd,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    RandGaussianNoised,
    Lambdad,
    RandAdjustContrastd,
    RandScaleIntensityd,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
    RandStdShiftIntensityd,
    RandGaussianSharpend,
    RandGibbsNoised
)
from monai.networks.nets import SwinUNETR, SegResNet, VNet, AttentionUnet, UNETR
from networks.models.ResUNetpp.model import ResUnetPlusPlus
from networks.models.UNet.model import UNet3D
from networks.models.UX_Net.network_backbone import UXNET
from networks.models.nnformer.nnFormer_tumor import nnFormer
try:
    from thesis.models.SegUXNet.model import SegUXNet
except ModuleNotFoundError:
    print('model not available, please train with other models')
    
from functools import partial
from utils.augment import DataAugmenter
from utils.schedulers import SegResNetScheduler, PolyDecayScheduler

# Configure logger
import logging
# ————————————————————————————————————————————————————————————————
# Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
os.makedirs("logger", exist_ok=True)
fh = logging.FileHandler("logger/train_logger.log")
fmt = logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S")
fh.setFormatter(fmt)
logger.addHandler(fh)
logger.addHandler(logging.StreamHandler())

# ————————————————————————————————————————————————————————————————
def init_random(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True      # kernels óptimos
    cudnn.deterministic = False

def create_dirs(exp_name: str):
    os.makedirs(exp_name, exist_ok=True)
    os.makedirs(f"{exp_name}/checkpoint", exist_ok=True)
    os.makedirs(f"{exp_name}/best-model", exist_ok=True)

def save_best_model(exp_name: str, model: nn.Module):
    torch.save(model.state_dict(), f"{exp_name}/best-model/best_model.pt")

def save_data(training_loss, et, wt, tc, mean_dice, epochs, cfg):
    data = {
        "training_loss": training_loss,
        "WT": wt,
        "ET": et,
        "TC": tc,
        "mean_dice": mean_dice,
        "epoch": epochs
    }
    df = pd.DataFrame(data)
    path = os.path.join(cfg.training.exp_name, "csv")
    os.makedirs(path, exist_ok=True)
    df.to_csv(os.path.join(path, "training_data.csv"), index=False)
    return df

# ————————————————————————————————————————————————————————————————
def train_epoch(model, loader, optimizer, loss_fn, scaler, device):
    """
    Entrena una época y loguea por batch:
     - imprime:  step/total_steps, train_loss, step time
     - wandb.log({"train/loss_batch": loss})
    """
    model.train()
    meter = AverageMeter()
    total_steps = len(loader)
    for step, batch in enumerate(loader, start=1):
        t0 = time.time()
        imgs = batch["image"].to(device, non_blocking=True)
        lbls = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad()
        with autocast(device_type="cuda"):
            preds = model(imgs)
            loss = loss_fn(preds, lbls)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        step_time = time.time() - t0
        loss_value = loss.item()
        meter.update(loss_value, n=imgs.size(0))

        # Logging por batch
        logger.info(f"{step}/{total_steps}, train_loss: {loss_value:.4f}, step time: {step_time:.4f}")
        wandb.log({"train/loss_batch": loss_value})
    return meter.avg
@torch.no_grad()
def validate(model, loader, inferer, post_sigmoid, post_pred, acc_fn, device):
    model.eval()
    all_preds, all_lbls = [], []
    for batch in loader:
        imgs = batch["image"].to(device, non_blocking=True)
        lbls = decollate_batch(batch["label"].to(device))
        logits = inferer(imgs)
        all_preds.extend(decollate_batch(logits))
        all_lbls.extend(lbls)
    processed = [post_pred(post_sigmoid(x)) for x in all_preds]
    acc_fn.reset()
    acc_fn(y_pred=processed, y=all_lbls)
    metrics, _ = acc_fn.aggregate()
    return metrics.cpu().numpy()  # array [TC, WT, ET]

def extract_tumor_labels_from_seg_masks(image):
    """
    Transforma una máscara de segmentación con etiquetas:
      - 1: Necrosis (NCR)
      - 2: Edema (ED)
      - 3: Enhancing Tumor (ET)
      - 4: Non-Enhancing Tumor (NET)
    en un tensor o array con tres canales binarios:
      0: ET (Enhancing Tumor)
      1: TC (Tumor Core = NCR ∪ NET ∪ ET)
      2: WT (Whole Tumor = TC ∪ ED)
    """
    # Definición de IDs según tu convención
    ncr_label = 1  # Necrosis
    ed_label  = 2  # Edema
    et_label  = 3  # Enhancing Tumor
    net_label = 4  # Non-Enhancing Tumor

    # Soportar tanto torch.Tensor como np.ndarray
    is_tensor = isinstance(image, torch.Tensor)
    if is_tensor:
        img = image.cpu().numpy()
    else:
        img = image.copy()

    # Si viene con canal extra de 1
    if img.ndim == 4 and img.shape[0] == 1:
        img = img.squeeze(0)

    # Crear máscaras binarias de cada componente
    et  = (img == et_label)
    ncr = (img == ncr_label)
    net = (img == net_label)
    ed  = (img == ed_label)

    # Tumor Core = NCR ∪ NET ∪ ET
    tc = np.logical_or(ncr, net)
    tc = np.logical_or(tc, et)

    # Whole Tumor = TC ∪ ED
    wt = np.logical_or(tc, ed)

    # Stack en orden [ET, TC, WT]
    seg_masks = np.stack([et, tc, wt]).astype(np.uint8)

    if is_tensor:
        return torch.from_numpy(seg_masks)
    else:
        return seg_masks
# ————————————————————————————————————————————————————————————————
@hydra.main(config_path="conf", config_name="configs", version_base=None)
def main(cfg: DictConfig):
    # setup
    init_random(cfg.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    create_dirs(cfg.training.exp_name)
    wandb.login(key=cfg.training.wand_api_key)
    wandb.init(
        project=cfg.training.project_name,
        id=cfg.training.exp_name,
        name=cfg.training.exp_name,
        resume=True,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    train_data_transform_func_list = [
                                  EnsureTyped(keys = "label", 
                                              dtype = np.int8),
                                  Lambdad(keys = "label", 
                                          func = extract_tumor_labels_from_seg_masks),
                                  Spacingd(keys = ["image", "label"], 
                                           pixdim = (1.0, 1.0, 1.0), 
                                           mode = ("area", "area")),
                                  Orientationd(keys = ["image", "label"], 
                                               axcodes = "RAS"),
                                  RandAdjustContrastd(keys = "image", prob=0.1, gamma=(0.5, 4.5)),
                                  NormalizeIntensityd(keys = ["image"],
                                                      nonzero = True, 
                                                      channel_wise = True),
                                  # HistogramNormalized(keys = "image"),
                                  RandScaleIntensityd(keys = ["image"], 
                                                      factors = (0, 10),
                                                      prob = 0.3),
                                  RandStdShiftIntensityd(keys = "image", 
                                                         factors = 0.1,
                                                         nonzero = True,
                                                         channel_wise = True,
                                                         prob = 0.7),
                                  RandFlipd(keys = ["image", "label"], 
                                            prob = 0.7, 
                                            spatial_axis = (0, 1)),
                                  RandZoomd(keys = ["image", "label"], 
                                            prob = 0.8),
                                  RandGaussianSharpend(keys = "image", 
                                                       prob = 0.3),
                                  RandGaussianNoised(keys = "image", 
                                                     prob = 0.8, 
                                                     mean = 0, 
                                                     std = np.random.uniform(0, 0.45)),
                                  RandGibbsNoised(keys = "image", 
                                                 prob = 0.5),
                                ]
    val_data_transform_func_list = [
                                  EnsureTyped(keys = "label", 
                                              dtype = np.int8),
                                  Lambdad(keys = "label", 
                                          func = extract_tumor_labels_from_seg_masks),
                                  Spacingd(keys = ["image", "label"], 
                                           pixdim = (1.0, 1.0, 1.0), 
                                           mode = ("area", "area")),
                                  Orientationd(keys = ["image", "label"], 
                                               axcodes = "RAS"),
                                  
                                ]

    # Data
    train_transform = Compose(train_data_transform_func_list)
    val_transform   = Compose(val_data_transform_func_list)  # sin augmentaciones

    train_ds = get_datasets(
        dataset_folder = cfg.dataset.dataset_folder,
        mode           = "train",
        target_size    = (128,128,128),
        version        = "brats2024",
        transform      = train_transform,
    )
    val_ds   = get_datasets(
        dataset_folder = cfg.dataset.dataset_folder,
        mode           = "val",
        target_size    = (128,128,128),
        version        = "brats2024",
        transform      = val_transform,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # Model
    arch = cfg.model.architecture
    if arch == "segres_net":
        model = SegResNet(spatial_dims=3, init_filters=32,
                          in_channels=4, out_channels=3,
                          dropout_prob=0.2,
                          blocks_down=(1,2,2,4),
                          blocks_up=(1,1,1))
    elif arch == "unet3d":
        model = UNet3D(in_channels=4, num_classes=3)
    elif arch == "v_net":
        model = VNet(spatial_dims=3, in_channels=4, out_channels=3)
    elif arch == "attention_unet":
        model = AttentionUnet(spatial_dims=3, in_channels=4, out_channels=3)
    elif arch == "resunet_pp":
        model = ResUnetPlusPlus(in_channels=4, out_channels=3)
    elif arch == "unet_r":
        model = UNETR(in_channels=4, out_channels=3, img_size=(128,128,128))
    elif arch == "swinunet_r":
        model = SwinUNETR(img_size=128, in_channels=4, out_channels=3,
                          feature_size=48, drop_rate=0.1, attn_drop_rate=0.2,
                          spatial_dims=3)
    elif arch == "ux_net":
        model = UXNET(in_chans=4, out_chans=3,
                      depths=[2,2,2,2], feat_size=[48,96,192,384],
                      spatial_dims=3)
    elif arch == "nn_former":
        model = nnFormer(crop_size=(128,128,128), embedding_dim=96,
                         input_channels=4, num_classes=3,
                         depths=[2,2,2,2], num_heads=[3,6,12,24],
                         deep_supervision=False)
    elif arch == "seg_uxnet" and SegUXNet:
        model = SegUXNet(spatial_dims=3, init_filters=32,
                         in_channels=4, out_channels=3)
    else:
        raise ValueError(f"Unknown arch {arch}")
    
    model = model.to(device)

    if cfg.training.pretrained:
        pretrained_path = cfg.training.pretrained_path  # agrégalo en tu config
        model = load_pretrained_model(
            model,
            state_path=pretrained_path,
            device=device,
            strict=False            # o True si tus keys coinciden exactamente
        )
    # Loss
    if cfg.training.loss_type == "dice":
        loss_fn = DiceLoss(to_onehot_y=False, sigmoid=True)
    else:
        loss_fn = DiceCELoss(to_onehot_y=False, sigmoid=True)

    # Metric / inferer
    post_sigmoid = Activations(sigmoid=True)
    post_pred    = AsDiscrete(argmax=False, threshold=0.5)
    acc_fn       = DiceMetric(include_background=True,
                              reduction=MetricReduction.MEAN_BATCH,
                              get_not_nans=True)
    inferer = partial(
        sliding_window_inference,
        roi_size=(128,128,128),
        sw_batch_size=cfg.training.sw_batch_size,
        predictor=model,
        overlap=cfg.model.infer_overlap,
    )

    # Optimizer / scheduler / scaler / augmenter
    optimizer = getattr(torch.optim, cfg.training.solver_name)(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay
    )
    if arch == "segres_net":
        scheduler = SegResNetScheduler(optimizer, cfg.training.max_epochs, cfg.training.learning_rate)
    elif arch == "nn_former":
        scheduler = PolyDecayScheduler(optimizer,
                                       total_epochs=cfg.training.max_epochs,
                                       initial_lr=cfg.training.learning_rate)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.training.max_epochs)
    scaler = GradScaler()
    #augmenter = DataAugmenter(prob=0.5).to(device)

    # Históricos
    training_losses, dices_tc, dices_wt, dices_et, dices_mean, epochs_list = [], [], [], [], [], []

    # Training loop
    best_mean = 0.0
    start_epoch = 0   
    ckpt_dir  = os.path.join(cfg.training.exp_name, "checkpoint")
    ckpt_path = os.path.join(ckpt_dir, "checkpoint.pth")
    if cfg.training.resume:
        if os.path.isfile(ckpt_path):
            start_epoch, best_mean = resume_training(
                model, optimizer, scheduler, ckpt_path, device
            )
            # si quieres aumentar el total de epochs al reanudar:
            if cfg.training.new_max_epochs is not None:
                cfg.training.max_epochs = cfg.training.new_max_epochs
        else:
            logger.warning(f"No se encontró checkpoint en {ckpt_path}; comenzando desde 0")

    for epoch in range(start_epoch, cfg.training.max_epochs):
        # Epoch training
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, scaler, device)
        t_train = time.time() - t0

        # Scheduler step
        scheduler.step()

        # Epoch validation
        v0 = time.time()
        metrics = validate(model, val_loader, inferer, post_sigmoid, post_pred, acc_fn, device)
        t_val = time.time() - v0

        tc, wt, et = metrics
        mean_d = metrics.mean()

        # Guardar mejor modelo
        if mean_d > best_mean:
            best_mean = mean_d
            save_best_model(cfg.training.exp_name, model)

        # Append históricos
        training_losses.append(train_loss)
        dices_tc.append(tc)
        dices_wt.append(wt)
        dices_et.append(et)
        dices_mean.append(mean_d)
        epochs_list.append(epoch)

        # Logging
        logger.info(
            f"Epoch {epoch+1}/{cfg.training.max_epochs} — "
            f"TrainLoss: {train_loss:.4f} ({t_train:.1f}s) — "
            f"ValDice: {mean_d:.4f} TC:{tc:.4f} WT:{wt:.4f} ET:{et:.4f} ({t_val:.1f}s) — "
            f"GPUmem: {torch.cuda.max_memory_allocated()/1e9:.2f}G — "
            f"CPU%: {psutil.cpu_percent()}%"
        )
        wandb.log({
            "train/loss": train_loss,
            "val/dice_mean": mean_d,
            "val/dice_tc": tc,
            "val/dice_wt": wt,
            "val/dice_et": et,
            "epoch": epoch + 1,
        })
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
            best_acc=best_mean,
            filename="checkpoint.pth",
            save_dir=ckpt_dir
        )

    # Guardar CSV
    save_data(training_losses, dices_et, dices_wt, dices_tc, dices_mean, epochs_list, cfg)

    # Curvas en W&B
    wandb.log({
        "Dice_TC_curve": wandb.plot.line_series(xs=epochs_list, ys=[dices_tc],
                                                 keys=["TC"], title="Dice TC", xname="Epoch"),
        "Dice_WT_curve": wandb.plot.line_series(xs=epochs_list, ys=[dices_wt],
                                                 keys=["WT"], title="Dice WT", xname="Epoch"),
        "Dice_ET_curve": wandb.plot.line_series(xs=epochs_list, ys=[dices_et],
                                                 keys=["ET"], title="Dice ET", xname="Epoch"),
        "Dice_mean_curve": wandb.plot.line_series(xs=epochs_list, ys=[dices_mean],
                                                   keys=["Mean"], title="Mean Dice", xname="Epoch"),
    })

    wandb.finish()

if __name__ == "__main__":
    main()
