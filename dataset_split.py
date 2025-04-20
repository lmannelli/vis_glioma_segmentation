#!/usr/bin/env python3
"""
Reorganiza un dataset de resonancias BraTS en estructuras de train/validation
manteniendo la integridad de pacientes (todas sus resonancias van al mismo split)
con estratificación exacta según número de estudios:

    Estudios | Total pacientes | Train (≈80%) | Validation (resto)
    1        | 144             | 115          | 29
    2        | 383             | 306          | 77
    3        | 55              | 44           | 11
    4        | 32              | 25           | 7
    5        | 22              | 17           | 5
    6        | 12              | 9            | 3
    7        | 5               | 4            | 1
    8        | 3               | 2            | 1
    10       | 1               | 1            | 0

Uso:
    python reorganize_dataset.py RAW_DIR OUTPUT_DIR [train_ratio] [seed]

Arguments:
    RAW_DIR:      Directorio con subcarpetas BraTS-GLI-... 
    OUTPUT_DIR:   Directorio donde se crearán 'train/' y 'val/'
    train_ratio:  Fracción de pacientes para train (default=0.8)
    seed:         Semilla aleatoria (default=42)
"""
import os
import sys
import shutil
import random
from collections import defaultdict

def gather_patient_studies(raw_dir):
    """
    Retorna dict {patient_id: [study_folder_names]}.
    Asume carpetas formato: BraTS-GLI-<patient_id>-<study_id>
    """
    patients = defaultdict(list)
    try:
        entries = sorted(os.listdir(raw_dir))
    except FileNotFoundError:
        print(f"Error: '{raw_dir}' no existe.")
        sys.exit(1)
    for entry in entries:
        path = os.path.join(raw_dir, entry)
        if not os.path.isdir(path):
            continue
        parts = entry.split('-')
        if len(parts) < 4:
            continue
        pid = parts[2]
        patients[pid].append(entry)
    return patients


def stratified_split(patients, train_ratio=0.8, seed=42):
    """
    Divide pacientes en train/val asegurando las cantidades exactas
    por grupo de número de estudios, usando regla:
      train_n = max(1, int(total_group * train_ratio))
    """
    # Agrupar por número de estudios
    strata = defaultdict(list)
    for pid, studies in patients.items():
        n = len(studies)
        strata[n].append(pid)

    train_ids, val_ids = [], []
    random.seed(seed)

    print("Estudios\tTotal\tTrain\tValidation")
    for n in sorted(strata.keys()):
        pids = strata[n]
        size = len(pids)
        random.shuffle(pids)
        # Exact split, garantizando al menos 1 en train
        train_n = max(1, int(size * train_ratio))
        val_n = size - train_n
        train_ids.extend(pids[:train_n])
        val_ids.extend(pids[train_n:])
        print(f"{n}\t{size}\t{train_n}\t{val_n}")

    print(f"\nTotal pacientes: {len(patients)}")
    print(f"-> Train: {len(train_ids)}  Validation: {len(val_ids)}\n")
    return set(train_ids), set(val_ids)


def copy_split(patients, raw_dir, out_dir, train_ids, val_ids):
    """
    Copia archivos a la nueva estructura sin renombrar:
    out_dir/train/images/<folder>/*.nii.gz
    out_dir/train/masks/*-seg.nii.gz
    out_dir/val/...
    """
    for split, ids in [('train', train_ids), ('val', val_ids)]:
        for pid in ids:
            for folder in patients[pid]:
                src_folder = os.path.join(raw_dir, folder)
                dest_img = os.path.join(out_dir, split, 'images', folder)
                dest_mask = os.path.join(out_dir, split, 'masks')
                os.makedirs(dest_img, exist_ok=True)
                os.makedirs(dest_mask, exist_ok=True)
                for fname in os.listdir(src_folder):
                    if not fname.endswith('.nii.gz'):
                        continue
                    src_file = os.path.join(src_folder, fname)
                    # segmentaciones en masks
                    if fname.endswith('-seg.nii.gz'):
                        dst = os.path.join(dest_mask, fname)
                    else:
                        dst = os.path.join(dest_img, fname)
                    shutil.copy2(src_file, dst)
        print(f"Split '{split}' completo.")


def main():
    if len(sys.argv) < 3:
        print("Uso: python reorganize_dataset.py RAW_DIR OUTPUT_DIR [train_ratio] [seed]")
        sys.exit(1)

    raw_dir = sys.argv[1]
    out_dir = sys.argv[2]
    train_ratio = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42

    patients = gather_patient_studies(raw_dir)
    train_ids, val_ids = stratified_split(patients, train_ratio, seed)
    copy_split(patients, raw_dir, out_dir, train_ids, val_ids)
    print("Reorganización completada.")

if __name__ == '__main__':
    main()

