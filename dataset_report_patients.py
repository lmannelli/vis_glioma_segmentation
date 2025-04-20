import os
import sys
import random
from collections import defaultdict, Counter

def gather_patient_studies(root_dir):
    """
    Recorre cada subcarpeta en root_dir y construye un diccionario:
    {patient_id: [study_folder_names]}
    Asume carpetas con formato: BraTS-GLI-<patient_id>-<study_id>
    """
    patient_dict = defaultdict(list)
    try:
        entries = os.listdir(root_dir)
    except FileNotFoundError:
        print(f"Error: El directorio '{root_dir}' no existe.")
        sys.exit(1)

    for entry in entries:
        path = os.path.join(root_dir, entry)
        if not os.path.isdir(path):
            continue
        parts = entry.split('-')
        if len(parts) < 4:
            continue
        patient_id = parts[2]
        patient_dict[patient_id].append(entry)

    return patient_dict


def report_distribution(patient_dict):
    """
    Imprime un reporte de cuántos pacientes tienen X cantidad de estudios.
    """
    counts = Counter(len(studies) for studies in patient_dict.values())
    print("Distribución de estudios por paciente:")
    for n_studies, n_patients in sorted(counts.items()):
        print(f"- {n_patients} paciente(s) con {n_studies} estudio(s)")


def recommend_split(patient_dict, train_ratio=0.8, seed=42):
    """
    Calcula y muestra cuántos pacientes irían en train/validation
en base al porcentaje dado, sin mover archivos.
    """
    patient_ids = list(patient_dict.keys())
    total = len(patient_ids)
    n_train = int(total * train_ratio)
    n_val = total - n_train

    random.seed(seed)
    train_patients = random.sample(patient_ids, n_train)
    val_patients = [pid for pid in patient_ids if pid not in train_patients]

    print(f"\nSplit recomendado ({train_ratio*100:.0f}% train):")
    print(f"Total pacientes: {total}")
    print(f"- Train: {len(train_patients)} pacientes")
    print(f"- Validation: {len(val_patients)} pacientes")

    # Si deseas listas de IDs, descomenta estas líneas:
    # print("\nIDs para Train:", train_patients)
    # print("\nIDs para Validation:", val_patients)

    return train_patients, val_patients


def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Uso: python dataset_report.py /ruta/a/dataset_brats2024 [train_ratio]")
        sys.exit(1)

    root_dir = sys.argv[1]
    train_ratio = float(sys.argv[2]) if len(sys.argv) == 3 else 0.8

    patient_dict = gather_patient_studies(root_dir)
    report_distribution(patient_dict)
    recommend_split(patient_dict, train_ratio)

if __name__ == '__main__':
    main()

