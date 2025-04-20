import os
import sys

def main(root_dir):
    """
    Recorre cada subcarpeta en root_dir y verifica que existan los archivos:
    - <folder>-seg.nii.gz
    - <folder>-t1c.nii.gz
    - <folder>-t1n.nii.gz
    - <folder>-t2f.nii.gz
    - <folder>-t2w.nii.gz
    Si falta alguno, almacena el nombre de la carpeta y lista los archivos faltantes.
    """
    required_suffixes = [
        "-seg.nii.gz",
        "-t1c.nii.gz",
        "-t1n.nii.gz",
        "-t2f.nii.gz",
        "-t2w.nii.gz",
    ]

    missing = []

    # Lista todas las entradas en el directorio raíz
    try:
        entries = os.listdir(root_dir)
    except FileNotFoundError:
        print(f"Error: El directorio '{root_dir}' no existe.")
        sys.exit(1)

    for entry in entries:
        subdir = os.path.join(root_dir, entry)
        if not os.path.isdir(subdir):
            continue

        files = set(os.listdir(subdir))
        faltantes = []

        for suf in required_suffixes:
            expected = f"{entry}{suf}"
            if expected not in files:
                faltantes.append(expected)

        if faltantes:
            missing.append((entry, faltantes))

    # Imprime resultado
    if missing:
        print("Carpetas con archivos faltantes:")
        for folder, faltantes in missing:
            print(f"- {folder}: faltan {', '.join(faltantes)}")
    else:
        print("Todas las carpetas contienen los archivos requeridos.")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Uso: python check_dataset.py /ruta/a/dataset_brats2024")
        sys.exit(1)

    root_directory = sys.argv[1]
    main(root_directory)

