"""Découpe les données en train / val / test_lab / test_field."""
import csv
import random
import shutil
import sys
from pathlib import Path

import yaml

params = yaml.safe_load(open("params.yaml", encoding="utf-8"))
SEED = params["data"]["seed"]
VAL = params["data"]["val_split"]
CLASSES = params["data"]["classes"]

INTERIM = Path("data/interim/plantvillage")
LOCAL = Path("data/raw/local_mada")
OUT = Path("data/processed")


def reset(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def count_images(path: Path) -> int:
    return len([
        p for p in path.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])


def main():
    if not INTERIM.exists():
        print(f"❌ {INTERIM} introuvable. Lancez d'abord prepare.py")
        sys.exit(1)

    random.seed(SEED)

    for split in ["train", "val", "test_lab", "test_field"]:
        reset(OUT / split)

    print("📦 Découpage PlantVillage...\n")

    for label in CLASSES:
        src = INTERIM / label

        if not src.exists():
            print(f"⚠️ Classe absente : {label}")
            continue

        imgs = [
            p for p in sorted(src.glob("*"))
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]

        random.shuffle(imgs)

        n = len(imgs)
        n_val = int(n * VAL)
        n_test = int(n * VAL)

        parts = {
            "val": imgs[:n_val],
            "test_lab": imgs[n_val:n_val + n_test],
            "train": imgs[n_val + n_test:],
        }

        for split, files in parts.items():
            dst = OUT / split / label
            dst.mkdir(parents=True, exist_ok=True)

            for f in files:
                shutil.copy2(f, dst / f.name)

        print(
            f"{label:35s} "
            f"train={len(parts['train']):5d} "
            f"val={len(parts['val']):4d} "
            f"test={len(parts['test_lab']):4d}"
        )

    labels_csv = LOCAL / "labels.csv"

    if labels_csv.exists():
        print("\n🇲🇬 Ajout des photos locales Madagascar...")

        n_local = 0

        with open(labels_csv, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)

            for row in reader:
                filename = row.get("filename", "").strip()
                label = row.get("label", "").strip()

                src_img = LOCAL / filename

                if label in CLASSES and src_img.exists():
                    dst = OUT / "test_field" / label
                    dst.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_img, dst / src_img.name)
                    n_local += 1

        print(f"✅ {n_local} photos locales ajoutées dans test_field")
    else:
        print("\n⚠️ Pas encore de photos locales.")
        print("Créez plus tard : data/raw/local_mada/labels.csv")

    print("\n" + "=" * 60)
    for split in ["train", "val", "test_lab", "test_field"]:
        n = count_images(OUT / split)
        print(f"{split:12s} : {n:6d} images")
    print("=" * 60)


if __name__ == "__main__":
    main()
