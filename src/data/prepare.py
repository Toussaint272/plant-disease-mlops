"""Filtre PlantVillage sur les 10 classes du projet."""
import shutil
import sys
from pathlib import Path

import yaml

params = yaml.safe_load(open("params.yaml", encoding="utf-8"))
CLASSES = params["data"]["classes"]

SRC = Path("data/raw/plantvillage/raw/color")
DST = Path("data/interim/plantvillage")

ALIASES = {
    "Corn_(maize)___healthy": "Corn___healthy",
    "Corn_(maize)___Common_rust_": "Corn___Common_rust",
}


def main():
    if not SRC.exists():
        print(f"❌ Dossier introuvable : {SRC}")
        sys.exit(1)

    if DST.exists():
        shutil.rmtree(DST)

    DST.mkdir(parents=True, exist_ok=True)

    found = []
    total = 0

    print("📦 Filtrage des classes PlantVillage...\n")

    for folder in sorted(SRC.iterdir()):
        if not folder.is_dir():
            continue

        label = ALIASES.get(folder.name, folder.name)

        if label not in CLASSES:
            continue

        out = DST / label
        out.mkdir(parents=True, exist_ok=True)

        n = 0
        for img in folder.glob("*"):
            if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                shutil.copy2(img, out / img.name)
                n += 1

        found.append(label)
        total += n
        print(f"{label:35s} : {n:5d} images")

    missing = set(CLASSES) - set(found)

    print("\n" + "=" * 60)
    print(f"Classes trouvées : {len(found)}/{len(CLASSES)}")
    print(f"Total images     : {total}")

    if missing:
        print(f"⚠️ Classes manquantes : {missing}")
    else:
        print("✅ Toutes les classes sont présentes")

    print("=" * 60)


if __name__ == "__main__":
    main()
