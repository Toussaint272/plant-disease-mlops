"""Télécharge PlantVillage et PlantDoc dans data/raw/."""
import subprocess
from pathlib import Path

RAW = Path("data/raw")

def clone(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"[skip] {dest} existe déjà")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=True)

if __name__ == "__main__":
    clone("https://github.com/spMohanty/PlantVillage-Dataset.git",
          RAW / "plantvillage")
    clone("https://github.com/pratikkayal/PlantDoc-Dataset.git",
          RAW / "plantdoc")
    print("✅ Téléchargement terminé")