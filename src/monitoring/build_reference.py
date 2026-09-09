"""Cree monitoring/reference.csv a partir de data/processed/val"""
from pathlib import Path
import pandas as pd
from torchvision import datasets
from api.inference import Predictor

def main():
    pred = Predictor()
    ds = datasets.ImageFolder("data/processed/val")
    rows = []
    total = len(ds.samples)
    print(f"Construction reference sur {total} images val...")
    for i, (path, label) in enumerate(ds.samples):
        raw = Path(path).read_bytes()
        res = pred.predict(raw)
        rows.append({
            "prediction": res["prediction"],
            "confidence": res["confidence"],
            "true_label": ds.classes[label],
            **res["image_stats"],
        })
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{total}")
    Path("monitoring").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv("monitoring/reference.csv", index=False)
    print(f"OK reference creee ({len(rows)} lignes)")

if __name__ == "__main__":
    main()
