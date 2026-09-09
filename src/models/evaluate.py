"""Évaluation du modèle."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from src.models.dataset import load_dataset
from src.models.model import build_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def has_images(split):
    p = Path(f"data/processed/{split}")
    if not p.exists():
        return False
    return len(list(p.rglob("*"))) > 0


def load_trained_model():
    ckpt = torch.load("models/model.pt", map_location=DEVICE)
    model = build_model(ckpt["backbone"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE)
    model.eval()
    return model, ckpt["classes"]


@torch.no_grad()
def predict_split(model, split):
    ds = load_dataset(split, train=False)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    y_true, y_pred, confs = [], [], []
    
    for x, y in loader:
        x = x.to(DEVICE)
        out = model(x)
        probs = torch.softmax(out, dim=1)
        c, preds = probs.max(dim=1)
        y_true.extend(y.tolist())
        y_pred.extend(preds.cpu().tolist())
        confs.extend(c.cpu().tolist())
    return y_true, y_pred, confs


def main():
    if not Path("models/model.pt").exists():
        print("❌ models/model.pt introuvable. Lancez d'abord python -m src.models.train")
        return
    
    print("Chargement du modèle...")
    model, classes = load_trained_model()
    
    Path("reports").mkdir(exist_ok=True)
    Path("reports/figures").mkdir(exist_ok=True)
    results = {}
    
    for split in ["test_lab", "test_field"]:
        if not has_images(split):
            print(f"\n⏭️  {split} vide, ignoré.")
            continue
        
        print(f"\n📊 Évaluation {split}...")
        yt, yp, cf = predict_split(model, split)
        
        acc = accuracy_score(yt, yp)
        f1 = f1_score(yt, yp, average="macro")
        
        results[split] = {
            "n": len(yt), "accuracy": round(acc,4),
            "f1_macro": round(f1,4), "mean_confidence": round(sum(cf)/len(cf),4),
            "report": classification_report(yt,yp,target_names=classes,output_dict=True,zero_division=0)
        }
        
        print(f"  Accuracy : {acc:.4f}")
        print(f"  F1 macro : {f1:.4f}")
        
        cm = confusion_matrix(yt, yp)
        plt.figure(figsize=(11,9))
        sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu",
                   xticklabels=classes, yticklabels=classes)
        plt.title(f"Matrice - {split}"); plt.xlabel("Predicted"); plt.ylabel("Real")
        plt.xticks(rotation=45, ha="right"); plt.tight_layout()
        plt.savefig(f"reports/figures/cm_{split}.png", dpi=140); plt.close()
    
    if "test_lab" in results and "test_field" in results:
        gap = results["test_lab"]["accuracy"] - results["test_field"]["accuracy"]
        results["domain_gap"] = round(gap,4)
        print(f"\n{'='*55}")
        print(f"  LABO   : {results['test_lab']['accuracy']:.2%}")
        print(f"  TERRAIN: {results['test_field']['accuracy']:.2%}")
        print(f"  GAP    : {gap:.2%}")
        print(f"{'='*55}")
    
    json.dump(results, open("reports/metrics.json","w"), indent=2)
    print("\n✅ Sauvegardé dans reports/metrics.json")


if __name__ == "__main__":
    main()
