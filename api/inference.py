"""Chargement du modèle et prediction."""
import io
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.models.model import build_model

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

ADVICE = {
    "healthy": "Aucun signe de maladie detecte. Continuez la surveillance.",
    "Early_blight": "Alternariose precoce probable : retirez les feuilles atteintes.",
    "Late_blight": "Mildiou probable : agissez vite, propagation rapide.",
    "Leaf_Mold": "Cladosporiose probable : aerez, reduisez l humidite.",
    "Septoria_leaf_spot": "Septoriose probable : eliminez les debris, rotation.",
    "Common_rust": "Rouille commune probable : surveillez l extension.",
}


class Predictor:
    def __init__(self, model_path="models/model.pt"):
        ckpt = torch.load(model_path, map_location="cpu")
        self.classes = ckpt["classes"]
        self.size = ckpt["image_size"]
        self.backbone = ckpt["backbone"]
        self.version = ckpt.get("mlflow_run_id", "local")[:8]

        self.model = build_model(self.backbone, len(self.classes), pretrained=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        self.tf = transforms.Compose([
            transforms.Resize(int(self.size * 1.14)),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    @staticmethod
    def image_stats(img):
        """Statistiques de l image (serviront a Evidently pour le drift)."""
        a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        gray = a.mean(axis=2)
        return {
            "width": img.width,
            "height": img.height,
            "brightness": float(gray.mean()),
            "contrast": float(gray.std()),
            "mean_r": float(a[:, :, 0].mean()),
            "mean_g": float(a[:, :, 1].mean()),
            "mean_b": float(a[:, :, 2].mean()),
        }

    @torch.no_grad()
    def predict(self, raw: bytes, top_k: int = 3) -> dict:
        t0 = time.perf_counter()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        stats = self.image_stats(img)

        x = self.tf(img).unsqueeze(0)
        probs = torch.softmax(self.model(x), dim=1)[0]
        k = min(top_k, len(self.classes))
        conf, idx = probs.topk(k)

        label = self.classes[idx[0]]
        advice = next(
            (v for key, v in ADVICE.items() if key in label),
            "Consultez un technicien agricole.",
        )

        return {
            "prediction": label,
            "confidence": round(float(conf[0]), 4),
            "top_k": [
                {"label": self.classes[i], "probability": round(float(c), 4)}
                for c, i in zip(conf, idx)
            ],
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "model_version": self.version,
            "advice": advice,
            "image_stats": stats,
        }
