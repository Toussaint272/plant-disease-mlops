"""Entraînement du modèle avec MLflow."""
import json
import time
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.models.dataset import load_dataset
from src.models.model import build_model, set_backbone_trainable

params = yaml.safe_load(open("params.yaml", encoding="utf-8"))

TRAIN_PARAMS = params["train"]
DATA_PARAMS = params["data"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None

    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    y_true = []
    y_pred = []

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)

            preds = outputs.argmax(dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    avg_loss = total_loss / len(y_true)
    acc = sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)
    f1 = f1_score(y_true, y_pred, average="macro")

    return avg_loss, acc, f1


def main():
    print(f"🚀 Device utilisé : {DEVICE}")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("plant-disease-detection")

    train_ds = load_dataset("train", train=True)
    val_ds = load_dataset("val", train=False)

    classes = train_ds.classes
    n_classes = len(classes)

    print(f"Classes : {n_classes}")
    print(classes)
    print(f"Train images : {len(train_ds)}")
    print(f"Val images   : {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=TRAIN_PARAMS["batch_size"],
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=TRAIN_PARAMS["batch_size"],
        shuffle=False,
        num_workers=0
    )

    model = build_model(
        backbone=TRAIN_PARAMS["backbone"],
        n_classes=n_classes,
        pretrained=TRAIN_PARAMS["pretrained"]
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAIN_PARAMS["lr"],
        weight_decay=TRAIN_PARAMS["weight_decay"]
    )

    Path("models").mkdir(exist_ok=True)

    best_val_f1 = -1.0

    with mlflow.start_run(run_name=f"run-{int(time.time())}") as run:
        mlflow.log_params({
            "backbone": TRAIN_PARAMS["backbone"],
            "pretrained": TRAIN_PARAMS["pretrained"],
            "epochs": TRAIN_PARAMS["epochs"],
            "batch_size": TRAIN_PARAMS["batch_size"],
            "lr": TRAIN_PARAMS["lr"],
            "weight_decay": TRAIN_PARAMS["weight_decay"],
            "image_size": DATA_PARAMS["image_size"],
            "n_classes": n_classes,
            "device": DEVICE,
            "n_train": len(train_ds),
            "n_val": len(val_ds),
        })

        mlflow.log_artifact("params.yaml")

        for epoch in range(TRAIN_PARAMS["epochs"]):
            trainable = epoch >= TRAIN_PARAMS["freeze_backbone_epochs"]
            set_backbone_trainable(model, trainable=trainable)

            train_loss, train_acc, train_f1 = run_epoch(
                model, train_loader, criterion, optimizer
            )

            val_loss, val_acc, val_f1 = run_epoch(
                model, val_loader, criterion, optimizer=None
            )

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_f1": train_f1,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
            }, step=epoch)

            print(
                f"[{epoch + 1}/{TRAIN_PARAMS['epochs']}] "
                f"train_acc={train_acc:.4f} train_f1={train_f1:.4f} | "
                f"val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1

                torch.save({
                    "state_dict": model.state_dict(),
                    "classes": classes,
                    "backbone": TRAIN_PARAMS["backbone"],
                    "image_size": DATA_PARAMS["image_size"],
                    "val_f1": best_val_f1,
                    "mlflow_run_id": run.info.run_id,
                }, "models/model.pt")

                with open("models/model_meta.json", "w", encoding="utf-8") as f:
                    json.dump({
                        "classes": classes,
                        "backbone": TRAIN_PARAMS["backbone"],
                        "image_size": DATA_PARAMS["image_size"],
                        "val_f1": best_val_f1,
                        "mlflow_run_id": run.info.run_id,
                    }, f, indent=2)

        mlflow.log_metric("best_val_f1", best_val_f1)
        mlflow.log_artifact("models/model.pt")
        mlflow.log_artifact("models/model_meta.json")

    print("\n✅ Entraînement terminé")
    print(f"Meilleur val_f1 : {best_val_f1:.4f}")
    print("Modèle sauvegardé dans models/model.pt")


if __name__ == "__main__":
    main()

