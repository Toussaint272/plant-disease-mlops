"""API FastAPI de detection des maladies des plantes."""
import csv
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from api.inference import Predictor

LOG_PATH = Path("monitoring/predictions_log.csv")
MAX_MB = 8
CONF_THRESHOLD = 0.60

app = FastAPI(
    title="Plant Disease Detection API - Madagascar",
    description="Detection de maladies foliaires : tomate, pomme de terre, mais.",
    version="1.0.0",
)

_predictor = None


def get_predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        _predictor = Predictor()
    return _predictor


class TopK(BaseModel):
    label: str
    probability: float


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    top_k: list[TopK]
    latency_ms: float
    model_version: str
    low_confidence: bool
    advice: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    n_classes: int


def log_prediction(res: dict):
    """Journalise chaque prediction pour le monitoring Evidently."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prediction": res["prediction"],
        "confidence": res["confidence"],
        "latency_ms": res["latency_ms"],
        "model_version": res["model_version"],
        **res["image_stats"],
    }
    new = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


@app.get("/")
def root():
    return {"message": "Plant Disease API", "docs": "/docs", "health": "/health"}


@app.get("/health", response_model=HealthResponse)
def health():
    p = get_predictor()
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_version=p.version,
        n_classes=len(p.classes),
    )


@app.get("/classes")
def classes():
    return {"classes": get_predictor().classes}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(415, "Le fichier doit etre une image (jpeg/png).")

    raw = await file.read()
    if len(raw) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"Image trop lourde (max {MAX_MB} Mo).")

    try:
        res = get_predictor().predict(raw)
    except Exception as e:
        raise HTTPException(400, f"Image illisible : {e}")

    log_prediction(res)
    res["low_confidence"] = res["confidence"] < CONF_THRESHOLD
    if res["low_confidence"]:
        res["advice"] = "Confiance faible. " + res["advice"]

    return res
