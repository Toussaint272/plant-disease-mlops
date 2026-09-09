"""Tests automatises de l API."""
import io

from fastapi.testclient import TestClient
from PIL import Image

from api.main import app

client = TestClient(app)


def fake_image() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (300, 300), (40, 120, 40)).save(buf, format="JPEG")
    return buf.getvalue()


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_classes():
    r = client.get("/classes")
    assert r.status_code == 200
    assert len(r.json()["classes"]) == 10


def test_predict_ok():
    r = client.post(
        "/predict",
        files={"file": ("t.jpg", fake_image(), "image/jpeg")},
    )
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["top_k"]) == 3


def test_predict_rejects_text():
    r = client.post(
        "/predict",
        files={"file": ("a.txt", b"hello", "text/plain")},
    )
    assert r.status_code == 415
