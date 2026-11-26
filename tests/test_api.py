"""API tests. They need the trained model in models/ and the dataset in data/ (both git-ignored)."""

import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import MODEL_NAME, MODELS_DIR, app

if not (MODELS_DIR / f"{MODEL_NAME}_best.pth").exists():
    pytest.skip("trained model not found", allow_module_level=True)

SAMPLES = {"C": Path("data/C/cataract_088.png"), "N": Path("data/N/2551_right.jpg")}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # "with" runs the lifespan: loads the model once
        yield c


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_web_page_is_served(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "Detección de enfermedades oculares" in page.text
    assert client.get("/app.js").status_code == 200
    assert client.get("/docs").status_code == 200  # the static mount must not hide the API docs


def test_metrics_json_is_consistent(client):
    metrics = client.get("/metrics.json").json()
    assert client.get("/metrics.js").status_code == 200
    matrix = metrics["confusion"]
    assert len(matrix) == len(metrics["classes"]) == 7
    assert sum(map(sum, matrix)) == metrics["n_images"]
    for row, cls in zip(matrix, metrics["classes"]):
        assert sum(row) == cls["support"]  # each row holds every real image of its class
    assert 0 < metrics["kpis"]["accuracy"] <= 1


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["calibrated"] is True
    assert set(body["classes"]) == {"A", "C", "D", "G", "H", "M", "N"}


@pytest.mark.parametrize("label", SAMPLES)
def test_predict_real_image(client, label):
    path = SAMPLES[label]
    if not path.exists():
        pytest.skip(f"{path} not found")
    response = client.post("/predict", files={"file": (path.name, path.read_bytes(), "image/jpeg")})
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"]["code"] == label
    assert abs(sum(p["probability"] for p in body["probabilities"]) - 1) < 0.01
    assert body["heatmap"].startswith("data:image/png;base64,")
    if label == "N":
        assert body["prediction"]["level"] in ("normal", "low")  # Normal is never green


def test_heatmap_can_be_skipped(client):
    path = SAMPLES["N"]
    if not path.exists():
        pytest.skip(f"{path} not found")
    response = client.post("/predict?heatmap=false", files={"file": (path.name, path.read_bytes())})
    assert response.json()["heatmap"] is None


@pytest.mark.parametrize("image", [
    Image.new("RGB", (600, 600)),                       # black
    Image.new("RGB", (600, 600), (128, 128, 128)),      # flat gray
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (400, 400, 3), dtype=np.uint8)),
])
def test_rejects_non_fundus(client, image):
    response = client.post("/predict", files={"file": ("x.png", png_bytes(image), "image/png")})
    assert response.status_code == 422
    assert response.json()["detail"]["issues"]


def test_rejects_non_image(client):
    response = client.post("/predict", files={"file": ("notes.txt", b"hola", "text/plain")})
    assert response.status_code == 400


def test_rejects_too_large(client):
    response = client.post("/predict", files={"file": ("big.png", b"0" * (10 * 1024 * 1024 + 1))})
    assert response.status_code == 413
