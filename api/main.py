"""Phase 7: REST API — upload a fundus photo, get the disease, the traffic light and the heatmap.

Run:  uv run uvicorn api.main:app --reload
Web:  http://127.0.0.1:8000/
Docs: http://127.0.0.1:8000/docs
"""

import base64
import io
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel

from src.calibrate import LEVEL_LABELS
from src.explorer import CLASS_NAMES
from src.gradcam import Explainer, overlay_heatmap
from src.preprocessing import preprocess_fundus
from src.quality import assess_quality

MODEL_NAME = os.getenv("MODEL_NAME", "v2s")
MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
WEB_DIR = Path(__file__).resolve().parent.parent / "web"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
DISCLAIMER = ("Herramienta de apoyo académica: NO es un diagnóstico médico. "
              "Ante cualquier duda, consultar a un oftalmólogo.")


class Prediction(BaseModel):
    code: str
    name: str
    confidence: float
    level: str
    message: str


class ClassProbability(BaseModel):
    code: str
    name: str
    probability: float


class PredictResponse(BaseModel):
    model: str
    prediction: Prediction
    probabilities: list[ClassProbability]
    warnings: list[str]
    heatmap: str | None  # PNG as a data URL, ready for <img src=...>
    disclaimer: str


state: dict = {}
inference_lock = threading.Lock()  # Grad-CAM keeps activations on the instance: one request at a time


@asynccontextmanager
async def lifespan(app: FastAPI):
    checkpoint = MODELS_DIR / f"{MODEL_NAME}_best.pth"
    if not checkpoint.is_file():  # clear message instead of a long PyTorch traceback
        raise RuntimeError(f"No se encontró el modelo '{checkpoint.resolve()}'. "
                           f"Copia {MODEL_NAME}_best.pth a la carpeta models/ (ver README).")
    state["explainer"] = Explainer(MODEL_NAME, MODELS_DIR)  # load once, not per request
    yield
    state.clear()


app = FastAPI(title="Detección de enfermedades oculares", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["GET", "POST"],
                   allow_headers=["*"])


def read_image(file: UploadFile) -> Image.Image:
    """Validate the upload and open it as RGB, or answer 400/413."""
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "La imagen supera el máximo de 10 MB.")
    try:
        with Image.open(io.BytesIO(data)) as img:
            img = ImageOps.exif_transpose(img)  # phone photos: apply the rotation stored in EXIF
            return img.convert("RGB")
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError):
        raise HTTPException(400, "El archivo no es una imagen válida (usa JPG o PNG).")


def heatmap_data_url(image: Image.Image, cam: np.ndarray) -> str:
    overlay = Image.fromarray((overlay_heatmap(image, cam) * 255).astype(np.uint8))
    buffer = io.BytesIO()
    overlay.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


@app.get("/health")
def health() -> dict:
    explainer: Explainer = state["explainer"]
    return {"status": "ok", "model": MODEL_NAME, "device": str(explainer.device),
            "calibrated": explainer.calibration is not None,
            "classes": {code: CLASS_NAMES[code] for code in explainer.classes}}


@app.post("/predict", response_model=PredictResponse)
def predict(file: UploadFile = File(...),
            heatmap: bool = Query(True, description="include the Grad-CAM image")) -> PredictResponse:
    image = read_image(file)
    clean = preprocess_fundus(image)
    quality = assess_quality(clean)
    if not quality.ok:
        raise HTTPException(422, {"message": "La imagen no sirve para el análisis.",
                                  "issues": quality.messages})

    explainer: Explainer = state["explainer"]
    with inference_lock:
        result = explainer.explain_image(clean)

    code = result["pred"]
    probabilities = sorted(
        (ClassProbability(code=c, name=CLASS_NAMES[c], probability=round(p, 4))
         for c, p in result["distribution"].items()),
        key=lambda item: item.probability, reverse=True)
    return PredictResponse(
        model=MODEL_NAME,
        prediction=Prediction(code=code, name=CLASS_NAMES[code],
                              confidence=round(result["confidence"], 4),
                              level=result["level"], message=LEVEL_LABELS[result["level"]]),
        probabilities=probabilities,
        warnings=quality.warnings,
        heatmap=heatmap_data_url(result["image"], result["cam"]) if heatmap else None,
        disclaimer=DISCLAIMER,
    )


# Phase 8: serve the web page from the same server (http://127.0.0.1:8000/).
# Mounted LAST so /health, /predict and /docs keep priority over the static files.
if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
