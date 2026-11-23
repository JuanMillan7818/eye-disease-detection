"""Use the exported model WITHOUT PyTorch and without this project's code.

Needs only:  pip install onnxruntime numpy pillow
Files:       v2s.onnx and v2s.onnx.json (same folder)
Run:         python onnx_predict.py v2s.onnx photo.jpg [more.jpg ...]

Academic project: the result is NOT a medical diagnosis.
"""

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw


def preprocess_fundus(img: Image.Image, size: int = 512) -> Image.Image:
    """Crop to the retina, square it, resize and mask (same as src/preprocessing.py)."""
    img = img.convert("RGB")
    gray = np.asarray(img.convert("L"))
    rows = np.flatnonzero((gray > 15).any(axis=1))
    cols = np.flatnonzero((gray > 15).any(axis=0))
    if rows.size and cols.size:
        img = img.crop((cols[0], rows[0], cols[-1] + 1, rows[-1] + 1))
    width, height = img.size
    side = min(width, height)
    left, top = (width - side) // 2, (height - side) // 2
    img = img.crop((left, top, left + side, top + side)).resize((size, size), Image.Resampling.LANCZOS)
    radius = 0.95 * size / 2
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).ellipse((size / 2 - radius, size / 2 - radius,
                                  size / 2 + radius, size / 2 + radius), fill=255)
    return Image.composite(img, Image.new("RGB", img.size), mask)


def to_tensor(img: Image.Image, meta: dict) -> np.ndarray:
    """Resize, scale to [0, 1], normalize, and reorder to (1, 3, H, W)."""
    size = meta["input"]["shape"][-1]
    pixels = np.asarray(img.resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255
    pixels = (pixels - np.array(meta["input"]["mean"], dtype=np.float32)) / np.array(meta["input"]["std"], dtype=np.float32)
    return pixels.transpose(2, 0, 1)[None].astype(np.float32)


def decide(probabilities: np.ndarray, calibrated: np.ndarray, meta: dict) -> tuple[str, float, str]:
    """Screening policy: ask for more evidence before saying Normal; Normal is never green."""
    policy, classes = meta["policy"], meta["classes"]
    scaled = probabilities.copy()
    scaled[classes.index(policy["normal_class"])] *= policy["normal_factor"]
    index = int(scaled.argmax())
    confidence = float(calibrated[index])
    if confidence < policy["low"]:
        level = "low"
    elif classes[index] == policy["normal_class"]:
        level = "normal"
    else:
        level = "high" if confidence >= policy["high"] else "medium"
    return classes[index], confidence, level


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    model_path = Path(sys.argv[1])
    meta = json.loads(model_path.with_suffix(".onnx.json").read_text(encoding="utf-8"))
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    for image_path in sys.argv[2:]:
        with Image.open(image_path) as img:
            tensor = to_tensor(preprocess_fundus(img), meta)
        probabilities, calibrated = session.run(None, {"image": tensor})
        code, confidence, level = decide(probabilities[0], calibrated[0], meta)
        print(f"{image_path}: {meta['class_names'][code]} {confidence:.1%} "
              f"| {meta['policy']['level_labels'][level]}")


if __name__ == "__main__":
    main()
