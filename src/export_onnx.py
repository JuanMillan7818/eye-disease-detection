"""Phase 9: export the calibrated model to ONNX, so it can run without PyTorch (any language).

Outputs (in models/):
  <model>.onnx       the network + the calibration, input (N, 3, H, W) float32 -> two outputs
  <model>.onnx.json  everything ONNX can't hold: classes, preprocessing and the screening policy

Run: uv run --group export python -m src.export_onnx
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from src.calibrate import HIGH_CONFIDENCE, LEVEL_LABELS, LOW_CONFIDENCE, NORMAL_CLASS, NORMAL_FACTOR, decide
from src.dataset import IMAGENET_MEAN, IMAGENET_STD, build_transforms, read_splits
from src.explorer import CLASS_NAMES, DATA_DIR
from src.model import load_checkpoint
from src.preprocessing import preprocess_fundus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="v2s", help="checkpoint name inside models/")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--check", type=int, default=100, help="test images used to compare ONNX vs PyTorch")
    return parser.parse_args()


class CalibratedNet(nn.Module):
    """Network + softmax + weight-corrected temperature scaling, all inside the ONNX graph."""

    def __init__(self, model: nn.Module, class_weights: torch.Tensor, temperature: float):
        super().__init__()
        self.model = model
        self.register_buffer("log_weights", class_weights.log())
        self.temperature = temperature

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.model(image)
        probabilities = logits.softmax(dim=1)
        # same as calibrated_confidence(): log(p / w) / T, then softmax (log p = logits + constant)
        calibrated = ((logits - self.log_weights) / self.temperature).softmax(dim=1)
        return probabilities, calibrated


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def export(args: argparse.Namespace) -> tuple[Path, CalibratedNet, dict]:
    model, checkpoint = load_checkpoint(args.models_dir / f"{args.model}_best.pth", "cpu")
    calibration = checkpoint.get("calibration")
    if calibration is None:
        raise SystemExit(f"Modelo sin calibrar: ejecuta primero 'python -m src.calibrate --model {args.model}'")

    net = CalibratedNet(model, torch.tensor(calibration["class_weights"], dtype=torch.float32),
                        calibration["temperature"]).eval()
    size = checkpoint["image_size"]
    onnx_path = args.models_dir / f"{args.model}.onnx"
    dummy = torch.randn(1, 3, size, size)
    torch.onnx.export(
        net, (dummy,), onnx_path,
        input_names=["image"], output_names=["probabilities", "calibrated"],
        dynamic_shapes={"image": {0: torch.export.Dim("batch", min=1, max=64)}},
        external_data=False,  # one self-contained file (weights inside)
        dynamo=True,
        verbose=False,  # its progress messages contain emoji that break some Windows consoles
    )
    return onnx_path, net, checkpoint


def write_metadata(onnx_path: Path, checkpoint: dict) -> Path:
    calibration = checkpoint["calibration"]
    classes = checkpoint["classes"]
    meta = {
        "model": checkpoint["model_name"],
        "file": onnx_path.name,
        "sha256": sha256(onnx_path),
        "classes": classes,
        "class_names": {c: CLASS_NAMES[c] for c in classes},
        "input": {
            "name": "image", "shape": ["batch", 3, checkpoint["image_size"], checkpoint["image_size"]],
            "dtype": "float32", "color": "RGB", "layout": "NCHW",
            "preprocessing": [
                "crop the black border around the retina (gray > 15)",
                "center-crop the largest square",
                "resize to 512x512 (Lanczos)",
                "black out everything outside a centered circle of radius 0.95 * 256",
                f"resize to {checkpoint['image_size']}x{checkpoint['image_size']} (bilinear)",
                "scale to [0, 1], then (x - mean) / std per channel",
            ],
            "mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD),
        },
        "outputs": {
            "probabilities": "raw softmax: use it for the DECISION",
            "calibrated": "calibrated probabilities: use them for the CONFIDENCE shown",
        },
        "policy": {
            "decision": "scale probabilities[normal_class] by normal_factor, then argmax",
            "confidence": "calibrated[decided class]",
            "normal_class": NORMAL_CLASS,
            "normal_factor": calibration.get("normal_factor", NORMAL_FACTOR),
            "levels": "low if confidence < low; normal if decided class is N; high if >= high; else medium",
            "high": HIGH_CONFIDENCE, "low": LOW_CONFIDENCE,
            "level_labels": LEVEL_LABELS,
        },
        "temperature": calibration["temperature"],
        "disclaimer": "Academic project: NOT a medical diagnosis.",
    }
    meta_path = onnx_path.with_suffix(".onnx.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta_path


@torch.no_grad()
def compare(onnx_path: Path, net: CalibratedNet, checkpoint: dict, n: int) -> None:
    """Same test images through PyTorch and ONNX Runtime: outputs and decisions must match."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    transform = build_transforms(checkpoint["image_size"], train=False)
    records = [r for r in read_splits() if r["split"] == "test"][::max(1, 1446 // n)][:n]
    batch = []
    for r in records:
        with Image.open(DATA_DIR / r["source"]) as img:
            batch.append(transform(preprocess_fundus(img)))
    images = torch.stack(batch)

    start = time.perf_counter()
    torch_probs, torch_cal = net(images)
    torch_time = time.perf_counter() - start
    start = time.perf_counter()
    onnx_probs, onnx_cal = session.run(None, {"image": images.numpy()})
    onnx_time = time.perf_counter() - start

    classes = checkpoint["classes"]
    torch_pred, _, _, torch_dist = decide(torch_probs, checkpoint["calibration"], classes)
    onnx_pred, _, _, _ = decide(torch.from_numpy(onnx_probs), checkpoint["calibration"], classes)
    print(f"Comparación con {len(records)} imágenes de test:")
    print(f"  diferencia máxima probabilities: {np.abs(onnx_probs - torch_probs.numpy()).max():.2e}")
    print(f"  diferencia máxima calibrated:    {np.abs(onnx_cal - torch_cal.numpy()).max():.2e}")
    print(f"  calibrated == calibrate.py:      {np.abs(torch_dist.numpy() - torch_cal.numpy()).max():.2e}")
    print(f"  misma decisión:                  {int((torch_pred == onnx_pred).sum())}/{len(records)}")
    print(f"  tiempo por imagen: PyTorch {torch_time / len(records) * 1000:.0f} ms | "
          f"ONNX Runtime {onnx_time / len(records) * 1000:.0f} ms")


def main() -> None:
    args = parse_args()
    onnx_path, net, checkpoint = export(args)
    meta_path = write_metadata(onnx_path, checkpoint)
    print(f"ONNX: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB) | metadatos: {meta_path}")
    if args.check:
        compare(onnx_path, net, checkpoint, args.check)


if __name__ == "__main__":
    main()
