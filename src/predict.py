"""Predict the disease of one or more fundus images with a trained (and calibrated) checkpoint."""

import argparse
from pathlib import Path

import torch
from PIL import Image

from src.calibrate import LEVEL_LABELS, calibrated_confidence, confidence_level
from src.dataset import build_transforms, read_splits
from src.explorer import CLASS_NAMES, DATA_DIR
from src.model import load_checkpoint
from src.preprocessing import preprocess_fundus
from src.train import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path, help="raw fundus images (e.g. data/D/123_left.jpg)")
    parser.add_argument("--model", default="v2s", help="checkpoint name inside models/")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    return parser.parse_args()


def find_split(image_path: Path, splits: dict[str, dict]) -> dict | None:
    """Look the image up in splits.csv to know its real label and which split it belongs to."""
    try:
        key = image_path.resolve().relative_to(DATA_DIR.resolve()).as_posix()
    except ValueError:  # image outside data/: unknown label
        return None
    return splits.get(key)


@torch.no_grad()
def predict_image(model: torch.nn.Module, image_path: Path, image_size: int,
                  device: torch.device) -> torch.Tensor:
    """Same pipeline as training: fundus preprocessing -> resize -> normalize -> softmax."""
    with Image.open(image_path) as img:
        clean = preprocess_fundus(img)
    tensor = build_transforms(image_size, train=False)(clean).unsqueeze(0).to(device)
    return model(tensor).softmax(dim=1).cpu()


def main() -> None:
    args = parse_args()
    device = get_device()
    model, checkpoint = load_checkpoint(args.models_dir / f"{args.model}_best.pth", device)
    classes, image_size = checkpoint["classes"], checkpoint["image_size"]
    calibration = checkpoint.get("calibration")
    splits = {r["source"]: r for r in read_splits()}
    print(f"Modelo: {args.model} ({image_size}px) | dispositivo: {device}")
    if calibration is None:
        print(f"⚠ Modelo sin calibrar: ejecuta 'python -m src.calibrate --model {args.model}'")

    for image_path in args.images:
        probs = predict_image(model, image_path, image_size, device)
        if calibration is None:
            confidence, pred = probs.max(dim=1)
            distribution = probs
        else:
            pred, confidence, distribution = calibrated_confidence(
                probs, torch.tensor(calibration["class_weights"]), calibration["temperature"])
        best, confidence, distribution = int(pred), float(confidence), distribution[0]
        record = find_split(image_path, splits)

        print(f"\n{image_path}")
        if record is None:
            print("  Etiqueta real: desconocida (la imagen no está en splits.csv)")
        else:
            verdict = "✅ acierto" if classes[best] == record["label"] else "❌ error"
            print(f"  Etiqueta real: {record['label']} {CLASS_NAMES[record['label']]} | "
                  f"conjunto: {record['split']} | {verdict}")
            if record["split"] == "train":
                print("  ⚠ El modelo ya vio esta imagen al entrenar: no sirve para medirlo")
        print(f"  Predicción: {classes[best]} {CLASS_NAMES[classes[best]]} ({confidence:.1%})")
        print(f"  Confianza: {LEVEL_LABELS[confidence_level(confidence)]}")
        for i in distribution.argsort(descending=True).tolist():
            bar = "█" * round(distribution[i].item() * 30)
            print(f"    {classes[i]} {CLASS_NAMES[classes[i]]:<27}{distribution[i]:>7.1%} {bar}")


if __name__ == "__main__":
    main()
