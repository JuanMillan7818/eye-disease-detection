"""Model KPIs and confusion matrix for the web page, computed from the saved TEST predictions.

The numbers follow what the app shows: decisions with the screening policy (decide()),
not the raw argmax. Output: web/metrics.json (small, versioned in git, no data needed).

Run (after 'evaluate --split test' and 'calibrate'):
    uv run python -m src.model_report --model v2s
"""

import argparse
import csv
import json
import time
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.calibrate import (LOW_CONFIDENCE, NORMAL_CLASS, decide, green_cut,
                           expected_calibration_error)
from src.dataset import read_splits
from src.evaluate import split_suffix
from src.explorer import CLASS_NAMES
from src.model import MODEL_SPECS, count_parameters, load_checkpoint

REPORTS_DIR = Path("reports")
OUTPUT = Path("web/metrics.json")
ARCHITECTURES = {"b0": "EfficientNet-B0", "v2s": "EfficientNetV2-S"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="v2s", help="checkpoint name inside models/")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def read_predictions(path: Path, classes: list[str]) -> tuple[list[str], np.ndarray, torch.Tensor]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    labels = np.array([classes.index(r["label"]) for r in rows])
    probs = torch.tensor([[float(r[f"p_{c}"]) for c in classes] for r in rows])
    return [r["file"] for r in rows], labels, probs


def by_origin(origins: np.ndarray, labels: np.ndarray, preds: np.ndarray, green: np.ndarray,
              classes: list[str]) -> dict:
    """Same KPIs per image source: a mixed test set can hide a weak one."""
    report = {}
    for origin in sorted(set(origins)):
        m = origins == origin
        present = sorted(set(labels[m]))  # the external data has only 4 of the 7 classes
        _, recall, f1, _ = precision_recall_fscore_support(
            labels[m], preds[m], labels=present, zero_division=0)
        report[origin] = {
            "n_images": int(m.sum()),
            "accuracy": float((labels[m] == preds[m]).mean()),
            "f1_macro": float(f1.mean()),
            "green_share": float(green[m].mean()),
            "green_accuracy": float((labels[m & green] == preds[m & green]).mean()),
            "recall": {classes[c]: float(r) for c, r in zip(present, recall)},
        }
    return report


@torch.no_grad()
def cpu_time_per_image(model: torch.nn.Module, image_size: int, repeats: int = 5) -> float:
    """Milliseconds per image on this computer's CPU (one image at a time, like the API)."""
    image = torch.randn(1, 3, image_size, image_size)
    model(image)  # warm-up
    start = time.perf_counter()
    for _ in range(repeats):
        model(image)
    return (time.perf_counter() - start) / repeats * 1000


def build_report(name: str, models_dir: Path) -> dict:
    path = models_dir / f"{name}_best.pth"
    model, checkpoint = load_checkpoint(path, "cpu")
    classes, calibration = checkpoint["classes"], checkpoint["calibration"]
    splits_file = Path(checkpoint.get("splits", "splits/splits.csv"))
    files, labels, probs = read_predictions(
        REPORTS_DIR / f"{name}_test{split_suffix(splits_file)}_predictions.csv", classes)

    preds, confidence, levels, _ = decide(probs, calibration, classes)
    preds, confidence, levels = preds.numpy(), confidence.numpy(), np.array(levels)
    correct = preds == labels
    normal = classes.index(NORMAL_CLASS)
    sick, said_sick = labels != normal, preds != normal
    green = levels == "high"

    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=range(len(classes)), zero_division=0)
    records = read_splits(splits_file)
    splits = Counter(r["split"] for r in records)
    origin_of = {r["file"]: r.get("origin", "original") for r in records}
    origins = np.array([origin_of[f] for f in files])
    total, _ = count_parameters(model)

    return {
        "model": name,
        "architecture": ARCHITECTURES.get(checkpoint["model_name"], checkpoint["model_name"]),
        "image_size": checkpoint["image_size"],
        "generated": date.today().isoformat(),
        "split": "test",
        "n_images": int(len(labels)),
        "kpis": {
            "accuracy": float(correct.mean()),
            "f1_macro": float(f1.mean()),
            "sensitivity": float((said_sick & sick).sum() / sick.sum()),
            "specificity": float((~said_sick & ~sick).sum() / (~sick).sum()),
            "missed_sick": int((sick & ~said_sick).sum()),
            "false_alarms": int((~sick & said_sick).sum()),
            "green_share": float(green.mean()),
            "green_accuracy": float(correct[green].mean()),
            "ece": float(expected_calibration_error(confidence, correct)),
            "parameters": int(total),
            "file_mb": round(path.stat().st_size / 1e6, 1),
            "cpu_ms_per_image": round(cpu_time_per_image(model, checkpoint["image_size"])),
            "best_epoch": checkpoint["metrics"].get("epoch"),
        },
        "data": {split: splits[split] for split in ("train", "val", "test")},
        "policy": {"green": green_cut(calibration), "red": LOW_CONFIDENCE,
                   "normal_factor": calibration.get("normal_factor"),
                   "temperature": calibration["temperature"]},
        "levels": {lv: {"share": float((levels == lv).mean()),
                        "accuracy": float(correct[levels == lv].mean()) if (levels == lv).any() else None}
                   for lv in ("high", "medium", "normal", "low")},
        "classes": [{"code": c, "name": CLASS_NAMES[c], "precision": float(p), "recall": float(r),
                     "f1": float(f), "support": int(s)}
                    for c, p, r, f, s in zip(classes, precision, recall, f1, support)],
        "confusion": confusion_matrix(labels, preds, labels=range(len(classes))).tolist(),
        "origins": by_origin(origins, labels, preds, green, classes),
    }


def main() -> None:
    args = parse_args()
    report = build_report(args.model, args.models_dir)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    k = report["kpis"]
    print(f"{args.model} | test {report['n_images']} imágenes | acc {k['accuracy']:.1%} | "
          f"F1 macro {k['f1_macro']:.3f} | sensibilidad {k['sensitivity']:.1%} | "
          f"especificidad {k['specificity']:.1%} | verde {k['green_share']:.0%} acierta {k['green_accuracy']:.1%}")
    for origin, o in report["origins"].items():
        print(f"  origen {origin:<9} n={o['n_images']:4} | acc {o['accuracy']:.1%} | F1 macro {o['f1_macro']:.3f} | "
              f"verde {o['green_share']:.0%} acierta {o['green_accuracy']:.1%}")
    print(f"Guardado en {args.output}")


if __name__ == "__main__":
    main()
