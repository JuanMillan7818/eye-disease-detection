"""Phase 5: evaluation — metrics per class, confusion matrix, TTA and error analysis."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.dataset import create_dataloaders
from src.explorer import CLASS_NAMES
from src.model import load_checkpoint
from src.prepare_data import PROCESSED_DIR
from src.train import get_device

REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["b0", "v2s"])
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--image-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()

@torch.no_grad()
def predict(model: torch.nn.Module, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (labels, probabilities, probabilities of the mirrored images)."""
    all_labels, all_probs, all_flipped = [], [], []
    for images, labels in loader:
        images = images.to(device)
        all_probs.append(model(images).softmax(dim=1).cpu())
        all_flipped.append(model(torch.flip(images, dims=[3])).softmax(dim=1).cpu())
        all_labels.append(labels)
    return (torch.cat(all_labels).numpy(),
            torch.cat(all_probs).numpy(),
            torch.cat(all_flipped).numpy())


def summarize(labels: np.ndarray, probs: np.ndarray) -> dict:
    preds = probs.argmax(axis=1)
    return {
        "acc": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }

def plot_confusion_matrix(labels: np.ndarray, preds: np.ndarray, classes: list[str],
                          title: str, path: Path) -> None:
    """Rows = real class, columns = prediction. Colour = % of the real class (recall)."""
    matrix = confusion_matrix(labels, preds, labels=range(len(classes)))
    row_share = matrix / matrix.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(row_share, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            color = "white" if row_share[i, j] > 0.5 else "black"
            ax.text(j, i, f"{matrix[i, j]}\n{row_share[i, j]:.0%}",
                    ha="center", va="center", fontsize=8, color=color)
    names = [f"{c}\n{CLASS_NAMES.get(c, c)}" for c in classes]
    ax.set_xticks(range(len(classes)), names, fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), names, fontsize=7)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Clase real")
    ax.set_title(title)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)

def save_predictions(records: list[dict], labels: np.ndarray, probs: np.ndarray,
                     classes: list[str], path: Path) -> None:
    """One row per image: real label, prediction, confidence and all probabilities."""
    preds = probs.argmax(axis=1)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "label", "pred", "confidence", *[f"p_{c}" for c in classes]])
        for record, label, pred, row in zip(records, labels, preds, probs):
            writer.writerow([record["file"], classes[label], classes[pred], f"{row[pred]:.4f}",
                             *[f"{p:.4f}" for p in row]])


def plot_top_errors(records: list[dict], labels: np.ndarray, probs: np.ndarray, classes: list[str],
                    image_dir: Path, title: str, path: Path, n: int = 12) -> None:
    """Show the mistakes the model was MOST confident about."""
    preds = probs.argmax(axis=1)
    wrong = np.flatnonzero(preds != labels)
    worst = wrong[np.argsort(-probs[wrong, preds[wrong]])][:n]

    cols = 4
    rows = max(1, math.ceil(len(worst) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.3), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, worst):
        with Image.open(image_dir / records[i]["file"]) as img:
            ax.imshow(img.convert("RGB"))
        ax.set_title(f"real {classes[labels[i]]} -> pred {classes[preds[i]]} "
                     f"({probs[i, preds[i]]:.0%})\n{records[i]['file']}", fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)

def evaluate_model(name: str, args: argparse.Namespace, device: torch.device) -> list[dict]:
    model, checkpoint = load_checkpoint(args.models_dir / f"{name}_best.pth", device)
    classes = checkpoint["classes"]
    loaders, data_classes, _ = create_dataloaders(
        image_size=checkpoint["image_size"], batch_size=args.batch_size,
        num_workers=args.workers, image_dir=args.image_dir,
    )
    assert data_classes == classes, f"Class order mismatch: {data_classes} vs {classes}"
    loader = loaders[args.split]
    records = loader.dataset.records

    print(f"\n=== {name} | {args.split}: {len(records)} imágenes | "
          f"mejor época de entrenamiento: {checkpoint['metrics'].get('epoch')} ===")
    labels, probs, probs_flipped = predict(model, loader, device)
    probs_tta = (probs + probs_flipped) / 2

    preds = probs.argmax(axis=1)
    names = [f"{c} {CLASS_NAMES.get(c, c)}" for c in classes]
    print(classification_report(labels, preds, target_names=names, digits=3))

    tag = f"{name}_{args.split}"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    save_predictions(records, labels, probs, classes, REPORTS_DIR / f"{tag}_predictions.csv")
    plot_confusion_matrix(labels, preds, classes, f"Matriz de confusión — {name} ({args.split})",
                          FIGURES_DIR / f"{tag}_confusion.png")
    plot_top_errors(records, labels, probs, classes, args.image_dir,
                    f"Errores más confiados — {name} ({args.split})", FIGURES_DIR / f"{tag}_errors.png")

    per_class_f1 = f1_score(labels, preds, average=None)
    return [
        {"modelo": name, "variante": "sin TTA", **summarize(labels, probs),
         **{f"f1_{c}": f for c, f in zip(classes, per_class_f1)}},
        {"modelo": name, "variante": "con TTA", **summarize(labels, probs_tta)},
    ]


def main() -> None:
    args = parse_args()
    device = get_device()
    rows = []
    for name in args.models:
        rows.extend(evaluate_model(name, args, device))

    print(f"\nComparación en {args.split}:")
    print(f"{'modelo':<7}{'variante':<10}{'acc':>8}{'F1 macro':>10}")
    for row in rows:
        print(f"{row['modelo']:<7}{row['variante']:<10}{row['acc']:>8.3f}{row['f1_macro']:>10.3f}")

    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with open(REPORTS_DIR / f"{args.split}_comparison.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nReportes en {REPORTS_DIR}/ y figuras en {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
