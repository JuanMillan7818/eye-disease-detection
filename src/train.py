"""Phase 4: two-stage fine-tuning with early stopping on validation macro F1."""

import argparse
import csv
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.dataset import create_dataloaders
from src.explorer import SEED
from src.model import MODEL_SPECS, create_model, save_checkpoint, set_backbone_trainable
from src.prepare_data import PROCESSED_DIR

FIGURES_DIR = Path("reports/figures")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_SPECS, default="b0")
    parser.add_argument("--head-epochs", type=int, default=3, help="stage 1: only the new head")
    parser.add_argument("--epochs", type=int, default=25, help="stage 2: whole network")
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="use N images per split (smoke test)")
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--image-dir", type=Path, default=PROCESSED_DIR,
                        help="folder with the preprocessed images")
    return parser.parse_args()

def set_seed(seed: int) -> None:
    """Make runs repeatable (same initial head, same shuffling)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_epoch(model: nn.Module, loader, criterion: nn.Module, device: torch.device,
              optimizer: torch.optim.Optimizer | None = None,
              scaler: torch.amp.GradScaler | None = None) -> dict:
    """One pass over the loader. Trains if an optimizer is given, otherwise only evaluates."""
    training = optimizer is not None
    model.train(training)
    use_amp = device.type == "cuda"
    total_loss, all_preds, all_labels = 0.0, [], []

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * labels.size(0)
            all_preds.append(logits.argmax(dim=1).cpu())
            all_labels.append(labels.cpu())

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    return {
        "loss": total_loss / len(labels),
        "acc": float((preds == labels).mean()),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }

def save_history(history: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def plot_history(history: list[dict], model_name: str, path: Path) -> None:
    """Loss and macro F1 per epoch, train vs validation."""
    epochs = [h["epoch"] for h in history]
    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(12, 4))
    for split, style in (("train", "-"), ("val", "--")):
        ax_loss.plot(epochs, [h[f"{split}_loss"] for h in history], style, label=split)
        ax_f1.plot(epochs, [h[f"{split}_f1_macro"] for h in history], style, label=split)
    ax_loss.set_title(f"Pérdida (loss) — {model_name}")
    ax_f1.set_title(f"F1 macro — {model_name}")
    for ax in (ax_loss, ax_f1):
        ax.set_xlabel("Época")
        ax.legend()
        ax.grid(alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)

def train(args: argparse.Namespace) -> None:
    set_seed(SEED)
    device = get_device()
    image_size = MODEL_SPECS[args.model].image_size
    print(f"Dispositivo: {device} | modelo: {args.model} | imagen: {image_size}px")

    loaders, classes, class_weights = create_dataloaders(
        image_size=image_size, batch_size=args.batch_size,
        num_workers=args.workers, limit=args.limit, image_dir=args.image_dir,
    )
    print(f"Train: {len(loaders['train'].dataset)} | Val: {len(loaders['val'].dataset)} imágenes")

    model = create_model(args.model, num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device),
                                    label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    checkpoint_path = args.output_dir / f"{args.model}_best.pth"

    stages = [
        ("1-cabeza", args.head_epochs, args.head_lr, False),
        ("2-completo", args.epochs, args.lr, True),
    ]
    history, best_f1, epoch = [], -1.0, 0

    for stage, n_epochs, lr, train_backbone in stages:
        set_backbone_trainable(model, train_backbone)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=max(n_epochs, 1))
        patience_left = args.patience
        print(f"\n--- Etapa {stage}: {n_epochs} épocas, lr={lr}, "
              f"{sum(p.numel() for p in params):,} parámetros entrenables ---")

        for _ in range(n_epochs):
            epoch += 1
            start = time.perf_counter()
            current_lr = optimizer.param_groups[0]["lr"]
            train_m = run_epoch(model, loaders["train"], criterion, device, optimizer, scaler)
            val_m = run_epoch(model, loaders["val"], criterion, device)
            scheduler.step()

            history.append({
                "epoch": epoch, "stage": stage, "lr": current_lr,
                **{f"train_{k}": v for k, v in train_m.items()},
                **{f"val_{k}": v for k, v in val_m.items()},
            })
            print(f"Época {epoch:>2} [{stage}] "
                  f"train loss {train_m['loss']:.3f} f1 {train_m['f1_macro']:.3f} | "
                  f"val loss {val_m['loss']:.3f} acc {val_m['acc']:.3f} f1 {val_m['f1_macro']:.3f} | "
                  f"{time.perf_counter() - start:.0f}s")

            if val_m["f1_macro"] > best_f1:
                best_f1 = val_m["f1_macro"]
                patience_left = args.patience
                save_checkpoint(model, checkpoint_path, args.model, classes, metrics={
                    "epoch": epoch, "val_f1_macro": val_m["f1_macro"], "val_acc": val_m["acc"],
                })
                print(f"  -> mejor modelo guardado (val F1 {best_f1:.3f})")
            else:
                patience_left -= 1
                if patience_left == 0:
                    print(f"  -> early stopping: {args.patience} épocas sin mejorar")
                    break

    save_history(history, args.output_dir / f"{args.model}_history.csv")
    plot_history(history, args.model, FIGURES_DIR / f"{args.model}_training_curves.png")
    print(f"\nMejor val F1 macro: {best_f1:.3f} | checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    train(parse_args())
