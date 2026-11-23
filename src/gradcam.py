"""Phase 6: Grad-CAM heatmaps — where in the retina does the model look?"""

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image, ImageFilter
from torch import nn

from src.calibrate import LEVEL_LABELS, NORMAL_CLASS, confidence_level, decide
from src.dataset import build_transforms, read_splits
from src.explorer import CLASS_NAMES, DATA_DIR, SEED
from src.model import load_checkpoint
from src.predict import find_split
from src.prepare_data import AUGMENTED_PATTERN, ODIR_PATTERN
from src.preprocessing import preprocess_fundus
from src.train import get_device

OUTPUT_DIR = Path("reports/figures/gradcam")
LEVEL_COLORS = {"high": "green", "medium": "darkgoldenrod", "normal": "darkgoldenrod",
                "low": "red"}  # figures cannot draw emoji


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="*", type=Path, help="raw fundus images (e.g. data/D/123_left.jpg)")
    parser.add_argument("--model", default="v2s", help="checkpoint name inside models/")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--grid", action="store_true", help="grid of random test images per class")
    parser.add_argument("--per-class", type=int, default=4)
    parser.add_argument("--errors", nargs=2, metavar=("REAL", "PRED"),
                        help="grid of test mistakes, e.g. --errors D N")
    parser.add_argument("--border-check", type=int, metavar="N",
                        help="measure heatmap share on the retina border for N test images per class")
    parser.add_argument("--mean-maps", type=int, metavar="N",
                        help="average heatmap per class and source over up to N test images each")
    parser.add_argument("--sanity", action="store_true",
                        help="feed blank images and blurred normal retinas to probe shortcuts")
    return parser.parse_args()


class GradCAM:
    """Grad-CAM on one convolutional layer, using PyTorch hooks."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        target_layer.register_forward_hook(self._save_activations)

    def _save_activations(self, module: nn.Module, inputs, output: torch.Tensor) -> None:
        self.activations = output.detach()
        if output.requires_grad:  # under torch.no_grad() there are no gradients to catch
            output.register_hook(self._save_gradients)

    def _save_gradients(self, grad: torch.Tensor) -> None:
        self.gradients = grad.detach()

    def __call__(self, image: torch.Tensor, class_index: int | None = None) -> tuple[np.ndarray, torch.Tensor]:
        """Return (heatmap in [0, 1] with the image's size, softmax probabilities)."""
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self.model(image)
            if class_index is None:
                class_index = int(logits.argmax(dim=1))
            logits[0, class_index].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)        # (1, C, 1, 1): importance per channel
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))  # (1, 1, h, w)
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam / cam.max() if cam.max() > 0 else cam
        return cam.cpu().numpy(), logits.softmax(dim=1).detach().cpu()


def overlay_heatmap(image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend the heatmap (jet colours) over the RGB image."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
    heat = colormaps["jet"](cam)[..., :3]
    return np.clip((1 - alpha) * rgb + alpha * heat, 0, 1)  # clip float rounding (1.0000001)


class Explainer:
    """Load a checkpoint once and explain any number of raw images."""

    def __init__(self, model_name: str, models_dir: Path):
        self.device = get_device()
        self.model, checkpoint = load_checkpoint(models_dir / f"{model_name}_best.pth", self.device)
        self.classes = checkpoint["classes"]
        self.image_size = checkpoint["image_size"]
        self.calibration = checkpoint.get("calibration")
        self.transform = build_transforms(self.image_size, train=False)
        self.gradcam = GradCAM(self.model, self.model.features[-1])

    def explain(self, image_path: Path) -> dict:
        with Image.open(image_path) as img:
            clean = preprocess_fundus(img)
        seen = clean.resize((self.image_size, self.image_size))  # what the model actually sees
        tensor = self.transform(clean).unsqueeze(0).to(self.device)
        cam, probs = self.gradcam(tensor)

        if self.calibration is None:
            confidence, pred = probs.max(dim=1)
            level = confidence_level(float(confidence), self.classes[int(pred)] == NORMAL_CLASS)
        else:
            pred, confidence, levels, _ = decide(probs, self.calibration, self.classes)
            level = levels[0]
            if int(pred) != int(probs.argmax()):  # the policy changed the answer: explain THAT class
                cam, _ = self.gradcam(tensor, int(pred))
        return {"image": seen, "cam": cam, "pred": self.classes[int(pred)],
                "confidence": float(confidence), "level": level}


def title_for(result: dict, record: dict | None) -> str:
    pred = result["pred"]
    text = f"pred {pred} {CLASS_NAMES[pred]} ({result['confidence']:.0%})"
    if record is not None:
        mark = "✓" if record["label"] == pred else "✗"
        text = f"real {record['label']} | {text} {mark}"
    return text


def save_single(result: dict, record: dict | None, path: Path) -> None:
    fig, (ax_img, ax_cam) = plt.subplots(1, 2, figsize=(9, 4.8))
    ax_img.imshow(result["image"])
    ax_img.set_title("Imagen que ve el modelo")
    ax_cam.imshow(overlay_heatmap(result["image"], result["cam"]))
    ax_cam.set_title("Grad-CAM (rojo = más importante)")
    for ax in (ax_img, ax_cam):
        ax.axis("off")
    level_text = LEVEL_LABELS[result["level"]].split(" ", 1)[1]  # drop the emoji
    fig.suptitle(f"{title_for(result, record)}\nConfianza {level_text}", fontsize=10,
                 color=LEVEL_COLORS[result["level"]])
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def save_grid(items: list[tuple[dict, dict]], title: str, path: Path, cols: int = 4) -> None:
    """Each cell: original on top, heatmap below."""
    rows = max(1, -(-len(items) // cols))
    fig, axes = plt.subplots(rows * 2, cols, figsize=(cols * 2.6, rows * 5.4), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for k, (result, record) in enumerate(items):
        r, c = divmod(k, cols)
        axes[2 * r, c].imshow(result["image"])
        axes[2 * r, c].set_title(title_for(result, record), fontsize=6.5,
                                 color=LEVEL_COLORS[result["level"]])
        axes[2 * r + 1, c].imshow(overlay_heatmap(result["image"], result["cam"]))
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def border_share(cam: np.ndarray, inner: float = 0.80, outer: float = 0.95) -> float:
    """Share of the heatmap that falls on the outer ring of the retina (inner..outer radius)."""
    size = cam.shape[0]
    yy, xx = np.mgrid[:size, :size]
    radius = np.hypot(yy - (size - 1) / 2, xx - (size - 1) / 2) / (size / 2)
    inside = radius <= outer
    ring = inside & (radius > inner)
    return float(cam[ring].sum() / max(cam[inside].sum(), 1e-8))


def source_type(record: dict) -> str:
    """Which collection an image comes from, guessed from its file name."""
    name = Path(record["source"]).stem
    if ODIR_PATTERN.match(name):
        return "odir"
    if AUGMENTED_PATTERN.match(name):
        return "aumentada"
    return "otra"


def save_mean_maps(explainer: "Explainer", n: int, rng: random.Random, path: Path) -> None:
    """Average heatmap per (class, source). A sharp spot in a fixed place = suspicious shortcut."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in test_records():
        groups.setdefault((r["label"], source_type(r)), []).append(r)
    panels = []
    for (label, source), records in sorted(groups.items()):
        if len(records) < 5:
            continue
        records = rng.sample(records, min(n, len(records)))
        results = [explainer.explain(DATA_DIR / r["source"]) for r in records]
        hits = np.mean([res["pred"] == label for res in results])
        panels.append((f"{label} · {source}\nn={len(records)} · acierto {hits:.0%}",
                       np.mean([res["cam"] for res in results], axis=0)))
        print(f"  {label} {source:<10} n={len(records):<3} acierto {hits:.0%}")

    cols = 5
    rows = -(-len(panels) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.7), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (title, cam) in zip(axes.flat, panels):
        ax.imshow(cam, cmap="jet", vmin=0, vmax=1)
        ax.set_title(title, fontsize=8)
    fig.suptitle("Grad-CAM promedio por clase y origen (test)")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


@torch.no_grad()
def predict_clean(explainer: "Explainer", image: Image.Image) -> str:
    """Prediction for an already preprocessed image (no Grad-CAM needed)."""
    tensor = explainer.transform(image).unsqueeze(0).to(explainer.device)
    return explainer.classes[int(explainer.model(tensor).argmax(dim=1))]


def sanity_check(explainer: "Explainer", rng: random.Random, n: int = 30) -> None:
    """What does the model answer for images with NO retina details at all?"""
    print("\nImágenes sin retina:")
    for name, color in [("negra", (0, 0, 0)), ("gris", (128, 128, 128)), ("naranja plana", (180, 90, 40))]:
        print(f"  {name:<14} -> {predict_clean(explainer, Image.new('RGB', (512, 512), color))}")

    print(f"\nRetinas normales (N de test) cada vez más borrosas, n={n}:")
    records = rng.sample(test_records("N"), n)
    images = []
    for r in records:
        with Image.open(DATA_DIR / r["source"]) as img:
            images.append(preprocess_fundus(img))
    for radius in (0, 2, 4, 8, 16):
        preds = [predict_clean(explainer, img.filter(ImageFilter.GaussianBlur(radius))) for img in images]
        counts = {c: preds.count(c) for c in explainer.classes if preds.count(c)}
        print(f"  desenfoque {radius:>2}px -> {counts}")


def test_records(label: str | None = None) -> list[dict]:
    return [r for r in read_splits() if r["split"] == "test" and (label is None or r["label"] == label)]


def main() -> None:
    args = parse_args()
    explainer = Explainer(args.model, args.models_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    splits = {r["source"]: r for r in read_splits()}
    rng = random.Random(SEED)
    print(f"Modelo: {args.model} ({explainer.image_size}px) | dispositivo: {explainer.device}")

    for image_path in args.images:
        result = explainer.explain(image_path)
        path = OUTPUT_DIR / f"{args.model}_{image_path.stem}.png"
        save_single(result, find_split(image_path, splits), path)
        print(f"{image_path}: {result['pred']} {result['confidence']:.1%} "
              f"{LEVEL_LABELS[result['level']]} -> {path}")

    if args.grid:
        for label in explainer.classes:
            records = rng.sample(test_records(label), args.per_class)
            items = [(explainer.explain(DATA_DIR / r["source"]), r) for r in records]
            path = OUTPUT_DIR / f"{args.model}_grid_{label}.png"
            save_grid(items, f"Grad-CAM — clase real {label} {CLASS_NAMES[label]} (test)", path)
            print(f"Cuadrícula {label} -> {path}")

    if args.errors:
        real, pred = args.errors
        candidates = test_records(real)
        rng.shuffle(candidates)
        items = []
        for r in candidates:
            result = explainer.explain(DATA_DIR / r["source"])
            if result["pred"] == pred:
                items.append((result, r))
                if len(items) == 8:
                    break
        path = OUTPUT_DIR / f"{args.model}_errors_{real}_as_{pred}.png"
        save_grid(items, f"Grad-CAM — errores: real {real} predicho {pred} (test)", path)
        print(f"{len(items)} errores {real} -> {pred} -> {path}")

    if args.border_check:
        expected = 1 - (0.80 / 0.95) ** 2
        print(f"\nCalor en el borde de la retina (anillo 80-95 % del radio). "
              f"Si el calor fuera uniforme: {expected:.0%}")
        for label in explainer.classes:
            records = rng.sample(test_records(label), args.border_check)
            shares = [border_share(explainer.explain(DATA_DIR / r["source"])["cam"]) for r in records]
            print(f"  {label} {CLASS_NAMES[label]:<27} {np.mean(shares):>5.0%} "
                  f"(mediana {np.median(shares):.0%}, n={len(shares)})")

    if args.mean_maps:
        path = OUTPUT_DIR / f"{args.model}_mean_maps.png"
        print("\nMapas promedio por clase y origen:")
        save_mean_maps(explainer, args.mean_maps, rng, path)
        print(f"-> {path}")

    if args.sanity:
        sanity_check(explainer, rng)


if __name__ == "__main__":
    main()
