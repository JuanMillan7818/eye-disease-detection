"""Phase 5: confidence calibration and screening decision policy (traffic light)."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import f1_score

from src.dataset import create_dataloaders
from src.evaluate import FIGURES_DIR, predict
from src.model import load_checkpoint
from src.prepare_data import PROCESSED_DIR
from src.train import get_device

HIGH_CONFIDENCE = 0.88  # green: chosen on val (green hits >= 97 %)
LOW_CONFIDENCE = 0.80   # red below this: chosen on val (hits ~50 %)
NORMAL_FACTOR = 0.25    # demand 4x more evidence before saying "Normal" (fewer missed patients)
NORMAL_CLASS = "N"
LEVEL_LABELS = {
    "high": "🟢 alta",
    "medium": "🟡 media — confirmar con un especialista",
    "normal": "🟡 sin signos claros de enfermedad — control anual con oftalmólogo",
    "low": "🔴 baja — resultado incierto, consultar a un especialista",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="v2s", help="checkpoint name inside models/")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--image-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--normal-factor", type=float, default=NORMAL_FACTOR)
    return parser.parse_args()


def calibrated_confidence(probs: torch.Tensor, class_weights: torch.Tensor,
                          temperature: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep the model's decision, but report a confidence without the class-weight bias.

    Returns (predicted class, its calibrated confidence, full calibrated distribution).
    """
    preds = probs.argmax(dim=1)
    corrected = ((probs.clamp_min(1e-12) / class_weights).log() / temperature).softmax(dim=1)
    confidence = corrected.gather(1, preds[:, None]).squeeze(1)
    return preds, confidence, corrected


def confidence_level(confidence: float, is_normal: bool = False) -> str:
    """Traffic light. A 'Normal' answer is never green: missing a disease is the costly error."""
    if confidence < LOW_CONFIDENCE:
        return "low"
    if is_normal:
        return "normal"
    return "high" if confidence >= HIGH_CONFIDENCE else "medium"


def decide(probs: torch.Tensor, calibration: dict,
           classes: list[str]) -> tuple[torch.Tensor, torch.Tensor, list[str], torch.Tensor]:
    """Screening policy: (decision, calibrated confidence, traffic-light level, calibrated distribution)."""
    normal = classes.index(NORMAL_CLASS)
    scaled = probs.clone()
    scaled[:, normal] *= calibration.get("normal_factor", NORMAL_FACTOR)
    preds = scaled.argmax(dim=1)  # the factor only moves the decision...

    _, _, distribution = calibrated_confidence(  # ...the shown % comes from the real probabilities
        probs, torch.tensor(calibration["class_weights"]), calibration["temperature"])
    confidence = distribution.gather(1, preds[:, None]).squeeze(1)
    levels = [confidence_level(float(c), int(p) == normal) for c, p in zip(confidence, preds)]
    return preds, confidence, levels, distribution


def correctness_nll(confidence: torch.Tensor, correct: torch.Tensor) -> float:
    """How surprised we are by the hits/misses given the confidence (lower is better)."""
    c = confidence.clamp(1e-7, 1 - 1e-7)
    return float(-(correct * c.log() + (~correct) * (1 - c).log()).mean())


def fit_temperature(probs: torch.Tensor, labels: torch.Tensor, class_weights: torch.Tensor) -> float:
    """Try many temperatures and keep the one whose confidence best matches reality."""
    candidates = np.linspace(0.3, 3.0, 271)
    losses = []
    for t in candidates:
        preds, confidence, _ = calibrated_confidence(probs, class_weights, float(t))
        losses.append(correctness_nll(confidence, preds == labels))
    return float(candidates[int(np.argmin(losses))])


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """Average gap between 'confidence said' and 'accuracy got', weighted by images per bin."""
    edges = np.linspace(0, 1, n_bins + 1)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > low) & (confidence <= high)
        if in_bin.any():
            error += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return error


def plot_reliability(curves: dict[str, tuple[np.ndarray, np.ndarray]], title: str, path: Path,
                     n_bins: int = 10) -> None:
    """Reliability diagram: a perfectly calibrated model lies on the diagonal."""
    edges = np.linspace(0, 1, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "k:", label="calibración perfecta")
    for name, (confidence, correct) in curves.items():
        bins = np.clip(np.digitize(confidence, edges[1:-1]), 0, n_bins - 1)
        accuracy = [correct[bins == b].mean() if (bins == b).any() else np.nan for b in range(n_bins)]
        ax.plot(centers, accuracy, "o-", label=name)
    ax.set_xlabel("Confianza que muestra el modelo")
    ax.set_ylabel("Aciertos reales")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def print_policy_report(labels: np.ndarray, preds: np.ndarray, levels: list[str], classes: list[str]) -> None:
    """Screening view: missed patients, false alarms and how reliable each colour is."""
    normal = classes.index(NORMAL_CLASS)
    correct = preds == labels
    sick = labels != normal
    levels = np.array(levels)
    print(f"\nF1 macro {f1_score(labels, preds, average='macro'):.3f} | aciertos {correct.mean():.1%}")
    print(f"Sensibilidad (enfermos detectados): {(preds[sick] != normal).mean():.1%} | "
          f"especificidad (sanos = Normal): {(preds[~sick] == normal).mean():.1%}")
    print(f"Enfermos que se escapan (dicen Normal): {(sick & (preds == normal)).sum()} | "
          f"sanos con falsa alarma: {(~sick & (preds != normal)).sum()}")
    print(f"\n{'nivel':<8}{'% imágenes':>11}{'aciertos':>10}")
    for level in LEVEL_LABELS:
        mask = levels == level
        if mask.any():
            print(f"{level:<8}{mask.mean():>11.0%}{correct[mask].mean():>10.1%}")
    print("\nVerde por clase predicha (cuántas / aciertos):")
    for i, c in enumerate(classes):
        mask = (levels == "high") & (preds == i)
        hits = f"{correct[mask].mean():.0%}" if mask.any() else "-"
        print(f"  {c}: {int(mask.sum())}/{int((preds == i).sum())} ({hits})")


def main() -> None:
    args = parse_args()
    device = get_device()
    path = args.models_dir / f"{args.model}_best.pth"
    model, checkpoint = load_checkpoint(path, device)
    loaders, _, class_weights = create_dataloaders(
        image_size=checkpoint["image_size"], batch_size=args.batch_size,
        num_workers=args.workers, image_dir=args.image_dir,
    )
    if checkpoint.get("loss", "torch") != "torch":
        class_weights = torch.ones_like(class_weights)  # the fixed loss leaves no class-weight bias to undo

    print(f"Calibrando {args.model} con val ({len(loaders['val'].dataset)} imágenes)...")
    labels, probs, _ = predict(model, loaders["val"], device)
    labels, probs = torch.from_numpy(labels), torch.from_numpy(probs)

    raw_conf, raw_preds = probs.max(dim=1)
    temperature = fit_temperature(probs, labels, class_weights)
    preds, confidence, _ = calibrated_confidence(probs, class_weights, temperature)
    assert torch.equal(preds, raw_preds), "calibration must not change the predictions"

    correct = (preds == labels).numpy()
    print(f"Temperatura: {temperature:.2f} | aciertos val: {correct.mean():.1%}")
    print(f"Confianza media: {raw_conf.mean():.1%} -> {confidence.mean():.1%}")
    print(f"ECE: {expected_calibration_error(raw_conf.numpy(), correct):.3f} -> "
          f"{expected_calibration_error(confidence.numpy(), correct):.3f}")

    calibration = {
        "temperature": temperature,
        "class_weights": class_weights.tolist(),
        "normal_factor": args.normal_factor,
        "fitted_on": "val",
    }
    final_preds, _, levels, _ = decide(probs, calibration, checkpoint["classes"])
    print(f"\nPolítica de tamizaje (factor Normal {args.normal_factor}, "
          f"verde >= {HIGH_CONFIDENCE:.0%}, rojo < {LOW_CONFIDENCE:.0%}):")
    print_policy_report(labels.numpy(), final_preds.numpy(), levels, checkpoint["classes"])

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_reliability({"sin calibrar": (raw_conf.numpy(), correct),
                      "calibrado": (confidence.numpy(), correct)},
                     f"Calibración — {args.model} (val)", FIGURES_DIR / f"{args.model}_calibration.png")

    checkpoint["calibration"] = calibration
    torch.save(checkpoint, path)
    print(f"\nCalibración guardada en {path}")


if __name__ == "__main__":
    main()
