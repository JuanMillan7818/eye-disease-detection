"""Collect everything about the model the app uses into one folder, ready to show.

Confusion matrices use the screening policy (decide()), exactly like the app and the web metrics tab.
"""

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

from src.calibrate import decide
from src.evaluate import FIGURES_DIR, plot_confusion_matrix, split_suffix
from src.model import load_checkpoint
from src.model_report import OUTPUT as METRICS_JSON
from src.model_report import REPORTS_DIR, read_predictions
from src.train import plot_history

OUTPUT_DIR = REPORTS_DIR / "modelo_final"
ORIGIN_NAMES = {"original": "fotos originales (ODIR)", "externo": "fotos externas (otras cámaras)"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="v2s", help="model the app uses")
    parser.add_argument("--previous", default="v2s_original", help="model it replaced (for comparison)")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def policy_predictions(name: str, models_dir: Path, splits_file: Path):
    """(files, real labels, app decisions) on the test set, with that model's own calibration."""
    _, checkpoint = load_checkpoint(models_dir / f"{name}_best.pth", "cpu")
    classes = checkpoint["classes"]
    files, labels, probs = read_predictions(
        REPORTS_DIR / f"{name}_test{split_suffix(splits_file)}_predictions.csv", classes)
    preds, _, _, _ = decide(probs, checkpoint["calibration"], classes)
    return classes, np.array(files), labels, preds.numpy()


def main() -> None:
    args = parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    _, checkpoint = load_checkpoint(args.models_dir / f"{args.model}_best.pth", "cpu")
    splits_file = Path(checkpoint.get("splits", "splits/splits.csv"))
    with open(splits_file, newline="", encoding="utf-8") as f:
        origin_of = {r["file"]: r.get("origin", "original") for r in csv.DictReader(f)}

    classes, files, labels, preds = policy_predictions(args.model, args.models_dir, splits_file)
    plot_confusion_matrix(labels, preds, classes, f"Modelo final — test ({len(labels)} imágenes)",
                          out / "1_matriz_confusion_test.png")
    origins = np.array([origin_of[f] for f in files])
    for i, origin in enumerate(sorted(set(origins)), start=2):
        mask = origins == origin
        shown = sorted(set(labels[mask]) | set(preds[mask]))  # the external data lacks A, H and M
        remap = {c: i for i, c in enumerate(shown)}
        plot_confusion_matrix(np.array([remap[v] for v in labels[mask]]),
                              np.array([remap[v] for v in preds[mask]]), [classes[c] for c in shown],
                              f"Modelo final — test, {ORIGIN_NAMES.get(origin, origin)} ({mask.sum()})",
                              out / f"{i}_matriz_confusion_{origin}.png")

    _, _, old_labels, old_preds = policy_predictions(args.previous, args.models_dir, splits_file)
    plot_confusion_matrix(old_labels, old_preds, classes, f"Modelo anterior — mismo test ({len(old_labels)})",
                          out / "4_matriz_confusion_modelo_anterior.png")

    with open(args.models_dir / f"{args.model}_history.csv", newline="", encoding="utf-8") as f:
        history = [{k: (v if k == "stage" else float(v)) for k, v in row.items()} for row in csv.DictReader(f)]
    plot_history(history, "modelo final", out / "5_curvas_entrenamiento.png")

    copies = {  # figures made by evaluate / calibrate under their technical names
        FIGURES_DIR / f"{args.model}_test{split_suffix(splits_file)}_errors.png": "6_ejemplos_de_errores.png",
        FIGURES_DIR / "v2s_ext_calibration.png": "7_calibracion.png",
        METRICS_JSON: "metricas.json",
    }
    for source, name in copies.items():
        if source.exists():
            shutil.copy2(source, out / name)

    metrics = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    correct = int((labels == preds).sum())
    print(f"{args.model}: {correct}/{len(labels)} aciertos en test ({metrics['kpis']['accuracy']:.1%})")
    print(f"Carpeta lista: {out}")


if __name__ == "__main__":
    main()
