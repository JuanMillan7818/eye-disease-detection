"""Phase 6: does the model work equally well on every image source? (source shortcut check)"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import f1_score

from src.dataset import read_splits
from src.evaluate import REPORTS_DIR
from src.gradcam import source_type


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="v2s")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = REPORTS_DIR / f"{args.model}_{args.split}_predictions.csv"
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    sources = {r["file"]: source_type(r) for r in read_splits()}

    by_source = defaultdict(list)
    for row in rows:
        by_source[sources[row["file"]]].append(row)

    print(f"Modelo {args.model} | {args.split} | {len(rows)} imágenes\n")
    for source, items in sorted(by_source.items()):
        labels = [r["label"] for r in items]
        preds = [r["pred"] for r in items]
        acc = sum(a == b for a, b in zip(labels, preds)) / len(items)
        print(f"Origen '{source}': {len(items)} imágenes | acc {acc:.1%}")
        for c in sorted(set(labels)):
            idx = [i for i, label in enumerate(labels) if label == c]
            hits = sum(preds[i] == c for i in idx)
            to_normal = sum(preds[i] == "N" for i in idx) if c != "N" else 0
            print(f"   {c}: {len(idx):>4} imágenes | recall {hits / len(idx):>5.0%} | llamadas Normal: {to_normal}")
        if len(set(labels)) > 1:
            print(f"   F1 macro de este origen: {f1_score(labels, preds, average='macro'):.3f}")
        print()


if __name__ == "__main__":
    main()
