"""Add an external dataset (other cameras / hospitals) to a NEW split file.

The original split file is never modified: every original image keeps its split, so the
current model stays comparable and can be restored at any time. External images that are
already in the original dataset (same picture, even resized) are dropped.

Run: uv run python -m src.prepare_external
"""

import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from src.explorer import DATA_DIR, SEED
from src.prepare_data import (ODIR_PATTERN, PROCESSED_DIR, SPLITS_FILE, IMAGE_SIZE, check_no_group_leak,
                              difference_hash, merge_near_duplicates, preprocess_all)

EXTERNAL_DIR = DATA_DIR / "dataseet externo"
EXTERNAL_SPLITS_FILE = Path("splits/splits_ext.csv")
FOLDER_LABELS = {"cataract": "C", "diabetic_retinopathy": "D", "glaucoma": "G", "normal": "N"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
DUPLICATE_DISTANCE = 6  # same threshold as merge_near_duplicates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-dir", type=Path, default=EXTERNAL_DIR,
                        help="one sub-folder per class (names in FOLDER_LABELS), inside data/")
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--splits-in", type=Path, default=SPLITS_FILE)
    parser.add_argument("--splits-out", type=Path, default=EXTERNAL_SPLITS_FILE)
    parser.add_argument("--test-folds", type=int, default=4, help="1/4 of the external images -> test")
    parser.add_argument("--val-folds", type=int, default=5, help="1/5 of the rest -> val")
    return parser.parse_args()


def external_group(path: Path) -> str:
    """ODIR files (.jpg) share patient ids with the original dataset; EyePACS (.jpeg) does not."""
    if match := ODIR_PATTERN.match(path.stem):
        prefix = "eyepacs" if path.suffix.lower() == ".jpeg" else "odir"
        return f"{prefix}_{match.group(1)}"
    return f"ext_{path.parent.name}_{path.stem}"


def build_external_records(external_dir: Path) -> list[dict]:
    """One record per unique external image; drop images that appear under two labels."""
    by_hash = defaultdict(list)
    for folder, label in FOLDER_LABELS.items():
        for path in sorted((external_dir / folder).iterdir()):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                by_hash[hashlib.md5(path.read_bytes()).hexdigest()].append((label, path))

    records, conflicts, copies = [], 0, 0
    for items in by_hash.values():
        if len({label for label, _ in items}) > 1:
            conflicts += len(items)
            continue
        label, path = items[0]
        copies += len(items) - 1
        records.append({
            "file": f"ext/{label}/{path.stem}.jpg",
            "source": path.relative_to(DATA_DIR).as_posix(),
            "label": label,
            "group": external_group(path),
            "patient": external_group(path),  # kept apart: merging near-duplicates renames "group"
            "origin": "externo",
        })
    print(f"Externas: {len(records)} únicas | {conflicts} con etiquetas contradictorias y "
          f"{copies} copias exactas descartadas")
    return sorted(records, key=lambda r: r["file"])


def drop_known_images(external: list[dict], original: list[dict], image_dir: Path) -> list[dict]:
    """Remove external images that already exist in the original dataset (perceptual hash)."""
    ours = np.array([difference_hash(image_dir / r["file"]) for r in original], dtype=np.float32)
    kept, known, other_label = [], 0, 0
    for r in external:
        bits = difference_hash(image_dir / r["file"]).astype(np.float32)
        distances = bits @ (1 - ours).T + (1 - bits) @ ours.T
        nearest = int(distances.argmin())
        if distances[nearest] <= DUPLICATE_DISTANCE:
            known += 1
            other_label += original[nearest]["label"] != r["label"]
        else:
            kept.append(r)
    print(f"Ya estaban en el dataset original: {known} (con otra etiqueta allí: {other_label}) -> descartadas")
    return kept


def split_external(records: list[dict], original: list[dict], test_folds: int, val_folds: int) -> None:
    """ODIR patients already in the original split keep that split; the rest is split by group."""
    patient_split = {}
    for r in original:
        if match := ODIR_PATTERN.match(Path(r["source"]).stem):
            patient_split[f"odir_{match.group(1)}"] = r["split"]

    by_group = defaultdict(list)
    for r in records:
        by_group[r["group"]].append(r)

    forced, free = 0, []
    for members in by_group.values():
        known = {patient_split[r["patient"]] for r in members if r["patient"] in patient_split}
        # a merged group could hold two known patients; the strictest split wins (no leak into train)
        split = next((s for s in ("test", "val", "train") if s in known), None)
        if split:
            forced += len(members)
            for r in members:
                r["split"] = split
        else:
            free.extend(members)
    print(f"Pacientes ODIR que ya estaban en el original (heredan su parte): {forced} imágenes")

    labels = np.array([r["label"] for r in free])
    groups = np.array([r["group"] for r in free])
    index = np.arange(len(free))
    rest, test = next(StratifiedGroupKFold(test_folds, shuffle=True, random_state=SEED)
                      .split(index, labels, groups))
    train, val = next(StratifiedGroupKFold(val_folds, shuffle=True, random_state=SEED)
                      .split(rest, labels[rest], groups[rest]))
    for positions, split in ((test, "test"), (rest[val], "val"), (rest[train], "train")):
        for i in positions:
            free[i]["split"] = split


def print_summary(records: list[dict]) -> None:
    labels = sorted({r["label"] for r in records})
    counts = Counter((r["origin"], r["split"], r["label"]) for r in records)
    print(f"\n{'origen':<10}{'parte':<7}" + "".join(f"{c:>6}" for c in labels) + f"{'total':>8}")
    for origin in ("original", "externo"):
        for split in ("train", "val", "test"):
            row = [counts[(origin, split, c)] for c in labels]
            print(f"{origin:<10}{split:<7}" + "".join(f"{n:>6}" for n in row) + f"{sum(row):>8}")
    print()


def main() -> None:
    args = parse_args()
    with open(args.splits_in, newline="", encoding="utf-8") as f:
        original = [{**r, "origin": "original"} for r in csv.DictReader(f)]

    external = build_external_records(args.external_dir)
    preprocess_all(external, DATA_DIR, args.output_dir, IMAGE_SIZE)
    external = drop_known_images(external, original, args.output_dir)
    merge_near_duplicates(external, args.output_dir)
    split_external(external, original, args.test_folds, args.val_folds)

    records = original + external
    check_no_group_leak(records)
    print_summary(records)

    args.splits_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.splits_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "source", "label", "group", "split", "origin"],
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(f"División nueva guardada en {args.splits_out} ({args.splits_in} no se modificó)")


if __name__ == "__main__":
    main()
