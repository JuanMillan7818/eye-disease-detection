"""Phase 2A: clean the dataset, assign groups, split it and preprocess the images."""

import argparse
import csv
import hashlib
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from src.explorer import DATA_DIR, SEED, list_images
from src.preprocessing import preprocess_fundus

PROCESSED_DIR = Path("data_processed")
SPLITS_FILE = Path("splits/splits.csv")
IMAGE_SIZE = 512

ODIR_PATTERN = re.compile(r"^(\d+)_(left|right)$")  # 1022_left -> patient 1022
AUGMENTED_PATTERN = re.compile(r"^_(\d+)_\d+$")     # _90_3742626 -> original 90


def assign_group(label: str, path: Path) -> str:
    """Images in the same group must always end up in the same split."""
    if match := ODIR_PATTERN.match(path.stem):
        return f"odir_{match.group(1)}"
    if match := AUGMENTED_PATTERN.match(path.stem):
        return f"aug_{label}_{match.group(1)}"
    return f"img_{label}_{path.stem}"

def build_records(classes: dict[str, list[Path]]) -> list[dict]:
    """One record per unique image; drop images that appear in more than one class."""
    by_hash = defaultdict(list)
    for label, images in classes.items():
        for path in images:
            digest = hashlib.md5(path.read_bytes()).hexdigest()
            by_hash[digest].append((label, path))

    records = []
    conflicts = copies = 0
    for items in by_hash.values():
        if len({label for label, _ in items}) > 1:  # same image, different labels
            conflicts += len(items)
            continue
        label, path = items[0]  # keep the first copy only
        copies += len(items) - 1
        records.append({
            "file": f"{label}/{path.stem}.jpg",
            "source": f"{label}/{path.name}",
            "label": label,
            "group": assign_group(label, path),
        })

    print(f"Descartadas {conflicts} imágenes con etiquetas contradictorias y {copies} copias duplicadas")
    print(f"Se conservan {len(records)} imágenes únicas\n")
    return sorted(records, key=lambda r: r["file"])

def difference_hash(path: Path, hash_size: int = 16) -> np.ndarray:
    """Perceptual hash: 1 bit per pixel = 'is the next pixel brighter?'."""
    with Image.open(path) as img:
        small = img.convert("L").resize((hash_size + 1, hash_size))
    pixels = np.asarray(small, dtype=np.int16)
    return (pixels[:, 1:] > pixels[:, :-1]).ravel()


def merge_near_duplicates(records: list[dict], image_dir: Path, max_distance: int = 6) -> None:
    """Put visually identical images (resized or re-saved copies) in the same group."""
    bits = np.array([difference_hash(image_dir / r["file"]) for r in records], dtype=np.float32)
    parent = {r["group"]: r["group"] for r in records}

    def find(group: str) -> str:
        while parent[group] != group:
            parent[group] = parent[parent[group]]  # path compression
            group = parent[group]
        return group

    pairs = 0
    for start in range(0, len(bits), 1000):  # chunks keep memory low
        chunk = bits[start:start + 1000]
        # Hamming distance = number of bits that differ
        distances = chunk @ (1 - bits).T + (1 - chunk) @ bits.T
        for i, j in zip(*np.nonzero(distances <= max_distance)):
            i += start
            if i < j:
                pairs += 1
                parent[find(records[i]["group"])] = find(records[j]["group"])

    for r in records:
        r["group"] = find(r["group"])
    print(f"Pares de casi-duplicados unidos en un mismo grupo: {pairs}\n")


def split_records(records: list[dict], seed: int = SEED) -> None:
    """Add a 'split' key (train/val/test ~ 70/15/15), stratified by label, grouped by group."""
    labels = np.array([r["label"] for r in records])
    groups = np.array([r["group"] for r in records])
    indices = np.arange(len(records))

    outer = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    trainval_idx, test_idx = next(outer.split(indices, labels, groups))

    inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
    train_pos, val_pos = next(inner.split(trainval_idx, labels[trainval_idx], groups[trainval_idx]))

    for i in test_idx:
        records[i]["split"] = "test"
    for i in trainval_idx[val_pos]:
        records[i]["split"] = "val"
    for i in trainval_idx[train_pos]:
        records[i]["split"] = "train"


def check_no_group_leak(records: list[dict]) -> None:
    """Fail loudly if any group appears in more than one split."""
    splits_per_group = defaultdict(set)
    for r in records:
        splits_per_group[r["group"]].add(r["split"])
    leaked = [g for g, splits in splits_per_group.items() if len(splits) > 1]
    assert not leaked, f"{len(leaked)} grupos aparecen en varias particiones, ej. {leaked[:3]}"
    print(f"OK: {len(splits_per_group)} grupos, ninguno compartido entre particiones\n")


def print_split_summary(records: list[dict]) -> None:
    counts = Counter((r["split"], r["label"]) for r in records)
    labels = sorted({r["label"] for r in records})
    print(f"{'parte':<7}" + "".join(f"{label:>7}" for label in labels) + f"{'total':>8}")
    for split in ("train", "val", "test"):
        row = [counts[(split, label)] for label in labels]
        print(f"{split:<7}" + "".join(f"{n:>7}" for n in row) + f"{sum(row):>8}")
    print()


def save_splits(records: list[dict], splits_file: Path) -> None:
    splits_file.parent.mkdir(parents=True, exist_ok=True)
    with open(splits_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "source", "label", "group", "split"])
        writer.writeheader()
        writer.writerows(records)
    print(f"División guardada en {splits_file}")

def preprocess_one(record: dict, data_dir: Path, output_dir: Path, size: int) -> None:
    target = output_dir / record["file"]
    if target.exists():  # already done in a previous run
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(data_dir / record["source"]) as img:
        preprocess_fundus(img, size).save(target, quality=95)


def preprocess_all(records: list[dict], data_dir: Path, output_dir: Path, size: int) -> None:
    """Preprocess every image once, in parallel threads (PIL releases the GIL)."""
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        futures = [pool.submit(preprocess_one, r, data_dir, output_dir, size) for r in records]
        for i, future in enumerate(futures, start=1):
            future.result()  # re-raises any error from the worker thread
            if i % 1000 == 0 or i == len(futures):
                print(f"  procesadas {i}/{len(futures)}")
    print(f"Imágenes guardadas en {output_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--size", type=int, default=IMAGE_SIZE)
    args = parser.parse_args()

    records = build_records(list_images(args.data_dir))
    preprocess_all(records, args.data_dir, args.output_dir, args.size)
    merge_near_duplicates(records, args.output_dir)
    split_records(records)
    check_no_group_leak(records)
    print_split_summary(records)
    save_splits(records, SPLITS_FILE)


if __name__ == "__main__":
    main()
