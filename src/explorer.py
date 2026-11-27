"""Fase 1: exploración del dataset de imágenes de fondo de ojo."""

import hashlib
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

DATA_DIR = Path("data")
OUTPUT_DIR = Path("reports/figures")
EXTENSIONS = {".jpg", ".jpeg", ".png"}
SEED = 42

# Display names shown to end users (Spanish on purpose)
CLASS_NAMES = {
    "A": "Degeneración macular",
    "C": "Catarata",
    "D": "Retinopatía diabética",
    "G": "Glaucoma",
    "H": "Retinopatía hipertensiva",
    "M": "Miopía",
    "N": "Normal",
}

def list_images(data_dir: Path) -> dict[str, list[Path]]:
    """Return {class_label: [image paths]}, reading one folder per class."""
    classes = {}
    for folder in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        images = sorted(f for f in folder.iterdir() if f.suffix.lower() in EXTENSIONS)
        if images:  # skip folders that hold other datasets (e.g. "dataseet externo/<class>/")
            classes[folder.name] = images
    return classes

def print_class_counts(classes: dict[str, list[Path]]) -> None:
    """Print images per class and the imbalance ratio."""
    total = sum(len(imgs) for imgs in classes.values())
    print(f"{'Clase':<6}{'Nombre':<26}{'Imgs':>7}{'%':>8}")
    for label, imgs in classes.items():
        print(f"{label:<6}{CLASS_NAMES.get(label, '?'):<26}{len(imgs):>7}{len(imgs) / total:>8.1%}")
    print(f"{'':<6}{'TOTAL':<26}{total:>7}")

    counts = [len(imgs) for imgs in classes.values()]
    print(f"Ratio de desbalance (mayor/menor): {max(counts) / min(counts):.1f}x\n")

def plot_class_distribution(classes: dict[str, list[Path]], output_dir: Path) -> None:
    """Save a bar chart with the number of images per class."""
    labels = [f"{c}\n{CLASS_NAMES.get(c, '?')}" for c in classes]
    counts = [len(imgs) for imgs in classes.values()]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(labels, counts, color="#4C72B0")
    ax.bar_label(bars)
    ax.set_title("Imágenes por clase")
    ax.set_ylabel("Cantidad")
    ax.tick_params(axis="x", labelsize=8)
    fig.savefig(output_dir / "class_distribution.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

def plot_samples(classes: dict[str, list[Path]], output_dir: Path, n: int = 4) -> None:
    """Save a grid with n random images per class (one row per class)."""
    rng = random.Random(SEED)
    fig, axes = plt.subplots(len(classes), n, figsize=(n * 2.5, len(classes) * 2.5))

    for row, (label, imgs) in enumerate(classes.items()):
        for col, path in enumerate(rng.sample(imgs, n)):
            ax = axes[row, col]
            with Image.open(path) as img:
                ax.imshow(img.convert("RGB"))
            ax.set_title(f"{label} · {path.name}" if col == 0 else path.name, fontsize=7)
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_dir / "samples_per_class.png", dpi=100)
    plt.close(fig)


def analyze_formats(classes: dict[str, list[Path]], sample_size: int = 100) -> None:
    """Check file extension, size and color mode on a sample of each class."""
    rng = random.Random(SEED)
    print(f"Formatos (muestra de {sample_size} imágenes por clase)")
    for label, imgs in classes.items():
        extensions = Counter(p.suffix.lower() for p in imgs)
        sizes = Counter()
        modes = Counter()
        for path in rng.sample(imgs, min(sample_size, len(imgs))):
            with Image.open(path) as img:  # lazy: reads only the header, fast
                sizes[img.size] += 1
                modes[img.mode] += 1
        print(f"  {label}: ext={dict(extensions)} modo={dict(modes)} "
              f"tamaños más comunes={sizes.most_common(2)}")
    print()
    
def analyze_patients(classes: dict[str, list[Path]]) -> None:
    """Find patient IDs in ODIR-style names: '1022_left.jpg' -> '1022'."""
    pattern = re.compile(r"^(\d+)_(left|right)$")
    patients_per_class = {}

    print("Pacientes (solo archivos con formato ID_left / ID_right)")
    for label, imgs in classes.items():
        ids = [m.group(1) for p in imgs if (m := pattern.match(p.stem))]
        counts = Counter(ids)
        both_eyes = sum(1 for times in counts.values() if times >= 2)
        patients_per_class[label] = set(counts)
        print(f"  {label}: {len(ids)}/{len(imgs)} con ID, {len(counts)} pacientes, "
              f"{both_eyes} con ambos ojos en la clase")

    occurrences = Counter(pid for ids in patients_per_class.values() for pid in ids)
    multi_class = sum(1 for times in occurrences.values() if times > 1)
    print(f"  Pacientes que aparecen en más de una clase: {multi_class}\n")

def analyze_duplicates(classes: dict[str, list[Path]]) -> None:
    """Find byte-identical files using an MD5 hash."""
    by_hash = defaultdict(list)
    for label, imgs in classes.items():
        for path in imgs:
            digest = hashlib.md5(path.read_bytes()).hexdigest()
            by_hash[digest].append(f"{label}/{path.name}")

    groups = [paths for paths in by_hash.values() if len(paths) > 1]
    cross_class = [g for g in groups if len({p.split("/")[0] for p in g}) > 1]
    print(f"Grupos de duplicados exactos: {len(groups)}")
    print(f"  ...con la MISMA imagen en clases distintas: {len(cross_class)}")
    for g in cross_class[:5]:
        print("   ", g)
    print()

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    classes = list_images(DATA_DIR)

    print_class_counts(classes)
    analyze_formats(classes)
    analyze_patients(classes)
    analyze_duplicates(classes)
    plot_class_distribution(classes, OUTPUT_DIR)
    plot_samples(classes, OUTPUT_DIR)
    print(f"Figuras guardadas en {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

