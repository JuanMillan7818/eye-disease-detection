"""Phase 2B: PyTorch Dataset, transforms and DataLoaders."""

import csv
import random
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from src.explorer import SEED
from src.prepare_data import PROCESSED_DIR, SPLITS_FILE

# Statistics of ImageNet: pretrained models expect inputs normalized with them
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def read_splits(splits_file: Path = SPLITS_FILE) -> list[dict]:
    with open(splits_file, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

class FundusDataset(Dataset):
    """Returns (image_tensor, class_index) pairs for one split."""

    def __init__(self, records: list[dict], image_dir: Path, class_to_idx: dict[str, int],
                 transform=None):
        self.records = records
        self.image_dir = image_dir
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        with Image.open(self.image_dir / record["file"]) as img:
            image = img.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.class_to_idx[record["label"]]

def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    """Train: random augmentation + normalization. Val/test: only resize + normalization."""
    if train:
        steps = [
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        ]
    else:
        steps = [transforms.Resize((image_size, image_size))]
    steps += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(steps)


def compute_class_weights(records: list[dict], classes: list[str]) -> torch.Tensor:
    """weight = total / (n_classes * count): rare classes weigh more in the loss."""
    counts = Counter(r["label"] for r in records)
    total = sum(counts.values())
    weights = [total / (len(classes) * counts[c]) for c in classes]
    return torch.tensor(weights, dtype=torch.float32)

def create_dataloaders(image_size: int = 224, batch_size: int = 32, num_workers: int = 0,
                       limit: int | None = None, image_dir: Path = PROCESSED_DIR,
                       splits_file: Path = SPLITS_FILE):
    """Return (loaders dict, class list, class weights computed on train)."""
    records = read_splits(splits_file)
    classes = sorted({r["label"] for r in records})
    class_to_idx = {label: i for i, label in enumerate(classes)}
    rng = random.Random(SEED)

    loaders = {}
    for split in ("train", "val", "test"):
        subset = [r for r in records if r["split"] == split]
        if limit:  # small random subset for quick local tests
            subset = rng.sample(subset, min(limit, len(subset)))
        is_train = split == "train"
        dataset = FundusDataset(subset, image_dir, class_to_idx, build_transforms(image_size, is_train))
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    train_records = [r for r in records if r["split"] == "train"]
    return loaders, classes, compute_class_weights(train_records, classes)

def save_batch_preview(images: torch.Tensor, path: Path) -> None:
    """Undo the normalization and save the batch as a grid image."""
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    images = (images * std + mean).clamp(0, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(make_grid(images, nrow=8), path)


def main() -> None:
    loaders, classes, class_weights = create_dataloaders(batch_size=16, limit=200)
    print("Clases:", classes)
    print("Pesos por clase:", [round(w, 2) for w in class_weights.tolist()])
    for split, loader in loaders.items():
        print(f"{split}: {len(loader.dataset)} imágenes, {len(loader)} lotes")

    images, labels = next(iter(loaders["train"]))
    print("Imágenes del lote:", tuple(images.shape), images.dtype)
    print("Etiquetas del lote:", labels.tolist())
    print(f"Rango de píxeles tras normalizar: {images.min():.2f} .. {images.max():.2f}")
    save_batch_preview(images, Path("reports/figures/augmented_batch.png"))
    print("Vista previa guardada en reports/figures/augmented_batch.png")


if __name__ == "__main__":
    main()
