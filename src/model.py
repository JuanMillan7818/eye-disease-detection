"""Phase 3: model definition with transfer learning (EfficientNet-B0 / EfficientNetV2-S)."""

import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torchvision import models


@dataclass(frozen=True)
class ModelSpec:
    builder: Callable[..., nn.Module]
    weights: models.WeightsEnum
    image_size: int


MODEL_SPECS = {
    "b0": ModelSpec(models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1, 224),
    "v2s": ModelSpec(models.efficientnet_v2_s, models.EfficientNet_V2_S_Weights.IMAGENET1K_V1, 384),
}

def create_model(name: str, num_classes: int, pretrained: bool = True, dropout: float = 0.3) -> nn.Module:
    """Load an ImageNet-pretrained EfficientNet and replace its last layer."""
    spec = MODEL_SPECS[name]
    model = spec.builder(weights=spec.weights if pretrained else None)

    in_features = model.classifier[-1].in_features  # 1280 for both models
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model

def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze (False) or unfreeze (True) the pretrained feature extractor."""
    for param in model.features.parameters():
        param.requires_grad = trainable


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (total, trainable) number of parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def save_checkpoint(model: nn.Module, path: Path, model_name: str, classes: list[str],
                    image_size: int | None = None, metrics: dict | None = None,
                    loss: str = "torch") -> None:
    """Save weights plus everything needed to rebuild and use the model later."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_name": model_name,
        "classes": classes,
        "image_size": image_size or MODEL_SPECS[model_name].image_size,
        "loss": loss,  # calibration needs to know if the class-weight bias exists
        "metrics": metrics or {},
        "state_dict": model.state_dict(),
    }, path)


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[nn.Module, dict]:
    """Rebuild the model from a checkpoint, ready for inference."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = create_model(checkpoint["model_name"], len(checkpoint["classes"]), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint

def main() -> None:
    classes = ["A", "C", "D", "G", "H", "M", "N"]

    for name, spec in MODEL_SPECS.items():
        print(f"=== {name} ===")
        model = create_model(name, num_classes=len(classes))
        print("Cabeza nueva:", model.classifier)

        total, trainable = count_parameters(model)
        print(f"Parámetros: {total:,} totales, {trainable:,} entrenables")
        set_backbone_trainable(model, False)
        total, trainable = count_parameters(model)
        print(f"Con backbone congelado: {trainable:,} entrenables ({trainable / total:.2%})")

        model.eval()
        images = torch.randn(2, 3, spec.image_size, spec.image_size)
        with torch.no_grad():
            model(images)  # warm-up: the first call includes one-time setup costs
            start = time.perf_counter()
            logits = model(images)
            elapsed = (time.perf_counter() - start) / len(images)
        probs = logits.softmax(dim=1)
        print(f"Entrada {tuple(images.shape)} -> logits {tuple(logits.shape)}")
        print(f"Probabilidades imagen 0: {[round(p, 3) for p in probs[0].tolist()]} (suma {probs[0].sum():.2f})")
        print(f"Tiempo por imagen en CPU: {elapsed * 1000:.0f} ms")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"{name}.pth"
            save_checkpoint(model, path, name, classes)
            size_mb = path.stat().st_size / 1e6
            loaded, info = load_checkpoint(path)
            with torch.no_grad():
                same = torch.allclose(loaded(images), logits, atol=1e-5)
        print(f"Checkpoint: {size_mb:.1f} MB, recargado igual: {same}, clases: {info['classes']}\n")


if __name__ == "__main__":
    main()
