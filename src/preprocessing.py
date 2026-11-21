"""Fundus image preprocessing: crop to the retina, pad to square, resize and mask."""

import numpy as np
from PIL import Image, ImageDraw


def crop_to_fundus(img: Image.Image, threshold: int = 15) -> Image.Image:
    """Crop the black background around the retina."""
    gray = np.asarray(img.convert("L"))
    foreground = gray > threshold
    rows = np.flatnonzero(foreground.any(axis=1))
    cols = np.flatnonzero(foreground.any(axis=0))
    if rows.size == 0 or cols.size == 0:  # completely dark image: leave it as is
        return img
    return img.crop((cols[0], rows[0], cols[-1] + 1, rows[-1] + 1))

def center_crop_square(img: Image.Image) -> Image.Image:
    """Crop the largest centered square, so every retina becomes a full circle."""
    width, height = img.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return img.crop((left, top, left + side, top + side))

def apply_circular_mask(img: Image.Image, radius_scale: float = 0.95) -> Image.Image:
    """Black out everything outside a centered circle (removes corner notches)."""
    width, height = img.size
    radius = radius_scale * min(width, height) / 2
    cx, cy = width / 2, height / 2
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=255)
    return Image.composite(img, Image.new("RGB", img.size), mask)

def preprocess_fundus(img: Image.Image, size: int = 512) -> Image.Image:
    """Full pipeline applied once to every image before training."""
    img = img.convert("RGB")
    img = crop_to_fundus(img)
    img = center_crop_square(img)
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    return apply_circular_mask(img)
