"""Phase 7: image quality filter — reject what is not a usable fundus photo before predicting.

Thresholds come from 1 050 train images (150 per class): every limit sits below the 0.5th
percentile (or above the maximum) of real fundus photos, so real images almost never fail.
Blur and whitish images are only WARNINGS: real cataracts look blurry or white (Phase 6),
and rejecting a real patient would be a false negative.
"""

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

MIN_LIT_SHARE = 0.3      # share of the retina circle that is not black (train p0.5: 0.94)
MIN_BRIGHTNESS = 18.0    # mean gray level inside the circle (train p0.5: 41; 15.7 = 85 % darker)
BRIGHT_WARNING = 210.0   # whitish: dense cataract OR too much flash -> warn, never reject
MIN_CONTRAST = 3.0       # gray std: a flat colour has 0 (train p0.5: 6.1)
MIN_RED_SHARE = 0.34     # fundus photos are red/orange: R / (R+G+B) (green-tinted cameras: 0.35, gray: 0.333)
MAX_BLUE_SHARE = 0.33    # (train max: 0.29; whitish cataracts: 0.31)
MAX_SHARPNESS = 1500.0   # random noise or a drawing, not a photo (train max: 723)
BLUR_WARNING = 4.0       # out-of-focus 4 px gives 3.4; real cataracts have a median of 5.1

ISSUE_MESSAGES = {
    "dark": "La imagen está demasiado oscura: no se ve la retina.",
    "flat": "La imagen no tiene detalles (parece un color plano).",
    "not_fundus": "La imagen no parece una foto de fondo de ojo (retina).",
}
BRIGHT_MESSAGE = ("Imagen muy clara: casi no se ven detalles de la retina. Puede ser una "
                  "catarata densa o exceso de luz en la foto: confirmar con un especialista.")
BLUR_MESSAGE = ("Imagen poco nítida. Una catarata real también se ve así, pero una foto "
                "desenfocada da el mismo resultado: si es posible, repetir la toma.")


@dataclass
class QualityReport:
    metrics: dict[str, float]
    issues: list[str] = field(default_factory=list)    # codes that block the prediction
    warnings: list[str] = field(default_factory=list)  # messages shown next to the result

    @property
    def ok(self) -> bool:
        return not self.issues

    @property
    def messages(self) -> list[str]:
        return [ISSUE_MESSAGES[code] for code in self.issues]


def image_metrics(clean: Image.Image) -> dict[str, float]:
    """Measure an image already passed through preprocess_fundus (square, masked)."""
    pixels = np.asarray(clean.convert("RGB"), dtype=np.float32)
    height, width, _ = pixels.shape
    yy, xx = np.mgrid[:height, :width]
    inside = (xx - width / 2) ** 2 + (yy - height / 2) ** 2 <= (0.9 * width / 2) ** 2
    rgb = pixels[inside]
    gray = rgb.mean(axis=1)
    total = rgb.sum(axis=1) + 1e-6

    # sharpness: variance of the Laplacian (edges) on a 256 px gray version
    small = np.asarray(clean.convert("L").resize((256, 256)), dtype=np.float32)
    laplacian = (4 * small[1:-1, 1:-1] - small[:-2, 1:-1] - small[2:, 1:-1]
                 - small[1:-1, :-2] - small[1:-1, 2:])
    small_inside = np.asarray(Image.fromarray(inside).resize((256, 256)))[1:-1, 1:-1]

    return {
        "lit_share": float((gray > 20).mean()),
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "red_share": float((rgb[:, 0] / total).mean()),
        "blue_share": float((rgb[:, 2] / total).mean()),
        "sharpness": float(laplacian[small_inside].var()),
    }


def assess_quality(clean: Image.Image) -> QualityReport:
    m = image_metrics(clean)
    report = QualityReport(metrics=m)
    if m["lit_share"] < MIN_LIT_SHARE or m["brightness"] < MIN_BRIGHTNESS:
        report.issues.append("dark")
    if m["contrast"] < MIN_CONTRAST:
        report.issues.append("flat")
    elif (m["red_share"] < MIN_RED_SHARE or m["blue_share"] > MAX_BLUE_SHARE
          or m["sharpness"] > MAX_SHARPNESS):
        report.issues.append("not_fundus")
    if report.ok and m["brightness"] > BRIGHT_WARNING:
        report.warnings.append(BRIGHT_MESSAGE)
    elif report.ok and m["sharpness"] < BLUR_WARNING:
        report.warnings.append(BLUR_MESSAGE)
    return report
