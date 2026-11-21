"""Fase 1: exploración del dataset de imágenes de fondo de ojo."""

import hashlib
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

DATA_DIR = Path("data")
SALIDA_DIR = Path("reports/figures")
EXTENSIONES = {".jpg", ".jpeg", ".png"}
SEMILLA = 42

NOMBRES = {
    "A": "Degeneración macular",
    "C": "Catarata",
    "D": "Retinopatía diabética",
    "G": "Glaucoma",
    "H": "Retinopatía hipertensiva",
    "M": "Miopía",
    "N": "Normal",
}


def listar_imagenes(data_dir: Path) -> dict[str, list[Path]]:
    """Devuelve {clase: [rutas de imágenes]} leyendo una carpeta por clase."""
    clases = {}
    for carpeta in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        imagenes = sorted(f for f in carpeta.iterdir() if f.suffix.lower() in EXTENSIONES)
        clases[carpeta.name] = imagenes
    return clases

def resumen_conteo(clases: dict[str, list[Path]]) -> None:
    """Imprime cuántas imágenes hay por clase y el ratio de desbalance."""
    total = sum(len(imgs) for imgs in clases.values())
    print(f"{'Clase':<6}{'Nombre':<26}{'Imgs':>6}{'%':>8}")
    for clase, imgs in clases.items():
        print(f"{clase:<6}{NOMBRES.get(clase, '?'):<26}{len(imgs):>6}{len(imgs) / total:>8.1%}")
    print(f"{'':<6}{'TOTAL':<26}{total:>6}")

    conteos = [len(imgs) for imgs in clases.values()]
    print(f"Ratio de desbalance (mayor/menor): {max(conteos) / min(conteos):.1f}x\n")

def resumen_conteo(clases: dict[str, list[Path]]) -> None:
    """Imprime cuántas imágenes hay por clase y el ratio de desbalance."""
    total = sum(len(imgs) for imgs in clases.values())
    print(f"{'Clase':<6}{'Nombre':<26}{'Imgs':>6}{'%':>8}")
    for clase, imgs in clases.items():
        print(f"{clase:<6}{NOMBRES.get(clase, '?'):<26}{len(imgs):>6}{len(imgs) / total:>8.1%}")
    print(f"{'':<6}{'TOTAL':<26}{total:>6}")

    conteos = [len(imgs) for imgs in clases.values()]
    print(f"Ratio de desbalance (mayor/menor): {max(conteos) / min(conteos):.1f}x\n")


def grafico_barras(clases: dict[str, list[Path]], salida: Path) -> None:
    """Guarda un gráfico de barras con la distribución de clases."""
    etiquetas = [f"{c}\n{NOMBRES.get(c, '?')}" for c in clases]
    conteos = [len(imgs) for imgs in clases.values()]

    fig, ax = plt.subplots(figsize=(11, 5))
    barras = ax.bar(etiquetas, conteos, color="#4C72B0")
    ax.bar_label(barras)
    ax.set_title("Imágenes por clase")
    ax.set_ylabel("Cantidad")
    ax.tick_params(axis="x", labelsize=8)
    fig.savefig(salida / "class_distribution.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

def grilla_ejemplos(clases: dict[str, list[Path]], salida: Path, n: int = 4) -> None:
    """Guarda una grilla con n imágenes al azar de cada clase (una fila por clase)."""
    rng = random.Random(SEMILLA)
    fig, ejes = plt.subplots(len(clases), n, figsize=(n * 2.5, len(clases) * 2.5))

    for fila, (clase, imgs) in enumerate(clases.items()):
        for col, ruta in enumerate(rng.sample(imgs, n)):
            ax = ejes[fila, col]
            with Image.open(ruta) as img:
                ax.imshow(img.convert("RGB"))
            ax.set_title(f"{clase} · {ruta.name}" if col == 0 else ruta.name, fontsize=7)
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(salida / "samples_per_class.png", dpi=100)
    plt.close(fig)


def analizar_formatos(clases: dict[str, list[Path]], muestra: int = 100) -> None:
    """Revisa extensión, tamaño y modo de color en una muestra de cada clase."""
    rng = random.Random(SEMILLA)
    print("Formatos (muestra de", muestra, "imágenes por clase)")
    for clase, imgs in clases.items():
        extensiones = Counter(p.suffix.lower() for p in imgs)
        tamanos = Counter()
        modos = Counter()
        for ruta in rng.sample(imgs, min(muestra, len(imgs))):
            with Image.open(ruta) as img:  # solo lee la cabecera: rápido
                tamanos[img.size] += 1
                modos[img.mode] += 1
        print(f"  {clase}: ext={dict(extensiones)} modo={dict(modos)} "
              f"tamaños más comunes={tamanos.most_common(2)}")
    print()

def analizar_pacientes(clases: dict[str, list[Path]]) -> None:
    """Detecta IDs de paciente en nombres tipo ODIR: '1022_left.jpg' -> '1022'."""
    patron = re.compile(r"^(\d+)_(left|right)$")
    pacientes_por_clase = {}

    print("Pacientes (solo archivos con formato ID_left / ID_right)")
    for clase, imgs in clases.items():
        ids = [m.group(1) for p in imgs if (m := patron.match(p.stem))]
        conteo = Counter(ids)
        ambos_ojos = sum(1 for veces in conteo.values() if veces >= 2)
        pacientes_por_clase[clase] = set(conteo)
        print(f"  {clase}: {len(ids)}/{len(imgs)} con ID, {len(conteo)} pacientes, "
              f"{ambos_ojos} con ambos ojos en la clase")

    apariciones = Counter(i for ids in pacientes_por_clase.values() for i in ids)
    en_varias = sum(1 for veces in apariciones.values() if veces > 1)
    print(f"  Pacientes que aparecen en más de una clase: {en_varias}\n")

def analizar_duplicados(clases: dict[str, list[Path]]) -> None:
    """Busca archivos idénticos byte a byte usando un hash MD5."""
    por_hash = defaultdict(list)
    for clase, imgs in clases.items():
        for ruta in imgs:
            huella = hashlib.md5(ruta.read_bytes()).hexdigest()
            por_hash[huella].append(f"{clase}/{ruta.name}")

    grupos = [rutas for rutas in por_hash.values() if len(rutas) > 1]
    cruzados = [g for g in grupos if len({r.split("/")[0] for r in g}) > 1]
    print(f"Grupos de duplicados exactos: {len(grupos)}")
    print(f"  ...con la MISMA imagen en clases distintas: {len(cruzados)}")
    for g in cruzados[:5]:
        print("   ", g)
    print()


def main() -> None:
    SALIDA_DIR.mkdir(parents=True, exist_ok=True)
    clases = listar_imagenes(DATA_DIR)

    resumen_conteo(clases)
    analizar_formatos(clases)
    analizar_pacientes(clases)
    analizar_duplicados(clases)
    grafico_barras(clases, SALIDA_DIR)
    grilla_ejemplos(clases, SALIDA_DIR)
    print(f"Figuras guardadas en {SALIDA_DIR}/")


if __name__ == "__main__":
    main()
