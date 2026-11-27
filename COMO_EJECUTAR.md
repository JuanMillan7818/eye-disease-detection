# Cómo ejecutar cada archivo

Orden completo, desde explorar los datos hasta predecir una imagen. Cada paso dice **qué necesita**, **el comando**, **qué genera** y **cuánto tarda**.

Todos los comandos se ejecutan **desde la carpeta del proyecto** y con `uv run`, que usa el Python y las librerías del proyecto sin activar nada. Los scripts se llaman como módulo (`python -m src.archivo`), no como `python src/archivo.py`: así funcionan los `from src... import`.

```
data/  ──1 explorer──► gráficas
  │
  ├──2 prepare_data──────► data_processed/ + splits/splits.csv
  ├──3 prepare_external──► data_processed/ext/ + splits/splits_ext.csv
  │
  4 dataset · 5 model        (pruebas rápidas, opcionales)
  │
  6 train  (Colab, GPU) ──► models/v2s_ext_best.pth
  7 calibrate ──► calibración guardada dentro del .pth
  8 evaluate  ──► reports/*_predictions.csv + matrices de confusión
  │
  9 model_report · 10 final_package · 11 export_onnx · 12 gradcam
  │
  13 predict  ·  14 la app web
```

> **¿Solo quieres probar el modelo?** Con `models/v2s_best.pth` puesto, ve directo a los pasos 13 y 14. No hacen falta los datos ni entrenar.

---

## 0. Preparar el entorno (una sola vez)

```powershell
uv sync                    # instala Python y las librerías de pyproject.toml
uv sync --group export     # opcional: lo necesario para el paso 11 (ONNX)
```

Carpetas que deben existir:

| Carpeta | Contenido | ¿Está en git? |
|---|---|---|
| `data/<clase>/` | Imágenes originales: `data/A/`, `data/C/`, ... `data/N/` | ❌ (pesado) |
| `data/dataseet externo/<carpeta>/` | Dataset externo: `cataract/`, `diabetic_retinopathy/`, `glaucoma/`, `normal/` | ❌ |
| `models/` | Modelos entrenados (`.pth`) | ❌ |

---

## 1. `explorer.py`: explorar el dataset

- **Necesita:** `data/<clase>/`
- **Comando:**
  ```powershell
  uv run python -m src.explorer
  ```
- **Muestra:** cuántas imágenes hay por clase, el desbalance, los formatos y tamaños, y los pacientes con los dos ojos.
- **Genera:** `reports/figures/class_distribution.png` y `reports/figures/samples_per_class.png`
- **Tarda:** segundos. No modifica nada.

## 2. `prepare_data.py`: limpiar, dividir y preprocesar

- **Necesita:** `data/<clase>/`
- **Comando:**
  ```powershell
  uv run python -m src.prepare_data
  ```
- **Hace:**
  1. Borra las copias exactas (MD5) y las imágenes con dos etiquetas.
  2. Agrupa por paciente y junta las casi duplicadas (dHash).
  3. Divide en train / val / test por grupo.
  4. Recorta la retina, la deja cuadrada, la reduce a 512 px y le pone la máscara circular.
- **Genera:**
  - `splits/splits.csv`: una fila por imagen, con su clase, su grupo y su conjunto.
  - `data_processed/<clase>/*.jpg`: las imágenes ya preprocesadas.
- **Tarda:** unos minutos.
- **Opciones:** `--size 512` (tamaño de salida), `--data-dir`, `--output-dir`.
- ⚠️ **Vuelve a sortear la división.** Si `splits/splits.csv` ya existe y hay modelos entrenados con él, no lo vuelvas a correr: los resultados dejarían de ser comparables.

## 3. `prepare_external.py`: sumar el dataset externo

- **Necesita:** el paso 2 hecho y `data/dataseet externo/`
- **Comando:**
  ```powershell
  uv run python -m src.prepare_external
  ```
- **Hace:**
  - Borra las copias exactas y las imágenes externas que ya estaban en el original (dHash ≤ 6).
  - Las de pacientes ODIR ya conocidos **heredan el conjunto** de ese paciente.
  - El resto se divide por paciente: 1/4 a test y 1/5 del resto a val.
- **Genera:**
  - `splits/splits_ext.csv`: las filas originales sin cambios, más las externas, con la columna `origin`.
  - `data_processed/ext/<clase>/*.jpg`
- **No toca** `splits/splits.csv`.
- **Tarda:** unos minutos.

## 4. `dataset.py`: probar la carga de datos (opcional)

- **Necesita:** `data_processed/` y `splits/splits.csv`
- **Comando:**
  ```powershell
  uv run python -m src.dataset
  ```
- **Muestra:** las clases, los pesos por clase, cuántas imágenes y lotes hay por conjunto, y la forma de un lote.
- **Genera:** `reports/figures/augmented_batch.png`, un lote con *data augmentation*.
- **Tarda:** segundos. Usa solo 200 imágenes. Sirve para comprobar que todo carga **antes** de entrenar.

## 5. `model.py`: ver los modelos (opcional)

- **Comando:**
  ```powershell
  uv run python -m src.model
  ```
- **Muestra:** para B0 y V2-S, la cabeza nueva, los parámetros totales y cuántos se entrenan con el backbone congelado.
- **Tarda:** segundos. La primera vez descarga los pesos de ImageNet.

## 6. `train.py`: entrenar

- **Necesita:** `data_processed/` y el archivo de división. **Recomendado: GPU.** Usa `notebooks/02_training_colab.ipynb`, sección 8.
- **Comando del modelo final:**
  ```powershell
  uv run python -m src.train --model v2s --splits splits/splits_ext.csv --tag ext --patience 25
  ```
- **Hace:**
  1. Etapa 1: entrena solo la cabeza nueva (`--head-epochs 3`).
  2. Etapa 2: entrena toda la red (`--epochs 25`).
  3. Se queda con la época de mejor F1 macro en val (*early stopping*).
- **Genera:**
  - `models/v2s_ext_best.pth`
  - `models/v2s_ext_history.csv`
  - `reports/figures/v2s_ext_training_curves.png`
- **Tarda:** ~2 h en una GPU L4. En CPU, días.
- **Opciones útiles:**

  | Opción | Para qué |
  |---|---|
  | `--model b0` o `v2s` | Arquitectura |
  | `--tag ext` | Sufijo del nombre: `v2s` → `v2s_ext` |
  | `--splits` | `splits/splits.csv` (solo el original) o `splits/splits_ext.csv` (con el externo) |
  | `--limit 50` | Prueba rápida con 50 imágenes por conjunto, para ver que no falla |
  | `--batch-size 16 --workers 2` | Para la GPU de Colab |

## 7. `calibrate.py`: calibrar la confianza y fijar el semáforo

- **Necesita:** el `.pth` entrenado y `data_processed/`
- **Comando:**
  ```powershell
  uv run python -m src.calibrate --model v2s_ext --green 0.90
  ```
- **Hace:**
  1. Ajusta la temperatura con val, para que "90 %" signifique "acierta 9 de cada 10".
  2. Aplica la política de tamizaje: exige 4× más evidencia antes de decir "Normal".
  3. Muestra cuánto acierta cada color del semáforo.
  4. Si val mezcla las dos fuentes, repite el reporte para el original y para el externo.
- **Genera:** la calibración, guardada **dentro** del `.pth`, y `reports/figures/v2s_ext_calibration.png`.
- **Tarda:** ~5–10 min en CPU.
- **Opciones:**

  | Opción | Para qué |
  |---|---|
  | `--green 0.90` | Corte del verde de este modelo (por defecto 0.88) |
  | `--report-only` | Solo mide con la calibración actual; no guarda nada |

## 8. `evaluate.py`: medir y comparar modelos

- **Necesita:** los modelos y `data_processed/`
- **Comandos:**
  ```powershell
  # Para DECIDIR (val; se puede repetir):
  uv run python -m src.evaluate --split val --models v2s_original v2s --splits splits/splits_ext.csv

  # Medición FINAL (test; una sola vez, con el modelo ya elegido):
  uv run python -m src.evaluate --split test --models v2s_original v2s --splits splits/splits_ext.csv
  ```
- **Muestra:** precisión, recall y F1 por clase; los resultados por origen; y la comparación sin TTA y con TTA.
- **Genera:**
  - `reports/<modelo>_<conjunto>_splits_ext_predictions.csv`: una fila por imagen, con todas las probabilidades. La usan los pasos 9 y 10.
  - `reports/figures/<modelo>_<conjunto>_splits_ext_confusion.png`: matriz de confusión con la clase más probable, **sin** la política.
  - `reports/figures/<modelo>_<conjunto>_splits_ext_errors.png`: ejemplos de errores.
  - `reports/<conjunto>_splits_ext_comparison.csv`
- **Tarda:** ~5–10 min por modelo en CPU. Hace dos pasadas por modelo: normal y con TTA.

## 9. `model_report.py`: métricas para la web

- **Necesita:** el paso 8 en **test**
- **Comando:**
  ```powershell
  uv run python -m src.model_report --model v2s
  ```
- **Hace:** aplica la política de tamizaje a las predicciones de test y calcula los KPI, la matriz de confusión, el semáforo, las métricas por clase y por origen, y el tiempo en CPU.
- **Genera:** `web/metrics.json`, que dibuja la pestaña **Métricas del modelo**.
- **Tarda:** segundos.

## 10. `final_package.py`: carpeta para mostrar el modelo final

- **Necesita:** los pasos 8 (test) y 9
- **Comando:**
  ```powershell
  uv run python -m src.final_package
  ```
- **Genera** `reports/modelo_final/`:
  - las matrices de confusión **con la política**: todo el test, solo original, solo externo y el modelo anterior;
  - las curvas de entrenamiento, los ejemplos de errores y la calibración;
  - `metricas.json` y `LEEME.md`.
- **Tarda:** segundos.

## 11. `export_onnx.py`: exportar sin PyTorch (opcional)

- **Necesita:** `uv sync --group export`
- **Comando:**
  ```powershell
  uv run --group export python -m src.export_onnx --model v2s
  ```
- **Genera:** `models/v2s.onnx` (la red con la calibración incluida) y `models/v2s.onnx.json` (las clases, el preprocesamiento y el semáforo).
- **Verifica:** compara ONNX contra PyTorch en 100 imágenes de test; deben tomar la misma decisión en 100/100.
- **Para usarlo:** `python examples/onnx_predict.py models/v2s.onnx foto.jpg`
- **Tarda:** ~1 min.

## 12. `gradcam.py`: mapas de calor (opcional)

- **Necesita:** el modelo
- **Comandos:**
  ```powershell
  uv run python -m src.gradcam data\D\123_left.jpg     # una o varias imágenes
  uv run python -m src.gradcam --grid                  # cuadrícula de imágenes de test por clase
  uv run python -m src.gradcam --errors D N            # casos reales D que el modelo dijo N
  ```
- **Genera:** `reports/figures/gradcam/v2s_*.png`
- **Muestra:** en qué zona de la retina se fijó el modelo para responder.

## 13. `predict.py`: predecir desde la consola

- **Necesita:** solo `models/v2s_best.pth`. Acepta fotos originales, no hace falta preprocesarlas.
- **Comandos:**
  ```powershell
  uv run python -m src.predict data\N\2691_right.jpg
  uv run python -m src.predict foto1.jpg foto2.png foto3.jpg     # varias a la vez
  uv run python -m src.predict pruebas\*.jpg                     # una carpeta completa
  uv run python -m src.predict foto.jpg --model v2s_original     # con el modelo anterior
  ```
- **Muestra:** para cada imagen,
  - la clase real y su conjunto, si la foto está en el dataset: `test` significa que el modelo nunca la vio;
  - la predicción y el semáforo;
  - la probabilidad de cada clase.

  Ejemplo:
  ```
  data\N\2691_right.jpg
    Etiqueta real: N Normal | conjunto: test | ✅ acierto
    Predicción: N Normal (92.0%)
    Confianza: 🟡 sin signos claros de enfermedad — control anual con oftalmólogo
  ```
- **Tarda:** ~1 s por imagen en CPU.

## 14. La app web (API + página)

- **Necesita:** solo `models/v2s_best.pth`
- **Comando:**
  ```powershell
  uv run uvicorn api.main:app
  ```
- Abre <http://127.0.0.1:8000>: pestaña **Analizar imagen** (arrastras la foto) y pestaña **Métricas del modelo**.
- La documentación de la API está en <http://127.0.0.1:8000/docs>.
- Si el puerto 8000 está ocupado: `--port 8001`.
- Para detenerla: `Ctrl + C`.

---

## Archivos que no se ejecutan solos

Los usan los demás scripts:

| Archivo | Qué hace | Lo usan |
|---|---|---|
| `preprocessing.py` | Recorta la retina, la deja cuadrada, la redimensiona y le pone la máscara | `prepare_data`, `predict`, `gradcam`, la API |
| `quality.py` | Filtro de calidad: rechaza fotos que no son retinas y avisa si están borrosas o claras | La API |
| `source_report.py` | Resultados por fuente de imagen. Solo funciona con la división original (`splits.csv`) | Fase 6: `uv run python -m src.source_report --split test` |

## Tests

```powershell
uv run pytest -q          # 11 tests de la API (necesitan models/ y data/)
```

## Resumen de comandos del modelo final

```powershell
uv run python -m src.explorer
uv run python -m src.prepare_data
uv run python -m src.prepare_external
# entrenamiento en Colab (sección 8) → descargar v2s_ext_best.pth y v2s_ext_history.csv a models/
uv run python -m src.calibrate --model v2s_ext --green 0.90
uv run python -m src.evaluate --split val --models v2s v2s_ext --splits splits/splits_ext.csv
Copy-Item models\v2s_best.pth models\v2s_original_best.pth          # respaldo del modelo anterior
Copy-Item models\v2s_ext_best.pth models\v2s_best.pth               # adoptar el nuevo
uv run python -m src.evaluate --split test --models v2s_original v2s --splits splits/splits_ext.csv
uv run python -m src.model_report --model v2s
uv run python -m src.final_package
uv run --group export python -m src.export_onnx --model v2s
uv run python -m src.predict data\N\2691_right.jpg
uv run uvicorn api.main:app
```
