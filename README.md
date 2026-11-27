# Detección de enfermedades oculares con IA

Aplicación web que analiza una **foto de fondo de ojo (retinografía)** e indica si ve signos de alguna de estas enfermedades:

| Código | Enfermedad |
|---|---|
| A | Degeneración macular asociada a la edad |
| C | Catarata |
| D | Retinopatía diabética |
| G | Glaucoma |
| H | Retinopatía hipertensiva |
| M | Miopía patológica |
| N | Normal |

Muestra la enfermedad más probable, un **semáforo de confianza** calibrado, la probabilidad de cada clase y un **mapa de calor (Grad-CAM)** con las zonas de la retina que más influyeron en la respuesta.

> ⚠️ **Aviso importante:** proyecto académico (Inteligencia Artificial, INTEP). **No es un dispositivo médico ni reemplaza el diagnóstico de un oftalmólogo.** No debe usarse para tomar decisiones clínicas. Ante cualquier síntoma o duda, consulte a un profesional de la salud, aunque el resultado sea "Normal".

![Captura de la aplicación web](docs/web_demo.png)

## Resultados

Modelo final: **EfficientNetV2-S** (preentrenado en ImageNet, *fine-tuning* completo, entrada de 384 px), entrenado con el dataset original y un dataset externo de otras cámaras. Medido **una sola vez** en el conjunto de test: 2 205 imágenes que el modelo nunca vio, separadas por paciente. Los números salen de lo que la app realmente responde, con la política de tamizaje:

| Métrica | Valor |
|---|---|
| Exactitud (*accuracy*) | 88,3 % |
| F1 macro | 0,885 |
| Sensibilidad (enfermos detectados como enfermos) | 93,4 % |
| Especificidad (sanos reconocidos como sanos) | 82,0 % |

| Clase | A | C | D | G | H | M | N |
|---|---|---|---|---|---|---|---|
| F1 | 0,97 | 0,98 | 0,90 | 0,88 | 0,79 | 0,87 | 0,81 |

**Por origen de las imágenes:**

| Origen | Imágenes | Exactitud | F1 macro | Verde acierta |
|---|---|---|---|---|
| Original (ODIR, 7 clases) | 1 446 | 86,9 % | 0,886 | 97,9 % |
| Externo (otras cámaras, 4 clases) | 759 | 91,2 % | 0,889 | 98,3 % |

El primer modelo, entrenado solo con el dataset original, se quedaba en 44 % de exactitud con las fotos externas: detectaba solo el 15 % de las retinopatías diabéticas y el 20 % de los glaucomas. Al sumar el dataset externo al entrenamiento, el modelo nuevo es igual o mejor en las dos fuentes. El modelo anterior se conserva como `v2s_original_best.pth`.

**Semáforo de confianza** (probabilidades calibradas con *temperature scaling*, ECE 0,05):

| Nivel | Regla | En test |
|---|---|---|
| 🟢 Alta | confianza ≥ 90 % | 52 % de las imágenes, **acierta el 98,1 %** |
| 🟡 Media | 80 – 90 % | confirmar con un especialista |
| 🟡 Normal | el modelo no ve enfermedad | **nunca se muestra en verde**: se recomienda control anual |
| 🔴 Baja | confianza < 80 % | resultado incierto, consultar a un especialista |

En la app, la pestaña **Métricas del modelo** muestra la matriz de confusión, estos indicadores y el desglose por origen. Se regeneran con `uv run python -m src.model_report --model v2s`.

**Política de tamizaje:** en una prueba de detección temprana, dejar pasar a un enfermo es peor que una falsa alarma. Por eso el modelo exige más evidencia antes de responder "Normal", y "Normal" nunca sale en verde. A cambio hay algunas falsas alarmas más.

### Limitaciones conocidas

- **Atajo de origen (ODIR):** la mayoría de los errores "enfermo → Normal" son imágenes del dataset ODIR, donde las etiquetas vienen del paciente y no siempre del ojo fotografiado. Para corregirlo hacen falta más imágenes normales de otras fuentes.
- **Catarata = imagen sin detalle:** el modelo aprendió que una imagen borrosa o blanca indica catarata. Una foto desenfocada de un ojo sano puede salir "Catarata". La app lo advierte.
- **Retinopatía diabética leve** y una retina normal se parecen mucho a la resolución usada. D y N son las clases con más confusión.
- El filtro de calidad se ajustó con las imágenes de este dataset. Con otras cámaras puede rechazar o aceptar imágenes de forma distinta.
- El dataset mezcla varias fuentes públicas. Con cámaras, hospitales o poblaciones distintas a las del entrenamiento, el rendimiento puede bajar, como pasó con el primer modelo y las fotos externas.
- **Corte del verde:** para este modelo se subió de 88 % a 90 % después de ver la validación, porque con 88 % el verde acertaba 96,1 % en las imágenes originales (la meta era ≥ 97 %). Se eligió con validación y se midió una sola vez en test, pero no estaba fijado antes de entrenar.

## Cómo funciona

```
foto → recorte de la retina → filtro de calidad → EfficientNetV2-S → calibración → política de tamizaje → JSON
       (preprocessing.py)     (quality.py)        (model.py)         (calibrate.py)                       (api/main.py)
                                                       └→ Grad-CAM (gradcam.py) ──────────────────────────┘
```

1. **Preprocesamiento:** recorta el fondo negro, centra la retina, la lleva a 512 px y aplica una máscara circular.
2. **Filtro de calidad:** rechaza imágenes oscuras, planas o que no parecen una retina. Las borrosas o muy claras pasan con advertencia, porque una catarata real también se ve así.
3. **Modelo:** EfficientNetV2-S con transfer learning en dos etapas (primero solo la cabeza nueva, luego toda la red). Se entrenó con pesos por clase, *label smoothing*, *data augmentation*, AMP y *early stopping* según el F1 macro de validación.
4. **Calibración:** *temperature scaling*, ajustado en validación.
5. **Explicabilidad:** Grad-CAM sobre la última capa convolucional.

**Datos:** dataset público de Kaggle ("Eye Diseases Classification"), basado en ODIR-5K y otras fuentes, con 10 449 imágenes en 7 clases. Se le sumó un dataset externo de 4 clases (catarata, retinopatía diabética, glaucoma y normal) tomado con otras cámaras. Quedan 13 671 imágenes, divididas en train/val/test (9 501 / 1 965 / 2 205) **por paciente** (`StratifiedGroupKFold`), con detección de imágenes casi duplicadas para que no haya fugas entre conjuntos (`splits/splits_ext.csv`).

| Clase | A | C | D | G | H | M | N | Total |
|---|---|---|---|---|---|---|---|---|
| Original | 511 | 1 008 | 3 112 | 1 490 | 412 | 586 | 2 997 | 10 116 |
| Externo | — | 803 | 1 092 | 920 | — | — | 740 | 3 555 |
| **Total** | **511** | **1 811** | **4 204** | **2 410** | **412** | **586** | **3 737** | **13 671** |
| Test | 73 | 324 | 718 | 413 | 59 | 84 | 534 | 2 205 |

## Probar la app (sin entrenar nada)

Solo hacen falta **el código** y **el archivo del modelo** (`v2s_best.pth`, 82 MB). No se necesitan los datos, ni GPU, ni Colab: el modelo ya viene entrenado y calibrado, y corre en el procesador.

**1. Instalar [uv](https://docs.astral.sh/uv/)** (una sola vez). uv instala por su cuenta Python y todas las librerías.

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Después de instalar uv, **cierra la terminal y abre una nueva**.

**2. Descargar el código**, de una de estas dos formas:

- Con git: `git clone https://github.com/JuanMillan7818/eye-disease-detection.git`
- Sin git: en GitHub, botón **Code → Download ZIP**, y descomprimir.

**3. Poner el modelo en `models/`.** El modelo **no está en el repositorio** por su tamaño. Descárgalo desde la sección *Releases* o desde la carpeta compartida, y cópialo así:

```
eye-disease-detection/
├── api/
├── models/
│   └── v2s_best.pth      ← aquí (crear la carpeta models si no existe)
├── src/
└── web/
```

Solo hace falta `v2s_best.pth`. Los demás archivos de la carpeta compartida (`.onnx`, `muestras/`) son opcionales.

**4. Instalar las dependencias y arrancar la app**, desde la carpeta del proyecto:

```bash
cd eye-disease-detection
uv sync                          # la primera vez tarda unos minutos (descarga PyTorch)
uv run uvicorn api.main:app
```

Cuando aparezca `Uvicorn running on http://127.0.0.1:8000`, abre **<http://127.0.0.1:8000>** en el navegador. Para detener la app: `Ctrl + C`.

**5. Probar:** arrastra una **foto de fondo de ojo** (JPG o PNG) y pulsa **Analizar**. Si no tienes fotos, usa las de la carpeta `muestras/` que viene con el modelo. Una foto común (una persona, un paisaje) se rechaza a propósito, porque la app solo analiza retinas.

**Problemas frecuentes:**

| Mensaje | Solución |
|---|---|
| `No se encontró el modelo '...\models\v2s_best.pth'` | El archivo no está en `models/`, o tiene otro nombre. Revisa el paso 3. |
| `uv: command not found` / `uv no se reconoce` | Cierra y vuelve a abrir la terminal después de instalar uv. |
| `[WinError 10013]` o `address already in use` | El puerto 8000 está ocupado: `uv run uvicorn api.main:app --port 8001` y abre <http://127.0.0.1:8001>. |
| La página dice "No se pudo conectar con la API" | La terminal con uvicorn se cerró o se detuvo: vuelve a ejecutar el paso 4. |

**Verificar que el modelo llegó completo** (opcional, PowerShell):

```powershell
Get-FileHash models\v2s_best.pth -Algorithm SHA256
# 7D17EA36625C59E31C09A3841052041D164AC0A4AC27691101248534A7534AD7
```

Si el hash no coincide, el archivo está incompleto o es otro: descárgalo de nuevo.

## Uso

### Aplicación web

```bash
uv run uvicorn api.main:app
```

Abre <http://127.0.0.1:8000>, arrastra una foto de fondo de ojo y pulsa **Analizar**.

### API REST

La documentación interactiva está en <http://127.0.0.1:8000/docs>.

```bash
curl -F "file=@foto.jpg" http://127.0.0.1:8000/predict
```

| Endpoint | Descripción |
|---|---|
| `GET /health` | Estado del modelo, dispositivo y clases |
| `POST /predict` | Recibe una imagen (`file`, JPG o PNG, ≤ 10 MB). `?heatmap=false` omite el mapa de calor |

Códigos de respuesta: `200` resultado · `400` no es una imagen · `413` más de 10 MB · `422` la imagen no sirve para el análisis (`detail.issues` explica el motivo).

Variables de entorno opcionales: `MODEL_NAME` (por defecto `v2s`), `MODELS_DIR` (por defecto `models`), `ALLOWED_ORIGINS` (CORS).

### Línea de comandos

```bash
uv run python -m src.predict ruta/a/foto.jpg
uv run python -m src.gradcam ruta/a/foto.jpg      # guarda el mapa de calor en reports/figures/gradcam/
```

### Sin PyTorch (ONNX)

El modelo también está exportado a **ONNX**, un formato estándar que se puede usar desde Python, C#, Java o JavaScript con [ONNX Runtime](https://onnxruntime.ai/). Se distribuye en dos archivos:

- `v2s.onnx`: la red con la calibración incluida.
- `v2s.onnx.json`: las clases, el preprocesamiento y la política del semáforo.

```bash
pip install onnxruntime numpy pillow
python examples/onnx_predict.py v2s.onnx foto.jpg
```

[examples/onnx_predict.py](examples/onnx_predict.py) no depende del resto del proyecto. Se verificó que da la misma respuesta que PyTorch en 100 de 100 imágenes de test. Tiene una diferencia: **no incluye el filtro de calidad**.

Para generar los archivos ONNX desde el checkpoint:

```bash
uv run --group export python -m src.export_onnx
```

## Reproducir el entrenamiento

```bash
uv run python -m src.explorer          # 1. exploración de los datos
uv run python -m src.prepare_data      # 2. preprocesamiento y división por paciente
uv run python -m src.prepare_external  # 3. suma el dataset externo (data/dataseet externo/)
# 4. entrenamiento (mejor en GPU: notebooks/02_training_colab.ipynb, sección 8)
uv run python -m src.train --model v2s --splits splits/splits_ext.csv --tag ext
uv run python -m src.calibrate --model v2s_ext --green 0.90
uv run python -m src.evaluate --split val --models v2s v2s_ext --splits splits/splits_ext.csv
```

El entrenamiento se hizo en Google Colab (GPU L4, ~1,5 h). El dataset va en `data/<clase>/` y el externo en `data/dataseet externo/<clase>/`. El modelo de la app es `v2s_ext_best.pth` copiado como `v2s_best.pth`.

## Tests

```bash
uv run pytest -q
```

Se prueban la API (predicción, rechazo de imágenes inválidas, límites de tamaño) y que la página web se sirva. Los tests que necesitan el modelo se saltan si `models/` no existe.

## Estructura

```
api/main.py            API REST (FastAPI) que además sirve la web
web/                   frontend: index.html, styles.css, app.js, metrics.js, metrics.json
src/
  explorer.py          exploración del dataset
  preprocessing.py     recorte y máscara de la retina
  prepare_data.py      división por paciente y preprocesamiento
  dataset.py           Dataset, transformaciones y pesos por clase
  model.py             EfficientNet-B0 / V2-S con transfer learning
  train.py             entrenamiento en dos etapas
  evaluate.py          métricas, matriz de confusión, TTA
  calibrate.py         temperature scaling y política de tamizaje
  predict.py           predicción por consola
  gradcam.py           mapas de calor Grad-CAM
  quality.py           filtro de calidad de imagen
  source_report.py     aciertos por origen de la imagen
  export_onnx.py       exportación a ONNX
  model_report.py      KPIs y matriz de confusión para la web (web/metrics.json)
  prepare_external.py  división con un dataset externo (otras cámaras)
examples/onnx_predict.py  uso del modelo sin PyTorch
notebooks/             exploración y entrenamiento en Colab
reports/               métricas, predicciones y figuras
tests/                 tests de la API
```

## Tecnologías

PyTorch · torchvision · scikit-learn · FastAPI · ONNX Runtime · Grad-CAM · HTML/CSS/JavaScript · uv · Google Colab
