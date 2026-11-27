# Modelo final: lo que usa la app

**Modelo:** EfficientNetV2-S a 384 px, entrenado con las fotos originales (ODIR, 7 clases) y un dataset externo de otras cámaras (4 clases). Archivo: `models/v2s_best.pth`.

**Medición:** test, **una sola vez**, con 2 205 imágenes que el modelo nunca vio (separadas por paciente). Las decisiones aplican la política de tamizaje, igual que la app.

| Indicador | Valor |
|---|---|
| Exactitud | 88,3 % (1 948 de 2 205) |
| F1 macro | 0,885 |
| Sensibilidad (enfermos detectados) | 93,4 % |
| Especificidad (sanos reconocidos) | 82,0 % |
| 🟢 Verde: cuánto acierta | 98,1 % (en el 52 % de las imágenes) |

## Imágenes por enfermedad

| Clase | Total | Train | Val | Test |
|---|---|---|---|---|
| A Degeneración macular | 511 | 365 | 73 | 73 |
| C Catarata | 1 811 | 1 225 | 262 | 324 |
| D Retinopatía diabética | 4 204 | 2 877 | 609 | 718 |
| G Glaucoma | 2 410 | 1 658 | 339 | 413 |
| H Retinopatía hipertensiva | 412 | 295 | 58 | 59 |
| M Miopía | 586 | 418 | 84 | 84 |
| N Normal | 3 737 | 2 663 | 540 | 534 |
| **Total** | **13 671** | **9 501** | **1 965** | **2 205** |

A, H y M solo existen en las fotos originales; C, D, G y N también tienen fotos externas.

## Archivos

| Archivo | Qué muestra |
|---|---|
| `1_matriz_confusion_test.png` | **Principal.** Matriz de todo el test |
| `2_matriz_confusion_externo.png` | Solo las fotos de otras cámaras (759) |
| `3_matriz_confusion_original.png` | Solo las fotos originales (1 446) |
| `4_matriz_confusion_modelo_anterior.png` | El modelo anterior en el mismo test, para comparar |
| `5_curvas_entrenamiento.png` | Pérdida y F1 en cada época (train y val) |
| `6_ejemplos_de_errores.png` | Imágenes en las que el modelo se equivocó |
| `7_calibracion.png` | Si la confianza que muestra coincide con lo que acierta |
| `metricas.json` | Todos los números (los mismos de la pestaña "Métricas del modelo") |

## Cómo leer la matriz de confusión

- **Filas:** la enfermedad **real**. **Columnas:** lo que **respondió** el modelo.
- **Cada celda** muestra cuántas imágenes y qué % de su fila representan. Cada fila suma 100 %.
- **La diagonal** (de arriba a la izquierda a abajo a la derecha) son los **aciertos**. Más oscura = mejor.
- **Todo lo que está fuera de la diagonal** es un error. Su fila dice qué era y su columna qué respondió el modelo.
- **La columna N (Normal)** son los errores graves: un enfermo al que se le dijo "sano".
- **La fila N** son falsas alarmas: un sano al que se le dijo "enfermo". Es un error menos grave en un tamizaje.

**Lo que dice esta matriz:**
- A (100 %) y C (99 %) casi no fallan.
- D (89 %) y G (87 %) fallan sobre todo hacia **Normal**: son casos leves, que se parecen a una retina sana.
- H (76 %) es la clase más difícil: tiene pocas imágenes (59) y se confunde con Normal.
- N (82 %): los sanos que fallan salen como D o G. Es el precio de exigir más evidencia antes de decir "Normal".

En la matriz externa, la fila H sale vacía porque ese dataset no tiene esa enfermedad. La columna está porque el modelo respondió H 3 veces.

**Comparación con el modelo anterior** (`4_...`): en las fotos externas, el anterior respondía Normal a la mayoría de las retinopatías y Catarata a la mayoría de los glaucomas.

## Regenerar

```powershell
uv run python -m src.evaluate --split test --models v2s_original v2s --splits splits/splits_ext.csv
uv run python -m src.model_report --model v2s
uv run python -m src.final_package
```
