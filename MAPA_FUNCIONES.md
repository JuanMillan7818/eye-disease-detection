# Mapa de funciones: ¿dónde se hace cada cosa?

Para responder rápido "¿en qué parte del código se hace X?". Los enlaces llevan a la línea exacta (clic en VS Code).

---

## Respuestas rápidas

| Pregunta | Archivo y función |
|---|---|
| ¿Dónde se **crea** el modelo? | [src/model.py:26](src/model.py#L26) `create_model()` |
| ¿Dónde se elige la **arquitectura** (EfficientNetV2-S) y el **tamaño de imagen** (384)? | [src/model.py:21](src/model.py#L21) `MODEL_SPECS` |
| ¿Dónde se usa **transfer learning** (pesos de ImageNet)? | [src/model.py:29](src/model.py#L29), dentro de `create_model()` |
| ¿Dónde se cambia la **última capa** por una de 7 clases? | [src/model.py:31-35](src/model.py#L31-L35) |
| ¿Dónde se **congela** el backbone? | [src/model.py:38](src/model.py#L38) `set_backbone_trainable()` |
| ¿Dónde se **guarda** y se **carga** el modelo (`.pth`)? | [src/model.py:50](src/model.py#L50) `save_checkpoint()` y [src/model.py:66](src/model.py#L66) `load_checkpoint()` |
| ¿Dónde están los **hiperparámetros** (lr, épocas, batch...)? | [src/train.py:25](src/train.py#L25) `parse_args()` |
| ¿Dónde está el **ciclo de entrenamiento**? | [src/train.py:143](src/train.py#L143) `train()` |
| ¿Dónde se hace el **forward / backward** de cada lote? | [src/train.py:82](src/train.py#L82) `run_epoch()` |
| ¿Dónde se define la **función de pérdida**? | [src/train.py:161-165](src/train.py#L161-L165) |
| ¿Dónde está el **optimizador** (AdamW) y el **scheduler**? | [src/train.py:178-179](src/train.py#L178-L179) |
| ¿Dónde están las **dos etapas** (cabeza → red completa)? | [src/train.py:169](src/train.py#L169) `stages` |
| ¿Dónde está el **early stopping**? | [src/train.py:202-213](src/train.py#L202-L213) |
| ¿Dónde se usa **AMP** (float16 en GPU)? | [src/train.py:96](src/train.py#L96) `autocast` y [src/train.py:166](src/train.py#L166) `GradScaler` |
| ¿Dónde se fija la **semilla** (resultados repetibles)? | [src/train.py:71](src/train.py#L71) `set_seed()` |
| ¿Dónde está el **data augmentation**? | [src/dataset.py:47](src/dataset.py#L47) `build_transforms()` |
| ¿Dónde se **normaliza** la imagen (media y desviación de ImageNet)? | [src/dataset.py:18-19](src/dataset.py#L18-L19) y `build_transforms()` |
| ¿Dónde se calculan los **pesos por clase** (desbalance)? | [src/dataset.py:62](src/dataset.py#L62) `compute_class_weights()` |
| ¿Dónde se **cargan las imágenes** en lotes? | [src/dataset.py:69](src/dataset.py#L69) `create_dataloaders()` |
| ¿Dónde se **recorta la retina** y se pone la máscara? | [src/preprocessing.py:34](src/preprocessing.py#L34) `preprocess_fundus()` |
| ¿Dónde se ajusta la **temperatura** (calibración)? | [src/calibrate.py:94](src/calibrate.py#L94) `fit_temperature()` |
| ¿Dónde se calcula la **confianza calibrada**? | [src/calibrate.py:46](src/calibrate.py#L46) `calibrated_confidence()` |
| ¿Dónde se decide el **color del semáforo**? | [src/calibrate.py:58](src/calibrate.py#L58) `confidence_level()` |
| ¿Dónde está la **política de tamizaje** (más evidencia para decir "Normal")? | [src/calibrate.py:72](src/calibrate.py#L72) `decide()` |
| ¿Dónde están los **cortes** del semáforo (88 %, 80 %, factor 0,25)? | [src/calibrate.py:17-20](src/calibrate.py#L17-L20) |
| ¿Dónde se mide el **ECE** (error de calibración)? | [src/calibrate.py:104](src/calibrate.py#L104) `expected_calibration_error()` |
| ¿Dónde se calcula la **matriz de confusión**? | [src/evaluate.py:57](src/evaluate.py#L57) `plot_confusion_matrix()` y [src/model_report.py](src/model_report.py) `build_report()` (versión web) |
| ¿Dónde está el **TTA** (promedio con la imagen volteada)? | [src/evaluate.py:37](src/evaluate.py#L37) `predict()` y [src/evaluate.py:152](src/evaluate.py#L152) |
| ¿Dónde se hace el **Grad-CAM**? | [src/gradcam.py:47](src/gradcam.py#L47) clase `GradCAM` y [src/gradcam.py:97](src/gradcam.py#L97) (última capa convolucional) |
| ¿Dónde se predice **una imagen** desde la consola? | [src/predict.py:35](src/predict.py#L35) `predict_image()` |
| ¿Dónde se **carga el modelo** en la API? | [api/main.py:65](api/main.py#L65) `lifespan()` |
| ¿Dónde está el endpoint **/predict**? | [api/main.py:108](api/main.py#L108) `predict()` |
| ¿Dónde se **rechazan** las fotos que no son retinas? | [src/quality.py:75](src/quality.py#L75) `assess_quality()` |

---

## 1. El modelo: [src/model.py](src/model.py)

| Línea | Función | Qué hace |
|---|---|---|
| [21](src/model.py#L21) | `MODEL_SPECS` | Tabla de modelos disponibles. Para cada uno: la función de torchvision que lo construye, sus pesos de ImageNet y su tamaño de entrada. `b0` → EfficientNet-B0, 224 px. `v2s` → EfficientNetV2-S, 384 px |
| [26](src/model.py#L26) | `create_model(name, num_classes, pretrained, dropout)` | **Crea la instancia del modelo.** Descarga EfficientNet con los pesos de ImageNet (*transfer learning*) y cambia el clasificador original de 1000 clases por `Dropout(0.3)` + `Linear(1280 → 7)` |
| [38](src/model.py#L38) | `set_backbone_trainable(model, trainable)` | Congela (`False`) o descongela (`True`) el extractor de características (`model.features`) con `requires_grad` |
| [44](src/model.py#L44) | `count_parameters(model)` | Cuenta los parámetros totales y los entrenables (V2-S: 20,2 M) |
| [50](src/model.py#L50) | `save_checkpoint(...)` | Guarda el `.pth`: los pesos (`state_dict`) **más** todo lo necesario para usarlo después (nombre, clases, tamaño de imagen, pérdida, división, métricas) |
| [66](src/model.py#L66) | `load_checkpoint(path, device)` | Reconstruye el modelo desde el `.pth`: llama a `create_model()` sin pesos de ImageNet, carga el `state_dict` y lo pone en modo `eval()` |

## 2. Los datos: [src/dataset.py](src/dataset.py)

| Línea | Función | Qué hace |
|---|---|---|
| [22](src/dataset.py#L22) | `read_splits(splits_file)` | Lee el CSV de la división (`splits.csv` o `splits_ext.csv`) |
| [26](src/dataset.py#L26) | `class FundusDataset` | Dataset de PyTorch. `__getitem__` abre la imagen, le aplica las transformaciones y devuelve `(tensor, índice_de_clase)` |
| [47](src/dataset.py#L47) | `build_transforms(image_size, train)` | **Data augmentation** (solo en train): recorte aleatorio de 85–100 %, volteo horizontal, rotación de ±15° y cambios de brillo, contraste y saturación. En val y test solo redimensiona. Siempre normaliza con la media y desviación de ImageNet |
| [62](src/dataset.py#L62) | `compute_class_weights(records, classes)` | **Pesos por clase:** `total / (7 × cantidad_de_la_clase)`. Las clases raras (H) pesan más en la pérdida |
| [69](src/dataset.py#L69) | `create_dataloaders(...)` | Arma los 3 `DataLoader` (train, val y test). Solo train se baraja (`shuffle`). Devuelve también las clases y los pesos, calculados **solo con train** |

## 3. El entrenamiento: [src/train.py](src/train.py)

| Línea | Función | Qué hace |
|---|---|---|
| [25](src/train.py#L25) | `parse_args()` | **Todos los hiperparámetros** con su valor por defecto: épocas, lr, weight decay, label smoothing, batch, patience, división, etc. |
| [51](src/train.py#L51) | `class WeightedSmoothCE` | Pérdida alternativa (`--loss weighted-smooth`), usada en el experimento `v2s_ws`. **No** la usa el modelo final |
| [71](src/train.py#L71) | `set_seed(seed)` | Fija la semilla 42 en random, numpy y torch, para que los resultados se puedan repetir |
| [79](src/train.py#L79) | `get_device()` | Usa la GPU (`cuda`) si hay, o la CPU si no |
| [82](src/train.py#L82) | `run_epoch(model, loader, criterion, device, optimizer, scaler)` | **Una pasada por todos los datos.** Con `optimizer`, entrena: forward → pérdida → `backward()` → `optimizer.step()`. Sin `optimizer`, solo evalúa. Devuelve la pérdida, la exactitud y el F1 macro |
| [118](src/train.py#L118) | `save_history(...)` | Guarda `models/<modelo>_history.csv` con las métricas de cada época |
| [126](src/train.py#L126) | `plot_history(...)` | Dibuja las curvas de pérdida y F1 (train y val) |
| [143](src/train.py#L143) | `train(args)` | **El entrenamiento completo**, paso a paso abajo |

**Qué hace `train()`, en orden:**

1. [L144](src/train.py#L144): fija la semilla y elige el dispositivo.
2. [L153](src/train.py#L153): carga los datos con `create_dataloaders()`.
3. [L160](src/train.py#L160): **crea el modelo** con `create_model()`.
4. [L161-165](src/train.py#L161-L165): **la pérdida**, `nn.CrossEntropyLoss(weight=pesos_por_clase, label_smoothing=0.1)`.
5. [L166](src/train.py#L166): `GradScaler` para AMP, solo en GPU.
6. [L169](src/train.py#L169): **las dos etapas**.
   - `1-cabeza`: 3 épocas, lr 1e-3, backbone congelado.
   - `2-completo`: 25 épocas, lr 2e-4, toda la red.
7. [L176-179](src/train.py#L176-L179): en cada etapa, congela o descongela, y crea **AdamW** (weight decay 1e-4) y **CosineAnnealingLR**.
8. [L188-190](src/train.py#L188-L190): en cada época, entrena con train, evalúa con val y baja el lr.
9. [L202-213](src/train.py#L202-L213): **early stopping**. Si el F1 de val mejora, guarda el `.pth`. Si no mejora en `patience` épocas, para.
10. [L215-216](src/train.py#L215-L216): guarda el historial y las curvas.

## 4. La calibración y el semáforo: [src/calibrate.py](src/calibrate.py)

| Línea | Función | Qué hace |
|---|---|---|
| [17-20](src/calibrate.py#L17-L20) | Constantes | `HIGH_CONFIDENCE = 0.88` (verde por defecto), `LOW_CONFIDENCE = 0.80` (rojo por debajo), `NORMAL_FACTOR = 0.25` (para decir "Normal" exige 4 veces más evidencia) |
| [46](src/calibrate.py#L46) | `calibrated_confidence(probs, class_weights, temperature)` | **Corrige la confianza.** Quita el sesgo de los pesos por clase y divide entre la temperatura: `softmax(log(p / pesos) / T)`. No cambia la clase elegida, solo el % que se muestra |
| [58](src/calibrate.py#L58) | `confidence_level(confidence, is_normal, green)` | **Color del semáforo.** Si la confianza es < 0,80 → rojo. Si la respuesta es "Normal" → amarillo (nunca verde). Si es ≥ el corte del verde → verde. Si no → amarillo |
| [67](src/calibrate.py#L67) | `green_cut(calibration)` | Corte del verde de **este** modelo (`high_confidence` guardado en el `.pth`; el modelo final usa 0,90) |
| [72](src/calibrate.py#L72) | `decide(probs, calibration, classes)` | **La política de tamizaje que usa la app:** multiplica la probabilidad de Normal por 0,25, elige la clase más alta, calcula la confianza calibrada y le asigna el color del semáforo |
| [88](src/calibrate.py#L88) | `correctness_nll(confidence, correct)` | Mide qué tan "sorprendido" queda el modelo con sus aciertos y errores. Es lo que minimiza la temperatura |
| [94](src/calibrate.py#L94) | `fit_temperature(probs, labels, class_weights)` | **Temperature scaling:** prueba 271 temperaturas entre 0,3 y 3,0 en val y se queda con la de menor `correctness_nll` (modelo final: T = 1,03) |
| [104](src/calibrate.py#L104) | `expected_calibration_error(confidence, correct)` | **ECE:** agrupa las predicciones por confianza en 10 rangos y promedia la diferencia entre la confianza y el acierto real |
| [115](src/calibrate.py#L115) | `plot_reliability(...)` | Diagrama de confiabilidad: un modelo calibrado queda sobre la diagonal |
| [135](src/calibrate.py#L135) | `print_policy_report(...)` | Reporte de tamizaje: enfermos que se escapan, falsas alarmas y cuánto acierta cada color |
| [182](src/calibrate.py#L182) | `main()` | Predice val → `fit_temperature()` → calcula el ECE antes y después → aplica `decide()` → **guarda la calibración dentro del `.pth`** ([L232-233](src/calibrate.py#L232-L233)) |

## 5. La evaluación: [src/evaluate.py](src/evaluate.py)

| Línea | Función | Qué hace |
|---|---|---|
| [37](src/evaluate.py#L37) | `predict(model, loader, device)` | Predice cada imagen **y también su versión volteada** (para el TTA) |
| [50](src/evaluate.py#L50) | `summarize(labels, probs)` | Exactitud y F1 macro |
| [57](src/evaluate.py#L57) | `plot_confusion_matrix(...)` | Dibuja la matriz: filas = real, columnas = predicción, color = % de la fila |
| [80](src/evaluate.py#L80) | `save_predictions(...)` | CSV con una fila por imagen y todas sus probabilidades |
| [92](src/evaluate.py#L92) | `plot_top_errors(...)` | Las imágenes con los errores más seguros |
| [138](src/evaluate.py#L138) | `evaluate_model(name, args, device)` | Todo lo anterior para un modelo: el reporte por clase (`classification_report`), los resultados por origen y la comparación sin TTA y con TTA ([L152](src/evaluate.py#L152)) |

## 6. Del archivo a la respuesta (inferencia)

| Archivo | Función | Qué hace |
|---|---|---|
| [src/preprocessing.py:7](src/preprocessing.py#L7) | `crop_to_fundus()` | Quita el fondo negro alrededor de la retina |
| [src/preprocessing.py:17](src/preprocessing.py#L17) | `center_crop_square()` | Recorta el cuadrado central más grande |
| [src/preprocessing.py:25](src/preprocessing.py#L25) | `apply_circular_mask()` | Pone en negro todo lo que queda fuera del círculo |
| [src/preprocessing.py:34](src/preprocessing.py#L34) | `preprocess_fundus()` | Los 3 pasos anteriores y redimensiona a 512 px. **Se usa igual al entrenar y al predecir** |
| [src/quality.py:75](src/quality.py#L75) | `assess_quality()` | Rechaza las imágenes oscuras, planas o que no son retinas. Las borrosas o muy claras pasan con advertencia |
| [src/predict.py:35](src/predict.py#L35) | `predict_image()` | Para la consola: preprocesa → transforma → modelo → softmax |
| [src/gradcam.py:47](src/gradcam.py#L47) | `class GradCAM` | Usa *hooks* para guardar las activaciones y los gradientes de la última capa convolucional. Con ellos calcula el mapa de calor |
| [src/gradcam.py:87](src/gradcam.py#L87) | `class Explainer` | Carga el modelo **una vez** y, para cada imagen, devuelve la predicción, la confianza, el semáforo y el mapa de calor (usa `decide()`) |
| [api/main.py:65](api/main.py#L65) | `lifespan()` | Cuando arranca la API, carga el modelo (`Explainer`) una sola vez |
| [api/main.py:108](api/main.py#L108) | `predict()` (`POST /predict`) | Lee la imagen → `preprocess_fundus()` → `assess_quality()` → `explain_image()` → responde en JSON |

---

## El recorrido completo de una foto en la app

```
foto subida
  → api/main.py: predict()            valida que sea una imagen de ≤ 10 MB
  → preprocess_fundus()               recorta la retina, máscara, 512 px
  → assess_quality()                  ¿es una retina usable? si no → error 422
  → Explainer.explain_image()
      → build_transforms(train=False) 384 px + normalización ImageNet
      → GradCAM(...)                  forward del modelo + mapa de calor
      → decide()                      política de tamizaje + calibración + semáforo
  → JSON: clase, confianza, color, probabilidades, mapa de calor
```

## El recorrido del entrenamiento

```
create_dataloaders()      datos + augmentation + pesos por clase
  → create_model()        EfficientNetV2-S de ImageNet + cabeza nueva de 7 clases
  → etapa 1: set_backbone_trainable(False) → run_epoch() × 3     (solo la cabeza)
  → etapa 2: set_backbone_trainable(True)  → run_epoch() × 25    (toda la red)
      cada época: si el F1 de val mejora → save_checkpoint()
  → calibrate: fit_temperature() → la calibración se guarda en el .pth
  → evaluate:  predict() en test → matriz de confusión y métricas
```
