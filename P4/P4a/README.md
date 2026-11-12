# Práctica 4a de Visión por Computador — Detección y Seguimiento de Vehículos, Personas y Matrículas

## Datos de la práctica

- **Asignatura:** Visión por Computador
- **Práctica:** P4a (Detección y seguimiento de objetos)
- **Autor:** Félix Miguel Velásquez
- **Cuaderno principal (YOLOv11):** [`VC_P4_FMV.ipynb`](./VC_P4_FMV.ipynb)
- **Cuaderno alternativo (RT-DETRv2):** [`VC_P4_FMV_RTDETR.ipynb`](./VC_P4_FMV_RTDETR.ipynb)
- **Resultados generados (vídeos y CSV):** [`results/`](./results/)

---

## Descripción

Esta primera parte de la Práctica 4 aborda el diseño e implementación de un prototipo completo de procesado de vídeo para la **detección, clasificación, seguimiento (*tracking*) y conteo de vehículos, peatones y matrículas**, integrando tanto arquitecturas convolucionales de la familia YOLOv11 (Ultralytics) como arquitecturas basadas en Transformers (RT-DETRv2).

En concreto, el trabajo desarrollado cubre los siguientes aspectos:

1. Verificación, limpieza y particionado de un conjunto de datos de matrículas (2406 imágenes y 4371 matrículas anotadas), adaptándolo automáticamente a los formatos YOLO y COCO.
2. Entrenamiento y evaluación de un detector de matrículas propio mediante *transfer learning* en dos familias de arquitecturas distintas (YOLOv11 y RT-DETRv2), analizando el impacto de la resolución de entrada (640 vs 960 píxeles), el tamaño del lote (*batch size*) y la capacidad del modelo (*nano* vs *small*).
3. Procesamiento de vídeo con detección dual y seguimiento multi-objeto, combinando un detector generalista para vehículos y personas (*person, car, motorbike, bus*) con el detector especialista de matrículas (*license_plate*).
4. Gestión de oclusiones de peatones (`PedestrianOcclusionManager`) mediante correlación de histogramas HSV y proximidad espacial para mantener la identidad de las personas cuando un vehículo cruza por delante.
5. Asociación geométrica vehículo-matrícula, estabilización de clase por moda, censura mediante desenfoque gaussiano, superposición de interfaz gráfica (HUD) en el vídeo de salida y exportación detallada fotograma a fotograma en formato CSV.

---

## Requisitos y entorno de ejecución

Las pruebas y entrenamientos se han ejecutado sobre el siguiente entorno hardware y software:

- **CPU:** Intel Core i7-10700 (16 hilos)
- **RAM:** 31.25 GB
- **GPU:** NVIDIA GeForce RTX 3060 Ti (8 GB VRAM)
- **Sistema Operativo:** Linux (Kernel 7.3.3-1-cachyos)
- **Entorno Python:** Python `3.11.8`, PyTorch `2.6.0+cu124`, OpenCV `4.10.0`, Ultralytics `8.4.202`

Para instalar las dependencias necesarias:

```bash
pip install -r requirements.txt
```

Para la compresión final de los vídeos anotados con el códec H.264 (`libx264`) se requiere tener `ffmpeg` instalado en el sistema.

---

## 1. Conjunto de datos y preprocesamiento

Los modelos preentrenados en el conjunto de datos COCO incluyen clases para personas y distintos medios de transporte, pero carecen de una clase específica para matrículas. Por ello, se parte de un conjunto de datos colaborativo compuesto por imágenes de vehículos con matrículas españolas y sus correspondientes ficheros JSON de anotación.

### 1.1. Verificación de integridad y limpieza (`LicensePlateDatasetLoader`)

En ambos cuadernos se implementa la clase `LicensePlateDatasetLoader`, encargada de inspeccionar el directorio del conjunto de datos, emparejar cada imagen `.jpg`/`.jpeg` con su archivo `.json` homónimo y aplicar reglas de limpieza de datos:

- **Normalización automática:** si alguna anotación contiene coordenadas expresadas en píxeles absolutos (`> 1.05`), se divide automáticamente por el ancho y alto reales de la imagen (`img_w`, `img_h`).
- **Recorte a límites válidos (*clipping*):** se garantiza que todas las cajas delimitadoras (`[cx, cy, w, h]`) queden contenidas dentro del rango normalizado `[0.0, 1.0]`.
- **Filtrado de ruido geométrico:** se descartan anotaciones distorsionadas o excesivamente verticales con una relación de aspecto real `w_px / h_px < 0.95`, así como cajas minúsculas consideradas ruido (`h_px < 6` o `w_px < 12` píxeles).

El informe de integridad obtenido sobre el dataset refleja una correspondencia perfecta 1:1:

| Métrica del conjunto de datos | Valor |
| --- | ---: |
| Imágenes totales encontradas | 2406 |
| Anotaciones JSON totales | 2406 |
| Muestras emparejadas válidas | 2406 |
| Imágenes sin anotación o corruptas | 0 |
| Total de cajas delimitadoras (*bounding boxes*) válidas | 4371 |

A continuación se muestra una cuadrícula con muestras aleatorias del conjunto de datos y sus cajas anotadas en verde (`SEED = 42`):

![Muestras anotadas del conjunto de datos de matrículas](./images/docs/plate_dataset.png)

### 1.2. División en entrenamiento, validación y test

Utilizando una semilla fija (`SEED = 42`) para garantizar la reproducibilidad entre todos los experimentos y arquitecturas, las 2406 imágenes se dividen de forma aleatoria con una proporción **75 % / 15 % / 10 %**:

| Subconjunto (*Split*) | Proporción | Imágenes | Matrículas anotadas |
| --- | ---: | ---: | ---: |
| **Train** (Entrenamiento) | 75.00 % | 1804 | 3243 |
| **Val** (Validación) | 15.00 % | 360 | 681 |
| **Test** (Evaluación final) | 10.00 % | 242 | 447 |
| **Total** | 100.00 % | 2406 | 4371 |

- **Preparación para YOLO (`prepare_yolo_dataset`):** crea la jerarquía `images/{train,val,test}` y `labels/{train,val,test}`, empleando enlaces simbólicos para no duplicar las imágenes en disco, convirtiendo cada etiqueta al formato `<class_id> <cx> <cy> <w> <h>` con `class_id = 0` (`license_plate`) y generando el fichero `data.yaml`.
- **Preparación para RT-DETR (`prepare_rtdetr_dataset`):** pre-reescala las imágenes en paralelo usando hilos a la resolución objetivo manteniendo una proporción `16:9` (por ejemplo, `960x540` para `IMGSZ = 960`) para evitar cuellos de botella de lectura/memoria en imágenes de alta resolución, y exporta los ficheros `instances_{train,val,test}.json` con coordenadas absolutas `[x1, y1, width, height]` en formato estándar COCO.

---

## 2. Entrenamiento y evaluación con YOLOv11

### 2.1. Configuración y experimentos realizados

Las matrículas suelen ocupar una fracción muy pequeña del área total del fotograma, especialmente cuando los vehículos se encuentran alejados. Para maximizar su detección con **YOLOv11**, se diseñaron varios experimentos variando el tamaño del modelo (`YOLOv11n.pt` frente a `YOLOv11s.pt`), la resolución de entrada (`imgsz = 640` frente a `imgsz = 960`) y el tamaño de lote (`batch = 8` frente a `batch = 16`).

En la configuración final seleccionada (`plate_detector_960_BATCH_16_small`) se emplearon los siguientes hiperparámetros clave:

```python
EPOCHS = 120
BATCH_SIZE = 16
IMGSZ = 960
PATIENCE = 30
```

- **Entrenamiento rectangular (`rect=True`) y precisión mixta (`amp=True`):** reduce el relleno innecesario en los bordes de las imágenes panorámicas y acelera el cálculo en GPU evitando desbordamientos numéricos.
- **Ponderación de la función de pérdida:** se incrementó la ganancia de regresión de cajas (`box = 8.5`, frente al valor por defecto `7.5`) junto con `dfl = 1.5` y `cls = 1.5` para forzar un ajuste más ceñido a los bordes de la matrícula.
- **Aumento de datos (*Data Augmentation*):** variaciones fotométricas en espacio HSV (`hsv_s = 0.6`, `hsv_v = 0.4`), rotaciones suaves (`degrees = 5.0`), traslación (`translate = 0.1`), escalado (`scale = 0.5`) y `mosaic = 0.5` desactivado durante las últimas 20 épocas (`close_mosaic = 20`) para estabilizar la convergencia final.

La siguiente tabla resume los resultados obtenidos en validación a lo largo de las distintas configuraciones exploradas en [`runs/detect/runs_plate/`](./runs/detect/runs_plate/):

| Experimento YOLOv11 | Modelo base | Resolución (`imgsz`) | Batch | Épocas (Mejor) | Precision (Val) | Recall (Val) | mAP@50 (Val) | mAP@50-95 (Val) |
| --- | --- | :---: | :---: | :---: | ---: | ---: | ---: | ---: |
| plate_detector_640_small | YOLOv11s.pt | 640 | 16 | 105 (75) | 87.03 % | 69.02 % | 71.65 % | 37.86 % |
| plate_detector_640_BATCH_8_small | YOLOv11s.pt | 640 | 8 | 120 (100) | 87.01 % | 77.93 % | 80.77 % | 45.44 % |
| plate_detector_960_BATCH_8_nano | YOLOv11n.pt | 960 | 8 | 94 (64) | 86.44 % | 80.54 % | 81.32 % | 42.91 % |
| plate_detector_960_BATCH_16_small | YOLOv11s.pt | 960 | 16 | 120 (120) | 85.32 % | 83.45 % | 82.97 % | 44.24 % |

Aunque el modelo a `640` con *batch* `16` obtiene una métrica alta sobre recortes estáticos del conjunto de validación, el modelo **`plate_detector_960_BATCH_16_small`** demostró un comportamiento más sólido al generalizar sobre vídeo completo en alta definición (`1920x1080`) gracias a su mayor resolución de entrada (`960x960`) y a la estabilidad aportada por el `batch = 16` frente a los entrenamientos con `batch = 8` o arquitectura *nano*.

### 2.2. Análisis de las gráficas de entrenamiento (plate_detector_960_BATCH_16_small)

![Evolución de pérdidas y métricas durante el entrenamiento de YOLOv11s (960, batch 16)](./runs/detect/runs_plate/plate_detector_960_BATCH_16_small/results.png)

Analizando las curvas de evolución del entrenamiento:

- **Funciones de pérdida (`box_loss`, `cls_loss`, `dfl_loss`):** tanto en entrenamiento como en validación muestran un descenso constante y suave a lo largo de las 120 épocas, sin síntomas de *overfitting*. En particular, `val/box_loss` desciende desde ~2.8 hasta estabilizarse en torno a 1.68, y `val/dfl_loss` baja de 1.85 a ~1.1, con un pequeño escalón de mejora adicional a partir de la época 100 cuando se desactiva el aumento por mosaico (`close_mosaic = 20`).
- **Evolución de métricas (`precision`, `recall`, `mAP50`, `mAP50-95`):** las curvas crecen rápidamente durante las primeras 35 épocas y continúan mejorando de forma progresiva hasta alcanzar el máximo en la época 120 (`mAP@50 = 81.22 %` en validación).

### 2.3. Evaluación en el conjunto de Test y predicciones visuales (plate_detector_960_BATCH_16_small)

Al evaluar los mejores pesos (`best.pt`) sobre las 242 imágenes del conjunto de test (447 matrículas) con un umbral IoU de 0.55 para suprimir detecciones duplicadas en objetos pequeños, se obtienen las siguientes métricas finales:

| Métrica (Conjunto de Test — YOLOv11s) | Valor |
| --- | ---: |
| Precision (P) | 85.29 % (0.8529) |
| Recall (R) | 83.45 % (0.8345) |
| mAP@50 | 83.11 % (0.8311) |
| mAP@50-95 | 44.36 % (0.4436) |

| Curva Precision-Recall (Test) | Matriz de confusión normalizada (Test) |
| :---: | :---: |
| ![Curva PR en Test de YOLOv11s](./runs/detect/runs_plate/plate_detector_960_BATCH_16_small/val-3/BoxPR_curve.png) | ![Matriz de confusión normalizada en Test de YOLOv11s](./runs/detect/runs_plate/plate_detector_960_BATCH_16_small/val-3/confusion_matrix_normalized.png) |

La curva *Precision-Recall* sobre el conjunto de test alcanza un área bajo la curva de **0.831 mAP50**, manteniendo una precisión superior al 85 % para niveles de *recall* de hasta el 80 %. Asimismo, la matriz de confusión normalizada confirma que el **92 %** de las matrículas reales son detectadas correctamente frente a tan solo un **8 %** de omisiones con el fondo.

A continuación se muestran las predicciones del modelo sobre 9 imágenes aleatorias del conjunto de test (`conf_threshold = 0.40`), indicando en el título de cada imagen las matrículas detectadas (`detected`), las válidas tras el filtrado del cargador (`total`) y las anotadas originalmente en bruto (`real`):

![Predicciones de YOLOv11s sobre muestras del conjunto de test](./images/docs/yolo_detections_960_16.png)

Se observa cómo el modelo localiza con alta precisión matrículas frontales y traseras en condiciones variadas de iluminación (día, sombra y escena nocturna con reflejos en última imagen), así como matrículas parcialmente ocluidas tras una valla metálica (quinta imagen). Incluso en casos como en la cuarta o última imagen, donde el cargador había descartado la anotación original por su tamaño/relación de aspecto (total: 0 | real: 1), el modelo entrenado logra detectar la matrícula correctamente (`detected: 1`).

### 2.4 Evaluación de detecciones en entorno real (vídeo de demostración)

Resultados de detecciones de `plate_detector_960_BATCH_16_small`:
```
------------------------------------------------------------
 TOTAL COUNT OF CONFIRMED UNIQUE OBJECTS:
------------------------------------------------------------
  * person         : 27
  * car            : 166
  * motorbike      : 3
  * bus            : 1
  * license_plate  : 46
------------------------------------------------------------
 RAW DETECTIONS PER FRAME:
  * person         : 3047 total detections
  * car            : 17813 total detections
  * motorbike      : 79 total detections
  * bus            : 217 total detections
  * license_plate  : 3936 total detections
============================================================
```

Resultados de detecciones de `plate_detector_640_small`:
```
------------------------------------------------------------
 TOTAL COUNT OF CONFIRMED UNIQUE OBJECTS:
------------------------------------------------------------
  * person         : 28
  * car            : 169
  * motorbike      : 3
  * bus            : 1
  * license_plate  : 45
------------------------------------------------------------
 RAW DETECTIONS PER FRAME:
  * person         : 3029 total detections
  * car            : 17874 total detections
  * motorbike      : 75 total detections
  * bus            : 217 total detections
  * license_plate  : 3921 total detections
============================================================
```

---

## 3. Alternativa basada en Transformers: RT-DETRv2 (Extra)

Como alternativa a la familia YOLO (cuya licencia AGPL-3.0 restringe su integración directa en proyectos cerrados), en el cuaderno [`VC_P4_FMV_RTDETR.ipynb`](./VC_P4_FMV_RTDETR.ipynb) se ha implementado el flujo completo de entrenamiento, evaluación y *tracking* utilizando **RT-DETRv2** (*Real-Time DEtection TRansformer*).

### 3.1. Configuración y entrenamiento de RT-DETRv2-S

Se seleccionó la variante **RT-DETRv2-S** con *backbone* **PResNet18** (`rtdetrv2_r18vd_120e_coco_rerun_48.1.pth`, ~20.08M parámetros entrenables), adaptando su cabezal de clasificación a `num_classes = 1` (`license_plate`) y entrenando mediante el criterio `RTDETRCriterionv2`, que combina:

- **Varifocal Loss (`loss_vfl`):** con peso `1.5` ($\alpha = 0.75, \gamma = 2.0$).
- **Pérdida de regresión L1 (`loss_bbox`):** con ganancia `8.5`.
- **Pérdida GIoU (`loss_giou`):** con peso `2.0`.
- **Emparejamiento bipartito (`HungarianMatcher`):** asigna predicciones a objetos reales de forma única sin necesidad de etapa de *Non-Maximum Suppression* (NMS).

Se realizaron dos entrenamientos en [`runs/detect/runs_plate/`](./runs/detect/runs_plate/) para analizar el efecto de la resolución espacial y el número de épocas:

| Experimento RT-DETRv2 | Modelo | Resolución | Batch | Épocas (Mejor) | mAP@50 (Val) | mAP@50-95 (Val) | AR@100 (Val) |
| --- | --- | :---: | :---: | :---: | ---: | ---: | ---: |
| `plate_detector_rtdetr_v2_s_640_BATCH_16` | RT-DETRv2-S | `640x640` | `16` | `60 (55)` | `71.82 %` | `37.96 %` | `48.63 %` |
| **`plate_detector_rtdetr_v2_s_960_BATCH_8`** | **RT-DETRv2-S** | **`960x960`** | **`8`** | **`120 (118)`** | **`85.03 %`** | **`51.30 %`** | **`63.17 %`** |

El salto de `640x640` a `960x960` con 120 épocas supone una mejora sustancial de **+13.21 puntos en `mAP@50`** y **+13.34 puntos en `mAP@50-95`**, confirmando que el codificador híbrido de RT-DETR aprovecha de forma muy notable el aumento de resolución espacial para resolver objetos pequeños como las matrículas.

![Evolución de pérdidas y métricas COCO durante el entrenamiento de RT-DETRv2-S (960, batch 8)](./runs/detect/runs_plate/plate_detector_rtdetr_v2_s_960_BATCH_8/results.png)

En las gráficas generadas a partir del registro `log.txt` del entrenamiento (`960x960`, `batch = 8`):

1. **Curvas de pérdida (*Training Loss per Epoch*):** la pérdida total (incluyendo las pérdidas auxiliares de las capas del decodificador y las ramas de *denoising*) cae drásticamente desde `~25.3` en la primera época hasta `~3.6` en la época 120, mientras que las pérdidas individuales `VFL`, `Bbox` y `GIoU` convergen de manera estable por debajo de `0.5`.
2. **Curvas de validación (`mAP` y `mAR`):** el modelo alcanza más de un `80 %` de `mAP@50` apenas en la época 12 y continúa refinando la localización estricta (`mAP@75` y `mAP@50-95`) hasta estabilizarse en torno al `85.0 %` de `mAP@50` y `51.3 %` de `mAP@50-95`.

### 3.2. Evaluación en el conjunto de Test y comparativa frente a YOLOv11s

Al evaluar el mejor *checkpoint* de **RT-DETRv2-S** (`best.pth`) sobre el mismo conjunto de test (`242` imágenes, `447` matrículas) mediante el evaluador oficial COCO (`faster_coco_eval`), se obtienen los siguientes resultados:

| Métrica en Conjunto de Test | YOLOv11s (`960`, `batch=16`) | RT-DETRv2-S (`960`, `batch=8`) |
| --- | ---: | ---: |
| **mAP@50** | `83.11 %` | **86.53 %** |
| **mAP@75** | — | **55.78 %** |
| **mAP@50-95** | `44.36 %` | **53.10 %** |
| **Recall / AR@50** | `83.45 %` | **94.00 %** (`AR@100: 65.21 %`) |

- **Desglose por escala en RT-DETRv2-S (Test):** el evaluador COCO reporta un `mAP@50-95` de `43.7 %` en matrículas pequeñas (*small*), `64.3 %` en medianas (*medium*) y `73.6 %` en grandes (*large*).
- **Comparación directa:** **RT-DETRv2-S** supera a **YOLOv11s** tanto en `mAP@50` (`86.53 %` vs `83.11 %`) como especialmente en **`mAP@50-95` (`53.10 %` vs `44.36 %`, +8.74 puntos)**, lo que indica que las cajas determinadas por el *Transformer* se ajustan con mayor exactitud espacial a los contornos reales de las matrículas.

![Predicciones de RT-DETRv2-S sobre las mismas muestras del conjunto de test](./images/docs/rtdetr_detections_960_8.png)

Comparando visualmente las predicciones de **RT-DETRv2-S** con las de **YOLOv11s** sobre las mismas 9 imágenes de test (`SEED = 42`), se aprecia que RT-DETRv2-S asigna puntuaciones de confianza más altas y definidas en la mayoría de los vehículos (por ejemplo, `0.96` frente a `0.83` en el primer coche, o `0.85` frente a `0.49` en la imagen inferior izquierda).

Cabe destacar que, a efectos teóricos, RT-DETRv2 obtiene mejores resultados frente a YOLOv11s. Está comparativa no se traslada al completo a la práctica. En el caso de la imágen 2, podemos ver como RT-DETRv2 a hecho una detección adicional innecesaria, confundiendo el texto de anuncio de la furgoneta blanca con su matrícula. Aún teniendo estos defectos prácticos, ambos modelos han presentado mejoras de reconocimiento, detectando matrículas en las imágenes de test, las cuales no fueron anotadas, pero se podrían reconocer como un acierto por el modelo.

---

## 4. Procesamiento de vídeo, seguimiento y extras (`LicensePlateVideoProcessor`)

Para procesar el vídeo de prueba (`plate_test_video.mp4`, de `2832` fotogramas a `25 FPS`, duración `01:53`), se ha diseñado la clase modular `LicensePlateVideoProcessor`, que coordina en cada fotograma la detección dual, el seguimiento multi-objeto, la resolución de oclusiones, la anonimización y el registro de resultados.

### 4.1. Arquitectura dual de detección y *tracking*

En cada fotograma se ejecutan dos modelos en paralelo con un umbral de confianza `CONFIDENCE_THRESHOLD = 0.35`:

1. **Detector y *tracker* de vehículos y peatones:** filtra exclusivamente las clases de interés del estándar COCO (`0: person`, `2: car`, `3: motorbike`, `5: bus`).
   - En el cuaderno **YOLOv11**, se utiliza `YOLOv11n.pt` integrado con el algoritmo **BoT-SORT** (`yolo_tracker_file.yaml`), configurado con compensación de movimiento global por flujo óptico disperso (`gmc_method: sparseOptFlow`), fusión de puntuación de confianza (`fuse_score: True`) y un búfer de memoria de `track_buffer: 90` fotogramas.
   - En el cuaderno **RT-DETR**, se implementa desde cero la clase `RTDETRTracker` dentro de `RTDETRPredictor`, que mantiene las trayectorias activas asociando las detecciones entre fotogramas consecutivos mediante matrices de **IoU** y asignación óptima con el **algoritmo húngaro** (`scipy.optimize.linear_sum_assignment`), conservando pistas perdidas hasta `max_lost_frames = 45`.
2. **Detector especialista de matrículas:** aplica el modelo entrenado (`best.pt` en YOLOv11s o `best.pth` en RT-DETRv2-S) sobre cada fotograma y asocia cada matrícula detectada a su vehículo correspondiente mediante `plate_belongs_to_vehicle()`, comprobando si el centro de la matrícula cae dentro del *bounding box* del vehículo o si la intersección supera el **50 %** del área de la matrícula. Si una matrícula no puede vincularse a ningún vehículo en ese instante, se registra como `orphan_plate`.

### 4.2. Gestión de oclusiones de peatones (`PedestrianOcclusionManager`)

Un problema clásico de los *trackers* multi-objeto ocurre cuando un coche o autobús pasa por delante de un peatón: al reaparecer el peatón varios fotogramas después, el *tracker* suele asignarle un `track_id` nuevo, inflando artificialmente el conteo total de personas. Para solucionarlo, se ha implementado la clase `PedestrianOcclusionManager`:

1. **Extracción de apariencia:** para cada peatón detectado se calcula y normaliza un histograma conjunto en el espacio de color **HSV** (`16 x 16` *bins* sobre los canales *Hue* y *Saturation*).
2. **Detección de desaparición por oclusión:** cuando un peatón activo deja de detectarse en el fotograma actual, se comprueba si su última caja conocida se solapaba con la caja de algún vehículo (`occluded_by_vehicle = True`) y se almacena en `lost_tracks` hasta un máximo de `max_gap_frames = 90` fotogramas.
3. **Re-identificación y remapeo de ID (`_check_lost_tracks`):** cuando aparece un nuevo ID de peatón, se compara con las pistas perdidas evaluando una distancia euclídea adaptativa (`min(220, 40.0 + gap * 3.5)` píxeles) y la correlación entre sus histogramas HSV (`cv2.compareHist` con `cv2.HISTCMP_CORREL`). Si supera el umbral de similitud (`>= 0.60`, o `> 0.35` a menos de `90 px` si fue ocluido por un vehículo), el nuevo ID se enlaza en `id_remap` con el `canonical_id` original de la persona.

### 4.3. Consenso temporal de clase y conteo de objetos únicos

Para evitar fluctuaciones esporádicas en la clasificación de un mismo objeto a lo largo de su trayectoria (por ejemplo, un coche que en un fotograma aislado se clasifica como autobús o furgoneta) y descartar falsos positivos fugaces:

- Se mantiene un historial `track_class_history[track_id]` con un contador de frecuencias (`Counter`) de las clases asignadas en cada fotograma.
- En cada instante, la clase mostrada para un `track_id` es la **moda estadística** (`most_common(1)`) de su historial.
- En el marcador superior (HUD) y en el conteo final solo se contabilizan como **objetos únicos confirmados** aquellas trayectorias que hayan sido detectadas durante al menos `FRAMES_TO_CONFIRM_CLASS = 5` fotogramas.

### 4.4. Anonimización de personas y vehículos (Extra)

Con el parámetro `APPLY_CENSORING = True`, el procesador aplica automáticamente un filtro de privacidad antes de volcar cada fotograma al vídeo de salida:

- **Peatones (`_apply_people_censor`):** aplica un desenfoque gaussiano intenso (`cv2.GaussianBlur` con kernel `25x25` y $\sigma = 30$) sobre el cuarto superior (`1/4`) de la caja delimitadora de cada persona para anonimizar su rostro.
- **Vehículos (`_apply_incoming_car_censor`):** aplica el mismo desenfoque gaussiano sobre el tercio superior (`1/3`, zona del parabrisas y ocupantes) de los vehículos detectados con confianza `>= 0.65`.

### 4.5. Resultados del procesamiento del vídeo y archivos generados

Tanto el vídeo anotado (con cajas **azules/cian** para vehículos, **naranjas** para peatones, **verdes** para matrículas y barra superior **HUD** con el recuento acumulado en vivo) como el fichero CSV se han generado con ambas arquitecturas en la carpeta [`results/`](./results/):

- **Pipeline YOLOv11 (`YOLOv11n.pt` + `YOLOv11s` matrículas):**
  - Vídeo comprimido: [`results/yolo_dual_detections_compressed.mp4`](./results/yolo_dual_detections_compressed.mp4)
  - Registro CSV: [`results/yolo_dual_detections.csv`](./results/yolo_dual_detections.csv)
- **Pipeline RT-DETRv2 (`RT-DETRv2-S` vehículos + `RT-DETRv2-S` matrículas):**
  - Vídeo comprimido: [`results/rtdetr_dual_detections_compressed.mp4`](./results/rtdetr_dual_detections_compressed.mp4)
  - Registro CSV: [`results/rtdetr_dual_detections.csv`](./results/rtdetr_dual_detections.csv)

La siguiente tabla compara el recuento total de **objetos únicos confirmados** (`>= 5` fotogramas) y el total de **detecciones brutas acumuladas** a lo largo de los `2832` fotogramas del vídeo:

| Clase de objeto | Objetos únicos (YOLOv11) | Detecciones brutas (YOLOv11) | Objetos únicos (RT-DETRv2) | Detecciones brutas (RT-DETRv2) |
| --- | ---: | ---: | ---: | ---: |
| **Personas (`person`)** | `27` | `3047` | `77` | `8862` |
| **Coches (`car`)** | `166` | `17813` | `310` | `30998` |
| **Motos (`motorbike`)** | `3` | `79` | `10` | `336` |
| **Autobuses (`bus`)** | `1` | `217` | `6` | `686` |
| **Matrículas confirmadas (`license_plate`)** | `46` | `3936` | `58` | `2734` |

#### Comparación del comportamiento en vídeo (YOLOv11 vs RT-DETRv2)

1. **Sensibilidad a objetos lejanos vs estabilidad de seguimiento:** el detector generalista **RT-DETRv2-S** es notablemente más sensible que `YOLOv11n.pt` ante vehículos aparcados al fondo y peatones lejanos, generando casi el doble de detecciones por fotograma (`30998` vs `17813` en coches). Sin embargo, al emplear un *tracker* basado únicamente en IoU + algoritmo húngaro frente a **BoT-SORT** (que integra filtro de Kalman y compensación de movimiento óptico `sparseOptFlow`), las pequeñas interrupciones en objetos lejanos producen una mayor fragmentación de trayectorias en RT-DETR (`310` IDs de coches frente a `166` en YOLOv11).
2. **Detección de matrículas asociadas a vehículos:** mientras que **YOLOv11s** acumula más detecciones brutas por fotograma (`3936`), **RT-DETRv2-S** consigue confirmar matrículas en un mayor número de vehículos distintos a lo largo del vídeo (**`58` vehículos con matrícula confirmada frente a `46`**), en consonancia con el mayor `mAP@50` (`86.53 %`) y `mAP@50-95` (`53.10 %`) observado en el conjunto de test.

#### Estructura del archivo CSV

Ambos ficheros CSV siguen exactamente la estructura de 15 columnas solicitada en el guion de la práctica:

```csv
fotograma,tipo_objeto,confianza,identificador_tracking,x1,y1,x2,y2,matrícula_en_su_caso,confianza,mx1,my1,mx2,my2,texto_matricula
1,bus,0.9357,1,1402,52,1918,476,yes,0.4412,1903,409,1920,429,bus_1_plate
1,car,0.8437,2,1321,312,1398,371,yes,0.4640,1355,339,1377,347,car_2_plate
1,car,0.4915,3,1231,313,1279,343,no,,,,,,
1,license_plate,0.3698,,977,321,997,329,yes,0.3698,977,321,997,329,orphan_plate
```

---

## 5. Limitaciones y posibles mejoras

- **Resolución de las matrículas a larga distancia:** aunque entrenar a `960x960` mejora drásticamente la localización respecto a `640x640`, los vehículos situados en el plano más alejado de la escena presentan matrículas de apenas unos pocos píxeles de altura. Una estrategia en cascada (recortar primero la caja del vehículo a su resolución nativa `1080p` y pasar dicho recorte al detector de matrículas) permitiría detectar matrículas aún más lejanas, a costa de un mayor tiempo de inferencia por fotograma.
- **Fragmentación de trayectorias en RT-DETR:** el `RTDETRTracker` basado en IoU y asignación húngara funciona bien en objetos cercanos y con movimiento suave, pero podría integrarse con un filtro de Kalman y compensación de movimiento de cámara (como en BoT-SORT o ByteTrack) para igualar la estabilidad de IDs lograda en el pipeline de YOLO.
- **Integración con OCR (Práctica 4b):** en esta primera parte (`P4a`), la columna `texto_matricula` almacena un identificador simbólico (`<tipo>_<track_id>_plate` u `orphan_plate`), dejando lista la estructura de asociación espacial y temporal para acoplar los modelos de reconocimiento óptico de caracteres (OCR clásicos y VLMs) en la **Práctica 4b**.

---

## 6. Referencias

- [Documentación oficial de Ultralytics YOLOv11](https://docs.ultralytics.com/)
- [ Modos de predicción y seguimiento (BoT-SORT / ByteTrack) en Ultralytics](https://docs.ultralytics.com/modes/track/)
- [Repositorio oficial de RT-DETR / RT-DETRv2 (LyUwenyu / Baidu)](https://github.com/lyuwenyu/RT-DETR)
- Zhao, Y. et al. (2023). *DETRs Beat YOLOs on Real-time Object Detection (RT-DETR)*. [arXiv:2304.08069](https://arxiv.org/abs/2304.08069)
