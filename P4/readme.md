# Práctica 4 de Visión por Computador — Detección, Seguimiento y Lectura de Matrículas

> **Autores:** Félix Miguel Velásquez y Cristina Santana Martín  
> **Asignatura:** Visión por Computador

Esta práctica aborda el desarrollo completo de un sistema de análisis de tráfico sobre vídeo e imágenes basado en aprendizaje profundo. El proyecto se divide en dos partes complementarias (**P4a** y **P4b**) que cubren desde la detección y el seguimiento multi-objeto hasta el reconocimiento automático de los caracteres de las matrículas:

1. **Parte A ([`P4a/`](./P4a/)):** Detección, clasificación, seguimiento (*tracking*) y conteo de vehículos, peatones y matrículas en vídeo usando **YOLOv11** y **RT-DETRv2**, incluyendo gestión de oclusiones de peatones, asociación vehículo-matrícula, censura mediante desenfoque y exportación de resultados.
2. **Parte B ([`P4b/`](./P4b/)):** Reconocimiento óptico de caracteres (**OCR**) sobre las matrículas detectadas para identificar vehículos, integrando la lectura de matrículas en el flujo de vídeo y comparando modelos OCR clásicos (**EasyOCR**, **PaddleOCR**, **Tesseract**) frente a modelos de lenguaje visual (**SmolVLM**).

---

## Documentación y READMEs de cada parte

Cada subproyecto cuenta con su propia documentación detallada:

| Parte | Documentación |
| --- | --- |
| **P4a** |  [`P4a/README.md`](./P4a/README.md) |
| **P4b** |  [`P4b/README.md`](./P4b/README.md) |


---

## Resumen de las actividades

### Parte A — Detección y seguimiento de vehículos, personas y matrículas (`P4a`)

- **Objetivo:** Detectar y seguir vehículos (*car, motorbike, bus*), peatones (*person*) y matrículas (*license_plate*) en vídeo, asociar cada matrícula a su vehículo, gestionar oclusiones y exportar estadísticas fotograma a fotograma.
- **Conjunto de datos:** 2406 imágenes y 4371 matrículas anotadas, verificadas y divididas con `SEED = 42` en Train (75 %), Val (15 %) y Test (10 %) para formatos YOLO y COCO.
- **Modelos:** Entrenamiento mediante *transfer learning* de **YOLOv11** (`VC_P4_FMV.ipynb`, mejor configuración `plate_detector_960_BATCH_16_small` con mAP@50 de 82.97 % en validación) y **RT-DETRv2** (`VC_P4_FMV_RTDETR.ipynb`).
- **Pipeline de vídeo:** Detección dual (detector generalista COCO + detector propio de matrículas), `PedestrianOcclusionManager` basado en histogramas HSV y proximidad espacial, estabilización de clase por moda, censura de matrículas con desenfoque gaussiano, HUD informativo y volcado a CSV/MP4 en [`P4a/results/`](./P4a/results/).
- **Cuadernos principales:** [`P4a/VC_P4_FMV.ipynb`](./P4a/VC_P4_FMV.ipynb) y [`P4a/VC_P4_FMV_RTDETR.ipynb`](./P4a/VC_P4_FMV_RTDETR.ipynb).

### Parte B — Reconocimiento de caracteres en matrículas (`P4b`)

- **Objetivo:** Ampliar el detector de la Parte A para extraer el texto alfanumérico de cada matrícula detectada, mostrarlo en el vídeo de salida, almacenarlo en el CSV y comparar el rendimiento entre OCR clásicos y VLMs.
- **Integración en vídeo:** El cuaderno [`P4b/VC_P4B_CSM_FMV.ipynb`](./P4b/VC_P4B_CSM_FMV.ipynb) añade la lectura de matrículas al flujo de detección y genera los resultados en [`P4b/results/`](./P4b/results/) (`yolo_dual_detections_ocr_plates.csv` y vídeo anotado).
- **Comparativa OCR vs. VLM:** El cuaderno [`P4b/VC_P4B_CSM_FMV_comparative.ipynb`](./P4b/VC_P4B_CSM_FMV_comparative.ipynb) evalúa **EasyOCR**, **PaddleOCR** (ejecutado en proceso independiente mediante [`P4b/paddle_ocr_worker.py`](./P4b/paddle_ocr_worker.py) para evitar conflictos de librerías nativas con PyTorch) y **SmolVLM** (`HuggingFaceTB/SmolVLM-Instruct`) sobre 79 recortes de 50 imágenes (`SEED = 42`).
  - **SmolVLM** logra la mayor tasa de acierto exacto (**62.03 %**, 461.24 ms OCR).
  - **PaddleOCR** obtiene una precisión muy cercana (**59.49 %**, 377.73 ms OCR en CPU) sin consumir VRAM.
  - **EasyOCR** es el más rápido (**25.79 ms** OCR en GPU, ~17 fps totales), aunque con menor acierto exacto (**27.85 %**).

---

## Dependencias

### Entorno base probado

- **Python:** `3.11.x`
- **PyTorch:** `2.6.0+cu124` (CUDA 12.4, instalar acorde al sistema desde [pytorch.org](https://pytorch.org/get-started/locally/))
- **Sistema:** `ffmpeg` (necesario para la compresión de los vídeos de salida con `libx264`) y, opcionalmente, los binarios de [Tesseract OCR](https://tesseract-ocr.github.io/tessdoc/Installation.html) añadidos al `PATH` si se ejecuta el demostrador con `pytesseract`.

### Instalación de dependencias Python

Las dependencias unificadas para ambas partes de la práctica se encuentran en [`requirements.txt`](./requirements.txt):

```bash
pip install -r requirements.txt
```

> **Notas de compatibilidad en P4b:**
> 
> - **OpenCV y EasyOCR:** Si al instalar `easyocr` se instala `opencv-python-headless` y fallan funciones de ventana de OpenCV (como `cv2.imshow`), reinstalar `opencv-python`:
>   ```bash
>   pip uninstall -y opencv-python opencv-python-headless
>   pip install opencv-python --upgrade
>   ```
> - **PaddlePaddle y PyTorch:** Ambos cargan versiones distintas de OpenMP, oneDNN y CUDA. Para evitar fallos de segmentación en Jupyter, `P4b` ejecuta PaddleOCR en un subproceso aislado mediante [`P4b/paddle_ocr_worker.py`](./P4b/paddle_ocr_worker.py).
