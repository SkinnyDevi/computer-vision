# Práctica 5 — Reconocimiento de Emociones Faciales (FER-2013)

## 1. Objetivo y planteamiento del problema

El objetivo de esta práctica es diseñar, entrenar y evaluar de forma comparativa distintos paradigmas de visión por computador para la clasificación multiclase de expresiones faciales sobre el conjunto de datos **FER-2013** (7 categorías emocionales: _angry_, _disgusted_, _fearful_, _happy_, _neutral_, _sad_ y _surprised_). Se contrastan cuatro familias de modelos bajo un protocolo experimental común y reproducible:

1. **Aprendizaje automático clásico** con descriptores manuales (PCA, HOG, LBP) y clasificadores KNN, SVM (kernel RBF) y _Stacking_.
2. **Deep Learning desde cero** mediante una red neuronal convolucional propia (`CNNFer`) en PyTorch.
3. **Transfer Learning** en dos fases con arquitecturas preentrenadas en ImageNet (**ResNet-18** y **EfficientNet-B0**).
4. **Ensamble de modelos** combinando las probabilidades de `CNNFer` y `ResNet-18` junto con _Test-Time Augmentation_ (TTA), además de una referencia externa preentrenada (**DeepFace**).

---

## 2. Arquitectura modular del proyecto (`fer_lib/`)

Para evitar duplicación de código y garantizar consistencia en preprocesado, particiones y métricas entre experimentos, la lógica común se encapsuló en el paquete interno `fer_lib/`:

| Módulo          | Responsabilidad principal                                                                                                                                                            |
| :-------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `config.py`     | Rutas relativas del proyecto, constantes globales (`SEED = 42`, `IMG_SIZE = 48`, `TL_IMG_SIZE = 112`, `VAL_SIZE = 0.20`), caché y librerías CUDA.                                    |
| `data.py`       | Lectura en escala de grises ($48\times 48$), partición estratificada _train/val_, cálculo de pesos de clase suavizados, `DataLoader` deterministas y extracción paralela de HOG/LBP. |
| `models.py`     | Definición de `ConvBlock`, `CNNFer`, `TransferFER`, `EnsembleFER` y carga unificada de checkpoints `.pt` (`load_checkpoint`).                                                        |
| `training.py`   | Fijado determinista de semillas (`set_seed`) y bucles reutilizables `fit_cnn` y `fit_transfer_two_phase` con _Early Stopping_ y planificación de tasa de aprendizaje.                |
| `evaluation.py` | Inferencia con soporte de TTA horizontal (`predict_probs`), medición de latencia sincronizada en GPU (`measure_probs`), matrices de confusión y registro homogéneo de métricas.      |

---

## 3. Protocolo experimental y preprocesado

- **Partición de datos:** El conjunto de entrenamiento original ($N = 28\,709$ imágenes en escala de grises de $48\times 48$ píxeles) se dividió de forma estratificada en **80 % entrenamiento ($22\,967$)** y **20 % validación ($5\,742$)** con semilla fija (`SEED = 42`). El conjunto de prueba independiente cuenta con $7\,178$ imágenes, extrayéndose una submuestra estratificada común de $1\,000$ imágenes para la evaluación de DeepFace.
- **Desbalanceo de clases:** FER-2013 presenta un fuerte desbalanceo (p. ej., `disgusted` dispone de 349 muestras en entrenamiento frente a 5 772 de `happy`). Para mitigarlo sin sobreamplificar el ruido de etiquetado, se aplicaron pesos de clase suavizados mediante raíz cuadrada:
  $$w_c' = \frac{w_c^{0.5}}{\frac{1}{C}\sum_{j=1}^{C} w_j^{0.5}}$$
  lo que asigna un peso de $2.44$ a `disgusted` y $0.60$ a `happy`, combinado con _label smoothing_ ($\epsilon = 0.10$) en la entropía cruzada.
- **Aumento de datos (Data Augmentation):** Integrado en el grafo del modelo únicamente durante el entrenamiento (`RandomHorizontalFlip`, `RandomAffine` con rotación $\pm 10.8^\circ$, traslación $\pm 8\%$ y escala $0.9\text{--}1.1$, `ColorJitter` de brillo/contraste $0.2$ y `RandomErasing` con $p = 0.25$).
- **Métricas de evaluación:** Dado el desbalanceo, el criterio principal de selección y análisis es el **F1-macro**, complementado con **Accuracy**, informe por clase y matrices de confusión normalizadas.

---

## 4. Resultados experimentales en validación

| Familia           | Modelo                               | Parámetros / Dim.     | Accuracy (%) | F1-macro (%) | Observaciones                                            |
| :---------------- | :----------------------------------- | :-------------------- | :----------: | :----------: | :------------------------------------------------------- |
| Clásico           | PCA + KNN ($k=5$)                    | 252 dims (95 % var.)  |      —       |    31.84     | Sensible a variaciones de pose e iluminación.            |
| Clásico           | PCA + SVM (RBF, $C=10$)              | 252 dims (cap 10k)    |      —       |    36.84     | Mejora linealidad en subespacio global.                  |
| Clásico           | LBP + KNN ($k=21$)                   | 18 dims ($R=2, P=16$) |      —       |    29.35     | Textura global insuficiente para gestos finos.           |
| Clásico           | LBP + SVM (RBF, $C=10$)              | 18 dims (cap 10k)     |      —       |    20.87     | Baja separabilidad en espacio reducido.                  |
| Clásico           | HOG + KNN ($k=5$)                    | 900 dims              |      —       |    45.41     | Captura eficaz de gradientes locales.                    |
| Clásico           | **HOG + SVM (RBF, $C=10$)**          | 900 dims (cap 10k)    |      —       |  **46.51**   | Mejor modelo clásico individual.                         |
| Clásico           | Stacked (PCA+HOG+LBP)                | 1 170 dims (cap 5k)   |      —       |    40.40     | Penalizado por el submuestreo a 5k muestras.             |
| Referencia        | DeepFace (`skip`, _test_ $n=1\,000$) | Preentrenado          |    55.90     |    54.55     | Latencia media de 5.03 ms/img.                           |
| Deep Learning     | **CNNFer (propia, TTA)**             | 2.74 M                |  **68.97**   |  **67.24**   | Mejor época 59/71 (10.2 min en RTX 3060 Ti).             |
| Transfer Learning | EfficientNet-B0 (TTA)                | 4.02 M                |    68.29     |    66.24     | $112\times 112$, mejor época 22 (7.5 min, 0.450 ms/img). |
| Transfer Learning | **ResNet-18 (TTA)**                  | 11.18 M               |  **68.77**   |  **66.93**   | $112\times 112$, mejor época 20 (5.8 min, 0.277 ms/img). |
| Ensamble          | **CNNFer + ResNet-18 (TTA)**         | 13.92 M               |  **70.41**   |  **68.71**   | **+1.47 pp en F1-macro** sobre el mejor individual.      |

---

## 5. Discusión y conclusiones científicas

1. **Superioridad de representaciones aprendidas frente a descriptores manuales:** Entre los métodos clásicos, HOG+SVM alcanza un F1-macro de $46.51\%$ al codificar gradientes de contornos faciales (cejas, boca, ojos), superando ampliamente a PCA y LBP. Sin embargo, las redes convolucionales elevan el F1-macro por encima del $67\%$ al aprender jerarquías de características invariantes a pequeñas deformaciones y cambios fotométricos.
2. **Arquitectura propia vs. Transfer Learning en baja resolución:** `CNNFer` ($2.74\text{ M}$ de parámetros, entrada nativa $48\times 48$) iguala y supera ligeramente a `ResNet-18` ($67.24\%$ frente a $66.93\%$ en F1-macro) y `EfficientNet-B0` ($66.24\%$). En imágenes faciales de muy baja resolución, una red compacta diseñada _ad hoc_ evita el sobreajuste temprano observado en _backbones_ preentrenados en ImageNet tras el descongelado.
3. **Efecto del ensamble y TTA:** El promediado de probabilidades (_soft voting_) entre `CNNFer` y `ResNet-18` con TTA horizontal reduce la varianza de predicción al combinar dos arquitecturas con sesgos inductivos distintos (resolución nativa $48\times 48$ vs. reescalado $112\times 112$ con bloques residuales), logrando el mejor rendimiento global (**70.41 % Accuracy / 68.71 % F1-macro**).
4. **Patrones de error por clase:** Las emociones con rasgos geométricos muy marcados (`happy`: F1 = 0.88, `surprised`: F1 = 0.80) se clasifican con alta fiabilidad, mientras que las emociones negativas sutiles (`fearful`: 0.58, `sad`: 0.58, `angry`: 0.62) concentran la mayor confusión mutua, un comportamiento consistente con la literatura de FER-2013.

---