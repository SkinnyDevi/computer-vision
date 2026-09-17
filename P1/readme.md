# Práctica 1 de Visión por Computador

## Datos de la práctica

- **Asignatura:** Visión por Computador
- **Práctica:** P1
- **Autores:** Félix Miguel Velásquez y Cristina Santana Martín
- **Cuaderno:** [`VC_P1_FMV_CSM.ipynb`](VC_P1_FMV_CSM.ipynb)

## Descripción

En esta práctica se trabajan distintas técnicas básicas de procesamiento de imágenes y vídeo con Python, NumPy, Matplotlib y OpenCV. Las actividades incluyen la generación de patrones geométricos, el análisis de los píxeles más claros y oscuros de una imagen de vídeo y la creación de efectos visuales inspirados en el arte Pop Art.

## Requisitos

Para ejecutar el cuaderno se necesita Python 3 y las siguientes bibliotecas:

```bash
pip install opencv-python numpy matplotlib
```

También se puede usar la lista `spec-list-macosx.txt` para inicializar un entorno anaconda.

La tarea de análisis de vídeo requiere una webcam disponible y permisos para que Python pueda utilizarla.

## Tareas realizadas

### 1. Generación de un tablero de ajedrez

#### Objetivo

Crear una imagen cuadrada de 800 x 800 píxeles con la textura de un tablero de ajedrez. La actividad se resuelve primero manualmente, sin herramientas de inteligencia artificial, y después mediante una solución generada con ayuda de un asistente de IA.

#### Resolución manual

La primera solución crea una imagen en escala de grises inicializada a cero. Después se recorre el tablero mediante dos bucles anidados. Cada casilla mide 100 x 100 píxeles y el valor de la variable `flip_color` determina si la casilla se pinta de negro o de blanco. Al terminar cada fila se invierte de nuevo el color inicial para conservar el patrón alterno.

También se incluye una segunda implementación manual que recorre los píxeles individuales. Para cada píxel se calcula la fila y la columna de la casilla mediante división entera:

```python
fila = i // 100
columna = j // 100
```

La suma de ambas posiciones permite decidir el color con `(fila + columna) % 2`.

#### Resolución con asistencia de IA

La solución automática utiliza `np.indices` para construir una matriz de posiciones, calcula el patrón alterno con el módulo 2 y emplea `np.kron` para ampliar cada casilla hasta el tamaño final. Finalmente, `np.where` convierte el patrón en valores de intensidad blanco y negro.

Esta versión guarda el resultado en `tablero_ajedrez_800x800.png` y lo muestra con Matplotlib. Frente a la implementación que visita cada píxel, esta solución es más compacta y aprovecha operaciones vectorizadas de NumPy.

### 2. Imagen inspirada en Mondrian

#### Objetivo

Crear una composición geométrica inspirada en las obras de Piet Mondrian utilizando las funciones de dibujo de OpenCV, sin emplear herramientas de IA para la generación del código.

#### Procedimiento

1. Se crea un lienzo blanco de 640 x 480 píxeles.
2. Se generan líneas horizontales y verticales negras con posiciones y longitudes parcialmente aleatorias.
3. Algunas líneas recorren todo el lienzo y otras terminan al encontrar una línea perpendicular.
4. Se convierte la imagen a escala de grises y se localizan las zonas blancas.
5. Se buscan los contornos de esas zonas y se comprueba que sean rectángulos completamente blancos.
6. Cada rectángulo detectado se rellena con un color de una paleta formada por rojo, azul, amarillo y blanco.

La función `find_white_rectangle` devuelve las coordenadas del siguiente rectángulo blanco encontrado. El proceso continúa hasta que no quedan regiones blancas sin colorear.

#### Resultado

El resultado es una composición abstracta con bloques de color separados por líneas negras gruesas. Debido al uso de valores aleatorios, la distribución concreta de las líneas y los colores puede cambiar en cada ejecución.

### 3. Detección del píxel más claro y más oscuro de la webcam

#### Objetivo

Detectar en cada fotograma capturado por la cámara las posiciones del píxel con menor intensidad y del píxel con mayor intensidad. Sobre la imagen original se dibuja un círculo rojo o azul para el píxel más oscuro y un círculo verde para el más claro.

#### Primera implementación

La primera versión convierte cada fotograma a escala de grises y utiliza `np.argmin` y `np.argmax` para localizar los extremos. Después transforma las coordenadas de NumPy, expresadas como `(fila, columna)`, al formato `(x, y)` que utiliza OpenCV para dibujar los círculos.

La ventana se mantiene abierta hasta pulsar la tecla `ESC`.

#### Implementación optimizada

La segunda versión encapsula la detección en `detectar_pixeles_claro_oscuro` y utiliza `cv2.minMaxLoc`, que devuelve directamente los valores mínimo y máximo junto con sus posiciones. Esta alternativa evita recorrer explícitamente la imagen desde Python y permite que el procesamiento sea más fluido.

Además de los círculos, se muestran en pantalla los valores de intensidad detectados. El objetivo de la comparación es comprobar que la vectorización y las funciones nativas de OpenCV reducen los saltos de la primera versión.

### 4. Propuesta propia de Pop Art

#### Objetivo

Crear un efecto visual propio inspirado en el Pop Art y en la estética de los cómics clásicos. La propuesta combina una división de la imagen en cuadrantes con una trama de puntos similar a las técnicas de impresión.

#### Filtro de color

La función `color_filter` divide cada fotograma en cuatro cuadrantes y modifica los canales BGR de cada región:

- Cuadrante superior izquierdo: se elimina el canal azul.
- Cuadrante superior derecho: se elimina el canal azul.
- Cuadrante inferior izquierdo: se conservan principalmente los tonos azules.
- Cuadrante inferior derecho: se eliminan los canales azul y verde.

El filtro modifica el propio fotograma, sin crear una imagen adicional para cada cuadrante.

#### Filtro de puntos

La función `point_filter` crea una máscara blanca y dibuja círculos negros a intervalos regulares. Las filas alternas se desplazan horizontalmente mediante la variable `move`, lo que produce una trama más parecida a una impresión analógica.

La imagen coloreada y la máscara se combinan mediante `cv2.bitwise_and`. El resultado se muestra en tiempo real hasta pulsar `ESC` o cerrar la ventana.

#### Efecto Pop Art con caracteres ASCII

Como ampliación de la propuesta anterior, se implementa un efecto que transforma cada fotograma de la webcam en una composición de caracteres ASCII coloreados.

##### Conversión de intensidad a caracteres

Se utiliza una tabla ordenada de caracteres ASCII según su brillo visual. Para evitar recalcular la conversión en cada fotograma, se construye previamente `ascii_intensity_table`, que contiene el carácter correspondiente para cada una de las 256 intensidades posibles.

El fotograma se reduce a una cuadrícula cuyo tamaño depende de `cell_size`. Un valor pequeño genera más detalle, pero requiere dibujar más caracteres y puede reducir el rendimiento. Un valor grande mejora la velocidad a costa de perder precisión.

##### Color por cuadrantes

Cada carácter se dibuja con un color distinto según su posición:

- Superior izquierdo: rojo.
- Superior derecho: verde.
- Inferior izquierdo: azul.
- Inferior derecho: amarillo.

El resultado final se muestra en la ventana `ASCII Pop Art Effect`. La ejecución termina al pulsar `ESC`.

## Fuentes y asistencia de IA

- Referencia de caracteres ASCII y valores de brillo: [Stack Overflow](https://stackoverflow.com/a/74186686), respuesta de chungaloider, con licencia CC BY-SA 4.0.
- Referencia adicional para el efecto de imagen ASCII: [Medium](https://medium.com/@aditiverma00300/ascii-style-image-7d75f5d0b019).
- Explicación de operadores bit a bit: [omes-va](https://omes-va.com/operadores-bitwise/).
- Inspiración para la imagen Mondrian: [Ejemplo de Mondrian](https://www3.gobiernodecanarias.org/medusa/ecoescuela/sa/2017/04/17/descubriendo-a-mondrian/).
- Conversación de IA sobre la detección de píxeles claro y oscuro: [Gemini](https://share.gemini.google/AOQYsRSd17rO).
- Conversación de IA sobre la precomputación de la conversión a ASCII: [Gemini](https://share.gemini.google/KK4OFTzBhdzP).

Las imágenes utilizadas en las explicaciones se encuentran en la carpeta de la práctica cuando están disponibles.
