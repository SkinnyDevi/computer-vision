"""Ejecuta PaddleOCR en un proceso independiente.

PaddlePaddle y PyTorch cargan librerías nativas incompatibles (OpenMP, oneDNN,
CUDA). Si ambos se importan en el mismo kernel de Jupyter, el proceso puede
morir con un segmentation fault. Este script se lanza con ``subprocess`` desde
el notebook para que PaddleOCR nunca comparta proceso con PyTorch.

Uso:
    python paddle_ocr_worker.py <carpeta_recortes> <salida.json> [--lang es] [--gpu]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Debe configurarse ANTES de importar paddle.
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


def crear_lector(lang, usar_gpu):
    from paddleocr import PaddleOCR

    try:
        # PaddleOCR 3.x
        return PaddleOCR(
            lang=lang,
            device="gpu:0" if usar_gpu else "cpu",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
        )
    except (TypeError, ValueError):
        # PaddleOCR 2.x
        return PaddleOCR(
            lang=lang,
            use_gpu=usar_gpu,
            use_angle_cls=False,
            show_log=False,
            enable_mkldnn=False,
        )


def extraer_textos(resultado):
    """Normaliza la salida de PaddleOCR 2.x y 3.x a una lista de textos."""
    if resultado is None:
        return []
    if isinstance(resultado, dict) or hasattr(resultado, "keys"):
        if "rec_texts" in resultado:
            return list(resultado["rec_texts"])
        if "res" in resultado:
            return extraer_textos(resultado["res"])
    if hasattr(resultado, "json"):
        datos = resultado.json
        datos = datos() if callable(datos) else datos
        if isinstance(datos, dict):
            return extraer_textos(datos)

    textos = []
    if isinstance(resultado, (list, tuple)):
        for item in resultado:
            # Formato 2.x: [caja, (texto, confianza)]
            if (
                isinstance(item, (list, tuple))
                and len(item) == 2
                and isinstance(item[1], (list, tuple))
                and len(item[1]) == 2
                and isinstance(item[1][0], str)
            ):
                textos.append(item[1][0])
            else:
                textos.extend(extraer_textos(item))
    return textos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("carpeta")
    parser.add_argument("salida")
    parser.add_argument("--lang", default="es")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    import cv2

    lector = crear_lector(args.lang, args.gpu)
    usa_predict = hasattr(lector, "predict")

    imagenes = sorted(Path(args.carpeta).glob("*.png"))

    # Calentamiento para no penalizar la primera inferencia.
    if imagenes:
        img = cv2.imread(str(imagenes[0]))
        lector.predict(img) if usa_predict else lector.ocr(img, cls=False)

    resultados = {}
    for ruta in imagenes:
        img = cv2.imread(str(ruta))
        t0 = time.perf_counter()
        salida = lector.predict(img) if usa_predict else lector.ocr(img, cls=False)
        t_ocr = time.perf_counter() - t0
        resultados[ruta.stem] = {
            "textos": [str(t) for t in extraer_textos(salida)],
            "tiempo_ocr_s": t_ocr,
        }

    with open(args.salida, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)
    print(f"PaddleOCR procesó {len(resultados)} recortes.", file=sys.stderr)


if __name__ == "__main__":
    main()
