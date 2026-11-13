import os, sys, subprocess, urllib.request
from pathlib import Path

# Configurar ruta absoluta al repositorio RT_DETR/rtdetrv2_pytorch
REPO_PATH = Path(__file__).resolve().parent / "RT_DETR" / "rtdetrv2_pytorch"
if not REPO_PATH.exists():
    parent_repo = Path(__file__).resolve().parent.parent / "RT_DETR" / "rtdetrv2_pytorch"
    if parent_repo.exists():
        REPO_PATH = parent_repo
    else:
        # Clonar repositorio si no existe
        clone_dir = Path(__file__).resolve().parent / "RT_DETR"
        print(f"Clonando RT-DETR repositorio oficial en {clone_dir}...")
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/lyuwenyu/RT-DETR.git", str(clone_dir)], check=True)

if str(REPO_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_PATH))

import cv2
import numpy as np
import torch
from PIL import Image
from src.core import YAMLConfig
from src.core.yaml_utils import load_config      # next to your YAMLConfig import

REPO = str(REPO_PATH)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Para usar el modelo, necesitamos descargar los pesos oficiales desde el Repositorio.
# Actualmente, existen 8 modelos entrenados, divididos en dos familias y cuatro tamaños:
# La familia RT-DETRv1 y la V2. 
# Cada familia tiene cuatro tamaños, de menor a mayor, S,M,L y X.

# (config, tag de la release, fichero de pesos COCO)
MODELS = {
    "v1": {
        "r18":  ("rtdetr/rtdetr_r18vd_6x_coco.yml",    "v0.1", "rtdetr_r18vd_dec3_6x_coco_from_paddle.pth"),
        "r34":  ("rtdetr/rtdetr_r34vd_6x_coco.yml",    "v0.1", "rtdetr_r34vd_dec4_6x_coco_from_paddle.pth"),
        "r50m": ("rtdetr/rtdetr_r50vd_m_6x_coco.yml",  "v0.1", "rtdetr_r50vd_m_6x_coco_from_paddle.pth"),
        "r50":  ("rtdetr/rtdetr_r50vd_6x_coco.yml",    "v0.1", "rtdetr_r50vd_6x_coco_from_paddle.pth"),
        "r101": ("rtdetr/rtdetr_r101vd_6x_coco.yml",   "v0.1", "rtdetr_r101vd_6x_coco_from_paddle.pth"),
    },
    "v2": {
        "r18":  ("rtdetrv2/rtdetrv2_r18vd_120e_coco.yml",  "v0.2", "rtdetrv2_r18vd_120e_coco_rerun_48.1.pth"),
        "r34":  ("rtdetrv2/rtdetrv2_r34vd_120e_coco.yml",  "v0.1", "rtdetrv2_r34vd_120e_coco_ema.pth"),
        "r50m": ("rtdetrv2/rtdetrv2_r50vd_m_7x_coco.yml",  "v0.1", "rtdetrv2_r50vd_m_7x_coco_ema.pth"),
        "r50":  ("rtdetrv2/rtdetrv2_r50vd_6x_coco.yml",    "v0.1", "rtdetrv2_r50vd_6x_coco_ema.pth"),
        "r101": ("rtdetrv2/rtdetrv2_r101vd_6x_coco.yml",   "v0.1", "rtdetrv2_r101vd_6x_coco_from_paddle.pth"),
    },
}
SIZE_ALIASES = {"s": "r18", "m": "r50m", "l": "r50", "x": "r101"}   # nombres cortos


def load_rtdetr(family="v2", size="s", device=DEVICE, weights_path=None, config_path=None, num_classes=None):
    """
    Carga un modelo RT-DETR oficial preentrenado o un checkpoint personalizado (fine-tuned).
    
    Args:
        family (str): Familia del modelo ('v1' o 'v2').
        size (str): Tamaño ('s', 'm', 'l', 'x' o 'r18', 'r34', 'r50m', 'r50', 'r101').
        device (torch.device): Dispositivo ('cuda' o 'cpu').
        weights_path (str or Path, optional): Ruta a un checkpoint .pth personalizado.
        config_path (str or Path, optional): Ruta al archivo .yml de configuración correspondiente.
        num_classes (int, optional): Número de clases para el modelo personalizado.
    """
    family = family.lower()
    size = size.lower()
    size = SIZE_ALIASES.get(size, size)

    # Si se proporciona checkpoint personalizado (por ejemplo, tras entrenar con matrículas)
    if weights_path is not None:
        weights_path = Path(weights_path)
        print(f"Cargando RT-DETR desde checkpoint personalizado: {weights_path} en {device}")
        if config_path is None:
            if family in MODELS and size in MODELS[family]:
                cfg_name, _, _ = MODELS[family][size]
                config_path = REPO_PATH / "configs" / cfg_name
            else:
                raise ValueError("Se requiere 'config_path' para cargar un checkpoint con configuración personalizada.")
        
        load_config.__defaults__[0].clear()
        kwargs = {}
        if num_classes is not None:
            kwargs["num_classes"] = num_classes
            kwargs["remap_mscoco_category"] = False

        cfg = YAMLConfig(str(config_path), **kwargs)
        ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=False)
        state_dict = ckpt["ema"]["module"] if ("ema" in ckpt and ckpt["ema"] is not None) else ckpt.get("model", ckpt)
        cfg.model.load_state_dict(state_dict)
        model = cfg.model.deploy().to(device)
        postprocessor = cfg.postprocessor.deploy().to(device)
        return model, postprocessor

    # Carga de modelo oficial con pesos COCO
    print(f"Cargando RT-DETR familia {family!r}, tamaño {size!r} en {device}")
    if family not in MODELS or size not in MODELS[family]:
        raise ValueError(f"Modelo desconocido: {family}/{size}. Inserte familia v2 o v1 y tamaño s,m,l,x o r18,r34,r50m,r50,r101")
    
    cfg_name, tag, wfile = MODELS[family][size]
    weights_dir = Path(__file__).resolve().parent / "rtdetr"
    WEIGHTS = weights_dir / wfile

    if not WEIGHTS.exists():
        print(f'No se detectaron los pesos del modelo {family}/{size}. Se estan descargando y guardando en {WEIGHTS}')
        weights_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(
            f"https://github.com/lyuwenyu/storage/releases/download/{tag}/{wfile}", str(WEIGHTS))

    load_config.__defaults__[0].clear()
    cfg = YAMLConfig(str(REPO_PATH / "configs" / cfg_name))
   
    ckpt = torch.load(str(WEIGHTS), map_location="cpu", weights_only=True)
    cfg.model.load_state_dict(ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"])
    model = cfg.model.deploy().to(device)
    postprocessor = cfg.postprocessor.deploy().to(device)
    return model, postprocessor
