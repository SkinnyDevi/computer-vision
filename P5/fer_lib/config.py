"""Rutas relativas, constantes globales y configuración de entorno para FER-2013."""

from __future__ import annotations

import ctypes
import glob
import json
import os
import site
from pathlib import Path
from typing import Any

P5_DIR: Path = Path(__file__).resolve().parent.parent


def _resolve_data_dir() -> Path:
    env_dir = os.environ.get("FER_DATA_DIR")
    candidates = [
        Path(env_dir) if env_dir else None,
        P5_DIR.parent / "image-dataset" / "prac5_dataset",
        P5_DIR / "data" / "prac5_dataset",
    ]
    for cand in candidates:
        if cand is not None and cand.is_dir():
            return cand.resolve()
    return (P5_DIR.parent / "image-dataset" / "prac5_dataset").resolve()


DATA_DIR: Path = _resolve_data_dir()
TRAIN_DIR: Path = DATA_DIR / "train"
TEST_DIR: Path = DATA_DIR / "test"

OUT_DIR: Path = P5_DIR / "outputs_p5"
MODELS_DIR: Path = OUT_DIR / "models"

OTHERS_DIR: Path = MODELS_DIR / "v1_otros"
CNN_DIR: Path = MODELS_DIR / "v1_CNN_torch"
TL_RESNET18_DIR: Path = MODELS_DIR / "v1_TL_resnet18"
TL_EFFNET_DIR: Path = MODELS_DIR / "v1_TL_efficientnet_b0"
ENSEMBLE_DIR: Path = MODELS_DIR / "v1_ensemble_cnn_resnet18"

SEED: int = 42
IMG_SIZE: int = 48
TL_IMG_SIZE: int = 112
VAL_SIZE: float = 0.20
DEEPFACE_N: int = 1000
IMG_EXT: set[str] = {".jpg", ".jpeg", ".png", ".bmp"}

CNN_CKPT: Path = CNN_DIR / f"cnn_fer_torch_seed{SEED}.pt"
RESNET18_CKPT: Path = TL_RESNET18_DIR / f"resnet18_fer_torch_seed{SEED}.pt"
EFFNET_CKPT: Path = TL_EFFNET_DIR / f"efficientnet_b0_fer_torch_seed{SEED}.pt"
ENSEMBLE_CKPT: Path = ENSEMBLE_DIR / f"ensemble_cnn_resnet18_fer_torch_seed{SEED}.pt"


def configure_runtime_env() -> None:
    """Configura caché local y librerías CUDA compartidas para DeepFace/TF/Matplotlib."""
    site_pkgs = site.getsitepackages()
    if site_pkgs:
        nv_root = os.path.join(site_pkgs[0], "nvidia")
        for pattern in ("cusolver/lib/libcusolver.so.11", "curand/lib/libcurand.so.10"):
            for lib_path in glob.glob(os.path.join(nv_root, pattern)):
                try:
                    ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")
    os.environ.setdefault("DEEPFACE_HOME", str(P5_DIR / ".deepface_cache"))


def save_json(data: Any, path: Path, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
