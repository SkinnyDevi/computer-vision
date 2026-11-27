"""Carga de imágenes FER-2013, particiones estratificadas, DataLoaders y descriptores clásicos."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import torch
from joblib import Parallel, delayed
from skimage.feature import hog, local_binary_pattern
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from .config import DEEPFACE_N, IMG_EXT, IMG_SIZE, SEED, VAL_SIZE


def read_gray(path: Path, img_size: int = IMG_SIZE) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is not None and img.shape != (img_size, img_size):
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return img


def load_split(
    split_dir: Path,
    class_names: list[str] | None = None,
    img_size: int = IMG_SIZE,
    show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if class_names is None:
        class_names = sorted(p.name for p in split_dir.iterdir() if p.is_dir())
    if not class_names:
        raise ValueError(f"No se encontraron subcarpetas de clases en {split_dir}")

    X, y = [], []
    for idx, cname in enumerate(class_names):
        cdir = split_dir / cname
        files = sorted(f for f in cdir.iterdir() if f.suffix.lower() in IMG_EXT)
        iterator = tqdm(files, desc=f"{split_dir.name}/{cname}", leave=False) if show_progress else files
        for fpath in iterator:
            img = read_gray(fpath, img_size=img_size)
            if img is not None:
                X.append(img)
                y.append(idx)

    if not X:
        raise ValueError(f"No se pudieron leer imágenes en {split_dir}")
    return np.stack(X).astype(np.uint8), np.asarray(y, dtype=np.int64), class_names


def split_train_val(
    X_full: np.ndarray,
    y_full: np.ndarray,
    val_size: float = VAL_SIZE,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx_tr, idx_val = train_test_split(
        np.arange(len(y_full)),
        test_size=val_size,
        stratify=y_full,
        random_state=seed,
    )
    return X_full[idx_tr], X_full[idx_val], y_full[idx_tr], y_full[idx_val]


def compute_smoothed_class_weights(
    y_tr: np.ndarray,
    n_classes: int,
    power: float = 0.5,
) -> tuple[np.ndarray, torch.Tensor]:
    cw = compute_class_weight("balanced", classes=np.arange(n_classes), y=y_tr)
    if power != 1.0:
        cw = cw ** power
        cw = cw / cw.mean()
    return cw, torch.tensor(cw, dtype=torch.float32)


def make_loader(
    X_arr: np.ndarray,
    y_arr: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = False,
    seed: int = SEED,
    pin_memory: bool = True,
) -> DataLoader:
    x_t = torch.from_numpy(X_arr[:, None, :, :])
    y_t = torch.from_numpy(y_arr).long()
    ds = TensorDataset(x_t, y_t)
    gen = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=gen,
        pin_memory=pin_memory,
    )


def get_common_test_indices(
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_samples: int | None = DEEPFACE_N,
    seed: int = SEED,
    idx_file: Path | None = None,
) -> np.ndarray:
    if idx_file is not None and idx_file.is_file():
        return np.load(idx_file)
    if n_samples is not None and n_samples < len(y_test):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=n_samples, random_state=seed)
        _, idx = next(sss.split(X_test, y_test))
        return np.sort(idx)
    return np.arange(len(y_test))


def extract_hog_single(img: np.ndarray) -> np.ndarray:
    return hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype(np.float32)


def extract_lbp_single(img: np.ndarray, radius: int = 2, n_points: int = 16) -> np.ndarray:
    lbp = local_binary_pattern(img, n_points, radius, method="uniform")
    n_bins = n_points + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float32)


def extract_hog_batch(X: np.ndarray, n_jobs: int = -1) -> np.ndarray:
    return np.vstack(Parallel(n_jobs=n_jobs)(delayed(extract_hog_single)(im) for im in X))


def extract_lbp_batch(X: np.ndarray, n_jobs: int = -1) -> np.ndarray:
    return np.vstack(Parallel(n_jobs=n_jobs)(delayed(extract_lbp_single)(im) for im in X))


def preparar_features_test(X_imgs: np.ndarray, pca_transformer) -> dict[str, tuple[np.ndarray, float]]:
    t0 = time.perf_counter()
    X_flat = (X_imgs.reshape(len(X_imgs), -1) / 255.0).astype(np.float32)
    X_pca = pca_transformer.transform(X_flat)
    t_pca = time.perf_counter() - t0

    t0 = time.perf_counter()
    X_hog = np.vstack([extract_hog_single(im) for im in X_imgs])
    t_hog = time.perf_counter() - t0

    t0 = time.perf_counter()
    X_lbp = np.vstack([extract_lbp_single(im) for im in X_imgs])
    t_lbp = time.perf_counter() - t0

    X_all = np.hstack([X_pca, X_hog, X_lbp])
    t_all = t_pca + t_hog + t_lbp
    return {
        "PCA": (X_pca, t_pca),
        "HOG": (X_hog, t_hog),
        "LBP": (X_lbp, t_lbp),
        "ALL": (X_all, t_all),
    }
