"""Inferencia con soporte TTA, medición de latencia, matrices de confusión y registro de métricas."""

from __future__ import annotations

import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from .models import EnsembleFER


@torch.inference_mode()
def predict_probs(
    net: nn.Module,
    X_arr: np.ndarray,
    device: torch.device | str,
    batch_size: int = 256,
    use_tta: bool = True,
) -> np.ndarray:
    dev = torch.device(device)
    net.eval()
    probs_list = []
    for i in range(0, len(X_arr), batch_size):
        xb = torch.from_numpy(X_arr[i : i + batch_size, None, :, :]).to(dev)
        if isinstance(net, EnsembleFER):
            probs = net.forward_probs(xb, use_tta=use_tta)
        else:
            probs = F.softmax(net(xb), dim=-1)
            if use_tta:
                probs_flip = F.softmax(net(torch.flip(xb, dims=[-1])), dim=-1)
                probs = 0.5 * (probs + probs_flip)
        probs_list.append(probs.cpu().numpy())
    return np.concatenate(probs_list, axis=0)


def measure_probs(
    net: nn.Module,
    X_arr: np.ndarray,
    device: torch.device | str,
    batch_size: int = 256,
    use_tta: bool = True,
    warmup_n: int = 16,
) -> tuple[np.ndarray, float]:
    dev = torch.device(device)
    if warmup_n > 0 and len(X_arr) > 0:
        _ = predict_probs(
            net,
            X_arr[:warmup_n],
            dev,
            batch_size=min(batch_size, warmup_n),
            use_tta=use_tta,
        )
        if dev.type == "cuda":
            torch.cuda.synchronize()

    t0 = time.perf_counter()
    probs = predict_probs(net, X_arr, dev, batch_size=batch_size, use_tta=use_tta)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    ms_per_img = ((time.perf_counter() - t0) / len(X_arr)) * 1000.0
    return probs, ms_per_img


def plot_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    title: str,
    save_path: Path | None = None,
    figsize: tuple[float, float] = (5.5, 5.0),
    ax=None,
) -> None:
    cm = confusion_matrix(
        y_true, y_pred, labels=range(len(class_names)), normalize="true"
    )
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45, values_format=".2f"
    )
    ax.set_title(title)
    if own_fig:
        plt.tight_layout()
        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=140)
        plt.show()


def registrar_evaluacion(
    resultados: list[dict],
    modelo: str,
    familia: str,
    conjunto: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ms_img: float,
    class_names: list[str],
    mostrar_reporte: bool = True,
    guardar_cm: bool = True,
    out_dir: Path | None = None,
) -> dict:
    acc = float(accuracy_score(y_true, y_pred))
    f1_mac = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_w = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    entry = {
        "modelo": modelo,
        "familia": familia,
        "conjunto": conjunto,
        "n": int(len(y_true)),
        "accuracy": acc,
        "f1_macro": f1_mac,
        "f1_weighted": f1_w,
        "ms_img": float(ms_img),
    }
    resultados.append(entry)
    print(
        f"[{conjunto}] {modelo} ({familia}) -> "
        f"Acc={acc:.4f} | F1-macro={f1_mac:.4f} | F1-w={f1_w:.4f} | {ms_img:.3f} ms/img"
    )
    if mostrar_reporte:
        print(
            classification_report(
                y_true, y_pred, target_names=class_names, zero_division=0, digits=3
            )
        )
    if guardar_cm:
        slug = re.sub(r"[^a-z0-9]+", "_", modelo.lower()).strip("_")
        save_path = (
            (out_dir / f"cm_{slug}_{conjunto}.png") if out_dir is not None else None
        )
        plot_confusion(
            y_true,
            y_pred,
            class_names,
            title=f"Matriz de confusión — {modelo} ({conjunto})",
            save_path=save_path,
        )
    return entry


def plot_split_preview(
    X_arr: np.ndarray,
    y_arr: np.ndarray,
    class_names: list[str],
    samples_per_class: int = 2,
    seed: int = 42,
    title: str = "Vista previa de muestras por emoción",
    save_path: Path | None = None,
) -> plt.Figure:
    """Muestra una cuadrícula compacta (samples_per_class x n_classes) en escala de grises."""
    rng = np.random.default_rng(seed)
    n_classes = len(class_names)
    fig, axes = plt.subplots(
        samples_per_class,
        n_classes,
        figsize=(1.85 * n_classes, 1.95 * samples_per_class),
    )
    axes = np.atleast_2d(axes)

    for col, cname in enumerate(class_names):
        idxs = np.where(y_arr == col)[0]
        chosen = rng.choice(idxs, size=min(samples_per_class, len(idxs)), replace=False)
        for row in range(samples_per_class):
            ax = axes[row, col]
            if row < len(chosen):
                ax.imshow(X_arr[chosen[row]], cmap="gray", vmin=0, vmax=255)
            if row == 0:
                ax.set_title(cname, fontsize=10, fontweight="bold")
            ax.axis("off")

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig


def plot_predictions_preview(
    X_arr: np.ndarray,
    y_true: np.ndarray,
    preds_by_model: dict[str, np.ndarray],
    class_names: list[str],
    n_samples: int = 12,
    seed: int = 42,
    cols: int = 6,
    title: str = "Vista previa de predicciones en test",
    pad_top: float = 0.12,
    pad_x: float = 0.05,
    pad_bottom: float = 0.04,
    wspace: float = 0.28,
    hspace: float = 0.50,
    save_path: Path | None = None,
) -> plt.Figure:
    """Muestra n_samples caras con margen superior y lateral configurable."""
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(y_true), size=min(n_samples, len(y_true)), replace=False)
    rows = (len(chosen) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.55 * cols, 3.1 * rows))
    axes = np.asarray(axes).reshape(-1)

    for ax, idx in zip(axes, chosen):
        real_lbl = class_names[int(y_true[idx])]
        lines = [f"Real: {real_lbl}"]
        all_ok = True
        for mname, preds in preds_by_model.items():
            pred_lbl = class_names[int(preds[idx])]
            ok = int(preds[idx]) == int(y_true[idx])
            all_ok = all_ok and ok
            mark = "OK" if ok else "X"
            lines.append(f"{mname}: {pred_lbl} [{mark}]")

        ax.imshow(X_arr[idx], cmap="gray", vmin=0, vmax=255)
        ax.set_title(
            "\n".join(lines),
            fontsize=8.5,
            loc="left",
            pad=6,
            color="#1b7837" if all_ok else "#b2182b",
        )
        ax.axis("off")

    for ax in axes[len(chosen) :]:
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=1.0 - (pad_top * 0.32))

    fig.subplots_adjust(
        left=pad_x,
        right=1.0 - pad_x,
        bottom=pad_bottom,
        top=1.0 - pad_top,
        wspace=wspace,
        hspace=hspace,
    )
    # Ancla el lienzo completo (0..1) para que Jupyter/bbox_inches='tight' respete los márgenes
    fig.add_artist(plt.Rectangle((0, 0), 1, 1, transform=fig.transFigure, fill=False, edgecolor="none"))

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches="tight", pad_inches=0.15)
    return fig
