"""Semillas deterministas y bucles de entrenamiento reutilizables para CNNFer y TransferFER."""

from __future__ import annotations

import copy
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import SEED


def set_seed(seed: int = SEED, warn_only: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=warn_only)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    loss_sum, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            if is_train:
                loss.backward()
                optimizer.step()

            bs = yb.size(0)
            loss_sum += loss.item() * bs
            correct += (logits.argmax(dim=1) == yb).sum().item()
            total += bs

    return loss_sum / total, correct / total


def fit_cnn(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epochs: int = 80,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    lr_patience: int = 4,
    es_patience: int = 12,
) -> tuple[dict[str, list[float]], int, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=lr_patience, min_lr=1e-6
    )

    history: dict[str, list[float]] = {
        "loss": [],
        "val_loss": [],
        "accuracy": [],
        "val_accuracy": [],
        "lr": [],
    }
    best_val_loss = float("inf")
    best_ep = 0
    best_weights = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        ep_t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        cur_lr = optimizer.param_groups[0]["lr"]

        history["loss"].append(tr_loss)
        history["accuracy"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["lr"].append(cur_lr)

        scheduler.step(val_loss)
        print(
            f"Epoch {epoch:02d}/{epochs} - {time.time() - ep_t0:.1f}s - "
            f"loss: {tr_loss:.4f} - accuracy: {tr_acc:.4f} - "
            f"val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f} - lr: {cur_lr:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ep = epoch - 1
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= es_patience:
                print(
                    f"Early stopping en época {epoch}. "
                    f"Restaurando mejores pesos de la época {best_ep + 1}."
                )
                break

    model.load_state_dict(best_weights)
    train_minutes = (time.time() - t0) / 60.0
    print(f"Entrenamiento terminado en {train_minutes:.1f} min (mejor época: {best_ep + 1})")
    return history, best_ep, train_minutes


def fit_transfer_two_phase(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    warmup_epochs: int = 5,
    finetune_epochs: int = 35,
    warmup_lr: float = 1e-3,
    finetune_lr: float = 3e-4,
    weight_decay: float = 1e-4,
    es_patience: int = 10,
) -> tuple[dict, int, float]:
    history = {
        "loss": [],
        "val_loss": [],
        "accuracy": [],
        "val_accuracy": [],
        "lr": [],
        "phase": [],
    }
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_ep = 0
    best_weights = copy.deepcopy(model.state_dict())

    t0 = time.time()

    model.set_backbone_trainable(False)
    opt_warm = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=warmup_lr,
        weight_decay=weight_decay,
    )
    for ep in range(1, warmup_epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, opt_warm)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        history["loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["accuracy"].append(tr_acc)
        history["val_accuracy"].append(val_acc)
        history["lr"].append(warmup_lr)
        history["phase"].append("warmup")
        print(
            f"[Warmup {ep:02d}/{warmup_epochs}] loss: {tr_loss:.4f} - acc: {tr_acc:.4f} | "
            f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            best_ep = len(history["loss"]) - 1

    model.set_backbone_trainable(True)
    opt_ft = torch.optim.AdamW(model.parameters(), lr=finetune_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_ft, T_max=finetune_epochs, eta_min=1e-6
    )
    epochs_no_improve = 0

    for ep in range(1, finetune_epochs + 1):
        ep_t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, opt_ft)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        cur_lr = opt_ft.param_groups[0]["lr"]
        scheduler.step()

        history["loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["accuracy"].append(tr_acc)
        history["val_accuracy"].append(val_acc)
        history["lr"].append(cur_lr)
        history["phase"].append("finetune")

        ep_idx = len(history["loss"]) - 1
        print(
            f"[FT {ep:02d}/{finetune_epochs}] {time.time() - ep_t0:.1f}s - "
            f"loss: {tr_loss:.4f} - acc: {tr_acc:.4f} | "
            f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f} - lr: {cur_lr:.6f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            best_ep = ep_idx

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= es_patience:
                print(
                    f"Early stopping en época FT {ep}. "
                    f"Restaurando mejor val_acc de época global {best_ep + 1}."
                )
                break

    model.load_state_dict(best_weights)
    train_minutes = (time.time() - t0) / 60.0
    print(
        f"Finalizado en {train_minutes:.1f} min | "
        f"Mejor época global: {best_ep + 1} (val_acc={best_val_acc:.4f})"
    )
    return history, best_ep, train_minutes
