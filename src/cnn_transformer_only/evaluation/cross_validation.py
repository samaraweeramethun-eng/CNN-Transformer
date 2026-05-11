"""Grouped k-fold cross-validation harness for IDS models.

Respects temporal block groupings to prevent data leakage within folds.
"""
from __future__ import annotations

import gc
import math
import os
import random
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from cnn_transformer_only.config import CNNTransformerConfig
from cnn_transformer_only.data import (
    IntelligentDataBalancer,
    binary_predictions_from_proba,
    calculate_comprehensive_metrics,
    find_best_f1_threshold,
)
from cnn_transformer_only.models.cnn_classifier import CNNClassifier
from cnn_transformer_only.models.cnn_transformer import CNNTransformerIDS
from cnn_transformer_only.utils.device import setup_device


def _set_seeds(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _train_one_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: CNNTransformerConfig,
    device: torch.device,
    class_weight_tensor: torch.Tensor,
) -> dict[str, Any]:
    """Train a model for one fold and return best-epoch metrics."""
    criterion = nn.CrossEntropyLoss(
        weight=class_weight_tensor.to(device),
        label_smoothing=config.label_smoothing,
    )
    optimizer = optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    steps_per_epoch = max(len(train_loader), 1)
    warmup_steps = getattr(config, "warmup_epochs", 2) * steps_per_epoch
    total_steps = config.epochs * steps_per_epoch

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    grad_clip = getattr(config, "grad_clip", 0.5)

    best_f1 = 0.0
    best_metrics: dict[str, float] = {}
    best_state = None
    patience = getattr(config, "patience", 4)
    no_improve = 0

    for _epoch in range(1, config.epochs + 1):
        # Train
        model.train()
        for data, target in train_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(data)
            loss = criterion(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            scheduler.step()

        # Validate
        model.eval()
        all_probs, all_targets = [], []
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                logits = model(data)
                probs = F.softmax(logits, dim=1)[:, 1]
                all_probs.append(probs.cpu().numpy())
                all_targets.append(target.cpu().numpy())

        y_prob = np.concatenate(all_probs)
        y_true = np.concatenate(all_targets)
        thr, _ = find_best_f1_threshold(y_true, y_prob)
        y_pred = binary_predictions_from_proba(y_prob, thr)
        metrics = calculate_comprehensive_metrics(y_true, y_pred, y_prob)

        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_metrics = {**metrics, "threshold": thr}
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_metrics


def _preprocess_fold(
    X_train_raw: np.ndarray,
    X_val_raw: np.ndarray,
    y_train: np.ndarray,
    config: CNNTransformerConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply per-fold preprocessing (impute → scale) fitted on training data."""
    from sklearn.preprocessing import StandardScaler

    # Impute NaN with training medians
    train_medians = np.nanmedian(X_train_raw, axis=0)
    for ci in range(X_train_raw.shape[1]):
        mask = np.isnan(X_train_raw[:, ci])
        if mask.any():
            fill = train_medians[ci] if np.isfinite(train_medians[ci]) else 0.0
            X_train_raw[mask, ci] = fill
    for ci in range(X_val_raw.shape[1]):
        mask = np.isnan(X_val_raw[:, ci])
        if mask.any():
            fill = train_medians[ci] if np.isfinite(train_medians[ci]) else 0.0
            X_val_raw[mask, ci] = fill

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_val = scaler.transform(X_val_raw).astype(np.float32)

    # Balance training data
    balancer = IntelligentDataBalancer(config.undersampling_ratio, config.random_state)
    X_train_bal, y_train_bal = balancer.balance_classes(X_train, y_train)

    return X_train_bal, X_val, y_train_bal, y_train


def grouped_kfold_cv(
    X_np: np.ndarray,
    y: np.ndarray,
    source_groups: np.ndarray,
    config: CNNTransformerConfig,
    *,
    n_folds: int = 5,
    model_type: Literal["cnn_classifier", "cnn_transformer"] = "cnn_transformer",
) -> dict[str, Any]:
    """Run grouped k-fold cross-validation respecting temporal block structure.

    Blocks are assigned to folds such that temporally adjacent blocks stay
    together (using the same chunk-based grouping as ``temporal_chunks``).

    Parameters
    ----------
    X_np : np.ndarray
        Feature matrix (after dedup / column cleaning, before imputation).
    y : np.ndarray
        Binary labels.
    source_groups : np.ndarray
        Block/group ID per row.
    config : CNNTransformerConfig
        Training configuration.
    n_folds : int
        Number of cross-validation folds (default 5).
    model_type : str
        ``"cnn_classifier"`` or ``"cnn_transformer"``.

    Returns
    -------
    dict with keys:
        ``"fold_metrics"`` — list of per-fold metric dicts.
        ``"mean_metrics"`` — dict of mean ± std for each metric.
        ``"n_folds"`` — number of folds actually run.
    """
    _set_seeds(config.random_state)
    device, _ = setup_device()

    unique_blocks = np.unique(source_groups)
    n_blocks = len(unique_blocks)
    chunk_size = getattr(config, "chunk_size_blocks", 5)

    # Group blocks into temporal chunks
    chunks: list[list[int]] = []
    for i in range(0, n_blocks, chunk_size):
        chunks.append(list(unique_blocks[i: i + chunk_size]))

    n_chunks = len(chunks)
    if n_chunks < n_folds:
        n_folds = n_chunks
        print(f"  Reduced to {n_folds} folds (only {n_chunks} chunks available)")

    # Assign chunks to folds (round-robin)
    fold_chunk_indices: list[list[int]] = [[] for _ in range(n_folds)]
    for i, _ in enumerate(chunks):
        fold_chunk_indices[i % n_folds].append(i)

    fold_metrics: list[dict[str, float]] = []

    for fold_idx in range(n_folds):
        print(f"\n{'='*60}")
        print(f"  FOLD {fold_idx + 1} / {n_folds}")
        print(f"{'='*60}")

        # Determine val chunks and train chunks
        val_chunk_ids = fold_chunk_indices[fold_idx]
        train_chunk_ids = [
            ci for fi in range(n_folds) if fi != fold_idx
            for ci in fold_chunk_indices[fi]
        ]

        # Convert chunk IDs to block IDs
        val_blocks = [b for ci in val_chunk_ids for b in chunks[ci]]
        train_blocks = [b for ci in train_chunk_ids for b in chunks[ci]]

        trn_mask = np.isin(source_groups, train_blocks)
        val_mask = np.isin(source_groups, val_blocks)

        X_trn_raw = X_np[trn_mask].copy()
        y_trn = y[trn_mask].copy()
        X_val_raw = X_np[val_mask].copy()
        y_val = y[val_mask].copy()

        print(
            f"  Train: {len(y_trn):,} samples ({y_trn.mean()*100:.1f}% attack), "
            f"Val: {len(y_val):,} samples ({y_val.mean()*100:.1f}% attack)"
        )

        # Per-fold preprocessing
        X_trn_bal, X_val_sc, y_trn_bal, _ = _preprocess_fold(
            X_trn_raw, X_val_raw, y_trn, config
        )
        del X_trn_raw, X_val_raw; gc.collect()

        input_dim = X_trn_bal.shape[1]

        # Class weights
        n_b = int((y_trn_bal == 0).sum())
        n_a = int((y_trn_bal == 1).sum())
        cw = torch.tensor([1.0, n_b / max(n_a, 1)], dtype=torch.float32)

        # Build dataloaders
        trn_ds = TensorDataset(torch.FloatTensor(X_trn_bal), torch.LongTensor(y_trn_bal))
        val_ds = TensorDataset(torch.FloatTensor(X_val_sc), torch.LongTensor(y_val))
        trn_loader = DataLoader(trn_ds, batch_size=config.batch_size, shuffle=True, drop_last=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=config.val_batch_size, shuffle=False, num_workers=0)

        del X_trn_bal, X_val_sc, y_trn_bal, y_val; gc.collect()

        # Build model
        if model_type == "cnn_classifier":
            model = CNNClassifier(
                input_dim=input_dim,
                conv_channels=config.conv_channels,
                fc_dim=config.cnn_fc_dim,
                dropout=config.dropout,
            ).to(device)
        else:
            model = CNNTransformerIDS(
                input_dim=input_dim,
                conv_channels=config.conv_channels,
                num_layers=config.num_layers,
                num_heads=config.num_heads,
                d_model=config.d_model,
                d_ff=config.d_ff,
                dropout=config.dropout,
            ).to(device)

        metrics = _train_one_fold(model, trn_loader, val_loader, config, device, cw)
        fold_metrics.append(metrics)

        print(
            f"  Fold {fold_idx + 1} results: "
            f"F1={metrics.get('f1_score', 0):.4f}, "
            f"ROC-AUC={metrics.get('auc_roc', 0):.4f}, "
            f"PR-AUC={metrics.get('auc_pr', 0):.4f}"
        )

        del model, trn_loader, val_loader
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

    # Aggregate results
    metric_keys = ["f1_score", "auc_roc", "auc_pr", "precision", "recall", "accuracy"]
    mean_metrics: dict[str, dict[str, float]] = {}

    print(f"\n{'='*60}")
    print(f"  CROSS-VALIDATION SUMMARY ({n_folds} folds, {model_type})")
    print(f"{'='*60}")
    print(f"  {'Metric':<12} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for key in metric_keys:
        values = [fm.get(key, 0.0) for fm in fold_metrics]
        mean_metrics[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "values": values,
        }
        print(
            f"  {key:<12} {np.mean(values):>8.4f} {np.std(values):>8.4f} "
            f"{np.min(values):>8.4f} {np.max(values):>8.4f}"
        )
    print(f"{'='*60}\n")

    return {
        "fold_metrics": fold_metrics,
        "mean_metrics": mean_metrics,
        "n_folds": n_folds,
    }
