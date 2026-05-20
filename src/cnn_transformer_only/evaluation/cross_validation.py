"""Cross-validation harnesses for IDS models.

Provides:
- ``grouped_kfold_cv``: Grouped k-fold CV respecting temporal block structure.
- ``walk_forward_cv``: Walk-forward (rolling window) validation for concept
  drift detection and realistic deployment simulation.
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


# ---------------------------------------------------------------------------
# Walk-Forward (Rolling Window) Validation
# ---------------------------------------------------------------------------

def walk_forward_cv(
    X_np: np.ndarray,
    y: np.ndarray,
    block_groups: np.ndarray,
    config: CNNTransformerConfig,
    *,
    train_window_blocks: int = 30,
    test_window_blocks: int = 5,
    step_blocks: int = 5,
    model_type: Literal["cnn_classifier", "cnn_transformer"] = "cnn_transformer",
    plot: bool = True,
) -> dict[str, Any]:
    """Walk-forward (rolling window) validation over sequential temporal blocks.

    Simulates real-world deployment by training on a fixed-size recent window
    of data and testing on the immediately following blocks, then sliding
    the window forward.

    Parameters
    ----------
    X_np : np.ndarray
        Feature matrix (after dedup / column cleaning, before imputation).
    y : np.ndarray
        Binary labels.
    block_groups : np.ndarray
        Sequential block ID per row (0 … N-1).
    config : CNNTransformerConfig
        Training configuration (epochs, lr, architecture, etc.).
    train_window_blocks : int
        Number of blocks in the training window (default 30).
    test_window_blocks : int
        Number of blocks in the test window (default 5).
    step_blocks : int
        How many blocks to slide the window forward each step (default 5).
    model_type : str
        ``"cnn_classifier"`` or ``"cnn_transformer"``.
    plot : bool
        If True, generate a performance-over-time plot.

    Returns
    -------
    dict with keys:
        ``"step_metrics"`` — list of per-step metric dicts (includes
        ``"train_blocks"`` and ``"test_blocks"`` ranges).
        ``"mean_metrics"`` — aggregated mean ± std across steps.
        ``"n_steps"`` — number of walk-forward steps executed.
        ``"degradation"`` — dict with trend info (slope of F1 over steps).
    """
    _set_seeds(config.random_state)
    device, _ = setup_device()

    unique_blocks = np.sort(np.unique(block_groups))
    n_blocks = len(unique_blocks)
    total_needed = train_window_blocks + test_window_blocks

    if n_blocks < total_needed:
        raise ValueError(
            f"Need at least {total_needed} blocks "
            f"(train={train_window_blocks} + test={test_window_blocks}), "
            f"but only {n_blocks} available."
        )

    # Compute step positions
    steps: list[tuple[list, list]] = []
    start = 0
    while start + total_needed <= n_blocks:
        train_blocks = list(unique_blocks[start: start + train_window_blocks])
        test_blocks = list(
            unique_blocks[
                start + train_window_blocks:
                start + train_window_blocks + test_window_blocks
            ]
        )
        steps.append((train_blocks, test_blocks))
        start += step_blocks

    n_steps = len(steps)
    if n_steps == 0:
        raise ValueError("No valid walk-forward steps possible with the given parameters.")

    print(f"\n{'='*70}")
    print(f"  WALK-FORWARD VALIDATION ({model_type})")
    print(f"  {n_steps} steps | train_window={train_window_blocks} blocks "
          f"| test_window={test_window_blocks} blocks | step={step_blocks} blocks")
    print(f"{'='*70}")

    step_metrics: list[dict[str, Any]] = []

    for step_idx, (train_blocks, test_blocks) in enumerate(steps):
        print(f"\n  Step {step_idx + 1}/{n_steps}  "
              f"Train blocks [{train_blocks[0]}–{train_blocks[-1]}] → "
              f"Test blocks [{test_blocks[0]}–{test_blocks[-1]}]")

        trn_mask = np.isin(block_groups, train_blocks)
        tst_mask = np.isin(block_groups, test_blocks)

        X_trn_raw = X_np[trn_mask].copy()
        y_trn = y[trn_mask].copy()
        X_tst_raw = X_np[tst_mask].copy()
        y_tst = y[tst_mask].copy()

        print(f"    Train: {len(y_trn):,} ({y_trn.mean()*100:.1f}% attack)  "
              f"Test: {len(y_tst):,} ({y_tst.mean()*100:.1f}% attack)")

        # Per-step preprocessing (fit on train only)
        X_trn_bal, X_tst_sc, y_trn_bal, _ = _preprocess_fold(
            X_trn_raw, X_tst_raw, y_trn, config
        )
        del X_trn_raw, X_tst_raw
        gc.collect()

        input_dim = X_trn_bal.shape[1]

        # Class weights
        n_b = int((y_trn_bal == 0).sum())
        n_a = int((y_trn_bal == 1).sum())
        cw = torch.tensor([1.0, n_b / max(n_a, 1)], dtype=torch.float32)

        # Dataloaders
        trn_ds = TensorDataset(torch.FloatTensor(X_trn_bal), torch.LongTensor(y_trn_bal))
        tst_ds = TensorDataset(torch.FloatTensor(X_tst_sc), torch.LongTensor(y_tst))
        trn_loader = DataLoader(
            trn_ds, batch_size=config.batch_size, shuffle=True,
            drop_last=True, num_workers=0,
        )
        tst_loader = DataLoader(
            tst_ds, batch_size=config.val_batch_size, shuffle=False,
            num_workers=0,
        )
        del X_trn_bal, X_tst_sc, y_trn_bal
        gc.collect()

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

        metrics = _train_one_fold(model, trn_loader, tst_loader, config, device, cw)
        metrics["train_block_range"] = (int(train_blocks[0]), int(train_blocks[-1]))
        metrics["test_block_range"] = (int(test_blocks[0]), int(test_blocks[-1]))
        metrics["step"] = step_idx + 1
        step_metrics.append(metrics)

        print(f"    → F1={metrics.get('f1_score', 0):.4f}  "
              f"ROC-AUC={metrics.get('auc_roc', 0):.4f}  "
              f"PR-AUC={metrics.get('auc_pr', 0):.4f}")

        del model, trn_loader, tst_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # ── Aggregate results ─────────────────────────────────────────────
    metric_keys = ["f1_score", "auc_roc", "auc_pr", "precision", "recall", "accuracy"]
    mean_metrics: dict[str, dict[str, float]] = {}

    print(f"\n{'='*70}")
    print(f"  WALK-FORWARD SUMMARY ({n_steps} steps, {model_type})")
    print(f"{'='*70}")
    print(f"  {'Metric':<12} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for key in metric_keys:
        values = [sm.get(key, 0.0) for sm in step_metrics]
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

    # ── Concept drift detection (F1 trend) ────────────────────────────
    f1_values = [sm.get("f1_score", 0.0) for sm in step_metrics]
    if n_steps >= 2:
        x_steps = np.arange(n_steps, dtype=np.float64)
        slope, intercept = np.polyfit(x_steps, f1_values, 1)
    else:
        slope, intercept = 0.0, f1_values[0] if f1_values else 0.0

    degradation = {
        "f1_slope_per_step": float(slope),
        "f1_total_change": float(slope * (n_steps - 1)) if n_steps > 1 else 0.0,
        "drifting": abs(slope) > 0.01,
    }

    drift_msg = "YES — performance is changing over time" if degradation["drifting"] else "NO"
    print(f"\n  Concept drift detected: {drift_msg}")
    print(f"  F1 trend slope: {slope:+.4f} per step "
          f"(total change: {degradation['f1_total_change']:+.4f})")
    print(f"{'='*70}\n")

    # ── Performance-over-time plot ────────────────────────────────────
    if plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Left: F1 / ROC-AUC / PR-AUC over steps
            step_labels = [
                f"{sm['test_block_range'][0]}–{sm['test_block_range'][1]}"
                for sm in step_metrics
            ]
            x = np.arange(n_steps)
            for key, color, marker in [
                ("f1_score", "tab:blue", "o"),
                ("auc_roc", "tab:green", "s"),
                ("auc_pr", "tab:orange", "^"),
            ]:
                vals = [sm.get(key, 0.0) for sm in step_metrics]
                axes[0].plot(x, vals, marker=marker, label=key, color=color, linewidth=2)

            # F1 trend line
            axes[0].plot(
                x, intercept + slope * x,
                "--", color="tab:red", linewidth=1.5, alpha=0.7,
                label=f"F1 trend ({slope:+.4f}/step)",
            )
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(step_labels, rotation=45, ha="right", fontsize=8)
            axes[0].set_xlabel("Test Blocks")
            axes[0].set_ylabel("Score")
            axes[0].set_title(f"Walk-Forward Performance — {model_type}")
            axes[0].legend(fontsize=8)
            axes[0].grid(True, alpha=0.3)
            axes[0].set_ylim(0, 1.05)

            # Right: Attack % in test window
            attack_pcts = []
            for sm in step_metrics:
                blk_range = sm["test_block_range"]
                tst_mask = np.isin(block_groups, list(range(blk_range[0], blk_range[1] + 1)))
                attack_pcts.append(float(y[tst_mask].mean() * 100))
            axes[1].bar(x, attack_pcts, color="tab:red", alpha=0.7)
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(step_labels, rotation=45, ha="right", fontsize=8)
            axes[1].set_xlabel("Test Blocks")
            axes[1].set_ylabel("Attack %")
            axes[1].set_title("Attack Distribution per Test Window")
            axes[1].grid(True, alpha=0.3, axis="y")

            plt.tight_layout()
            out_dir = getattr(config, "output_dir", None)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
                fig.savefig(
                    os.path.join(out_dir, f"walk_forward_{model_type}.png"),
                    dpi=160, bbox_inches="tight",
                )
            plt.show()
        except Exception as e:
            print(f"  [plot skipped: {e}]")

    return {
        "step_metrics": step_metrics,
        "mean_metrics": mean_metrics,
        "n_steps": n_steps,
        "degradation": degradation,
    }
