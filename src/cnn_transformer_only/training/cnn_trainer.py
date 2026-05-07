import gc
import math
import os
import random

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader, TensorDataset

from cnn_transformer_only.config import CNNTransformerConfig
from cnn_transformer_only.data import (
    IntelligentDataBalancer,
    binary_predictions_from_proba,
    build_dataloaders,
    calculate_comprehensive_metrics,
    find_best_f1_threshold,
    load_cicids_feature_matrix,
    prepare_training_data,
)
from cnn_transformer_only.interpretability.grad_cam import generate_gradcam_report
from cnn_transformer_only.interpretability.integrated_gradients import generate_ig_report
from cnn_transformer_only.models.cnn_transformer import CNNTransformerIDS
from cnn_transformer_only.utils.device import setup_device


def _set_seeds(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# ─── Stability diagnostics ────────────────────────────────────────────────────

def _check_nan_inf(tensor) -> bool:
    """Return True if tensor contains NaN or Inf."""
    if tensor is None:
        return False
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        return True
    return False


def _compute_grad_norm(model) -> float:
    """Compute total gradient L2 norm across all parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


def _check_model_params_health(model) -> bool:
    """Return True if any model parameter contains NaN/Inf."""
    for name, p in model.named_parameters():
        if torch.isnan(p).any() or torch.isinf(p).any():
            return True
    return False


# ─── Threshold search ──────────────────────────────────────────────────────────

def _find_best_threshold_fine(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Search thresholds 0.05–0.95 in 0.01 steps; return (best_threshold, best_f1)."""
    from sklearn.metrics import f1_score as sk_f1
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.arange(0.05, 0.96, 0.01):
        preds = (y_prob >= thr).astype(int)
        f1 = sk_f1(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr, best_f1


# ─── Training epoch with diagnostics ──────────────────────────────────────────

def _train_epoch(model, loader, criterion, optimizer, scheduler, device, grad_clip: float = 0.5):
    """Train one epoch; returns (avg_loss, grad_norm_before_clip, grad_norm_after_clip, nan_detected)."""
    model.train()
    running_loss = 0.0
    grad_norms_pre = []
    grad_norms_post = []
    nan_detected = False

    for data, target in loader:
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(data)
        loss = criterion(logits, target)

        # Check loss health
        if _check_nan_inf(loss):
            nan_detected = True
            break

        loss.backward()

        # Gradient norm before clipping
        norm_pre = _compute_grad_norm(model)
        grad_norms_pre.append(norm_pre)

        # Check gradient health
        if not np.isfinite(norm_pre):
            nan_detected = True
            break

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        norm_post = _compute_grad_norm(model)
        grad_norms_post.append(norm_post)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        running_loss += loss.item()

        # Check parameter health after update
        if _check_model_params_health(model):
            nan_detected = True
            break

    avg_loss = running_loss / max(len(loader), 1)
    avg_norm_pre = float(np.mean(grad_norms_pre)) if grad_norms_pre else 0.0
    avg_norm_post = float(np.mean(grad_norms_post)) if grad_norms_post else 0.0
    return avg_loss, avg_norm_pre, avg_norm_post, nan_detected


# ─── Validation with full threshold search ─────────────────────────────────────

def _eval_epoch(model, loader, criterion, device):
    """Evaluate; returns (loss, metrics_at_0.5, all_probs, all_targets)."""
    model.eval()
    losses = []
    all_probs, all_targets = [], []
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            logits = model(data)
            loss = criterion(logits, target)
            losses.append(loss.item())
            probs = F.softmax(logits, dim=1)[:, 1]
            all_probs.append(probs.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    if not all_probs:
        empty_metrics = {k: 0.0 for k in ["accuracy", "auc_roc", "auc_pr", "f1_score", "precision", "recall"]}
        return 0.0, empty_metrics, np.array([]), np.array([])

    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_targets)
    y_pred = binary_predictions_from_proba(y_prob, threshold=0.5)
    metrics = calculate_comprehensive_metrics(y_true, y_pred, y_prob)
    return (
        float(np.mean(losses)) if losses else 0.0,
        metrics,
        y_prob,
        y_true,
    )


def _eval_epoch_with_threshold(model, loader, criterion, device, threshold: float):
    """Evaluate using a custom threshold; returns (loss, metrics, probs, targets)."""
    model.eval()
    losses = []
    all_probs, all_targets = [], []
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            logits = model(data)
            loss = criterion(logits, target)
            losses.append(loss.item())
            probs = F.softmax(logits, dim=1)[:, 1]
            all_probs.append(probs.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    if not all_probs:
        empty_metrics = {k: 0.0 for k in ["accuracy", "auc_roc", "auc_pr", "f1_score", "precision", "recall"]}
        return 0.0, empty_metrics, np.array([]), np.array([])

    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_targets)
    y_pred = binary_predictions_from_proba(y_prob, threshold=threshold)
    metrics = calculate_comprehensive_metrics(y_true, y_pred, y_prob)
    return float(np.mean(losses)) if losses else 0.0, metrics, y_prob, y_true


def train_cnn_transformer(config: CNNTransformerConfig | None = None):
    config = config or CNNTransformerConfig()
    _set_seeds(config.random_state)

    device, multi_gpu = setup_device()
    if multi_gpu:
        config.batch_size = 512
        config.val_batch_size = 1024

    os.makedirs(config.output_dir, exist_ok=True)
    grad_clip = getattr(config, "grad_clip", 0.5)

    print("Loading dataset for CNN-Transformer training...")
    X, y, feature_cols, _, source_groups = load_cicids_feature_matrix(
        config.input_path,
        max_rows=getattr(config, "max_rows", 0),
        chunksize=getattr(config, "csv_chunksize", 200_000),
        return_source_groups=True,
    )
    print(f"Loaded rows: {len(y):,} | Features: {len(feature_cols)}")

    print("Running enhanced preprocessing pipeline...")
    (X_train, X_val, X_test, y_train, y_val, y_test,
     scaler, medians, feature_cols, prep_meta) = prepare_training_data(
        X, y, feature_cols, config, source_groups=source_groups,
    )
    del X
    gc.collect()

    balancer = IntelligentDataBalancer(config.undersampling_ratio, config.random_state)
    X_train_bal, y_train_bal = balancer.balance_classes(X_train, y_train)

    # ── Class weights ───────────────────────────────────────────────────────
    attack_weight = getattr(config, "attack_class_weight", 0.0)
    if attack_weight > 0:
        class_weight_tensor = torch.tensor([1.0, attack_weight], dtype=torch.float32)
        print(f"Class weights (configured): benign=1.00, attack={attack_weight:.2f}")
    else:
        # No extra class weights — balanced data handles imbalance
        class_weight_tensor = None
        print("Class weights: disabled (using balanced data only)")

    input_dim = X_train.shape[1]
    del X_train, y_train
    gc.collect()

    train_loader, val_loader, _ = build_dataloaders(
        X_train_bal,
        y_train_bal,
        X_val,
        y_val,
        batch_size=config.batch_size,
        val_batch_size=config.val_batch_size,
        num_workers=config.num_workers,
    )

    # Build test loader for held-out evaluation
    test_loader = None
    if len(y_test) > 0:
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.val_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=config.num_workers > 0,
        )

    print(f"Training:   {len(train_loader.dataset)} samples, {len(train_loader)} batches/epoch")
    print(f"Validation: {len(val_loader.dataset)} samples")
    print(f"Test:       {len(y_test)} samples (held-out, never seen during training)")

    del X_train_bal, y_train_bal, X_test, y_test
    gc.collect()

    model = CNNTransformerIDS(
        input_dim=input_dim,
        d_model=config.d_model,
        conv_channels=config.conv_channels,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        dropout=config.dropout,
    ).to(device)

    if multi_gpu:
        model = DataParallel(model)

    # Loss function — optionally with class weights
    if class_weight_tensor is not None:
        criterion = nn.CrossEntropyLoss(
            weight=class_weight_tensor.to(device),
            label_smoothing=config.label_smoothing,
        )
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # Warmup + cosine annealing (critical for Transformer stability)
    warmup_epochs = getattr(config, "warmup_epochs", 2)
    steps_per_epoch = max(len(train_loader), 1)
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = config.epochs * steps_per_epoch

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda) if total_steps > 0 else None

    # ── Training loop with best checkpoint, diagnostics, early stopping ────
    early_stop_metric = getattr(config, "early_stop_metric", "roc_auc")
    patience = getattr(config, "patience", 4)
    best_metric_value = 0.0
    best_epoch = 0
    best_state = None
    best_val_threshold = 0.5
    no_improve = 0
    epoch_history: list[dict] = []
    nan_abort = False

    print(f"\n{'─'*70}")
    print(f"Training config: lr={config.lr:.1e} | grad_clip={grad_clip} | "
          f"warmup={warmup_epochs} | patience={patience}")
    print(f"Early stopping metric: {early_stop_metric} | "
          f"Attack class weight: {'disabled' if class_weight_tensor is None else f'{attack_weight:.2f}'}")
    print(f"{'─'*70}\n")

    for epoch in range(1, config.epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        train_loss, grad_norm_pre, grad_norm_post, nan_detected = _train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, grad_clip=grad_clip
        )

        if nan_detected:
            print(f"\n⚠️  NaN/Inf detected at epoch {epoch}! Stopping training safely.")
            nan_abort = True
            break

        # ── Validate ───────────────────────────────────────────────────────
        val_loss, val_metrics, val_probs, val_targets = _eval_epoch(
            model, val_loader, criterion, device
        )

        # Fine threshold search on validation
        val_best_thr, val_best_f1 = _find_best_threshold_fine(val_targets, val_probs)
        val_f1_at_05 = val_metrics["f1_score"]  # F1 at default 0.5

        # Re-compute metrics at the best threshold for logging
        val_preds_tuned = binary_predictions_from_proba(val_probs, val_best_thr)
        val_metrics_tuned = calculate_comprehensive_metrics(val_targets, val_preds_tuned, val_probs)

        # Confusion matrix values
        tn = int(((val_preds_tuned == 0) & (val_targets == 0)).sum())
        fp = int(((val_preds_tuned == 1) & (val_targets == 0)).sum())
        fn = int(((val_preds_tuned == 0) & (val_targets == 1)).sum())
        tp = int(((val_preds_tuned == 1) & (val_targets == 1)).sum())

        current_lr = optimizer.param_groups[0]["lr"]

        # Determine monitored metric value
        if early_stop_metric == "f1":
            current_monitor = val_best_f1
        else:
            current_monitor = val_metrics_tuned["auc_roc"]

        # ── Logging ────────────────────────────────────────────────────────
        print(
            f"Epoch {epoch:02d} | Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | "
            f"ROC-AUC {val_metrics_tuned['auc_roc']:.4f} | PR-AUC {val_metrics_tuned['auc_pr']:.4f} | "
            f"F1@0.5 {val_f1_at_05:.4f} | Best-F1 {val_best_f1:.4f}@{val_best_thr:.2f} | "
            f"P {val_metrics_tuned['precision']:.4f} R {val_metrics_tuned['recall']:.4f} | "
            f"LR {current_lr:.2e}"
        )
        print(
            f"       Grad norm: pre-clip={grad_norm_pre:.2f} post-clip={grad_norm_post:.2f} | "
            f"CM: TN={tn} FP={fp} FN={fn} TP={tp}"
        )

        epoch_history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "roc_auc": val_metrics_tuned["auc_roc"], "pr_auc": val_metrics_tuned["auc_pr"],
            "f1": val_best_f1, "f1_at_05": val_f1_at_05,
            "precision": val_metrics_tuned["precision"], "recall": val_metrics_tuned["recall"],
            "best_threshold": val_best_thr,
            "grad_norm_pre": grad_norm_pre, "grad_norm_post": grad_norm_post,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        })

        # ── Best checkpoint saving ────────────────────────────────────────
        if current_monitor > best_metric_value:
            best_metric_value = current_monitor
            best_epoch = epoch
            best_val_threshold = val_best_thr
            no_improve = 0
            state_dict = model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict()
            preprocess_state = {
                "type": "standard_scaler",
                "medians": medians.to_dict(),
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
                "log1p_columns": prep_meta["log1p_columns"],
                "indicator_source_columns": prep_meta["indicator_source_columns"],
                "csv_columns": prep_meta["csv_feature_cols"],
            }
            best_state = {
                "model_state_dict": state_dict,
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": val_metrics_tuned,
                "config": config.__dict__,
                "feature_columns": feature_cols,
                "preprocessor": preprocess_state,
                "model_type": "cnn_transformer",
                "best_epoch": best_epoch,
                "best_threshold": best_val_threshold,
                "best_val_f1": val_best_f1,
                "best_val_roc_auc": val_metrics_tuned["auc_roc"],
                "best_val_pr_auc": val_metrics_tuned["auc_pr"],
            }
            print(f"       ✓ New best ({early_stop_metric}={current_monitor:.4f}) — saved checkpoint")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={patience}, "
                      f"best {early_stop_metric}={best_metric_value:.4f} at epoch {best_epoch})")
                break

    # ── Handle NaN abort ───────────────────────────────────────────────────
    if nan_abort and best_state is None:
        print("\n⚠️  Training aborted due to NaN/Inf before any valid checkpoint was saved.")
        diag_path = os.path.join(config.output_dir, "cnn_transformer_nan_diagnostic.txt")
        with open(diag_path, "w") as f:
            f.write(f"Training aborted at epoch {epoch} due to NaN/Inf.\n")
            f.write(f"Config: lr={config.lr}, grad_clip={grad_clip}, "
                    f"dropout={config.dropout}, d_model={config.d_model}\n")
            f.write("Epoch history:\n")
            for h in epoch_history:
                f.write(f"  {h}\n")
        print(f"Diagnostic saved -> {diag_path}")
        return None

    # ── Plot training curves ──────────────────────────────────────────────
    if epoch_history:
        if best_state is not None:
            best_state["epoch_history"] = epoch_history
        epochs_arr = [h["epoch"] for h in epoch_history]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].plot(epochs_arr, [h["train_loss"] for h in epoch_history], label="Train Loss")
        axes[0].plot(epochs_arr, [h["val_loss"] for h in epoch_history], label="Val Loss")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
        axes[0].set_title("CNN-Transformer — Loss"); axes[0].legend(); axes[0].grid(True)

        axes[1].plot(epochs_arr, [h["roc_auc"] for h in epoch_history], label="Val ROC-AUC", color="tab:orange")
        axes[1].plot(epochs_arr, [h["pr_auc"] for h in epoch_history], label="Val PR-AUC", color="tab:purple", linestyle="--")
        axes[1].axhline(y=best_metric_value, color="r", linestyle="--", alpha=0.5,
                        label=f"Best {early_stop_metric}={best_metric_value:.4f}")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("AUC")
        axes[1].set_title("CNN-Transformer — ROC-AUC & PR-AUC"); axes[1].legend(); axes[1].grid(True)
        axes[1].set_ylim(0, 1.05)

        axes[2].plot(epochs_arr, [h["f1"] for h in epoch_history], label="Val Best-F1", color="tab:green")
        axes[2].plot(epochs_arr, [h["precision"] for h in epoch_history], label="Val Precision", color="tab:blue", linestyle="--")
        axes[2].plot(epochs_arr, [h["recall"] for h in epoch_history], label="Val Recall", color="tab:red", linestyle="--")
        axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Score")
        axes[2].set_title("CNN-Transformer — F1 / Precision / Recall"); axes[2].legend(); axes[2].grid(True)
        axes[2].set_ylim(0, 1.05)

        plt.tight_layout()
        curve_path = os.path.join(config.output_dir, "cnn_transformer_training_curves.png")
        plt.savefig(curve_path, dpi=160, bbox_inches="tight")
        plt.show()
        print(f"Saved training curves -> {curve_path}")

    if best_state is None:
        print("Training failed to improve beyond initialization.")
        return None

    # ── Reload best checkpoint for final evaluation ───────────────────────
    print(f"\nReloading best checkpoint from epoch {best_epoch} "
          f"({early_stop_metric}={best_metric_value:.4f})")
    final_model = model.module if isinstance(model, DataParallel) else model
    final_model.load_state_dict(best_state["model_state_dict"])

    # ── Final evaluation on held-out test set ─────────────────────────────
    if test_loader is not None and len(test_loader.dataset) > 0:
        test_loss, test_metrics, test_probs, test_targets = _eval_epoch_with_threshold(
            model, test_loader, criterion, device, threshold=best_val_threshold
        )
        # Test confusion matrix
        test_preds = binary_predictions_from_proba(test_probs, best_val_threshold)
        test_tn = int(((test_preds == 0) & (test_targets == 0)).sum())
        test_fp = int(((test_preds == 1) & (test_targets == 0)).sum())
        test_fn = int(((test_preds == 0) & (test_targets == 1)).sum())
        test_tp = int(((test_preds == 1) & (test_targets == 1)).sum())

        print(
            f"\n{'='*70}\n"
            f"  TEST SET RESULTS (held-out, never used for training/validation)\n"
            f"{'='*70}\n"
            f"  Best Epoch:      {best_epoch}\n"
            f"  Threshold:       {best_val_threshold:.4f} (tuned on validation, max F1)\n"
            f"  ROC-AUC:         {test_metrics['auc_roc']:.4f}\n"
            f"  PR-AUC:          {test_metrics['auc_pr']:.4f}\n"
            f"  F1-Score:        {test_metrics['f1_score']:.4f}\n"
            f"  Precision:       {test_metrics['precision']:.4f}\n"
            f"  Recall:          {test_metrics['recall']:.4f}\n"
            f"  Accuracy:        {test_metrics['accuracy']:.4f}\n"
            f"  Confusion Matrix:\n"
            f"    TN={test_tn:>8,}  FP={test_fp:>8,}\n"
            f"    FN={test_fn:>8,}  TP={test_tp:>8,}\n"
            f"{'='*70}"
        )
        best_state["test_metrics"] = test_metrics
        best_state["test_confusion_matrix"] = {"tn": test_tn, "fp": test_fp, "fn": test_fn, "tp": test_tp}
    else:
        print("No held-out test set configured; skipping test evaluation.")

    model_path = os.path.join(config.output_dir, "cnn_transformer_ids.pth")
    torch.save(best_state, model_path)
    print(f"Saved CNN-Transformer checkpoint -> {model_path}")

    preprocess_artifacts = {
        "feature_columns": feature_cols,
        "medians": medians.to_dict(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    preprocess_path = os.path.join(config.output_dir, "cnn_transformer_preprocess.pkl")
    joblib.dump(preprocess_artifacts, preprocess_path)
    print(f"Saved preprocessing artifacts -> {preprocess_path}")

    generate_ig_report(
        final_model,
        X_val,
        feature_cols,
        config.output_dir,
        steps=config.ig_steps,
        sample_size=config.ig_samples,
        seed=config.random_state,
    )

    generate_gradcam_report(
        final_model,
        X_val,
        feature_cols,
        config.output_dir,
        sample_size=config.ig_samples,
        seed=config.random_state,
    )

    return model_path
