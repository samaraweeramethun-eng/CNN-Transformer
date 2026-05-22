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
from cnn_transformer_only.models.cnn_classifier import CNNClassifier
from cnn_transformer_only.utils.device import setup_device


def _set_seeds(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# _prepare_scaled_data removed — replaced by prepare_training_data in data.py


def _check_nan_inf(tensor) -> bool:
    """Return True if tensor contains NaN or Inf."""
    if tensor is None:
        return False
    return bool(torch.isnan(tensor).any() or torch.isinf(tensor).any())


def _compute_grad_norm(model) -> float:
    """Compute total gradient L2 norm across all parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


def _train_epoch(model, loader, criterion, optimizer, scheduler, device, grad_clip: float = 0.5):
    """Train one epoch; returns (avg_loss, nan_detected)."""
    model.train()
    running_loss = 0.0
    nan_detected = False

    for data, target in loader:
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(data)
        loss = criterion(logits, target)

        if _check_nan_inf(loss):
            nan_detected = True
            break

        loss.backward()

        norm_pre = _compute_grad_norm(model)
        if not np.isfinite(norm_pre):
            nan_detected = True
            break

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        running_loss += loss.item()

    return running_loss / max(len(loader), 1), nan_detected


def _eval_epoch(model, loader, criterion, device):
    model.eval()
    losses = []
    all_preds, all_probs, all_targets = [], [], []
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            logits = model(data)
            loss = criterion(logits, target)
            losses.append(loss.item())
            probs = F.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    metrics = calculate_comprehensive_metrics(
        np.array(all_targets), np.array(all_preds), np.array(all_probs)
    )
    return (
        np.mean(losses) if losses else 0.0,
        metrics,
        np.array(all_probs),
        np.array(all_targets),
    )


def _eval_epoch_with_threshold(model, loader, criterion, device, threshold: float):
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
            all_probs.append(probs.detach().cpu().numpy())
            all_targets.append(target.detach().cpu().numpy())

    if not all_probs:
        return (
            0.0,
            {key: 0.0 for key in ["accuracy", "auc_roc", "auc_pr", "f1_score", "precision", "recall"]},
            np.array([]),
            np.array([]),
        )

    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_targets)
    y_pred = binary_predictions_from_proba(y_prob, threshold=threshold)
    metrics = calculate_comprehensive_metrics(y_true, y_pred, y_prob)
    return float(np.mean(losses)) if losses else 0.0, metrics, y_prob, y_true


# ── public entry point ──────────────────────────────────────────────

def train_cnn_classifier(config: CNNTransformerConfig | None = None):
    """Train a standalone CNN classifier and return the checkpoint path."""
    config = config or CNNTransformerConfig()
    _set_seeds(config.random_state)

    device, multi_gpu = setup_device()
    if multi_gpu:
        config.batch_size = 512
        config.val_batch_size = 1024

    os.makedirs(config.output_dir, exist_ok=True)
    grad_clip = getattr(config, "grad_clip", 0.5)

    print("Loading dataset for CNN classifier training...")
    X, y, feature_cols, _, source_groups = load_cicids_feature_matrix(
        config.input_path,
        max_rows=getattr(config, "max_rows", 0),
        chunksize=getattr(config, "csv_chunksize", 200_000),
        return_source_groups=True,
    )
    print(f"Loaded rows: {len(y):,} | Features: {len(feature_cols)}")

    print("Running enhanced preprocessing pipeline...")
    (X_train, X_val, X_test, y_train, y_val, y_test,
     scaler, medians, feature_cols, prep_meta, test_block_map) = prepare_training_data(
        X, y, feature_cols, config, source_groups=source_groups,
    )
    del X; gc.collect()

    balancer = IntelligentDataBalancer(config.undersampling_ratio, config.random_state)
    X_train_bal, y_train_bal = balancer.balance_classes(X_train, y_train)

    # Compute class weights for cost-sensitive loss (before deleting labels)
    n_benign_train = int((y_train_bal == 0).sum())
    n_attack_train = int((y_train_bal == 1).sum())
    class_weight_tensor = torch.tensor(
        [1.0, n_benign_train / max(n_attack_train, 1)],
        dtype=torch.float32,
    )
    print(f"Class weights: benign=1.00, attack={class_weight_tensor[1]:.2f}")

    input_dim = X_train.shape[1]
    del X_train, y_train; gc.collect()

    train_loader, val_loader, _ = build_dataloaders(
        X_train_bal, y_train_bal, X_val, y_val,
        batch_size=config.batch_size,
        val_batch_size=config.val_batch_size,
        num_workers=config.num_workers,
    )

    # Build test loader
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

    del X_train_bal, y_train_bal, X_test, y_test; gc.collect()

    model = CNNClassifier(
        input_dim=input_dim,
        conv_channels=config.conv_channels,
        fc_dim=config.cnn_fc_dim,
        dropout=config.dropout,
    ).to(device)

    if multi_gpu:
        model = DataParallel(model)

    criterion = nn.CrossEntropyLoss(
        weight=class_weight_tensor.to(device),
        label_smoothing=config.label_smoothing,
    )
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # Warmup + cosine annealing
    warmup_epochs = getattr(config, "warmup_epochs", 3)
    steps_per_epoch = max(len(train_loader), 1)
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = config.epochs * steps_per_epoch

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda) if total_steps > 0 else None

    checkpoint_metric = getattr(config, "checkpoint_metric", "best_f1")
    patience = getattr(config, "patience", 7)

    best_roc_auc = {"value": 0.0, "epoch": 0, "state": None, "threshold": 0.5}
    best_f1 = {"value": 0.0, "epoch": 0, "state": None, "threshold": 0.5}
    best_val_loss = {"value": float("inf"), "epoch": 0, "state": None, "threshold": 0.5}

    no_improve = 0
    epoch_history: list[dict] = []
    nan_abort = False

    for epoch in range(1, config.epochs + 1):
        train_loss, nan_detected = _train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, grad_clip=grad_clip
        )

        if nan_detected:
            print(f"\n⚠️  NaN/Inf detected at epoch {epoch}! Stopping training safely.")
            nan_abort = True
            break

        val_loss, _, val_probs, val_targets = _eval_epoch(model, val_loader, criterion, device)

        # Fine threshold search on validation
        val_best_thr, val_best_f1 = find_best_f1_threshold(val_targets, val_probs)
        val_preds_tuned = binary_predictions_from_proba(val_probs, val_best_thr)
        val_metrics_tuned = calculate_comprehensive_metrics(val_targets, val_preds_tuned, val_probs)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:02d} | Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | "
            f"ROC-AUC {val_metrics_tuned['auc_roc']:.4f} | F1 {val_best_f1:.4f}@{val_best_thr:.3f} | "
            f"LR {current_lr:.2e}"
        )
        epoch_history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "roc_auc": val_metrics_tuned["auc_roc"], "f1": val_best_f1,
            "precision": val_metrics_tuned["precision"], "recall": val_metrics_tuned["recall"],
            "best_threshold": val_best_thr,
        })

        # ── Multi-checkpoint saving ────────────────────────────────────────
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

        def _make_checkpoint():
            return {
                "model_state_dict": state_dict.copy(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": val_metrics_tuned,
                "config": config.__dict__,
                "feature_columns": feature_cols,
                "preprocessor": preprocess_state,
                "model_type": "cnn_classifier",
                "epoch": epoch,
                "threshold": val_best_thr,
            }

        if val_metrics_tuned["auc_roc"] > best_roc_auc["value"]:
            best_roc_auc["value"] = val_metrics_tuned["auc_roc"]
            best_roc_auc["epoch"] = epoch
            best_roc_auc["state"] = _make_checkpoint()
            best_roc_auc["threshold"] = val_best_thr

        if val_best_f1 > best_f1["value"]:
            best_f1["value"] = val_best_f1
            best_f1["epoch"] = epoch
            best_f1["state"] = _make_checkpoint()
            best_f1["threshold"] = val_best_thr

        if val_loss < best_val_loss["value"]:
            best_val_loss["value"] = val_loss
            best_val_loss["epoch"] = epoch
            best_val_loss["state"] = _make_checkpoint()
            best_val_loss["threshold"] = val_best_thr

        # Early stopping based on checkpoint_metric
        metric_improved = False
        if checkpoint_metric == "best_f1" and best_f1["epoch"] == epoch:
            metric_improved = True
        elif checkpoint_metric == "val_loss" and best_val_loss["epoch"] == epoch:
            metric_improved = True
        elif checkpoint_metric in ("roc_auc", "pr_auc") and best_roc_auc["epoch"] == epoch:
            metric_improved = True

        if metric_improved:
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience})")
                break

    # ── Handle NaN abort ───────────────────────────────────────────────
    if nan_abort and best_f1["state"] is None and best_roc_auc["state"] is None:
        print("\n⚠️  Training aborted due to NaN/Inf before any valid checkpoint was saved.")
        return None

    # ── Plot training curves (loss, ROC-AUC, F1) ─────────────────────
    # Select best checkpoint based on configured metric
    if checkpoint_metric == "best_f1" and best_f1["state"] is not None:
        best_state = best_f1["state"]
        best_threshold = best_f1["threshold"]
    elif checkpoint_metric == "val_loss" and best_val_loss["state"] is not None:
        best_state = best_val_loss["state"]
        best_threshold = best_val_loss["threshold"]
    elif best_roc_auc["state"] is not None:
        best_state = best_roc_auc["state"]
        best_threshold = best_roc_auc["threshold"]
    elif best_f1["state"] is not None:
        best_state = best_f1["state"]
        best_threshold = best_f1["threshold"]
    else:
        best_state = None
        best_threshold = 0.5

    if epoch_history:
        if best_state is not None:
            best_state["epoch_history"] = epoch_history
        epochs_arr = [h["epoch"] for h in epoch_history]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].plot(epochs_arr, [h["train_loss"] for h in epoch_history], label="Train Loss")
        axes[0].plot(epochs_arr, [h["val_loss"] for h in epoch_history], label="Val Loss")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
        axes[0].set_title("CNN Classifier — Loss"); axes[0].legend(); axes[0].grid(True)

        axes[1].plot(epochs_arr, [h["roc_auc"] for h in epoch_history], label="Val ROC-AUC", color="tab:orange")
        axes[1].axhline(y=best_roc_auc["value"], color="r", linestyle="--", alpha=0.5, label=f"Best={best_roc_auc['value']:.4f}")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("ROC-AUC")
        axes[1].set_title("CNN Classifier — ROC-AUC"); axes[1].legend(); axes[1].grid(True)
        axes[1].set_ylim(0, 1.05)

        axes[2].plot(epochs_arr, [h["f1"] for h in epoch_history], label="Val F1", color="tab:green")
        axes[2].plot(epochs_arr, [h["precision"] for h in epoch_history], label="Val Precision", color="tab:blue", linestyle="--")
        axes[2].plot(epochs_arr, [h["recall"] for h in epoch_history], label="Val Recall", color="tab:red", linestyle="--")
        axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Score")
        axes[2].set_title("CNN Classifier — F1 / Precision / Recall"); axes[2].legend(); axes[2].grid(True)
        axes[2].set_ylim(0, 1.05)

        plt.tight_layout()
        curve_path = os.path.join(config.output_dir, "cnn_classifier_training_curves.png")
        plt.savefig(curve_path, dpi=160, bbox_inches="tight")
        plt.show()
        print(f"Saved training curves -> {curve_path}")

    if best_state is None:
        print("Training failed to improve beyond initialization.")
        return None

    best_state["best_threshold"] = best_threshold

    # Final evaluation on held-out test set
    if test_loader is not None and len(test_loader.dataset) > 0:
        final_model_eval = model.module if isinstance(model, DataParallel) else model
        final_model_eval.load_state_dict(best_state["model_state_dict"])
        test_loss, test_metrics, test_probs, test_targets = _eval_epoch_with_threshold(
            model, test_loader, criterion, device, threshold=best_threshold
        )
        
        # Test confusion matrix
        test_preds = binary_predictions_from_proba(test_probs, best_threshold)
        test_tn = int(((test_preds == 0) & (test_targets == 0)).sum())
        test_fp = int(((test_preds == 1) & (test_targets == 0)).sum())
        test_fn = int(((test_preds == 0) & (test_targets == 1)).sum())
        test_tp = int(((test_preds == 1) & (test_targets == 1)).sum())
        
        print(
            f"\n{'='*70}\n"
            f"CNN CLASSIFIER — TEST SET RESULTS\n"
            f"{'='*70}\n"
            f"  Loss:      {test_loss:.4f}\n"
            f"  Threshold: {best_threshold:.4f} (tuned on validation, max F1)\n"
            f"  ROC-AUC:   {test_metrics['auc_roc']:.4f}\n"
            f"  PR-AUC:    {test_metrics['auc_pr']:.4f}\n"
            f"  F1-Score:  {test_metrics['f1_score']:.4f}\n"
            f"  Precision: {test_metrics['precision']:.4f}\n"
            f"  Recall:    {test_metrics['recall']:.4f}\n"
            f"  Accuracy:  {test_metrics['accuracy']:.4f}\n"
            f"  Confusion Matrix:\n"
            f"    TN={test_tn:>8,}  FP={test_fp:>8,}\n"
            f"    FN={test_fn:>8,}  TP={test_tp:>8,}\n"
            f"{'='*70}"
        )
        best_state["test_metrics"] = test_metrics
        best_state["test_confusion_matrix"] = {"tn": test_tn, "fp": test_fp, "fn": test_fn, "tp": test_tp}
        
        # ── Per-block evaluation (if block assignments available) ─────────
        if test_block_map is not None:
            print(f"\n{'='*70}")
            print(f"  PER-BLOCK TEST EVALUATION")
            print(f"{'='*70}")
            
            unique_blocks = np.unique(test_block_map)
            print(f"  Evaluating {len(unique_blocks)} test blocks individually...")
            
            block_results = []
            for block_id in unique_blocks:
                block_mask = test_block_map == block_id
                block_targets = test_targets[block_mask]
                block_probs = test_probs[block_mask]
                block_preds = test_preds[block_mask]
                
                if len(block_targets) == 0:
                    continue
                
                # Compute metrics for this block
                block_metrics = calculate_comprehensive_metrics(
                    block_targets,
                    block_preds,
                    block_probs
                )
                
                # Confusion matrix
                b_tn = int(((block_preds == 0) & (block_targets == 0)).sum())
                b_fp = int(((block_preds == 1) & (block_targets == 0)).sum())
                b_fn = int(((block_preds == 0) & (block_targets == 1)).sum())
                b_tp = int(((block_preds == 1) & (block_targets == 1)).sum())
                
                block_results.append({
                    "block_id": int(block_id),
                    "n_samples": len(block_targets),
                    "attack_pct": float(block_targets.mean() * 100),
                    "roc_auc": block_metrics['auc_roc'],
                    "pr_auc": block_metrics['auc_pr'],
                    "f1": block_metrics['f1_score'],
                    "precision": block_metrics['precision'],
                    "recall": block_metrics['recall'],
                    "tn": b_tn, "fp": b_fp, "fn": b_fn, "tp": b_tp,
                })
            
            # Print results
            print(f"\n  {'Block':<7} {'Samples':>8} {'Attack%':>8} {'ROC-AUC':>8} {'PR-AUC':>8} {'F1':>8} {'Prec':>8} {'Rec':>8}")
            print(f"  {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
            
            for res in block_results:
                print(f"  {res['block_id']:<7} {res['n_samples']:>8,} {res['attack_pct']:>7.1f}% "
                      f"{res['roc_auc']:>8.4f} {res['pr_auc']:>8.4f} {res['f1']:>8.4f} "
                      f"{res['precision']:>8.4f} {res['recall']:>8.4f}")
            
            # Check for problematic blocks
            print(f"\n  Per-block diagnostics:")
            low_f1_blocks = [r for r in block_results if r['f1'] < 0.80]
            if low_f1_blocks:
                print(f"    ⚠️  {len(low_f1_blocks)} block(s) with F1 < 0.80:")
                for r in low_f1_blocks:
                    print(f"      Block {r['block_id']}: F1={r['f1']:.4f}, samples={r['n_samples']}, attack%={r['attack_pct']:.1f}%")
            else:
                print(f"    ✓ All blocks have F1 ≥ 0.80")
            
            # Check variance across blocks
            f1_scores = [r['f1'] for r in block_results]
            f1_std = float(np.std(f1_scores))
            f1_min = float(min(f1_scores))
            f1_max = float(max(f1_scores))
            print(f"    F1 range: [{f1_min:.4f}, {f1_max:.4f}], std={f1_std:.4f}")
            if f1_std > 0.10:
                print(f"    ⚠️  High F1 variance across blocks (std={f1_std:.4f})")
            else:
                print(f"    ✓ Consistent performance across blocks")
            
            best_state["per_block_results"] = block_results
        else:
            print(f"\n  Per-block evaluation: skipped (no block assignments available)")
    else:
        print("No held-out test set configured; skipping test evaluation.")

    model_path = os.path.join(config.output_dir, "cnn_classifier.pth")
    torch.save(best_state, model_path)
    print(f"Saved CNN classifier checkpoint -> {model_path}")

    preprocess_artifacts = {
        "feature_columns": feature_cols,
        "medians": medians.to_dict(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    preprocess_path = os.path.join(config.output_dir, "cnn_classifier_preprocess.pkl")
    joblib.dump(preprocess_artifacts, preprocess_path)
    print(f"Saved preprocessing artifacts -> {preprocess_path}")

    final_model = model.module if isinstance(model, DataParallel) else model

    generate_ig_report(
        final_model,
        X_val,
        feature_cols,
        config.output_dir,
        steps=config.ig_steps,
        sample_size=config.ig_samples,
        seed=config.random_state,
        prefix="cnn_classifier",
    )

    generate_gradcam_report(
        final_model,
        X_val,
        feature_cols,
        config.output_dir,
        sample_size=config.ig_samples,
        seed=config.random_state,
        prefix="cnn_classifier",
    )

    return model_path
