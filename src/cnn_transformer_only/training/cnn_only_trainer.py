import gc
import os
import random

import joblib
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


def _train_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    for data, target in loader:
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(data)
        loss = criterion(logits, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        running_loss += loss.item()
    return running_loss / max(len(loader), 1)


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
     scaler, medians, feature_cols, prep_meta) = prepare_training_data(
        X, y, feature_cols, config, source_groups=source_groups,
    )
    del X; gc.collect()

    balancer = IntelligentDataBalancer(config.undersampling_ratio, config.random_state)
    X_train_bal, y_train_bal = balancer.balance_classes(X_train, y_train)

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

    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = (
        optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.lr,
            epochs=config.epochs,
            steps_per_epoch=max(len(train_loader), 1),
        )
        if len(train_loader) > 0
        else None
    )

    best_auc = 0.0
    best_state = None

    for epoch in range(1, config.epochs + 1):
        train_loss = _train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        val_loss, metrics, _, _ = _eval_epoch(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch:02d} | Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | "
            f"ROC-AUC {metrics['auc_roc']:.4f} | F1 {metrics['f1_score']:.4f}"
        )

        if metrics["auc_roc"] > best_auc:
            best_auc = metrics["auc_roc"]
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
                "metrics": metrics,
                "config": config.__dict__,
                "feature_columns": feature_cols,
                "preprocessor": preprocess_state,
                "model_type": "cnn_classifier",
            }

    if best_state is None:
        print("Training failed to improve beyond initialization.")
        return None

    # Tune probability threshold on validation set (maximize F1)
    best_threshold = 0.5
    best_val_f1 = 0.0
    if val_loader is not None and len(val_loader.dataset) > 0:
        final_model_eval = model.module if isinstance(model, DataParallel) else model
        final_model_eval.load_state_dict(best_state["model_state_dict"])
        _, _, val_probs, val_targets = _eval_epoch(model, val_loader, criterion, device)
        best_threshold, best_val_f1 = find_best_f1_threshold(val_targets, val_probs)
        best_state["best_threshold"] = best_threshold
        best_state["best_val_f1_at_threshold"] = best_val_f1
        print(
            f"Best val threshold (maximize F1): {best_threshold:.4f} (val F1 {best_val_f1:.4f})"
        )

    # Final evaluation on held-out test set
    if test_loader is not None and len(test_loader.dataset) > 0:
        final_model_eval = model.module if isinstance(model, DataParallel) else model
        final_model_eval.load_state_dict(best_state["model_state_dict"])
        test_loss, test_metrics, _, _ = _eval_epoch_with_threshold(
            model, test_loader, criterion, device, threshold=best_threshold
        )
        print(
            f"\n{'='*60}\n"
            f"CNN CLASSIFIER — TEST SET RESULTS\n"
            f"{'='*60}\n"
            f"  Loss:      {test_loss:.4f}\n"
            f"  Threshold: {best_threshold:.4f} (tuned on validation, max F1)\n"
            f"  ROC-AUC:   {test_metrics['auc_roc']:.4f}\n"
            f"  PR-AUC:    {test_metrics['auc_pr']:.4f}\n"
            f"  F1-Score:  {test_metrics['f1_score']:.4f}\n"
            f"  Precision: {test_metrics['precision']:.4f}\n"
            f"  Recall:    {test_metrics['recall']:.4f}\n"
            f"  Accuracy:  {test_metrics['accuracy']:.4f}\n"
            f"{'='*60}"
        )
        best_state["test_metrics"] = test_metrics
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
