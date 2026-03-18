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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader, TensorDataset

from cnn_transformer_only.config import CNNTransformerConfig
from cnn_transformer_only.data import (
    IntelligentDataBalancer,
    binary_predictions_from_proba,
    build_dataloaders,
    calculate_comprehensive_metrics,
    detect_label_column,
    find_best_f1_threshold,
    prepare_features,
)
from cnn_transformer_only.interpretability.grad_cam import generate_gradcam_report
from cnn_transformer_only.interpretability.integrated_gradients import generate_ig_report
from cnn_transformer_only.models.cnn_transformer import CNNTransformerIDS
from cnn_transformer_only.utils.device import setup_device


def _set_seeds(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _prepare_scaled_data(X_np: np.ndarray, y: np.ndarray, config: CNNTransformerConfig):
    """Three-way split (train/val/test), scale, return float32 arrays."""
    val_ratio = getattr(config, "val_size", 0.1)
    test_ratio = config.test_size
    holdout_ratio = val_ratio + test_ratio

    X_train_raw, X_holdout, y_train, y_holdout = train_test_split(
        X_np,
        y,
        test_size=holdout_ratio,
        stratify=y,
        random_state=config.random_state,
    )

    if val_ratio > 0 and test_ratio > 0 and len(y_holdout) > 0:
        test_fraction = test_ratio / holdout_ratio
        X_val_raw, X_test_raw, y_val, y_test = train_test_split(
            X_holdout,
            y_holdout,
            test_size=test_fraction,
            stratify=y_holdout,
            random_state=config.random_state,
        )
    elif val_ratio > 0:
        X_val_raw, y_val = X_holdout, y_holdout
        X_test_raw = np.empty((0, X_np.shape[1]), dtype=np.float32)
        y_test = np.empty(0, dtype=np.int64)
    else:
        X_test_raw, y_test = X_holdout, y_holdout
        X_val_raw = np.empty((0, X_np.shape[1]), dtype=np.float32)
        y_val = np.empty(0, dtype=np.int64)

    del X_holdout, y_holdout
    gc.collect()

    train_medians = pd.Series(np.nanmedian(X_train_raw, axis=0))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    del X_train_raw
    gc.collect()

    X_val = scaler.transform(X_val_raw).astype(np.float32)
    del X_val_raw
    gc.collect()

    X_test = (
        scaler.transform(X_test_raw).astype(np.float32)
        if len(X_test_raw) > 0
        else np.empty((0, X_train.shape[1]), dtype=np.float32)
    )
    del X_test_raw
    gc.collect()

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, train_medians


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


def train_cnn_transformer(config: CNNTransformerConfig | None = None):
    config = config or CNNTransformerConfig()
    _set_seeds(config.random_state)

    device, multi_gpu = setup_device()
    if multi_gpu:
        config.batch_size = 512
        config.val_batch_size = 1024

    os.makedirs(config.output_dir, exist_ok=True)

    print("Loading dataset for CNN-Transformer training...")
    df = pd.read_csv(config.input_path, low_memory=False)
    label_col = detect_label_column(df)
    X, y, feature_cols = prepare_features(df, label_col)
    del df
    gc.collect()

    X_train, X_val, X_test, y_train, y_val, y_test, scaler, medians = _prepare_scaled_data(
        X, y, config
    )
    del X
    gc.collect()

    balancer = IntelligentDataBalancer(config.undersampling_ratio, config.random_state)
    X_train_bal, y_train_bal = balancer.balance_classes(X_train, y_train)

    input_dim = X_train.shape[1]
    del X_train, y_train
    gc.collect()

    max_samples = getattr(config, "max_train_samples", 0)
    train_loader, val_loader, _ = build_dataloaders(
        X_train_bal,
        y_train_bal,
        X_val,
        y_val,
        batch_size=config.batch_size,
        val_batch_size=config.val_batch_size,
        num_workers=config.num_workers,
        max_train_samples=max_samples,
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
            }
            best_state = {
                "model_state_dict": state_dict,
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
                "config": config.__dict__,
                "feature_columns": feature_cols,
                "preprocessor": preprocess_state,
                "model_type": "cnn_transformer",
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
            f"TEST SET RESULTS (held-out, never used for training/validation)\n"
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

    final_model = model.module if isinstance(model, DataParallel) else model

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
