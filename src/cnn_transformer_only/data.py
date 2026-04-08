from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


class IntelligentDataBalancer:
    def __init__(self, undersampling_ratio: float = 0.12, random_state: int = 42):
        self.undersampling_ratio = undersampling_ratio
        self.random_state = random_state
        self._rng = np.random.RandomState(random_state)

    def balance_classes(self, X: np.ndarray, y: np.ndarray):
        majority_idx = np.where(y == 0)[0]
        minority_idx = np.where(y == 1)[0]
        if len(majority_idx) == 0 or len(minority_idx) == 0:
            return X, y
        target_majority = max(
            len(minority_idx) * 3,
            int(len(majority_idx) * self.undersampling_ratio),
        )
        if len(majority_idx) > target_majority:
            minority_center = X[minority_idx].mean(axis=0)
            CHUNK = 100_000
            distances = np.empty(len(majority_idx), dtype=np.float32)
            for start in range(0, len(majority_idx), CHUNK):
                end = min(start + CHUNK, len(majority_idx))
                diff = X[majority_idx[start:end]] - minority_center
                distances[start:end] = np.linalg.norm(diff, axis=1).astype(np.float32)
                del diff
            weights = 1.0 / (distances.astype(np.float64) + 1e-8)
            weights /= weights.sum()
            selected_majority = self._rng.choice(
                majority_idx,
                size=target_majority,
                replace=False,
                p=weights,
            )
            del distances, weights
        else:
            selected_majority = majority_idx
        combined_idx = np.concatenate([selected_majority, minority_idx])
        return X[combined_idx], y[combined_idx]


def resolve_cicids_csv_paths(input_path: str) -> list[str]:
    """Resolve input into a sorted list of CSV files.

    Supports:
    - Single CSV file path
    - Directory containing CSV files
    - Glob pattern (e.g. data/cicids2018/*.csv)
    - Comma-separated list of CSV paths
    """
    if not input_path or not input_path.strip():
        raise ValueError("Input path is empty")

    raw = input_path.strip()
    paths: list[str] = []

    if "," in raw:
        for item in raw.split(","):
            item = item.strip()
            if item:
                paths.append(item)
    elif os.path.isdir(raw):
        paths = sorted(glob.glob(os.path.join(raw, "*.csv")))
    elif any(ch in raw for ch in ["*", "?", "["]):
        paths = sorted(glob.glob(raw))
    else:
        paths = [raw]

    expanded = [os.path.abspath(p) for p in paths]
    csv_paths = [p for p in expanded if p.lower().endswith(".csv")]

    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found for input: {input_path}")

    missing = [p for p in csv_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing CSV file(s): {missing}")

    return sorted(csv_paths)


def load_cicids_dataframe(input_path: str) -> pd.DataFrame:
    """Load one or many CICIDS CSV files into a single dataframe."""
    csv_paths = resolve_cicids_csv_paths(input_path)
    if len(csv_paths) == 1:
        df = pd.read_csv(csv_paths[0], low_memory=False)
        df.columns = [str(col).strip() for col in df.columns]
        return df

    frames: list[pd.DataFrame] = []
    for path in csv_paths:
        df = pd.read_csv(path, low_memory=False)
        # CICIDS exports can include spacing inconsistencies in headers.
        df.columns = [str(col).strip() for col in df.columns]
        frames.append(df)

    merged = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    return merged


def load_cicids_feature_matrix(
    input_path: str,
    max_rows: int = 0,
    chunksize: int = 200_000,
) -> tuple[np.ndarray, np.ndarray, list[str], str]:
    """Load CICIDS CSV(s) into float32 feature matrix + binary labels using chunked IO.

    This avoids creating one giant dataframe in memory, which is important for Kaggle
    and other constrained environments.
    """
    csv_paths = resolve_cicids_csv_paths(input_path)

    header_df = pd.read_csv(csv_paths[0], nrows=0)
    header_df.columns = [str(col).strip() for col in header_df.columns]
    label_col = detect_label_column(header_df)

    blacklist = {label_col, "Flow ID", "Source IP", "Destination IP", "Timestamp"}
    feature_cols = [col for col in header_df.columns if col not in blacklist]
    feature_count = len(feature_cols)

    if feature_count == 0:
        raise ValueError("No feature columns found after excluding metadata/label columns")

    use_cap = max_rows is not None and int(max_rows) > 0
    cap = int(max_rows) if use_cap else 0

    if use_cap:
        x_data = np.empty((cap, feature_count), dtype=np.float32)
        y_data = np.empty((cap,), dtype=np.int8)
    else:
        x_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []

    rows_written = 0
    for path in csv_paths:
        for chunk in pd.read_csv(path, low_memory=False, chunksize=chunksize):
            chunk.columns = [str(col).strip() for col in chunk.columns]
            if label_col not in chunk.columns:
                raise ValueError(f"Label column '{label_col}' missing in file: {path}")

            labels = chunk[label_col].astype(str).str.strip().str.upper()
            y_chunk = (labels != "BENIGN").astype(np.int8).to_numpy()

            x_chunk_df = chunk.reindex(columns=feature_cols)
            obj_cols = x_chunk_df.select_dtypes(include=["object"]).columns
            if len(obj_cols) > 0:
                x_chunk_df = x_chunk_df.copy()
                for col in obj_cols:
                    x_chunk_df[col] = pd.to_numeric(x_chunk_df[col], errors="coerce")

            x_chunk = x_chunk_df.to_numpy(dtype=np.float32, copy=False)
            np.nan_to_num(x_chunk, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

            if use_cap:
                remaining = cap - rows_written
                if remaining <= 0:
                    break
                take = min(len(y_chunk), remaining)
                if take > 0:
                    x_data[rows_written:rows_written + take] = x_chunk[:take]
                    y_data[rows_written:rows_written + take] = y_chunk[:take]
                    rows_written += take
            else:
                x_parts.append(x_chunk)
                y_parts.append(y_chunk)
                rows_written += len(y_chunk)

            del chunk, x_chunk_df, x_chunk, y_chunk

        if use_cap and rows_written >= cap:
            break

    if rows_written == 0:
        raise ValueError("No rows were loaded from the provided CICIDS input")

    if use_cap:
        return x_data[:rows_written], y_data[:rows_written], feature_cols, label_col

    x_all = np.concatenate(x_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)
    return x_all, y_all, feature_cols, label_col


def detect_label_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if "label" in col.lower():
            return col
    raise ValueError("Label column not found in dataset")


def prepare_features(df: pd.DataFrame, label_col: str):
    """Extract features as a float32 numpy array to minimise memory."""
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    label_col = label_col.strip()
    label_series = df[label_col].astype(str).str.strip().str.upper()
    binary_label = (label_series != "BENIGN").astype(np.int8).values
    blacklist = {label_col, "Flow ID", "Source IP", "Destination IP", "Timestamp"}
    feature_cols = [col for col in df.columns if col not in blacklist]
    X = df[feature_cols]
    obj_cols = X.select_dtypes(include=["object"]).columns
    if len(obj_cols) > 0:
        X = X.copy()
        for col in obj_cols:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    X_np = X.values.astype(np.float32)
    np.nan_to_num(X_np, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return X_np, binary_label, feature_cols


def build_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
    val_batch_size: int,
    num_workers: int,
    pin_memory: bool | None = None,
):
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    use_persistent = num_workers > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=use_persistent,
        prefetch_factor=4 if use_persistent else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent,
        prefetch_factor=4 if use_persistent else None,
    )
    return train_loader, val_loader, val_dataset


def calculate_comprehensive_metrics(y_true, y_pred, y_prob):
    if len(y_true) == 0:
        return {key: 0.0 for key in ["accuracy", "auc_roc", "auc_pr", "f1_score", "precision", "recall"]}
    accuracy = float(np.mean(y_true == y_pred))
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        return {
            "accuracy": accuracy,
            "auc_roc": 0.5,
            "auc_pr": float(np.mean(y_true)),
            "f1_score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }

    from sklearn.metrics import (
        roc_auc_score,
        precision_recall_curve,
        auc,
        f1_score,
        precision_score,
        recall_score,
    )

    auc_roc = roc_auc_score(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auc_pr = auc(recall, precision)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    return {
        "accuracy": accuracy,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "f1_score": f1,
        "precision": prec,
        "recall": rec,
    }


def binary_predictions_from_proba(y_prob: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    probs = np.asarray(y_prob, dtype=np.float64)
    return (probs >= float(threshold)).astype(np.int64)


def find_best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64)
    if y_true_arr.size == 0 or y_prob_arr.size == 0:
        return 0.5, 0.0
    if np.unique(y_true_arr).size < 2:
        return 0.5, 0.0

    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true_arr, y_prob_arr)
    if thresholds.size == 0:
        return 0.5, 0.0

    p = precision[1:]
    r = recall[1:]
    denom = (p + r)
    f1 = np.where(denom > 0, 2 * p * r / denom, 0.0)
    best_idx = int(np.argmax(f1))
    return float(thresholds[best_idx]), float(f1[best_idx])
