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
    return_source_groups: bool = False,
):
    """Load CICIDS CSV(s) into float32 feature matrix + binary labels using chunked IO.

    NaN values are preserved (inf is replaced with NaN) for proper downstream
    imputation.  When *return_source_groups* is True an additional int32 array
    is returned that maps each row to its source CSV file index (useful for
    grouped splitting to reduce data leakage).
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
        g_data = np.empty((cap,), dtype=np.int32) if return_source_groups else None
    else:
        x_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        g_parts: list[np.ndarray] = [] if return_source_groups else None

    rows_written = 0
    for file_idx, path in enumerate(csv_paths):
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
            # Replace inf/-inf with NaN; keep NaN for proper imputation downstream
            x_chunk[np.isinf(x_chunk)] = np.nan

            if use_cap:
                remaining = cap - rows_written
                if remaining <= 0:
                    break
                take = min(len(y_chunk), remaining)
                if take > 0:
                    x_data[rows_written:rows_written + take] = x_chunk[:take]
                    y_data[rows_written:rows_written + take] = y_chunk[:take]
                    if return_source_groups:
                        g_data[rows_written:rows_written + take] = file_idx
                    rows_written += take
            else:
                x_parts.append(x_chunk)
                y_parts.append(y_chunk)
                if return_source_groups:
                    g_parts.append(np.full(len(y_chunk), file_idx, dtype=np.int32))
                rows_written += len(y_chunk)

            del chunk, x_chunk_df, x_chunk, y_chunk

        if use_cap and rows_written >= cap:
            break

    if rows_written == 0:
        raise ValueError("No rows were loaded from the provided CICIDS input")

    if use_cap:
        result = (x_data[:rows_written], y_data[:rows_written], feature_cols, label_col)
        if return_source_groups:
            result = result + (g_data[:rows_written],)
        return result

    x_all = np.concatenate(x_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)
    result = (x_all, y_all, feature_cols, label_col)
    if return_source_groups:
        g_all = np.concatenate(g_parts, axis=0)
        result = result + (g_all,)
    return result


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

    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Replace NaN/inf in probabilities to prevent sklearn errors
    nan_mask = ~np.isfinite(y_prob)
    if nan_mask.any():
        y_prob = np.nan_to_num(y_prob, nan=0.0, posinf=1.0, neginf=0.0)

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


def prepare_training_data(
    X_np: np.ndarray,
    y: np.ndarray,
    feature_cols: list[str],
    config,
    source_groups: np.ndarray | None = None,
):
    """Enhanced preprocessing: dedup → clean → split → impute → transform → scale.

    All transforms are fit on training data only to prevent data leakage.

    Returns
    -------
    tuple of (X_train, X_val, X_test, y_train, y_val, y_test,
              scaler, train_medians, feature_cols, preprocess_meta)
    """
    import gc

    from scipy.stats import skew as compute_skew
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
    from sklearn.preprocessing import StandardScaler

    feature_cols = list(feature_cols)

    # ── 1. Replace inf/-inf with NaN ──────────────────────────────────
    inf_count = int(np.isinf(X_np).sum())
    if inf_count > 0:
        X_np = X_np.copy()
        X_np[np.isinf(X_np)] = np.nan
        print(f"  Replaced {inf_count:,} inf values with NaN")

    # ── 2. Remove duplicate rows ──────────────────────────────────────
    n_before = len(y)
    row_view = np.ascontiguousarray(X_np).view(
        np.dtype((np.void, X_np.dtype.itemsize * X_np.shape[1]))
    ).ravel()
    _, uniq_idx = np.unique(row_view, return_index=True)
    uniq_idx.sort()
    if len(uniq_idx) < n_before:
        X_np = X_np[uniq_idx]
        y = y[uniq_idx]
        if source_groups is not None:
            source_groups = source_groups[uniq_idx]
        print(f"  Removed {n_before - len(uniq_idx):,} duplicate rows "
              f"({n_before:,} -> {len(uniq_idx):,})")
    del row_view

    # ── 3. Drop constant / zero-variance columns ─────────────────────
    col_var = np.nanvar(X_np, axis=0)
    const_mask = col_var == 0
    if const_mask.any():
        dropped = [feature_cols[i] for i, m in enumerate(const_mask) if m]
        keep = ~const_mask
        feature_cols = [feature_cols[i] for i, m in enumerate(const_mask) if not m]
        X_np = X_np[:, keep]
        print(f"  Dropped {len(dropped)} zero-variance column(s): {dropped[:5]}")

    # ── 4. Drop duplicate columns ─────────────────────────────────────
    n_cols = X_np.shape[1]
    dup_set: set[int] = set()
    sample_n = min(5_000, len(X_np))
    col_hash: dict[int, int] = {}
    for i in range(n_cols):
        h = hash(np.nan_to_num(X_np[:sample_n, i], nan=-999.0).tobytes())
        if h in col_hash:
            j = col_hash[h]
            if np.array_equal(
                np.nan_to_num(X_np[:, i], nan=-999.0),
                np.nan_to_num(X_np[:, j], nan=-999.0),
            ):
                dup_set.add(i)
        else:
            col_hash[h] = i
    if dup_set:
        keep_idx = [i for i in range(n_cols) if i not in dup_set]
        dropped = [feature_cols[i] for i in sorted(dup_set)]
        feature_cols = [feature_cols[i] for i in keep_idx]
        X_np = X_np[:, keep_idx]
        print(f"  Dropped {len(dropped)} duplicate column(s): {dropped[:5]}")

    # Save CSV-level feature list (before indicators & correlation drop)
    csv_feature_cols = list(feature_cols)

    # ── 5. Split (grouped or stratified) ──────────────────────────────
    val_ratio = getattr(config, "val_size", 0.1)
    test_ratio = config.test_size
    holdout_ratio = val_ratio + test_ratio
    use_grouped = (
        getattr(config, "grouped_split", True)
        and source_groups is not None
    )

    split_done = False
    if use_grouped:
        unique_groups = np.unique(source_groups)
        if len(unique_groups) >= 3:
            gss = GroupShuffleSplit(
                n_splits=1, test_size=holdout_ratio,
                random_state=config.random_state,
            )
            trn_idx, hld_idx = next(gss.split(X_np, y, source_groups))
            X_train_raw = X_np[trn_idx]
            y_train = y[trn_idx]
            X_holdout = X_np[hld_idx]
            y_holdout = y[hld_idx]

            if val_ratio > 0 and test_ratio > 0 and len(y_holdout) > 1:
                test_frac = test_ratio / holdout_ratio
                try:
                    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
                        X_holdout, y_holdout, test_size=test_frac,
                        stratify=y_holdout, random_state=config.random_state,
                    )
                except ValueError:
                    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
                        X_holdout, y_holdout, test_size=test_frac,
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

            split_done = True
            trn_atk = y_train.sum() / len(y_train) * 100
            print(f"  Grouped split ({len(unique_groups)} source files) -> "
                  f"train {len(y_train):,} ({trn_atk:.1f}% attack), "
                  f"val {len(y_val):,}, test {len(y_test):,}")
        else:
            print(f"  Too few groups ({len(unique_groups)}) for grouped split; "
                  f"falling back to stratified")

    if not split_done:
        X_train_raw, X_holdout, y_train, y_holdout = train_test_split(
            X_np, y, test_size=holdout_ratio,
            stratify=y, random_state=config.random_state,
        )
        if val_ratio > 0 and test_ratio > 0 and len(y_holdout) > 0:
            test_frac = test_ratio / holdout_ratio
            X_val_raw, X_test_raw, y_val, y_test = train_test_split(
                X_holdout, y_holdout, test_size=test_frac,
                stratify=y_holdout, random_state=config.random_state,
            )
        elif val_ratio > 0:
            X_val_raw, y_val = X_holdout, y_holdout
            X_test_raw = np.empty((0, X_np.shape[1]), dtype=np.float32)
            y_test = np.empty(0, dtype=np.int64)
        else:
            X_test_raw, y_test = X_holdout, y_holdout
            X_val_raw = np.empty((0, X_np.shape[1]), dtype=np.float32)
            y_val = np.empty(0, dtype=np.int64)
        print(f"  Stratified split -> train {len(y_train):,}, "
              f"val {len(y_val):,}, test {len(y_test):,}")

    del X_np
    try:
        del X_holdout, y_holdout
    except NameError:
        pass
    gc.collect()

    # ── 6. Missing indicators + median imputation (train-fit) ─────────
    train_medians_arr = np.nanmedian(X_train_raw, axis=0)

    # Find columns with NaN in training data
    cols_with_nan = [
        i for i in range(X_train_raw.shape[1])
        if np.isnan(X_train_raw[:, i]).any()
    ]

    # Create missing-indicator flags BEFORE imputing
    indicator_source_names: list[str] = [feature_cols[i] for i in cols_with_nan]
    indicator_names: list[str] = [f"{feature_cols[i]}_missing" for i in cols_with_nan]

    if cols_with_nan:
        train_ind = np.isnan(X_train_raw[:, cols_with_nan]).astype(np.float32)
        val_ind = (
            np.isnan(X_val_raw[:, cols_with_nan]).astype(np.float32)
            if len(X_val_raw) > 0
            else np.empty((0, len(cols_with_nan)), dtype=np.float32)
        )
        test_ind = (
            np.isnan(X_test_raw[:, cols_with_nan]).astype(np.float32)
            if len(X_test_raw) > 0
            else np.empty((0, len(cols_with_nan)), dtype=np.float32)
        )
        print(f"  Created {len(cols_with_nan)} missing-indicator column(s): "
              f"{indicator_names[:5]}")

    # Impute with train medians
    def _impute(X, med_arr):
        for ci in range(X.shape[1]):
            mask = np.isnan(X[:, ci])
            if mask.any():
                fill = med_arr[ci] if np.isfinite(med_arr[ci]) else 0.0
                X[mask, ci] = fill

    _impute(X_train_raw, train_medians_arr)
    if len(X_val_raw) > 0:
        _impute(X_val_raw, train_medians_arr)
    if len(X_test_raw) > 0:
        _impute(X_test_raw, train_medians_arr)

    # Save medians for inference imputation (pre-log1p, pre-scale)
    imputation_medians = pd.Series(train_medians_arr, index=feature_cols[:len(train_medians_arr)])

    # Append indicator columns
    if cols_with_nan:
        X_train_raw = np.hstack([X_train_raw, train_ind])
        if len(X_val_raw) > 0:
            X_val_raw = np.hstack([X_val_raw, val_ind])
        if len(X_test_raw) > 0:
            X_test_raw = np.hstack([X_test_raw, test_ind])
        feature_cols = feature_cols + indicator_names
        del train_ind, val_ind, test_ind

    # ── 7. Drop highly correlated features (train-based) ─────────────
    corr_thresh = getattr(config, "correlation_threshold", 0.95)
    if corr_thresh < 1.0 and X_train_raw.shape[1] > 1:
        n_corr_sample = min(50_000, len(X_train_raw))
        corr = np.corrcoef(X_train_raw[:n_corr_sample], rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        to_drop: set[int] = set()
        n_feats = corr.shape[0]
        for i in range(n_feats):
            if i in to_drop:
                continue
            for j in range(i + 1, n_feats):
                if j in to_drop:
                    continue
                if abs(corr[i, j]) > corr_thresh:
                    to_drop.add(j)
        if to_drop:
            keep = sorted(set(range(n_feats)) - to_drop)
            dropped = [feature_cols[i] for i in sorted(to_drop)]
            feature_cols = [feature_cols[i] for i in keep]
            X_train_raw = X_train_raw[:, keep]
            if len(X_val_raw) > 0:
                X_val_raw = X_val_raw[:, keep]
            if len(X_test_raw) > 0:
                X_test_raw = X_test_raw[:, keep]
            print(f"  Dropped {len(dropped)} correlated feature(s) "
                  f"(|r| > {corr_thresh}): {dropped[:5]}")
        del corr

    # ── 8. Log1p on highly skewed features (train-based) ──────────────
    skew_thresh = getattr(config, "skew_threshold", 5.0)
    log1p_col_names: list[str] = []
    if skew_thresh > 0 and X_train_raw.shape[0] > 10:
        skewness = compute_skew(X_train_raw, axis=0, nan_policy="omit")
        log1p_indices: list[int] = []
        for i, s in enumerate(skewness):
            name = feature_cols[i]
            if name in indicator_names:
                continue
            if abs(s) > skew_thresh and X_train_raw[:, i].min() >= 0:
                log1p_indices.append(i)
                log1p_col_names.append(name)
        if log1p_indices:
            for i in log1p_indices:
                X_train_raw[:, i] = np.log1p(X_train_raw[:, i])
                if len(X_val_raw) > 0:
                    X_val_raw[:, i] = np.log1p(np.clip(X_val_raw[:, i], 0, None))
                if len(X_test_raw) > 0:
                    X_test_raw[:, i] = np.log1p(np.clip(X_test_raw[:, i], 0, None))
            print(f"  Applied log1p to {len(log1p_indices)} skewed feature(s): "
                  f"{log1p_col_names[:5]}")

    # ── 9. Scale (fit on train only) ──────────────────────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    del X_train_raw; gc.collect()

    X_val = (
        scaler.transform(X_val_raw).astype(np.float32)
        if len(X_val_raw) > 0
        else np.empty((0, X_train.shape[1]), dtype=np.float32)
    )
    del X_val_raw; gc.collect()

    X_test = (
        scaler.transform(X_test_raw).astype(np.float32)
        if len(X_test_raw) > 0
        else np.empty((0, X_train.shape[1]), dtype=np.float32)
    )
    del X_test_raw; gc.collect()

    # Build medians series covering ALL final feature columns
    full_med: dict[str, float] = imputation_medians.to_dict()
    for name in indicator_names:
        if name in feature_cols:
            full_med[name] = 0.0
    train_medians = pd.Series({col: full_med.get(col, 0.0) for col in feature_cols})

    preprocess_meta = {
        "csv_feature_cols": csv_feature_cols,
        "log1p_columns": log1p_col_names,
        "indicator_source_columns": indicator_source_names,
    }

    print(f"  Final feature count: {len(feature_cols)}")

    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        scaler, train_medians, feature_cols, preprocess_meta,
    )
