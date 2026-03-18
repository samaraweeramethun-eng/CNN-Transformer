import os
import numpy as np
import pandas as pd
import torch

from cnn_transformer_only.models.cnn_transformer import CNNTransformerIDS
from cnn_transformer_only.utils.device import setup_device
from cnn_transformer_only.data import detect_label_column


def load_checkpoint(checkpoint_path: str):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def _build_cnn_model(state, cfg, device):
    positional = state.get("positional")
    if positional is None:
        raise KeyError("CNN checkpoint missing positional embeddings; cannot infer input dimension")
    input_dim = positional.shape[1] - 1
    model = CNNTransformerIDS(
        input_dim=input_dim,
        d_model=cfg.get("d_model", 192),
        conv_channels=cfg.get("conv_channels", 96),
        num_layers=cfg.get("num_layers", 3),
        num_heads=cfg.get("num_heads", 8),
        d_ff=cfg.get("d_ff", 768),
        dropout=cfg.get("dropout", 0.2),
    ).to(device)
    return model


def build_model_from_ckpt(ckpt, device):
    state = ckpt["model_state_dict"]
    cfg = ckpt.get("config", {})
    model = _build_cnn_model(state, cfg, device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


class StandardScalerPreprocessor:
    def __init__(self, state: dict, feature_cols: list[str]):
        self.feature_cols = feature_cols
        medians = state.get("medians", {})
        self.fill_values = {col: medians.get(col, 0.0) for col in feature_cols}
        self.mean = np.array(state.get("mean", []), dtype=np.float32)
        self.scale = np.array(state.get("scale", []), dtype=np.float32)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        filled = df.fillna(self.fill_values)
        arr = filled.to_numpy(dtype=np.float32, copy=False)
        if self.mean.size != arr.shape[1] or self.scale.size != arr.shape[1]:
            return arr
        denom = np.where(self.scale == 0, 1.0, self.scale)
        return (arr - self.mean) / denom


def resolve_preprocessor(state, feature_cols):
    if state is None:
        return None
    if hasattr(state, "transform"):
        return state
    if isinstance(state, dict) and state.get("type") == "standard_scaler":
        return StandardScalerPreprocessor(state, feature_cols)
    return None


def prepare_eval_matrix(
    data_path: str,
    label_col: str,
    feature_cols: list,
    preprocessor=None,
    sample_size: int | None = None,
    random_state: int = 42,
    chunksize: int = 50_000,
):
    usecols = list(dict.fromkeys(feature_cols + [label_col]))
    rng = np.random.RandomState(random_state)
    reservoir_X: list[np.ndarray] = []
    reservoir_y: list[int] = []
    total_rows = 0

    for chunk in pd.read_csv(data_path, usecols=usecols, chunksize=chunksize):
        chunk["binary_label"] = (chunk[label_col] != "BENIGN").astype(int)
        X_chunk = chunk[feature_cols].copy()
        for col in X_chunk.select_dtypes(include=["object"]).columns:
            X_chunk[col] = pd.to_numeric(X_chunk[col], errors="coerce")
        X_chunk = X_chunk.replace([np.inf, -np.inf], np.nan).fillna(X_chunk.median())
        X_vals = X_chunk.to_numpy(dtype=np.float32, copy=False)
        y_vals = chunk["binary_label"].to_numpy()

        for row, label in zip(X_vals, y_vals, strict=False):
            if sample_size is None or len(reservoir_X) < sample_size:
                reservoir_X.append(row.copy())
                reservoir_y.append(int(label))
            else:
                swap_idx = rng.randint(0, total_rows + 1)
                if swap_idx < sample_size:
                    reservoir_X[swap_idx] = row.copy()
                    reservoir_y[swap_idx] = int(label)
            total_rows += 1

    if not reservoir_X:
        return np.empty((0, len(feature_cols))), np.empty(0, dtype=int), total_rows

    X_array = np.stack(reservoir_X).astype(np.float32, copy=False)
    y_array = np.array(reservoir_y, dtype=int)
    perm = rng.permutation(len(y_array))
    X_array = X_array[perm]
    y_array = y_array[perm]

    if preprocessor is not None:
        X_df = pd.DataFrame(X_array, columns=feature_cols)
        X_proc = preprocessor.transform(X_df)
    else:
        from sklearn.preprocessing import QuantileTransformer

        qt = QuantileTransformer(output_distribution="uniform", random_state=random_state)
        X_proc = qt.fit_transform(X_array)

    return X_proc, y_array, total_rows


def run_shap(
    checkpoint_path: str,
    data_path: str,
    output_dir: str,
    chunk_size: int = 256,
    background_size: int = 2000,
    eval_size: int = 2000,
    eval_pool: int = 150_000,
    random_seed: int = 42,
    plot_topk: int = 20,
):
    os.makedirs(output_dir, exist_ok=True)

    device, _ = setup_device()
    ckpt = load_checkpoint(checkpoint_path)
    model = build_model_from_ckpt(ckpt, device)

    saved_feature_cols = ckpt.get("feature_columns") or []
    df_head = pd.read_csv(data_path, nrows=5)
    label_col = detect_label_column(df_head)

    if saved_feature_cols:
        missing = [col for col in saved_feature_cols if col not in df_head.columns]
        if missing:
            raise ValueError("Saved feature columns missing in provided data: " + ", ".join(missing))
        feature_cols = list(saved_feature_cols)
    else:
        exclude_cols = [
            label_col,
            "binary_label",
            "Flow ID",
            "Source IP",
            "Destination IP",
            "Timestamp",
        ]
        feature_cols = [col for col in df_head.columns if col not in exclude_cols]

    preprocessor_state = ckpt.get("preprocessor")
    preprocessor = resolve_preprocessor(preprocessor_state, feature_cols)

    sample_cap = max(eval_pool, background_size, eval_size)
    X_all, y_all, total_rows = prepare_eval_matrix(
        data_path,
        label_col,
        feature_cols,
        preprocessor,
        sample_size=sample_cap,
        random_state=random_seed,
    )
    if len(y_all) == 0:
        raise RuntimeError("No rows sampled from the provided dataset; verify data path and columns.")

    pool_n = min(eval_pool, len(y_all))
    rng = np.random.RandomState(random_seed)
    pool_idx = rng.choice(len(y_all), size=pool_n, replace=False)
    X_pool = X_all[pool_idx]
    y_pool = y_all[pool_idx]

    bg_size = min(background_size, X_pool.shape[0])
    eval_n = min(eval_size, X_pool.shape[0])
    background = torch.from_numpy(X_pool[:bg_size]).float().to(device)
    eval_tensor = torch.from_numpy(X_pool[:eval_n]).float().to(device)

    try:
        import shap
    except ImportError as exc:
        raise ImportError("shap must be installed to run SHAP analysis") from exc

    class LogitsOnly(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, x):
            out = self.base(x)
            return out[0] if isinstance(out, (tuple, list)) else out

    logits_model = LogitsOnly(model).to(device)
    explainer = shap.GradientExplainer(logits_model, background)

    shap_values = []
    start = 0
    while start < eval_tensor.shape[0]:
        end = min(start + chunk_size, eval_tensor.shape[0])
        raw_sv = explainer.shap_values(eval_tensor[start:end])
        if isinstance(raw_sv, (list, tuple)):
            class_sv = raw_sv[1]
        elif isinstance(raw_sv, np.ndarray) and raw_sv.ndim == 3:
            class_sv = raw_sv[:, :, 1]
        else:
            raise ValueError(f"Unsupported SHAP output shape: {getattr(raw_sv, 'shape', 'unknown')}")
        shap_values.append(class_sv)
        start = end

    sv = np.concatenate(shap_values, axis=0)
    mean_abs = np.abs(sv).mean(axis=0)

    print(f"Sampled {len(y_all)} rows (of ~{total_rows}) for SHAP | eval: {sv.shape}")

    importance = pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs})
    importance = importance.sort_values("mean_abs_shap", ascending=False)
    csv_path = os.path.join(output_dir, "shap_global_importance_attack.csv")
    importance.to_csv(csv_path, index=False)
    print(f"Saved SHAP CSV -> {csv_path}")

    try:
        import matplotlib.pyplot as plt

        shap.summary_plot(
            sv,
            features=eval_tensor.detach().cpu().numpy(),
            feature_names=feature_cols,
            show=False,
        )
        plt.tight_layout()
        summary_path = os.path.join(output_dir, "shap_summary_attack.png")
        plt.savefig(summary_path, dpi=160, bbox_inches="tight")
        plt.close()
        print(f"Saved SHAP summary plot -> {summary_path}")
    except Exception as exc:
        print(f"Skipping SHAP plot: {exc}")

    probs = torch.softmax(logits_model(eval_tensor), dim=1)[:, 1].detach().cpu().numpy()
    attack_idx = np.where(y_pool[:eval_n] == 1)[0]
    if len(attack_idx) > 0:
        target_idx = attack_idx[np.argmax(probs[attack_idx])]
    else:
        target_idx = int(np.argmax(probs))

    try:
        with torch.no_grad():
            base_output = logits_model(background)
            base_value = float(base_output[:, 1].mean().cpu().item())

        explanation = shap.Explanation(
            values=sv[target_idx],
            base_values=base_value,
            data=eval_tensor[target_idx].detach().cpu().numpy(),
            feature_names=feature_cols,
        )
        shap.plots.waterfall(explanation, max_display=plot_topk, show=False)
        import matplotlib.pyplot as plt

        waterfall_path = os.path.join(output_dir, "shap_waterfall_attack.png")
        plt.tight_layout()
        plt.savefig(waterfall_path, dpi=160, bbox_inches="tight")
        plt.close()
        print(f"Saved SHAP waterfall plot -> {waterfall_path}")
    except Exception as exc:
        print(f"Skipping SHAP waterfall: {exc}")

    return csv_path
