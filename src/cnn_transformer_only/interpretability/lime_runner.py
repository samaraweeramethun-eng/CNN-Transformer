"""LIME (Local Interpretable Model-agnostic Explanations) for IDS models.

Generates local feature attributions by fitting a linear surrogate model
in the neighbourhood of each sample input.  Average absolute LIME weights
across a random sample of instances are used as a global feature importance
proxy, consistent with the Integrated Gradients and Grad-CAM outputs.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def _predict_proba(model: torch.nn.Module, device: torch.device, data: np.ndarray) -> np.ndarray:
    """Return (N, 2) probability array from a PyTorch model."""
    model.eval()
    results = []
    with torch.no_grad():
        for chunk in np.array_split(data, max(1, len(data) // 256)):
            t = torch.FloatTensor(chunk).to(device)
            logits = model(t)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            results.append(probs)
    return np.vstack(results).astype(np.float64)


def generate_lime_report(
    model: torch.nn.Module,
    X_val: np.ndarray,
    feature_names: list[str],
    output_dir: str,
    sample_size: int = 256,
    num_samples: int = 1000,
    seed: int = 42,
    prefix: str = "cnn_transformer",
) -> str:
    """Generate a LIME global importance report by averaging local explanations.

    Parameters
    ----------
    model : torch.nn.Module
        Trained PyTorch model (eval mode will be enforced).
    X_val : np.ndarray
        Validation/background feature matrix, shape ``(N, n_features)``.
        Should already be scaled (same preprocessing as training).
    feature_names : list[str]
        Feature column names matching the columns of ``X_val``.
    output_dir : str
        Directory to save the CSV and optional bar plot.
    sample_size : int
        Number of validation instances to explain (default 256).
    num_samples : int
        Number of perturbed neighbourhood samples per LIME explanation
        (default 1000).  Higher values give more stable weights but are slower.
    seed : int
        Random seed for reproducibility.
    prefix : str
        File-name prefix for saved outputs.

    Returns
    -------
    str
        Absolute path to the saved CSV ranking file.
    """
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError as exc:
        raise ImportError(
            "LIME is not installed. Run: pip install lime"
        ) from exc

    if X_val.shape[0] == 0:
        return ""

    os.makedirs(output_dir, exist_ok=True)
    device = next(model.parameters()).device

    rng = np.random.RandomState(seed)
    n = min(sample_size, X_val.shape[0])
    idx = rng.choice(X_val.shape[0], n, replace=False)
    X_sample = X_val[idx].astype(np.float32)

    # Replace NaN / inf for LIME (it doesn't tolerate them)
    X_background = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    X_sample_clean = np.nan_to_num(X_sample, nan=0.0, posinf=0.0, neginf=0.0)

    predict_fn = lambda data: _predict_proba(model, device, data.astype(np.float32))

    explainer = LimeTabularExplainer(
        training_data=X_background,
        feature_names=feature_names,
        class_names=["Benign", "Attack"],
        mode="classification",
        discretize_continuous=False,
        random_state=seed,
    )

    print(f"Running LIME on {n} samples ({num_samples} perturbations each)...")
    all_weights = np.zeros(len(feature_names), dtype=np.float64)

    for i, row in enumerate(X_sample_clean):
        exp = explainer.explain_instance(
            row,
            predict_fn,
            num_features=len(feature_names),
            num_samples=num_samples,
            labels=(1,),  # explain the Attack class
        )
        weight_map = dict(exp.as_list(label=1))
        for feat_idx, feat_name in enumerate(feature_names):
            # LIME may truncate/bin feature names; match by prefix
            matched = next(
                (v for k, v in weight_map.items() if feat_name in k), 0.0
            )
            all_weights[feat_idx] += abs(matched)

        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"  {i + 1}/{n} explained")

    mean_weights = all_weights / n

    importance_df = (
        pd.DataFrame({"feature": feature_names, "lime_importance": mean_weights})
        .sort_values("lime_importance", ascending=False)
        .reset_index(drop=True)
    )

    csv_path = os.path.join(output_dir, f"{prefix}_lime.csv")
    importance_df.to_csv(csv_path, index=False)
    print(f"Saved LIME feature ranking -> {csv_path}")

    # Bar plot of top-20 features
    try:
        import matplotlib.pyplot as plt

        top_k = min(20, len(importance_df))
        top = importance_df.head(top_k)
        plt.figure(figsize=(10, 6))
        plt.barh(top["feature"][::-1], top["lime_importance"][::-1], color="steelblue")
        plt.xlabel("Mean |LIME weight| (Attack class)")
        plt.title(f"Top {top_k} Features — LIME ({prefix})")
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"{prefix}_lime_importance.png")
        plt.savefig(plot_path, dpi=160, bbox_inches="tight")
        plt.close()
        print(f"Saved LIME plot -> {plot_path}")
    except Exception as exc:
        print(f"Skipping LIME plot: {exc}")

    return csv_path
