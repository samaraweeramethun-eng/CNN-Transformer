"""Confusion matrix visualisation and error analysis for binary IDS classifiers."""
from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    title: str = "Confusion Matrix",
    class_names: tuple[str, str] = ("Benign", "Attack"),
    output_dir: str | None = None,
    filename: str = "confusion_matrix.png",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot a 2×2 confusion matrix heatmap with counts and percentages.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels (0 / 1).
    y_pred : array-like
        Predicted binary labels (0 / 1).
    title : str
        Plot title.
    class_names : tuple[str, str]
        Names for the negative and positive classes.
    output_dir : str or None
        If provided, saves the figure to ``output_dir/filename``.
    filename : str
        File name used when *output_dir* is set.
    ax : matplotlib Axes or None
        If provided, draws on this axes; otherwise creates a new figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    cm = np.array([[tn, fp], [fn, tp]])
    total = cm.sum()

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=[f"Pred {c}" for c in class_names],
        yticklabels=[f"True {c}" for c in class_names],
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            pct = 100.0 * cm[i, j] / max(total, 1)
            ax.text(
                j, i,
                f"{cm[i, j]:,}\n({pct:.1f}%)",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12,
            )

    if own_fig:
        plt.tight_layout()
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, filename), dpi=160, bbox_inches="tight")
    return fig


def error_analysis_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    *,
    block_map: np.ndarray | None = None,
    output_dir: str | None = None,
    prefix: str = "model",
) -> dict[str, Any]:
    """Produce an error-analysis report with plots.

    Analyses
    --------
    1. **Confidence histogram**: probability distributions for correct vs
       incorrect predictions, split by FP / FN.
    2. **Per-block error rates**: if *block_map* is provided, shows error
       rate and FP/FN breakdown per temporal block.
    3. **High-confidence errors**: lists samples where the model was very
       confident (prob > 0.9 or < 0.1) but wrong.

    Parameters
    ----------
    y_true, y_pred : array-like
        Ground-truth and predicted binary labels.
    y_prob : array-like
        Predicted probability for the positive (attack) class.
    block_map : array-like or None
        Block ID for each sample (same length as y_true).
    output_dir : str or None
        Directory to save figures.
    prefix : str
        File-name prefix for saved figures.

    Returns
    -------
    dict with keys ``"n_fp"``, ``"n_fn"``, ``"high_conf_errors"``,
    ``"per_block_errors"`` (if block_map given).
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()

    fp_mask = (y_pred == 1) & (y_true == 0)
    fn_mask = (y_pred == 0) & (y_true == 1)
    correct_mask = y_pred == y_true

    n_fp = int(fp_mask.sum())
    n_fn = int(fn_mask.sum())

    report: dict[str, Any] = {"n_fp": n_fp, "n_fn": n_fn}

    # ── 1. Confidence histogram ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(y_prob[correct_mask], bins=50, alpha=0.7, label="Correct", color="tab:green")
    axes[0].hist(y_prob[fp_mask], bins=50, alpha=0.7, label=f"FP (n={n_fp})", color="tab:orange")
    axes[0].hist(y_prob[fn_mask], bins=50, alpha=0.7, label=f"FN (n={n_fn})", color="tab:red")
    axes[0].set_xlabel("Predicted Probability (Attack)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Prediction Confidence Distribution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # FP vs FN probability box plot
    fp_probs = y_prob[fp_mask] if n_fp > 0 else np.array([])
    fn_probs = y_prob[fn_mask] if n_fn > 0 else np.array([])
    data_to_plot = [d for d in [fp_probs, fn_probs] if len(d) > 0]
    labels_to_plot = [l for d, l in zip([fp_probs, fn_probs], ["FP", "FN"]) if len(d) > 0]
    if data_to_plot:
        axes[1].boxplot(data_to_plot, labels=labels_to_plot)
        axes[1].set_ylabel("Predicted Probability")
        axes[1].set_title("Error Confidence Spread")
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(
            os.path.join(output_dir, f"{prefix}_error_analysis.png"),
            dpi=160, bbox_inches="tight",
        )
    plt.show()

    # ── 2. High-confidence errors ────────────────────────────────────
    high_conf_fp = int(((y_prob > 0.9) & fp_mask).sum())
    high_conf_fn = int(((y_prob < 0.1) & fn_mask).sum())
    report["high_conf_errors"] = {
        "fp_above_0.9": high_conf_fp,
        "fn_below_0.1": high_conf_fn,
    }

    print(f"\n{'='*60}")
    print(f"  ERROR ANALYSIS — {prefix}")
    print(f"{'='*60}")
    print(f"  Total errors:         {n_fp + n_fn:,}")
    print(f"    False Positives:    {n_fp:,}")
    print(f"    False Negatives:    {n_fn:,}")
    print(f"  High-confidence FP (prob > 0.9): {high_conf_fp:,}")
    print(f"  High-confidence FN (prob < 0.1): {high_conf_fn:,}")

    # ── 3. Per-block error rates ─────────────────────────────────────
    if block_map is not None:
        block_map = np.asarray(block_map).ravel()
        unique_blocks = np.unique(block_map)
        block_errors = []
        for bid in unique_blocks:
            mask = block_map == bid
            n = int(mask.sum())
            n_err = int((y_pred[mask] != y_true[mask]).sum())
            n_fp_b = int(fp_mask[mask].sum())
            n_fn_b = int(fn_mask[mask].sum())
            block_errors.append({
                "block_id": int(bid),
                "n_samples": n,
                "n_errors": n_err,
                "error_rate": n_err / max(n, 1),
                "n_fp": n_fp_b,
                "n_fn": n_fn_b,
            })

        report["per_block_errors"] = block_errors

        print("\n  Per-block error rates:")
        print(f"  {'Block':<7} {'Samples':>8} {'Errors':>8} {'Rate':>8} {'FP':>6} {'FN':>6}")
        print(f"  {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*6}")
        for be in block_errors:
            print(
                f"  {be['block_id']:<7} {be['n_samples']:>8,} {be['n_errors']:>8,} "
                f"{be['error_rate']:>7.2%} {be['n_fp']:>6,} {be['n_fn']:>6,}"
            )

        # Per-block bar chart
        fig2, ax2 = plt.subplots(figsize=(max(8, len(block_errors) * 0.6), 5))
        bids = [be["block_id"] for be in block_errors]
        fps = [be["n_fp"] for be in block_errors]
        fns = [be["n_fn"] for be in block_errors]
        x = np.arange(len(bids))
        ax2.bar(x, fps, label="FP", color="tab:orange")
        ax2.bar(x, fns, bottom=fps, label="FN", color="tab:red")
        ax2.set_xticks(x)
        ax2.set_xticklabels(bids, rotation=45 if len(bids) > 10 else 0)
        ax2.set_xlabel("Block ID")
        ax2.set_ylabel("Error Count")
        ax2.set_title(f"Per-Block Errors — {prefix}")
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        if output_dir is not None:
            fig2.savefig(
                os.path.join(output_dir, f"{prefix}_per_block_errors.png"),
                dpi=160, bbox_inches="tight",
            )
        plt.show()

    print(f"{'='*60}\n")
    return report
