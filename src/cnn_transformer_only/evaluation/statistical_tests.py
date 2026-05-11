"""Statistical hypothesis testing for preprocessing and model comparisons.

Provides:
- ``compare_preprocessing``: paired comparison of metrics before/after a
  preprocessing change (e.g. duplicate removal) across k-fold CV.
- ``compare_models_statistical``: paired comparison of two models trained on
  identical CV folds.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _paired_tests(
    values_a: list[float],
    values_b: list[float],
    metric_name: str,
    label_a: str = "A",
    label_b: str = "B",
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Run paired t-test and Wilcoxon signed-rank test on fold-level scores.

    Parameters
    ----------
    values_a, values_b : list[float]
        Per-fold metric values from condition A and B. Must be same length.
    metric_name : str
        Name of the metric (for reporting).
    label_a, label_b : str
        Human-readable labels for the two conditions.
    alpha : float
        Significance threshold.

    Returns
    -------
    dict with test results.
    """
    from scipy.stats import ttest_rel, wilcoxon

    a = np.array(values_a, dtype=np.float64)
    b = np.array(values_b, dtype=np.float64)
    n = len(a)

    if n != len(b):
        raise ValueError(
            f"Mismatched fold counts: {len(values_a)} vs {len(values_b)}"
        )

    diff = a - b
    mean_a, std_a = float(np.mean(a)), float(np.std(a, ddof=1)) if n > 1 else 0.0
    mean_b, std_b = float(np.mean(b)), float(np.std(b, ddof=1)) if n > 1 else 0.0
    mean_diff = float(np.mean(diff))

    result: dict[str, Any] = {
        "metric": metric_name,
        "n_folds": n,
        "mean_a": mean_a,
        "std_a": std_a,
        "mean_b": mean_b,
        "std_b": std_b,
        "mean_diff": mean_diff,
        "label_a": label_a,
        "label_b": label_b,
    }

    # Paired t-test (parametric)
    if n >= 2 and np.std(diff) > 0:
        t_stat, p_value = ttest_rel(a, b)
        result["paired_t"] = {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": float(p_value) < alpha,
        }
    else:
        result["paired_t"] = {
            "t_statistic": float("nan"),
            "p_value": 1.0,
            "significant": False,
            "note": "Insufficient variance or folds for t-test",
        }

    # Wilcoxon signed-rank (non-parametric)
    if n >= 6 and np.any(diff != 0):
        try:
            stat, p_value = wilcoxon(a, b)
            result["wilcoxon"] = {
                "statistic": float(stat),
                "p_value": float(p_value),
                "significant": float(p_value) < alpha,
            }
        except ValueError:
            result["wilcoxon"] = {
                "statistic": float("nan"),
                "p_value": 1.0,
                "significant": False,
                "note": "Wilcoxon test could not be computed",
            }
    else:
        result["wilcoxon"] = {
            "statistic": float("nan"),
            "p_value": 1.0,
            "significant": False,
            "note": f"Need ≥6 folds with non-zero differences (got {n} folds)",
        }

    return result


def _print_test_result(res: dict[str, Any]) -> None:
    """Pretty-print a single paired-test result."""
    metric = res["metric"]
    la, lb = res["label_a"], res["label_b"]
    print(f"\n  {metric}:")
    print(f"    {la}: {res['mean_a']:.4f} ± {res['std_a']:.4f}")
    print(f"    {lb}: {res['mean_b']:.4f} ± {res['std_b']:.4f}")
    print(f"    Mean diff ({la} − {lb}): {res['mean_diff']:+.4f}")

    pt = res["paired_t"]
    sig_t = "YES" if pt["significant"] else "no"
    print(f"    Paired t-test:   t={pt['t_statistic']:.4f}, p={pt['p_value']:.4f} ({sig_t})")
    if "note" in pt:
        print(f"      Note: {pt['note']}")

    wt = res["wilcoxon"]
    sig_w = "YES" if wt["significant"] else "no"
    print(f"    Wilcoxon:        W={wt['statistic']:.4f}, p={wt['p_value']:.4f} ({sig_w})")
    if "note" in wt:
        print(f"      Note: {wt['note']}")


def compare_preprocessing(
    cv_before: dict[str, Any],
    cv_after: dict[str, Any],
    *,
    label_before: str = "Before Dedup",
    label_after: str = "After Dedup",
    metrics: list[str] | None = None,
    alpha: float = 0.05,
) -> list[dict[str, Any]]:
    """Compare model performance before/after a preprocessing change.

    Both ``cv_before`` and ``cv_after`` should be the dict returned by
    ``grouped_kfold_cv`` (with ``mean_metrics[key]["values"]``).

    Parameters
    ----------
    cv_before, cv_after : dict
        Output of ``grouped_kfold_cv`` for the two preprocessing conditions.
    label_before, label_after : str
        Labels for the two conditions.
    metrics : list[str] or None
        Metrics to compare (default: f1_score, auc_roc, auc_pr).
    alpha : float
        Significance threshold.

    Returns
    -------
    list of per-metric test result dicts.
    """
    if metrics is None:
        metrics = ["f1_score", "auc_roc", "auc_pr"]

    print(f"\n{'='*60}")
    print(f"  STATISTICAL COMPARISON: {label_before} vs {label_after}")
    print(f"{'='*60}")

    results = []
    for m in metrics:
        vals_before = cv_before["mean_metrics"][m]["values"]
        vals_after = cv_after["mean_metrics"][m]["values"]
        res = _paired_tests(vals_before, vals_after, m, label_before, label_after, alpha)
        _print_test_result(res)
        results.append(res)

    # Overall verdict
    any_sig = any(
        r["paired_t"]["significant"] or r["wilcoxon"]["significant"]
        for r in results
    )
    print(f"\n  {'='*58}")
    if any_sig:
        print(f"  VERDICT: Statistically significant difference detected (α={alpha})")
    else:
        print(f"  VERDICT: No statistically significant difference (α={alpha})")
    print(f"  {'='*58}\n")

    return results


def compare_models_statistical(
    cv_model_a: dict[str, Any],
    cv_model_b: dict[str, Any],
    *,
    label_a: str = "CNN Classifier",
    label_b: str = "CNN-Transformer",
    metrics: list[str] | None = None,
    alpha: float = 0.05,
) -> list[dict[str, Any]]:
    """Compare two models using paired statistical tests on identical CV folds.

    Both models must have been evaluated on the **same** k-fold split structure
    (same fold assignments) so that fold-level pairing is valid.

    Parameters
    ----------
    cv_model_a, cv_model_b : dict
        Output of ``grouped_kfold_cv`` for each model.
    label_a, label_b : str
        Model labels.
    metrics : list[str] or None
        Metrics to compare (default: f1_score, auc_roc, auc_pr, precision, recall).
    alpha : float
        Significance threshold.

    Returns
    -------
    list of per-metric test result dicts.
    """
    if metrics is None:
        metrics = ["f1_score", "auc_roc", "auc_pr", "precision", "recall"]

    n_a = cv_model_a["n_folds"]
    n_b = cv_model_b["n_folds"]
    if n_a != n_b:
        raise ValueError(
            f"Fold counts differ: {label_a} has {n_a} folds, "
            f"{label_b} has {n_b} folds. Use identical CV splits."
        )

    print(f"\n{'='*60}")
    print(f"  MODEL COMPARISON: {label_a} vs {label_b}")
    print(f"  ({n_a}-fold cross-validation, paired tests)")
    print(f"{'='*60}")

    results = []
    for m in metrics:
        vals_a = cv_model_a["mean_metrics"][m]["values"]
        vals_b = cv_model_b["mean_metrics"][m]["values"]
        res = _paired_tests(vals_a, vals_b, m, label_a, label_b, alpha)
        _print_test_result(res)
        results.append(res)

    # Summary table
    print(f"\n  {'Metric':<12} {'Diff':>8} {'t-test p':>10} {'Wilcoxon p':>12} {'Winner':>15}")
    print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*12} {'-'*15}")
    for r in results:
        diff = r["mean_diff"]
        tp = r["paired_t"]["p_value"]
        wp = r["wilcoxon"]["p_value"]
        sig = r["paired_t"]["significant"] or r["wilcoxon"]["significant"]
        if sig:
            winner = label_a if diff > 0 else label_b
        else:
            winner = "no sig. diff."
        print(f"  {r['metric']:<12} {diff:>+8.4f} {tp:>10.4f} {wp:>12.4f} {winner:>15}")

    print(f"{'='*60}\n")
    return results
