# =========================
# Model Evaluation and Statistical Testing
# =========================

import numpy as np
from typing import Tuple, Dict


def paired_bootstrap_delta(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 42
) -> Dict[str, float]:
    """
    Paired bootstrap comparison of two models on the same test set.
    
    Args:
        y: True labels
        p_a: Predictions from model A
        p_b: Predictions from model B
        n_bootstrap: Number of bootstrap samples
        seed: Random seed
        
    Returns:
        Dictionary with delta statistics (B - A)
    """
    rng = np.random.default_rng(seed)
    N = len(y)
    idx = rng.integers(0, N, size=(n_bootstrap, N))
    
    def brier(y0, p0):
        return np.mean((p0 - y0) ** 2)
    
    def acc(y0, p0):
        return np.mean((p0 >= 0.5) == y0)
    
    delta_brier = []
    delta_acc = []
    
    for s in idx:
        y_s = y[s]
        # Negative delta means B is better (lower Brier)
        delta_brier.append(brier(y_s, p_b[s]) - brier(y_s, p_a[s]))
        delta_acc.append(acc(y_s, p_b[s]) - acc(y_s, p_a[s]))
    
    return {
        "delta_brier_mean": float(np.mean(delta_brier)),
        "delta_brier_ci95": (
            float(np.percentile(delta_brier, 2.5)),
            float(np.percentile(delta_brier, 97.5))
        ),
        "delta_acc_mean": float(np.mean(delta_acc)),
        "delta_acc_ci95": (
            float(np.percentile(delta_acc, 2.5)),
            float(np.percentile(delta_acc, 97.5))
        ),
    }


def bootstrap_confidence_interval(
    y: np.ndarray,
    p: np.ndarray,
    metric: str = "brier",
    n_bootstrap: int = 2000,
    seed: int = 42
) -> Tuple[float, Tuple[float, float]]:
    """
    Calculate bootstrap confidence interval for a single metric.
    
    Args:
        y: True labels
        p: Predicted probabilities
        metric: Metric to calculate ('brier', 'acc', 'logloss')
        n_bootstrap: Number of bootstrap samples
        seed: Random seed
        
    Returns:
        Tuple of (mean, (lower_ci, upper_ci))
    """
    rng = np.random.default_rng(seed)
    N = len(y)
    idx = rng.integers(0, N, size=(n_bootstrap, N))
    
    if metric == "brier":
        metric_fn = lambda y0, p0: np.mean((p0 - y0) ** 2)
    elif metric == "acc":
        metric_fn = lambda y0, p0: np.mean((p0 >= 0.5) == y0)
    elif metric == "logloss":
        metric_fn = lambda y0, p0: -np.mean(y0 * np.log(p0 + 1e-15) + (1 - y0) * np.log(1 - p0 + 1e-15))
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    scores = [metric_fn(y[s], p[s]) for s in idx]
    
    return (
        float(np.mean(scores)),
        (float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5)))
    )


def print_evaluation_results(results: Dict[str, dict], alpha: float = 0.75):
    """
    Pretty-print evaluation results.
    
    Args:
        results: Dictionary mapping model name to evaluation dict
        alpha: Alpha parameter for objective function
    """
    print(f"\n{'Model':<15} | {'N':>5} | {'Acc':>8} | {'Brier':>9} | {'LogLoss':>9} | {'Objective':>10}")
    print("-" * 85)
    
    for name, met in results.items():
        obj = alpha * met["brier"] + (1 - alpha) * (1 - met["acc"])
        print(
            f"{name:<15} | {met['n']:>5d} | "
            f"{met['acc']:>8.4f} | {met['brier']:>9.5f} | "
            f"{met['logloss']:>9.4f} | {obj:>10.5f}"
        )


def print_bootstrap_comparison(
    baseline_name: str,
    comparison_name: str,
    bootstrap_result: dict,
    n_common: int
):
    """
    Pretty-print paired bootstrap comparison results.
    
    Args:
        baseline_name: Name of baseline model
        comparison_name: Name of comparison model
        bootstrap_result: Output from paired_bootstrap_delta
        n_common: Number of common test samples
    """
    print(f"\n{comparison_name} vs {baseline_name}:")
    print(f"  Common samples: {n_common}")
    print(f"  ΔBrier: {bootstrap_result['delta_brier_mean']:+.6f} "
          f"(95% CI: [{bootstrap_result['delta_brier_ci95'][0]:+.6f}, "
          f"{bootstrap_result['delta_brier_ci95'][1]:+.6f}])")
    print(f"  ΔAcc:   {bootstrap_result['delta_acc_mean']:+.4f} "
          f"(95% CI: [{bootstrap_result['delta_acc_ci95'][0]:+.4f}, "
          f"{bootstrap_result['delta_acc_ci95'][1]:+.4f}])")
    
    # Interpretation
    if bootstrap_result['delta_brier_ci95'][1] < 0:
        print(f"  → {comparison_name} significantly better (lower Brier)")
    elif bootstrap_result['delta_brier_ci95'][0] > 0:
        print(f"  → {baseline_name} significantly better (lower Brier)")
    else:
        print(f"  → No significant difference")