"""Official direct-inhibition selection metrics for the OpenADMET CYP challenge."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def soft_threshold_rae(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    y_true_lower: Sequence[float],
    y_true_upper: Sequence[float],
) -> float:
    """Match the organiser's per-endpoint soft-threshold RAE calculation."""
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    lower = np.asarray(y_true_lower, dtype=float)
    upper = np.asarray(y_true_upper, dtype=float)
    if not (true.shape == pred.shape == lower.shape == upper.shape):
        raise ValueError("True values, predictions, and credible bounds must align")
    if true.size == 0 or not np.isfinite(np.concatenate((true, pred, lower, upper))).all():
        return float("inf")
    if np.any(lower > upper):
        raise ValueError("Credible-interval lower bounds must not exceed upper bounds")

    soft_error = np.clip(pred - upper, 0, None) + np.clip(lower - pred, 0, None)
    mean_true = float(np.mean(true))
    baseline_error = (
        np.clip(mean_true - upper, 0, None)
        + np.clip(lower - mean_true, 0, None)
    )
    denominator = float(np.sum(baseline_error))
    return float(np.sum(soft_error) / denominator) if denominator > 0 else float("inf")


def macro_soft_threshold_rae(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    y_true_lower: Sequence[float],
    y_true_upper: Sequence[float],
    endpoint_indices: Sequence[int],
    endpoint_names: Sequence[str],
) -> tuple[float, dict[str, float]]:
    """Compute the equal-weight macro average of endpoint ST-RAE values."""
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    lower = np.asarray(y_true_lower, dtype=float)
    upper = np.asarray(y_true_upper, dtype=float)
    endpoints = np.asarray(endpoint_indices, dtype=int)
    if not (true.shape == pred.shape == lower.shape == upper.shape == endpoints.shape):
        raise ValueError("All MA-ST-RAE inputs must have the same shape")

    per_endpoint = {}
    for index, name in enumerate(endpoint_names):
        selected = endpoints == index
        per_endpoint[str(name)] = soft_threshold_rae(
            true[selected], pred[selected], lower[selected], upper[selected]
        )
    values = np.asarray(list(per_endpoint.values()), dtype=float)
    macro = float(np.mean(values)) if np.isfinite(values).all() else float("inf")
    return macro, per_endpoint


def bootstrap_macro_soft_threshold_rae(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    y_true_lower: Sequence[float],
    y_true_upper: Sequence[float],
    endpoint_indices: Sequence[int],
    endpoint_names: Sequence[str],
    n_bootstrap_samples: int = 1000,
    seed: int = 0,
) -> tuple[float, dict[str, float]]:
    """Return the organiser-style mean of 1,000 bootstrapped MA-ST-RAE scores."""
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    lower = np.asarray(y_true_lower, dtype=float)
    upper = np.asarray(y_true_upper, dtype=float)
    endpoints = np.asarray(endpoint_indices, dtype=int)
    if n_bootstrap_samples < 1:
        raise ValueError("n_bootstrap_samples must be positive")

    endpoint_bootstraps = []
    per_endpoint = {}
    for index, name in enumerate(endpoint_names):
        selected = endpoints == index
        endpoint_true = true[selected]
        endpoint_pred = pred[selected]
        endpoint_lower = lower[selected]
        endpoint_upper = upper[selected]
        if endpoint_true.size == 0:
            return float("inf"), {str(endpoint): float("inf")
                                  for endpoint in endpoint_names}
        rng = np.random.default_rng(seed)
        sample_indices = rng.choice(
            endpoint_true.size,
            size=(n_bootstrap_samples, endpoint_true.size),
            replace=True,
        )
        values = np.asarray([
            soft_threshold_rae(
                endpoint_true[sample], endpoint_pred[sample],
                endpoint_lower[sample], endpoint_upper[sample]
            )
            for sample in sample_indices
        ])
        per_endpoint[str(name)] = float(np.mean(values))
        endpoint_bootstraps.append(values)
    macro_values = np.mean(np.stack(endpoint_bootstraps), axis=0)
    return float(np.mean(macro_values)), per_endpoint
