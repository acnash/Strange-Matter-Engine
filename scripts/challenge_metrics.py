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


def bootstrap_regression_report(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    y_true_lower: Sequence[float],
    y_true_upper: Sequence[float],
    endpoint_indices: Sequence[int],
    endpoint_names: Sequence[str],
    n_bootstrap_samples: int = 1000,
    seed: int = 0,
) -> dict:
    """Return macro and per-endpoint bootstrap summaries for all challenge metrics."""
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    lower = np.asarray(y_true_lower, dtype=float)
    upper = np.asarray(y_true_upper, dtype=float)
    endpoints = np.asarray(endpoint_indices, dtype=int)
    names = [str(name) for name in endpoint_names]
    draws = {name: {metric: [] for metric in
                    ("st_rae", "mae", "r2", "spearman_rho", "kendall_tau")}
             for name in names}

    def ranks(values):
        order = np.argsort(values, kind="mergesort")
        ranked = np.empty(len(values), dtype=float)
        start = 0
        while start < len(values):
            end = start + 1
            while end < len(values) and values[order[end]] == values[order[start]]:
                end += 1
            ranked[order[start:end]] = 0.5 * (start + end - 1)
            start = end
        return ranked

    def correlation(left, right):
        left = left - np.mean(left); right = right - np.mean(right)
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        return float(np.dot(left, right) / denominator) if denominator else np.nan

    def kendall_tau_b(left, right):
        i, j = np.triu_indices(len(left), 1)
        dx, dy = np.sign(left[i] - left[j]), np.sign(right[i] - right[j])
        concordance = float(np.sum(dx * dy))
        denominator = float(np.sqrt(np.sum(dx != 0) * np.sum(dy != 0)))
        return concordance / denominator if denominator else np.nan

    for index, name in enumerate(names):
        selected = endpoints == index
        yt, yp = true[selected], pred[selected]
        lo, hi = lower[selected], upper[selected]
        if yt.size < 2:
            raise ValueError(f"Endpoint {name} has too few observations")
        rng = np.random.default_rng(seed)
        samples = rng.choice(yt.size, size=(n_bootstrap_samples, yt.size), replace=True)
        for sample in samples:
            sy, sp, sl, sh = yt[sample], yp[sample], lo[sample], hi[sample]
            draws[name]["st_rae"].append(soft_threshold_rae(sy, sp, sl, sh))
            draws[name]["mae"].append(float(np.mean(np.abs(sp - sy))))
            denominator = float(np.sum((sy - sy.mean()) ** 2))
            draws[name]["r2"].append(
                1.0 - float(np.sum((sp - sy) ** 2)) / denominator
                if denominator > 0 else np.nan
            )
            draws[name]["spearman_rho"].append(correlation(ranks(sy), ranks(sp)))
            draws[name]["kendall_tau"].append(kendall_tau_b(sy, sp))

    def summarize(values):
        array = np.asarray(values, dtype=float)
        finite = array[np.isfinite(array)]
        if not finite.size:
            return {"mean": float("nan"), "ci_low": float("nan"),
                    "ci_high": float("nan")}
        return {"mean": float(np.mean(finite)),
                "ci_low": float(np.quantile(finite, 0.025)),
                "ci_high": float(np.quantile(finite, 0.975))}

    per_endpoint = {
        name: {metric: summarize(values) for metric, values in metrics.items()}
        for name, metrics in draws.items()
    }
    macro = {}
    for metric in next(iter(draws.values())):
        stacked = np.stack([np.asarray(draws[name][metric], dtype=float) for name in names])
        macro_values = [float(np.mean(column[np.isfinite(column)]))
                        if np.isfinite(column).any() else np.nan
                        for column in stacked.T]
        macro[metric] = summarize(macro_values)
    return {"n_bootstrap_samples": n_bootstrap_samples, "seed": seed,
            "macro": macro, "per_endpoint": per_endpoint}
