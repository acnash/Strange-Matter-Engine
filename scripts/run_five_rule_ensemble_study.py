#!/usr/bin/env python3
"""Retune five graph-CA rules and build a leakage-safe prediction ensemble."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from challenge_metrics import bootstrap_regression_report, macro_soft_threshold_rae
from runtime_device import python_executable, requested_device


ROOT = Path(__file__).resolve().parents[1]
MEMBER_RUNNER = ROOT / "scripts" / "run_production_transition_study.py"
STUDY = "production_five_rule_ensemble_v1"
VERSION = "ensemble_v1"
RULES = (
    "gated_residual",
    "delayed_memory",
    "inertial_reaction_diffusion",
    "kuramoto_sakaguchi",
    "fitzhugh_nagumo",
)
SEEDS = (1701, 2909, 4211)
FOLDS = range(5)
ENDPOINTS = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")
FINAL_SEED_AVERAGE = False


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def project_simplex(values):
    values = np.asarray(values, dtype=float)
    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    indices = np.arange(1, len(values) + 1)
    valid = ordered - cumulative / indices > 0
    rho = indices[valid][-1]
    threshold = cumulative[rho - 1] / rho
    return np.maximum(values - threshold, 0.0)


def point_metrics(rows, predictions):
    true = np.asarray([float(row["experimental_pic50"]) for row in rows])
    lower = np.asarray([float(row["credible_interval_low"]) for row in rows])
    upper = np.asarray([float(row["credible_interval_high"]) for row in rows])
    endpoint_names = [row["cyp_target"] for row in rows]
    endpoint_indices = np.asarray([ENDPOINTS.index(name) for name in endpoint_names])
    ma_st_rae, per_endpoint = macro_soft_threshold_rae(
        true, predictions, lower, upper, endpoint_indices, ENDPOINTS
    )
    rmse = float(np.sqrt(np.mean((predictions - true) ** 2)))
    return ma_st_rae, rmse, per_endpoint


def optimise_weights(rows, matrix, iterations=1200):
    """Projected subgradient solution of convex MA-ST-RAE on the simplex."""
    true = np.asarray([float(row["experimental_pic50"]) for row in rows])
    lower = np.asarray([float(row["credible_interval_low"]) for row in rows])
    upper = np.asarray([float(row["credible_interval_high"]) for row in rows])
    endpoint = np.asarray([row["cyp_target"] for row in rows])
    denominators = {}
    active_endpoints = [name for name in ENDPOINTS if np.any(endpoint == name)]
    for name in active_endpoints:
        selected = endpoint == name
        baseline = true[selected].mean()
        denominators[name] = np.maximum(baseline - upper[selected], 0).sum()
        denominators[name] += np.maximum(lower[selected] - baseline, 0).sum()
    starts = [np.full(matrix.shape[1], 1.0 / matrix.shape[1])]
    starts.extend(np.eye(matrix.shape[1]))
    best_weights, best_score = starts[0], float("inf")
    for start in starts:
        weights = start.copy()
        local_best, local_score = weights.copy(), float("inf")
        for step in range(1, iterations + 1):
            pred = matrix @ weights
            gradient = np.zeros(matrix.shape[1])
            for name in active_endpoints:
                selected = endpoint == name
                if not np.any(selected) or denominators[name] <= 0:
                    continue
                high = pred[selected] > upper[selected]
                low = pred[selected] < lower[selected]
                signed = high.astype(float) - low.astype(float)
                contribution = (signed[:, None] * matrix[selected]).sum(0)
                gradient += contribution / (
                    len(active_endpoints) * denominators[name]
                )
            learning_rate = 0.08 / math.sqrt(step)
            weights = project_simplex(weights - learning_rate * gradient)
            if step == 1 or step % 20 == 0:
                candidate = matrix @ weights
                endpoint_scores = []
                for name in active_endpoints:
                    selected = endpoint == name
                    error = np.maximum(candidate[selected] - upper[selected], 0).sum()
                    error += np.maximum(lower[selected] - candidate[selected], 0).sum()
                    endpoint_scores.append(error / denominators[name])
                score = float(np.mean(endpoint_scores))
                if score < local_score:
                    local_score, local_best = score, weights.copy()
        if local_score < best_score:
            best_score, best_weights = local_score, local_best
    return best_weights


def combine(rows, matrices, method, train_rows=None, train_matrix=None):
    if method == "equal":
        weights = np.full(len(RULES), 1.0 / len(RULES))
        return matrices @ weights, {"global": weights.tolist()}
    if method == "global":
        weights = optimise_weights(train_rows, train_matrix)
        return matrices @ weights, {"global": weights.tolist()}
    predictions = np.empty(len(rows))
    weights_by_endpoint = {}
    for endpoint in ENDPOINTS:
        train_selected = np.asarray([row["cyp_target"] == endpoint for row in train_rows])
        selected = np.asarray([row["cyp_target"] == endpoint for row in rows])
        weights = optimise_weights(
            [row for row, keep in zip(train_rows, train_selected) if keep],
            train_matrix[train_selected],
        )
        predictions[selected] = matrices[selected] @ weights
        weights_by_endpoint[endpoint] = weights.tolist()
    return predictions, weights_by_endpoint


def winner_confirmation_index(rule):
    member_dir = ROOT / "results" / f"production_{rule}_{VERSION}"
    summary = json.loads((member_dir / "study_summary.json").read_text())
    winner = summary["winner"]
    for index in (1, 2):
        metrics = json.loads((member_dir / "runs" /
                              f"confirm_{index:02d}_seed_{SEEDS[0]}_fold_0" /
                              "metrics.json").read_text())
        if metrics.get("study_config") == winner:
            return index
    raise RuntimeError(f"Could not identify confirmation winner for {rule}")


def member_oof_rows(rule):
    member_dir = ROOT / "results" / f"production_{rule}_{VERSION}"
    candidate = winner_confirmation_index(rule)
    accumulated = {}
    for seed in SEEDS:
        for fold in FOLDS:
            path = (member_dir / "runs" /
                    f"confirm_{candidate:02d}_seed_{seed}_fold_{fold}" /
                    "validation_predictions.csv")
            for row in read_csv(path):
                if row["split"] != "validation":
                    continue
                key = (fold, row["molecule_id"], row["cyp_target"])
                entry = accumulated.setdefault(key, {"rows": [], "predictions": []})
                entry["rows"].append(row)
                entry["predictions"].append(float(row["predicted_pic50"]))
    result = {}
    for key, entry in accumulated.items():
        row = dict(entry["rows"][0])
        row["fold"] = key[0]
        row["predicted_pic50"] = float(np.mean(entry["predictions"]))
        result[key] = row
    return result


def member_final_rows(rule):
    labels = ["final_model"]
    if FINAL_SEED_AVERAGE:
        labels.extend(f"final_seed_{seed}" for seed in SEEDS[1:])
    accumulated = {}
    for label in labels:
        path = (ROOT / "results" / f"production_{rule}_{VERSION}" /
                "runs" / label / "validation_predictions.csv")
        for row in read_csv(path):
            if row["split"] != "validation":
                continue
            key = (row["molecule_id"], row["cyp_target"])
            entry = accumulated.setdefault(key, {"row": row, "predictions": []})
            entry["predictions"].append(float(row["predicted_pic50"]))
    result = {}
    for key, entry in accumulated.items():
        row = dict(entry["row"])
        row["predicted_pic50"] = float(np.mean(entry["predictions"]))
        result[key] = row
    return result


def aligned_members(loader):
    by_rule = {rule: loader(rule) for rule in RULES}
    keys = sorted(set.intersection(*(set(rows) for rows in by_rule.values())))
    if any(len(rows) != len(keys) for rows in by_rule.values()):
        raise RuntimeError("Member prediction keys do not align")
    rows = [dict(by_rule[RULES[0]][key]) for key in keys]
    matrix = np.asarray([[float(by_rule[rule][key]["predicted_pic50"])
                          for rule in RULES] for key in keys])
    return rows, matrix


def main():
    global STUDY, VERSION, FINAL_SEED_AVERAGE
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--python", dest="python_path", default=None)
    parser.add_argument("--report-python", default=None)
    parser.add_argument("--skip-members", action="store_true")
    parser.add_argument("--member-version", default=VERSION)
    parser.add_argument("--study", default=STUDY)
    parser.add_argument("--enhanced-search", action="store_true")
    parser.add_argument("--interval-loss-search", action="store_true")
    parser.add_argument("--seed-average-final", action="store_true")
    args = parser.parse_args()
    VERSION = args.member_version
    STUDY = args.study
    FINAL_SEED_AVERAGE = args.seed_average_final
    device = requested_device(args.device)
    worker = python_executable(args.python_path)
    report_python = python_executable(args.report_python or worker)
    study_dir = ROOT / "results" / STUDY
    study_dir.mkdir(parents=True, exist_ok=True)
    progress_path = study_dir / "progress.json"
    started = time.time()

    if not args.skip_members:
        if args.enhanced_search:
            os.environ["SME_ENHANCED_TRAJECTORY_SEARCH"] = "1"
        if args.interval_loss_search:
            os.environ["SME_INTERVAL_LOSS_SEARCH"] = "1"
        for index, rule in enumerate(RULES, 1):
            write_json(progress_path, {"stage": "member_tuning", "member": rule,
                                      "completed_members": index - 1,
                                      "total_members": len(RULES)})
            subprocess.run([
                str(worker), str(MEMBER_RUNNER), "--rule", rule,
                "--device", device, "--python", str(worker),
                "--report-python", str(report_python),
                "--search-version", VERSION,
            ], cwd=ROOT, check=True)

    if FINAL_SEED_AVERAGE:
        import run_production_transition_study as member_search
        for rule in RULES:
            member_dir = ROOT / "results" / f"production_{rule}_{VERSION}"
            summary = json.loads((member_dir / "study_summary.json").read_text())
            winner = summary["winner"]
            for seed in SEEDS[1:]:
                member_search.run_fit(
                    rule, f"production_{rule}_{VERSION}", f"final_seed_{seed}",
                    winner, seed, 15, 5, 999999, 999999, device, worker,
                    analyse=False,
                )

    oof_rows, oof_matrix = aligned_members(member_oof_rows)
    meta_predictions = {method: np.empty(len(oof_rows))
                        for method in ("equal", "global", "per_endpoint")}
    for fold in FOLDS:
        train = np.asarray([int(row["fold"]) != fold for row in oof_rows])
        held = ~train
        train_rows = [row for row, keep in zip(oof_rows, train) if keep]
        held_rows = [row for row, keep in zip(oof_rows, held) if keep]
        for method in meta_predictions:
            pred, _ = combine(held_rows, oof_matrix[held], method,
                              train_rows, oof_matrix[train])
            meta_predictions[method][held] = pred
    meta_scores = {}
    for method, predictions in meta_predictions.items():
        ma_st_rae, rmse, per_endpoint = point_metrics(oof_rows, predictions)
        meta_scores[method] = {"ma_st_rae": ma_st_rae, "rmse": rmse,
                               "per_endpoint_st_rae": per_endpoint}
    selected_method = min(meta_scores, key=lambda method: meta_scores[method]["ma_st_rae"])

    final_rows, final_matrix = aligned_members(member_final_rows)
    final_predictions, weights = combine(
        final_rows, final_matrix, selected_method, oof_rows, oof_matrix
    )
    final_point, final_rmse, final_per_endpoint = point_metrics(final_rows, final_predictions)
    true = np.asarray([float(row["experimental_pic50"]) for row in final_rows])
    lower = np.asarray([float(row["credible_interval_low"]) for row in final_rows])
    upper = np.asarray([float(row["credible_interval_high"]) for row in final_rows])
    endpoints = np.asarray([ENDPOINTS.index(row["cyp_target"]) for row in final_rows])
    bootstrap = bootstrap_regression_report(
        true, final_predictions, lower, upper, endpoints, ENDPOINTS, 1000, 0
    )

    output_rows = []
    for row, member_values, prediction in zip(final_rows, final_matrix, final_predictions):
        output = dict(row)
        output["predicted_pic50"] = float(prediction)
        output["residual"] = float(prediction - float(row["experimental_pic50"]))
        for rule, value in zip(RULES, member_values):
            output[f"prediction_{rule}"] = float(value)
        output_rows.append(output)
    write_csv(study_dir / "validation_predictions.csv", output_rows)
    summary = {
        "study": STUDY,
        "members": list(RULES),
        "member_search_version": VERSION,
        "selection_metric": "MA-ST-RAE",
        "meta_cv_scores": meta_scores,
        "selected_ensemble_method": selected_method,
        "final_member_seed_averaging": list(SEEDS) if FINAL_SEED_AVERAGE else [SEEDS[0]],
        "weights": weights,
        "final_validation_point_ma_st_rae": final_point,
        "final_validation_rmse": final_rmse,
        "final_validation_point_st_rae_by_cyp": final_per_endpoint,
        "validation_bootstrap_metrics": bootstrap,
        "blind_data_used": False,
        "elapsed_seconds": time.time() - started,
        "device": device,
    }
    write_json(study_dir / "study_summary.json", summary)
    write_json(progress_path, {"stage": "complete", "summary": summary})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
