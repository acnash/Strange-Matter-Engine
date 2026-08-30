#!/usr/bin/env python3
"""Select target-specific subsets from ten retrained dual-scale Graph-CA rules."""

from __future__ import annotations

import csv
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_cross_fitted_target_calibrated_ensemble as meta
import run_five_rule_ensemble_study as ensemble
import run_overnight_retrained_ten_rule_cft_ds_gcae as campaign
from challenge_metrics import bootstrap_regression_report, soft_threshold_rae


STUDY = campaign.STUDY
RULES = campaign.RULES
ENDPOINTS = campaign.ENDPOINTS
PENALTIES = meta.RIDGE_PENALTIES


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(meta.serialise(payload), indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def rule_subsets():
    for size in range(1, len(RULES) + 1):
        yield from itertools.combinations(range(len(RULES)), size)


def columns_for(subset) -> np.ndarray:
    # Feature order is all original rules followed by all multiscale rules.
    return np.asarray(tuple(subset) + tuple(index + len(RULES) for index in subset))


def endpoint_data(rows, matrix, endpoint):
    selected = np.asarray([row["cyp_target"] == endpoint for row in rows])
    endpoint_rows = [row for row, keep in zip(rows, selected) if keep]
    return endpoint_rows, matrix[selected]


def fold_cache(rows: list[dict], matrix: np.ndarray):
    target, _, _ = meta.endpoint_arrays(rows)
    folds = np.asarray([int(row["fold"]) for row in rows])
    cache = []
    for fold in sorted(set(folds)):
        held = folds == fold
        train = ~held
        mean = matrix[train].mean(axis=0)
        scale = matrix[train].std(axis=0)
        scale[scale < 1e-8] = 1.0
        train_x = (matrix[train] - mean) / scale
        held_x = (matrix[held] - mean) / scale
        target_mean = float(target[train].mean())
        cache.append({
            "held": held, "gram": train_x.T @ train_x,
            "rhs": train_x.T @ (target[train] - target_mean),
            "held_x": held_x, "target_mean": target_mean,
        })
    return cache


def search(rows: list[dict], matrix: np.ndarray) -> dict:
    cache = fold_cache(rows, matrix)
    true, low, high = meta.endpoint_arrays(rows)
    best = {"score": float("inf")}
    by_size = {}
    for subset in rule_subsets():
        columns = columns_for(subset)
        identity = np.eye(len(columns))
        for penalty in PENALTIES:
            prediction = np.empty(len(rows))
            for fold in cache:
                coefficients = np.linalg.solve(
                    fold["gram"][np.ix_(columns, columns)] + penalty * identity,
                    fold["rhs"][columns],
                )
                prediction[fold["held"]] = (
                    fold["target_mean"] + fold["held_x"][:, columns] @ coefficients
                )
            score = float(soft_threshold_rae(true, prediction, low, high))
            if score < by_size.get(len(subset), float("inf")):
                by_size[len(subset)] = score
            if score < best["score"]:
                best = {"score": score, "rules": [RULES[i] for i in subset],
                        "rule_indices": list(subset), "penalty": float(penalty)}
    best["best_score_by_subset_size"] = {str(key): value
                                         for key, value in sorted(by_size.items())}
    return best


def fit_selected(rows: list[dict], matrix: np.ndarray, selected: dict) -> dict:
    columns = columns_for(selected["rule_indices"])
    target = meta.endpoint_arrays(rows)[0]
    state = meta.fit_ridge(matrix[:, columns], target, selected["penalty"])
    return {"selection": selected, "columns": columns.tolist(), "ridge": state,
            "calibration": {"slope": 1.0, "intercept": 0.0,
                            "penalty": "identity"}}


def predict_models(rows: list[dict], matrix: np.ndarray, models: dict) -> np.ndarray:
    prediction = np.empty(len(rows))
    endpoints = np.asarray([row["cyp_target"] for row in rows])
    for endpoint, model in models.items():
        selected = endpoints == endpoint
        columns = np.asarray(model["columns"])
        prediction[selected] = meta.predict_ridge(
            matrix[selected][:, columns], model["ridge"]
        )
    return prediction


def nested_cross_fit(rows: list[dict], matrix: np.ndarray) -> tuple[np.ndarray, dict]:
    result = np.empty(len(rows))
    parameters = {}
    folds = np.asarray([int(row["fold"]) for row in rows])
    endpoints = np.asarray([row["cyp_target"] for row in rows])
    for outer_fold in sorted(set(folds)):
        parameters[str(outer_fold)] = {}
        for endpoint in ENDPOINTS:
            train = (folds != outer_fold) & (endpoints == endpoint)
            held = (folds == outer_fold) & (endpoints == endpoint)
            train_rows = [row for row, keep in zip(rows, train) if keep]
            campaign.progress("nested_rule_subset_selection", outer_fold=int(outer_fold),
                              endpoint=endpoint)
            selected = search(train_rows, matrix[train])
            model = fit_selected(train_rows, matrix[train], selected)
            result[held] = meta.predict_ridge(
                matrix[held][:, np.asarray(model["columns"])], model["ridge"]
            )
            parameters[str(outer_fold)][endpoint] = selected
    return result, parameters


def main() -> None:
    started = time.time()
    repeats = [campaign.repeat_oof(repeat)
               for repeat in range(len(campaign.SPLIT_SEEDS))]
    rows, matrix = campaign.averaged_oof(repeats)
    nested_prediction, nested_parameters = nested_cross_fit(rows, matrix)
    nested_point, nested_rmse, nested_by_cyp = ensemble.point_metrics(
        rows, nested_prediction
    )
    models, final_selections = {}, {}
    for endpoint in ENDPOINTS:
        endpoint_rows, endpoint_matrix = endpoint_data(rows, matrix, endpoint)
        campaign.progress("final_rule_subset_selection", endpoint=endpoint)
        selected = search(endpoint_rows, endpoint_matrix)
        final_selections[endpoint] = selected
        models[endpoint] = fit_selected(endpoint_rows, endpoint_matrix, selected)

    holdout_rows, holdout_matrix = campaign.reserved_holdout()
    holdout_prediction = predict_models(holdout_rows, holdout_matrix, models)
    point, rmse, by_cyp = ensemble.point_metrics(
        holdout_rows, holdout_prediction
    )
    true, low, high = meta.endpoint_arrays(holdout_rows)
    endpoint_indices = np.asarray([ENDPOINTS.index(row["cyp_target"])
                                   for row in holdout_rows])
    bootstrap = bootstrap_regression_report(
        true, holdout_prediction, low, high, endpoint_indices, ENDPOINTS, 1000, 0
    )
    output_rows = []
    for row, prediction in zip(holdout_rows, holdout_prediction):
        output = dict(row)
        output["predicted_pic50"] = float(prediction)
        output["selected_rules"] = "|".join(final_selections[row["cyp_target"]]["rules"])
        output_rows.append(output)
    write_csv(STUDY / "subset_selected_holdout_predictions.csv", output_rows)
    summary = {
        "architecture": "target_specific_subset_selected_dual_scale_ten_rule_gcae",
        "candidate_rule_combinations_per_endpoint": 2 ** len(RULES) - 1,
        "nested_development_ma_st_rae": nested_point,
        "nested_development_rmse": nested_rmse,
        "nested_development_st_rae_by_cyp": nested_by_cyp,
        "nested_fold_selections": nested_parameters,
        "final_selections": final_selections,
        "final_models": models,
        "reserved_holdout_point_ma_st_rae": point,
        "reserved_holdout_rmse": rmse,
        "reserved_holdout_st_rae_by_cyp": by_cyp,
        "reserved_holdout_bootstrap_metrics": bootstrap,
        "reserved_holdout_used_for_selection": False,
        "blind_data_used": False,
        "elapsed_seconds": time.time() - started,
    }
    write_json(STUDY / "subset_selection_summary.json", summary)
    campaign.progress("subset_selection_complete", summary=summary)
    print(json.dumps(meta.serialise(summary), indent=2))


if __name__ == "__main__":
    main()
