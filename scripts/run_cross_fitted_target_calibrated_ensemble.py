#!/usr/bin/env python3
"""Build a leakage-safe target-calibrated ridge stack over Graph-CA experts."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_five_rule_ensemble_study as ensemble
from challenge_metrics import bootstrap_regression_report, soft_threshold_rae


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "results" / "production_cross_fitted_target_calibrated_gcae_v1"
ENDPOINTS = ensemble.ENDPOINTS
FOLDS = tuple(ensemble.FOLDS)
RIDGE_PENALTIES = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
CALIBRATION_PENALTIES = (0.0, 1.0, 10.0, 100.0, 1000.0, float("inf"))


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fieldnames=None) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def fit_ridge(x: np.ndarray, y: np.ndarray, penalty: float) -> dict:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    standardized = (x - mean) / scale
    target_mean = float(y.mean())
    gram = standardized.T @ standardized
    coefficients = np.linalg.solve(
        gram + float(penalty) * np.eye(x.shape[1]),
        standardized.T @ (y - target_mean),
    )
    return {
        "feature_mean": mean,
        "feature_scale": scale,
        "coefficients": coefficients,
        "intercept": target_mean,
        "penalty": float(penalty),
    }


def predict_ridge(x: np.ndarray, state: dict) -> np.ndarray:
    return state["intercept"] + (
        (x - state["feature_mean"]) / state["feature_scale"]
    ) @ state["coefficients"]


def fit_affine_calibration(prediction: np.ndarray, target: np.ndarray,
                           penalty: float) -> dict:
    if np.isinf(penalty):
        return {"slope": 1.0, "intercept": 0.0, "penalty": "identity"}
    design = np.column_stack((prediction, np.ones(len(prediction))))
    regularizer = np.diag([float(penalty), float(penalty) * 0.04])
    prior = np.asarray([1.0, 0.0])
    parameters = np.linalg.solve(
        design.T @ design + regularizer,
        design.T @ target + regularizer @ prior,
    )
    return {
        "slope": float(np.clip(parameters[0], 0.5, 1.5)),
        "intercept": float(np.clip(parameters[1], -1.0, 1.0)),
        "penalty": float(penalty),
    }


def apply_calibration(prediction: np.ndarray, state: dict) -> np.ndarray:
    return float(state["slope"]) * prediction + float(state["intercept"])


def endpoint_arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray([float(row["experimental_pic50"]) for row in rows]),
        np.asarray([float(row["credible_interval_low"]) for row in rows]),
        np.asarray([float(row["credible_interval_high"]) for row in rows]),
    )


def endpoint_score(rows: list[dict], prediction: np.ndarray) -> float:
    true, low, high = endpoint_arrays(rows)
    return soft_threshold_rae(true, prediction, low, high)


def inner_oof_ridge(rows: list[dict], x: np.ndarray, penalty: float) -> np.ndarray:
    result = np.empty(len(rows))
    fold_values = np.asarray([int(row["fold"]) for row in rows])
    for fold in sorted(set(fold_values)):
        held = fold_values == fold
        state = fit_ridge(x[~held], endpoint_arrays([row for row, keep in zip(rows, ~held)
                                                    if keep])[0], penalty)
        result[held] = predict_ridge(x[held], state)
    return result


def select_ridge_penalty(rows: list[dict], x: np.ndarray) -> tuple[float, np.ndarray, dict]:
    scores, predictions = {}, {}
    for penalty in RIDGE_PENALTIES:
        prediction = inner_oof_ridge(rows, x, penalty)
        predictions[penalty] = prediction
        scores[str(penalty)] = endpoint_score(rows, prediction)
    selected = min(RIDGE_PENALTIES, key=lambda value: scores[str(value)])
    return selected, predictions[selected], scores


def select_calibration(rows: list[dict], raw_oof: np.ndarray) -> tuple[dict, dict]:
    target, _, _ = endpoint_arrays(rows)
    fold_values = np.asarray([int(row["fold"]) for row in rows])
    scores = {}
    for penalty in CALIBRATION_PENALTIES:
        calibrated = np.empty(len(rows))
        for fold in sorted(set(fold_values)):
            held = fold_values == fold
            state = fit_affine_calibration(raw_oof[~held], target[~held], penalty)
            calibrated[held] = apply_calibration(raw_oof[held], state)
        label = "identity" if np.isinf(penalty) else str(penalty)
        scores[label] = endpoint_score(rows, calibrated)
    selected_label = min(scores, key=scores.get)
    selected_penalty = (float("inf") if selected_label == "identity"
                        else float(selected_label))
    return fit_affine_calibration(raw_oof, target, selected_penalty), scores


def cross_fitted_predictions(rows: list[dict], x: np.ndarray) -> tuple[np.ndarray, dict]:
    predictions = np.empty(len(rows))
    parameters = {}
    fold_values = np.asarray([int(row["fold"]) for row in rows])
    endpoint_values = np.asarray([row["cyp_target"] for row in rows])
    for outer_fold in FOLDS:
        parameters[str(outer_fold)] = {}
        for endpoint in ENDPOINTS:
            train = (fold_values != outer_fold) & (endpoint_values == endpoint)
            held = (fold_values == outer_fold) & (endpoint_values == endpoint)
            train_rows = [row for row, keep in zip(rows, train) if keep]
            penalty, raw_inner, ridge_scores = select_ridge_penalty(train_rows, x[train])
            calibration, calibration_scores = select_calibration(train_rows, raw_inner)
            target = endpoint_arrays(train_rows)[0]
            ridge = fit_ridge(x[train], target, penalty)
            predictions[held] = apply_calibration(predict_ridge(x[held], ridge), calibration)
            parameters[str(outer_fold)][endpoint] = {
                "ridge_penalty": penalty,
                "ridge_inner_scores": ridge_scores,
                "calibration": calibration,
                "calibration_inner_scores": calibration_scores,
            }
    return predictions, parameters


def final_models(rows: list[dict], x: np.ndarray) -> dict:
    models = {}
    endpoint_values = np.asarray([row["cyp_target"] for row in rows])
    for endpoint in ENDPOINTS:
        selected = endpoint_values == endpoint
        endpoint_rows = [row for row, keep in zip(rows, selected) if keep]
        penalty, raw_oof, ridge_scores = select_ridge_penalty(endpoint_rows, x[selected])
        calibration, calibration_scores = select_calibration(endpoint_rows, raw_oof)
        ridge = fit_ridge(x[selected], endpoint_arrays(endpoint_rows)[0], penalty)
        models[endpoint] = {
            "ridge": ridge,
            "calibration": calibration,
            "ridge_cv_scores": ridge_scores,
            "calibration_cv_scores": calibration_scores,
        }
    return models


def serialise(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: serialise(item) for key, item in value.items()}
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def align_matrix(reference_rows, other_rows, matrix):
    lookup = {(row.get("fold", ""), row["molecule_id"], row["cyp_target"]): values
              for row, values in zip(other_rows, matrix)}
    return np.asarray([lookup[(row.get("fold", ""), row["molecule_id"], row["cyp_target"])]
                       for row in reference_rows])


def load_experts():
    ensemble.VERSION, ensemble.FINAL_SEED_AVERAGE = "ensemble_v1", False
    oof_rows, original_oof = ensemble.aligned_members(ensemble.member_oof_rows)
    final_rows, original_final = ensemble.aligned_members(ensemble.member_final_rows)
    ensemble.VERSION, ensemble.FINAL_SEED_AVERAGE = "multiscale_ensemble_v2", True
    multi_oof_rows, multi_oof = ensemble.aligned_members(ensemble.member_oof_rows)
    multi_final_rows, multi_final = ensemble.aligned_members(ensemble.member_final_rows)
    return (oof_rows, np.column_stack((original_oof,
                                       align_matrix(oof_rows, multi_oof_rows, multi_oof))),
            final_rows, np.column_stack((original_final,
                                         align_matrix(final_rows, multi_final_rows, multi_final))))


def predict_with_models(rows: list[dict], x: np.ndarray, models: dict) -> np.ndarray:
    result = np.empty(len(rows))
    endpoints = np.asarray([row["cyp_target"] for row in rows])
    for endpoint, model in models.items():
        selected = endpoints == endpoint
        result[selected] = apply_calibration(
            predict_ridge(x[selected], model["ridge"]), model["calibration"]
        )
    return result


def main() -> None:
    started = time.time()
    STUDY.mkdir(parents=True, exist_ok=True)
    write_json(STUDY / "progress.json", {"stage": "loading_graph_ca_experts"})
    oof_rows, oof_x, final_rows, final_x = load_experts()
    if oof_x.shape[1] != 10 or final_x.shape[1] != 10:
        raise RuntimeError("Expected ten Graph-CA expert predictions")

    write_json(STUDY / "progress.json", {"stage": "nested_cross_fitting"})
    oof_prediction, fold_parameters = cross_fitted_predictions(oof_rows, oof_x)
    oof_point, oof_rmse, oof_per_endpoint = ensemble.point_metrics(oof_rows, oof_prediction)
    models = final_models(oof_rows, oof_x)
    final_prediction = predict_with_models(final_rows, final_x, models)
    final_point, final_rmse, final_per_endpoint = ensemble.point_metrics(final_rows, final_prediction)

    true, low, high = endpoint_arrays(final_rows)
    endpoint_indices = np.asarray([ENDPOINTS.index(row["cyp_target"]) for row in final_rows])
    bootstrap = bootstrap_regression_report(
        true, final_prediction, low, high, endpoint_indices, ENDPOINTS, 1000, 0
    )
    feature_names = ([f"original_{rule}" for rule in ensemble.RULES]
                     + [f"multiscale_{rule}" for rule in ensemble.RULES])
    output_rows = []
    for row, features, prediction in zip(final_rows, final_x, final_prediction):
        output = dict(row)
        output.update({name: float(value) for name, value in zip(feature_names, features)})
        output["predicted_pic50"] = float(prediction)
        output["residual"] = float(prediction - float(row["experimental_pic50"]))
        output_rows.append(output)
    write_csv(STUDY / "validation_predictions.csv", output_rows)

    summary = {
        "study": STUDY.name,
        "architecture": "cross_fitted_target_calibrated_graph_cellular_automata_ensemble",
        "feature_names": feature_names,
        "selection_metric": "MA-ST-RAE",
        "nested_group_folds": len(FOLDS),
        "development_oof_ma_st_rae": oof_point,
        "development_oof_rmse": oof_rmse,
        "development_oof_st_rae_by_cyp": oof_per_endpoint,
        "final_validation_point_ma_st_rae": final_point,
        "final_validation_rmse": final_rmse,
        "final_validation_point_st_rae_by_cyp": final_per_endpoint,
        "validation_bootstrap_metrics": bootstrap,
        "fold_parameters": serialise(fold_parameters),
        "final_models": serialise(models),
        "blind_data_used": False,
        "elapsed_seconds": time.time() - started,
    }
    write_json(STUDY / "study_summary.json", summary)
    write_json(STUDY / "progress.json", {"stage": "validation_complete", "summary": summary})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
