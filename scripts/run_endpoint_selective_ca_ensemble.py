#!/usr/bin/env python3
"""Cross-fit an endpoint-selective blend of two five-rule CA ensembles."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

import run_five_rule_ensemble_study as ensemble
from challenge_metrics import bootstrap_regression_report


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "results" / "production_endpoint_selective_ca_ensemble_v1"
ENDPOINTS = ensemble.ENDPOINTS
FOLDS = ensemble.FOLDS


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)


def expert_predictions(version, method, seed_average_final):
    ensemble.VERSION = version
    ensemble.FINAL_SEED_AVERAGE = seed_average_final
    oof_rows, oof_matrix = ensemble.aligned_members(ensemble.member_oof_rows)
    oof_predictions = np.empty(len(oof_rows))
    for fold in FOLDS:
        train = np.asarray([int(row["fold"]) != fold for row in oof_rows])
        held = ~train
        train_rows = [row for row, keep in zip(oof_rows, train) if keep]
        held_rows = [row for row, keep in zip(oof_rows, held) if keep]
        prediction, _ = ensemble.combine(
            held_rows, oof_matrix[held], method,
            train_rows, oof_matrix[train],
        )
        oof_predictions[held] = prediction
    final_rows, final_matrix = ensemble.aligned_members(ensemble.member_final_rows)
    final_predictions, weights = ensemble.combine(
        final_rows, final_matrix, method, oof_rows, oof_matrix,
    )
    return oof_rows, oof_predictions, final_rows, final_predictions, weights


def align_experts(base_rows, other_rows, other_predictions):
    lookup = {
        (row.get("fold", ""), row["molecule_id"], row["cyp_target"]): prediction
        for row, prediction in zip(other_rows, other_predictions)
    }
    return np.asarray([
        lookup[(row.get("fold", ""), row["molecule_id"], row["cyp_target"])]
        for row in base_rows
    ])


def best_alpha(rows, base, alternative):
    best = (float("inf"), 0.0)
    for alpha in np.linspace(0.0, 1.0, 41):
        prediction = (1.0 - alpha) * base + alpha * alternative
        score, _, _ = ensemble.point_metrics(rows, prediction)
        if score < best[0]:
            best = (score, float(alpha))
    return best[1]


def cross_fitted_blend(rows, base, alternative, per_endpoint):
    predictions = np.empty(len(rows))
    fold_parameters = {}
    for fold in FOLDS:
        train = np.asarray([int(row["fold"]) != fold for row in rows])
        held = ~train
        if per_endpoint:
            fold_parameters[str(fold)] = {}
            for endpoint in ENDPOINTS:
                train_endpoint = train & np.asarray([
                    row["cyp_target"] == endpoint for row in rows
                ])
                held_endpoint = held & np.asarray([
                    row["cyp_target"] == endpoint for row in rows
                ])
                train_rows = [row for row, keep in zip(rows, train_endpoint) if keep]
                alpha = best_alpha(
                    train_rows, base[train_endpoint], alternative[train_endpoint]
                )
                predictions[held_endpoint] = (
                    (1.0 - alpha) * base[held_endpoint]
                    + alpha * alternative[held_endpoint]
                )
                fold_parameters[str(fold)][endpoint] = alpha
        else:
            train_rows = [row for row, keep in zip(rows, train) if keep]
            alpha = best_alpha(train_rows, base[train], alternative[train])
            predictions[held] = (1.0 - alpha) * base[held] + alpha * alternative[held]
            fold_parameters[str(fold)] = alpha
    return predictions, fold_parameters


def final_blend(oof_rows, base_oof, alternative_oof,
                final_rows, base_final, alternative_final, per_endpoint):
    predictions = np.empty(len(final_rows))
    if not per_endpoint:
        alpha = best_alpha(oof_rows, base_oof, alternative_oof)
        return ((1.0 - alpha) * base_final + alpha * alternative_final,
                {"global": alpha})
    parameters = {}
    for endpoint in ENDPOINTS:
        train_selected = np.asarray([row["cyp_target"] == endpoint for row in oof_rows])
        final_selected = np.asarray([row["cyp_target"] == endpoint for row in final_rows])
        train_rows = [row for row, keep in zip(oof_rows, train_selected) if keep]
        alpha = best_alpha(
            train_rows, base_oof[train_selected], alternative_oof[train_selected]
        )
        predictions[final_selected] = (
            (1.0 - alpha) * base_final[final_selected]
            + alpha * alternative_final[final_selected]
        )
        parameters[endpoint] = alpha
    return predictions, parameters


def main():
    started = time.time()
    STUDY.mkdir(parents=True, exist_ok=True)
    write_json(STUDY / "progress.json", {"stage": "loading_experts"})
    base = expert_predictions("ensemble_v1", "equal", False)
    multi = expert_predictions("multiscale_ensemble_v2", "global", True)
    base_oof_rows, base_oof, base_final_rows, base_final, base_weights = base
    multi_oof_rows, multi_oof_raw, multi_final_rows, multi_final_raw, multi_weights = multi
    multi_oof = align_experts(base_oof_rows, multi_oof_rows, multi_oof_raw)
    multi_final = align_experts(base_final_rows, multi_final_rows, multi_final_raw)

    meta_predictions = {"base": base_oof, "multiscale": multi_oof}
    meta_parameters = {}
    for name, per_endpoint in (("global_blend", False), ("endpoint_blend", True)):
        prediction, parameters = cross_fitted_blend(
            base_oof_rows, base_oof, multi_oof, per_endpoint
        )
        meta_predictions[name] = prediction
        meta_parameters[name] = parameters
    meta_scores = {}
    for name, prediction in meta_predictions.items():
        score, rmse, per_endpoint = ensemble.point_metrics(base_oof_rows, prediction)
        meta_scores[name] = {
            "ma_st_rae": score, "rmse": rmse,
            "per_endpoint_st_rae": per_endpoint,
        }
    selected = min(meta_scores, key=lambda name: meta_scores[name]["ma_st_rae"])
    if selected in ("base", "multiscale"):
        final_prediction = base_final if selected == "base" else multi_final
        final_parameters = {"selected_expert": selected}
    else:
        final_prediction, final_parameters = final_blend(
            base_oof_rows, base_oof, multi_oof, base_final_rows,
            base_final, multi_final, selected == "endpoint_blend",
        )
    point, rmse, per_endpoint = ensemble.point_metrics(base_final_rows, final_prediction)
    true = np.asarray([float(row["experimental_pic50"]) for row in base_final_rows])
    low = np.asarray([float(row["credible_interval_low"]) for row in base_final_rows])
    high = np.asarray([float(row["credible_interval_high"]) for row in base_final_rows])
    endpoint_indices = np.asarray([
        ENDPOINTS.index(row["cyp_target"]) for row in base_final_rows
    ])
    bootstrap = bootstrap_regression_report(
        true, final_prediction, low, high, endpoint_indices, ENDPOINTS, 1000, 0
    )
    output_rows = []
    for row, base_value, multi_value, prediction in zip(
            base_final_rows, base_final, multi_final, final_prediction):
        output = dict(row)
        output.update({
            "prediction_original_ensemble": float(base_value),
            "prediction_multiscale_ensemble": float(multi_value),
            "predicted_pic50": float(prediction),
            "residual": float(prediction - float(row["experimental_pic50"])),
        })
        output_rows.append(output)
    write_csv(STUDY / "validation_predictions.csv", output_rows)
    summary = {
        "study": STUDY.name,
        "architecture": "endpoint_selective_blend_of_two_five_rule_graph_ca_ensembles",
        "selection_metric": "MA-ST-RAE",
        "meta_cv_scores": meta_scores,
        "meta_cv_parameters": meta_parameters,
        "selected_method": selected,
        "final_parameters": final_parameters,
        "base_weights": base_weights,
        "multiscale_weights": multi_weights,
        "final_validation_point_ma_st_rae": point,
        "final_validation_rmse": rmse,
        "final_validation_point_st_rae_by_cyp": per_endpoint,
        "validation_bootstrap_metrics": bootstrap,
        "blind_data_used": False,
        "elapsed_seconds": time.time() - started,
    }
    write_json(STUDY / "study_summary.json", summary)
    write_json(STUDY / "progress.json", {"stage": "complete", "summary": summary})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
