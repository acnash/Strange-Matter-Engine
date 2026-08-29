#!/usr/bin/env python3
"""Production search for a cross-fitted residual Graph-CA ensemble expert."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np

import run_production_transition_study as production
from challenge_metrics import bootstrap_regression_report
from run_five_rule_ensemble_study import (
    ENDPOINTS,
    aligned_members,
    member_final_rows,
    member_oof_rows,
    point_metrics,
)
from runtime_device import python_executable, requested_device


ROOT = Path(__file__).resolve().parents[1]
STUDY = "production_residual_ca_ensemble_v1"
BASE_VERSION = "ensemble_v1"
RESIDUAL_RULES = (
    "gated_residual",
    "inertial_reaction_diffusion",
    "fitzhugh_nagumo",
)
SEEDS = (1701, 2909, 4211)
FOLDS = range(5)


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)


def build_residual_targets(study_dir: Path) -> Path:
    """Combine leakage-safe OOF base predictions with final holdout predictions."""
    global BASE_VERSION
    import run_five_rule_ensemble_study as base

    base.VERSION = BASE_VERSION
    oof_rows, oof_matrix = aligned_members(member_oof_rows)
    final_rows, final_matrix = aligned_members(member_final_rows)
    rows = []
    for split, source_rows, matrix in (
        ("development_oof", oof_rows, oof_matrix),
        ("reserved_holdout", final_rows, final_matrix),
    ):
        base_prediction = matrix.mean(axis=1)
        for row, prediction in zip(source_rows, base_prediction):
            experimental = float(row["experimental_pic50"])
            rows.append({
                "split": split,
                "fold": row.get("fold", ""),
                "molecule_id": row["molecule_id"],
                "cyp_target": row["cyp_target"],
                "experimental_pic50": experimental,
                "credible_interval_low": row["credible_interval_low"],
                "credible_interval_high": row["credible_interval_high"],
                "base_prediction": float(prediction),
                "residual_target": experimental - float(prediction),
            })
    target_path = study_dir / "cross_fitted_residual_targets.csv"
    write_csv(target_path, rows)
    return target_path


def candidates(seed: int) -> list[tuple[str, dict]]:
    """Broad, rule-balanced search including residual correction strength."""
    result = []
    for rule_index, rule in enumerate(RESIDUAL_RULES):
        production.SEARCH_SEED = seed + 1009 * rule_index
        sampled = production.sampled_candidates(36)
        rng = random.Random(seed + 7919 * (rule_index + 1))
        for config in sampled:
            config = dict(config)
            config["residual_alpha"] = rng.choice((0.15, 0.25, 0.4, 0.6, 0.8, 1.0))
            result.append((rule, config))
    return result


def finite_metric(metrics, key="restored_validation_point_ma_st_rae"):
    value = float(metrics.get(key, float("inf")))
    return value if math.isfinite(value) else float("inf")


def mean_metric(metrics_list, key="restored_validation_point_ma_st_rae"):
    values = [finite_metric(metrics, key) for metrics in metrics_list]
    return float(np.mean(values)) if all(math.isfinite(v) for v in values) else float("inf")


def mean_rmse(metrics_list):
    return float(np.mean([
        float(metrics.get("restored_validation_rmse", float("inf")))
        for metrics in metrics_list
    ]))


def run_candidate(rule, study_name, label, config, seed, epochs, patience,
                  fit_limit, validation_limit, device, worker, folds):
    metrics = []
    for fold in folds:
        suffix = f"_fold_{fold}" if fold is not None else ""
        metrics.append(production.run_fit(
            rule, study_name, f"{label}{suffix}", config, seed,
            epochs, patience, fit_limit, validation_limit, device, worker,
            cv_fold=fold,
        ))
    return metrics


def progress(path, stage, completed, total, collection):
    payload = {"stage": stage, "completed": completed, "total": total}
    if collection:
        best = min(collection, key=lambda item: mean_metric(item[2]))
        payload.update({
            "best_rule": best[0],
            "best_ma_st_rae": mean_metric(best[2]),
            "best_rmse": mean_rmse(best[2]),
            "best_residual_alpha": best[1]["residual_alpha"],
        })
    write_json(path, payload)


def main() -> None:
    global STUDY, BASE_VERSION
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--python", dest="python_path", default=None)
    parser.add_argument("--study", default=STUDY)
    parser.add_argument("--base-version", default=BASE_VERSION)
    parser.add_argument("--search-seed", type=int, default=260827)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    STUDY, BASE_VERSION = args.study, args.base_version
    device = requested_device(args.device)
    worker = python_executable(args.python_path)
    study_dir = ROOT / "results" / STUDY
    study_dir.mkdir(parents=True, exist_ok=True)
    progress_path = study_dir / "progress.json"
    started = time.time()

    production.ensure_graph_cache(worker)
    target_path = build_residual_targets(study_dir)
    os.environ["SME_RESIDUAL_TARGETS"] = str(target_path)
    os.environ["SME_SKIP_REPORT"] = "1"
    os.environ["SME_SAVE_TRAJECTORY_CASES"] = "64"

    broad_candidates = candidates(args.search_seed)
    if args.smoke:
        broad_candidates = broad_candidates[:3]
    write_json(study_dir / "search_space.json", {
        "search_seed": args.search_seed,
        "base_ensemble": "equal five-rule OOF ensemble",
        "residual_rules": list(RESIDUAL_RULES),
        "configurations": [{"rule": rule, **config}
                           for rule, config in broad_candidates],
    })

    stage1 = []
    for index, (rule, config) in enumerate(broad_candidates, 1):
        metrics = run_candidate(
            rule, STUDY, f"stage1_{index:03d}", config, 1701,
            3, 2, 600, 200, device, worker, (0,),
        )
        stage1.append((rule, config, metrics))
        progress(progress_path, "broad_search", index, len(broad_candidates), stage1)

    per_rule_promoted = []
    for rule in RESIDUAL_RULES:
        rule_rows = sorted(
            [item for item in stage1 if item[0] == rule],
            key=lambda item: mean_metric(item[2]),
        )
        per_rule_promoted.extend(rule_rows[:(1 if args.smoke else 6)])

    stage2 = []
    for index, (rule, config, _) in enumerate(per_rule_promoted, 1):
        metrics = run_candidate(
            rule, STUDY, f"stage2_{index:02d}", config, 1701,
            6, 3, 2400, 600, device, worker,
            (0,) if args.smoke else (0, 1, 2),
        )
        stage2.append((rule, config, metrics))
        progress(progress_path, "multifold_promotion", index,
                 len(per_rule_promoted), stage2)

    finalists = sorted(stage2, key=lambda item: mean_metric(item[2]))[:(
        1 if args.smoke else 4
    )]
    confirmations = []
    confirmation_seeds = (1701,) if args.smoke else SEEDS
    confirmation_folds = (0,) if args.smoke else FOLDS
    for index, (rule, config, _) in enumerate(finalists, 1):
        metrics = []
        for seed in confirmation_seeds:
            metrics.extend(run_candidate(
                rule, STUDY, f"confirm_{index:02d}_seed_{seed}", config, seed,
                10, 4, 999999, 999999, device, worker, confirmation_folds,
            ))
        score_values = [finite_metric(m, "restored_validation_ma_st_rae")
                        for m in metrics]
        score_mean = float(np.mean(score_values))
        score_sd = float(np.std(score_values))
        confirmations.append((rule, config, metrics, score_mean, score_sd,
                              score_mean + 0.25 * score_sd))
        write_json(progress_path, {
            "stage": "confirmation", "completed": index, "total": len(finalists),
            "latest_rule": rule, "latest_mean_ma_st_rae": score_mean,
            "latest_mean_rmse": mean_rmse(metrics), "latest_seed_sd": score_sd,
        })

    winner = min(confirmations, key=lambda item: item[5])
    rule, config, _, confirmation_mean, confirmation_sd, robust_score = winner
    final_metrics = production.run_fit(
        rule, STUDY, "final_model", config, 1701, 20, 6,
        999999, 999999, device, worker, analyse=True,
    )
    prediction_path = study_dir / "runs" / "final_model" / "validation_predictions.csv"
    with prediction_path.open(newline="", encoding="utf-8") as handle:
        final_rows = [row for row in csv.DictReader(handle)
                      if row["split"] == "validation"]
    true = np.asarray([float(row["experimental_pic50"]) for row in final_rows])
    pred = np.asarray([float(row["predicted_pic50"]) for row in final_rows])
    lower = np.asarray([float(row["credible_interval_low"]) for row in final_rows])
    upper = np.asarray([float(row["credible_interval_high"]) for row in final_rows])
    endpoint = np.asarray([ENDPOINTS.index(row["cyp_target"]) for row in final_rows])
    point_score, rmse, per_endpoint = point_metrics(final_rows, pred)
    bootstrap = bootstrap_regression_report(
        true, pred, lower, upper, endpoint, ENDPOINTS, 1000, 0,
    )
    summary = {
        "study": STUDY,
        "architecture": "five_rule_equal_ensemble_plus_cross_fitted_residual_graph_ca",
        "base_member_version": BASE_VERSION,
        "residual_rule": rule,
        "winner": config,
        "confirmation_mean_ma_st_rae": confirmation_mean,
        "confirmation_ma_st_rae_sd": confirmation_sd,
        "confirmation_robust_score": robust_score,
        "final_validation_point_ma_st_rae": point_score,
        "final_validation_rmse": rmse,
        "final_validation_point_st_rae_by_cyp": per_endpoint,
        "validation_bootstrap_metrics": bootstrap,
        "final_metrics": final_metrics,
        "cross_fitted_residual_targets": str(target_path.relative_to(ROOT)),
        "blind_data_used": False,
        "elapsed_seconds": time.time() - started,
        "device": device,
    }
    write_json(study_dir / "study_summary.json", summary)
    write_json(progress_path, {"stage": "complete", "summary": summary})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
