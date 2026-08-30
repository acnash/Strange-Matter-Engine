#!/usr/bin/env python3
"""Resumable production campaign for CYP-specialist Graph cellular automata.

Each nonlinear Graph-CA is trained by backpropagation against observations from
one CYP endpoint only.  A genuine differentiable ridge solve remains the
readout during training.  All ten rules are screened, the three strongest per
endpoint are confirmed across scaffold folds and seeds, and nested ridge subset
selection chooses one or more specialists for the final prediction.
"""

from __future__ import annotations

import csv
import itertools
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_cross_fitted_target_calibrated_ensemble as meta
import run_five_rule_ensemble_study as ensemble
import run_overnight_retrained_ten_rule_cft_ds_gcae as previous
import run_production_transition_study as production
from challenge_metrics import bootstrap_regression_report, soft_threshold_rae

STUDY_NAME = "production_cv_cyp_specialist_gca_v1"
STUDY = ROOT / "results" / STUDY_NAME
RULES = previous.RULES
ENDPOINTS = ensemble.ENDPOINTS
SCREEN_FOLDS = (0, 1)
FOLDS = tuple(range(5))
SEEDS = (7103, 8209)
PENALTIES = meta.RIDGE_PENALTIES
RUNNER = ROOT / "scripts" / "run_graph_ca_visual_prototype.py"
BLIND_CACHE = ROOT / "tmp" / "strange_matter_graph_ca_graphs_with_blind.pkl"
TEST_CSV = ROOT / "data" / "openadmet-cyp-challenge-2026" / "cyp-challenge-TEST-BLINDED.csv"


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta.serialise(value), indent=2) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields=None) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or rows[0].keys())
        writer.writeheader(); writer.writerows(rows)


def progress(stage: str, **details) -> None:
    write_json(STUDY / "progress.json", {
        "stage": stage, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **details,
    })


def base_config(rule: str) -> dict:
    return deepcopy(previous.config_for(rule, "multiscale"))


def specialist_config(rule: str, endpoint: str) -> dict:
    config = base_config(rule)
    profiles = {
        "CYP1A2": "electronic_local", "CYP2C9": "comprehensive",
        "CYP2D6": "valence_electronic", "CYP3A4": "periodic_electronic",
    }
    generations = (16, 32, 64, 125, 250, 500)
    current = int(config.get("generations", 64))
    nearest = min(range(len(generations)), key=lambda i: abs(generations[i] - current))
    direction = 1 if endpoint in ("CYP2C9", "CYP3A4") else -1
    config.update({
        "atom_feature_profile": profiles[endpoint],
        "generations": generations[max(0, min(len(generations) - 1, nearest + direction))],
        "ridge": min(10.0, max(0.001, float(config.get("ridge", 0.1)) *
                               (10.0 if endpoint == "CYP2D6" else 0.1))),
        "trajectory_pooling": "multiscale", "ridge_mode": "shared",
    })
    return config


def run_label(stage, endpoint, rule, variant, seed, fold):
    return f"{stage}_{endpoint}_{rule}_{variant}_seed_{seed}_fold_{fold}"


def run_path(label: str) -> Path:
    return STUDY / "runs" / label


def fit(worker: Path, stage, endpoint, rule, variant, config, seed, fold,
        epochs, patience):
    label = run_label(stage, endpoint, rule, variant, seed, fold)
    return production.run_fit(
        rule, STUDY_NAME, label, config, seed, epochs, patience,
        999999, 999999, "cuda", worker, analyse=False,
        cv_fold=fold, cv_folds=len(FOLDS), active_cyp=endpoint,
    )


def screen(worker: Path) -> dict:
    jobs = [(endpoint, rule, variant, fold)
            for endpoint in ENDPOINTS for rule in RULES
            for variant in ("base", "specialist") for fold in SCREEN_FOLDS]
    scores = {(endpoint, rule, variant): [] for endpoint, rule, variant, _ in jobs}
    durations = []
    for number, (endpoint, rule, variant, fold) in enumerate(jobs, 1):
        config = base_config(rule) if variant == "base" else specialist_config(rule, endpoint)
        progress("specialist_screen", completed=number - 1, total=len(jobs),
                 endpoint=endpoint, rule=rule, variant=variant, fold=fold,
                 estimated_remaining_seconds=(np.mean(durations) * (len(jobs)-number+1)
                                              if durations else None))
        started = time.time()
        metrics = fit(worker, "screen", endpoint, rule, variant, config, 6101, fold, 18, 5)
        durations.append(time.time() - started)
        scores[(endpoint, rule, variant)].append(float(
            metrics.get("restored_validation_point_ma_st_rae", np.inf)))
    selected = {}
    for endpoint in ENDPOINTS:
        candidates = []
        for rule in RULES:
            for variant in ("base", "specialist"):
                values = scores[(endpoint, rule, variant)]
                candidates.append((float(np.mean(values)), rule, variant, values))
        best_by_rule = {}
        for candidate in sorted(candidates):
            best_by_rule.setdefault(candidate[1], candidate)
        selected[endpoint] = [
            {"rule": rule, "variant": variant, "screen_ma_st_rae": score,
             "fold_scores": values,
             "config": (base_config(rule) if variant == "base"
                        else specialist_config(rule, endpoint))}
            for score, rule, variant, values in sorted(best_by_rule.values())[:3]
        ]
    write_json(STUDY / "screening_summary.json", selected)
    return selected


def confirm(worker: Path, selected: dict) -> None:
    jobs = [(endpoint, candidate, seed, fold)
            for endpoint, candidates in selected.items() for candidate in candidates
            for seed in SEEDS for fold in FOLDS]
    durations = []
    for number, (endpoint, candidate, seed, fold) in enumerate(jobs, 1):
        progress("cross_validation_confirmation", completed=number - 1, total=len(jobs),
                 endpoint=endpoint, rule=candidate["rule"], seed=seed, fold=fold,
                 estimated_remaining_seconds=(np.mean(durations) * (len(jobs)-number+1)
                                              if durations else None))
        started = time.time()
        fit(worker, "confirm", endpoint, candidate["rule"], candidate["variant"],
            candidate["config"], seed, fold, 35, 8)
        durations.append(time.time() - started)


def prediction_rows(endpoint, candidate, seed, fold, split):
    label = run_label("confirm", endpoint, candidate["rule"], candidate["variant"], seed, fold)
    return [row for row in read_csv(run_path(label) / "validation_predictions.csv")
            if row["split"] == split and row["cyp_target"] == endpoint]


def matrices(selected: dict, split: str):
    all_rows, all_matrix = [], []
    for endpoint in ENDPOINTS:
        rules = selected[endpoint]
        maps = []
        for candidate in rules:
            values = {}
            for seed in SEEDS:
                for fold in FOLDS:
                    for row in prediction_rows(endpoint, candidate, seed, fold, split):
                        key = ((int(row.get("fold", fold)) if split == "validation" else 0),
                               row["molecule_id"], endpoint)
                        values.setdefault(key, []).append(float(row["predicted_pic50"]))
            maps.append(values)
        keys = sorted(set.intersection(*(set(mapping) for mapping in maps)))
        for key in keys:
            reference = None
            for seed in SEEDS:
                for fold in FOLDS:
                    rows = prediction_rows(endpoint, rules[0], seed, fold, split)
                    reference = next((r for r in rows if r["molecule_id"] == key[1]), reference)
            row = dict(reference); row["fold"] = key[0]
            all_rows.append(row)
            all_matrix.append([float(np.mean(mapping[key])) for mapping in maps])
    return all_rows, np.asarray(all_matrix)


def select_subset(rows, matrix, endpoint):
    selected = np.asarray([row["cyp_target"] == endpoint for row in rows])
    endpoint_rows = [row for row, keep in zip(rows, selected) if keep]
    x = matrix[selected]
    y, low, high = meta.endpoint_arrays(endpoint_rows)
    folds = np.asarray([int(row["fold"]) for row in endpoint_rows])
    best = {"score": float("inf")}
    for size in range(1, x.shape[1] + 1):
        for columns in itertools.combinations(range(x.shape[1]), size):
            for penalty in PENALTIES:
                prediction = np.empty(len(x))
                for fold in sorted(set(folds)):
                    train, held = folds != fold, folds == fold
                    state = meta.fit_ridge(x[train][:, columns], y[train], penalty)
                    prediction[held] = meta.predict_ridge(x[held][:, columns], state)
                score = float(soft_threshold_rae(y, prediction, low, high))
                if score < best["score"]:
                    best = {"score": score, "columns": list(columns),
                            "penalty": float(penalty)}
    best["ridge"] = meta.fit_ridge(x[:, best["columns"]], y, best["penalty"])
    return best


def assemble(selected: dict) -> dict:
    oof_rows, oof_matrix = matrices(selected, "validation")
    holdout_rows, holdout_matrix = matrices(selected, "reserved_holdout")
    models, prediction = {}, np.empty(len(holdout_rows))
    for endpoint in ENDPOINTS:
        model = select_subset(oof_rows, oof_matrix, endpoint)
        model["rules"] = [selected[endpoint][i]["rule"] for i in model["columns"]]
        models[endpoint] = model
        held = np.asarray([row["cyp_target"] == endpoint for row in holdout_rows])
        prediction[held] = meta.predict_ridge(
            holdout_matrix[held][:, model["columns"]], model["ridge"])
    point, rmse, by_cyp = ensemble.point_metrics(holdout_rows, prediction)
    true, low, high = meta.endpoint_arrays(holdout_rows)
    endpoint_indices = np.asarray([ENDPOINTS.index(row["cyp_target"]) for row in holdout_rows])
    bootstrap = bootstrap_regression_report(
        true, prediction, low, high, endpoint_indices, ENDPOINTS, 1000, 0)
    output = []
    for row, value in zip(holdout_rows, prediction):
        item = dict(row); item["predicted_pic50"] = float(value)
        item["selected_rules"] = "|".join(models[row["cyp_target"]]["rules"])
        output.append(item)
    write_csv(STUDY / "reserved_holdout_predictions.csv", output)
    summary = {
        "method": "CV-CYP-GCA", "architecture": "cross_validated_cyp_specialist_graph_ca",
        "specialisation_stage": "nonlinear_ca_backpropagation_and_ridge_readout",
        "screened_rules_per_endpoint": len(RULES), "selected_candidates": selected,
        "final_models": models, "reserved_holdout_point_ma_st_rae": point,
        "reserved_holdout_rmse": rmse, "reserved_holdout_st_rae_by_cyp": by_cyp,
        "reserved_holdout_bootstrap_metrics": bootstrap,
        "reserved_holdout_used_for_selection": False, "blind_labels_loaded": False,
    }
    write_json(STUDY / "study_summary.json", summary)
    return summary


def main() -> None:
    worker = Path(os.environ.get(
        "SME_PYTHON", "C:/Users/Anthony/anaconda3/envs/strange-matter-gpu/python.exe"))
    STUDY.mkdir(parents=True, exist_ok=True)
    production.ensure_graph_cache(worker)
    screening_path = STUDY / "screening_summary.json"
    selected = (json.loads(screening_path.read_text(encoding="utf-8"))
                if screening_path.exists() else screen(worker))
    confirm(worker, selected)
    progress("assembling_validation")
    summary = assemble(selected)
    progress("validation_complete", summary=summary)
    print(json.dumps(meta.serialise(summary), indent=2))


if __name__ == "__main__":
    main()
