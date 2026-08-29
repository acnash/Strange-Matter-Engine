#!/usr/bin/env python3
"""Resumable overnight retraining of a ten-rule, dual-scale Graph-CA stack."""

from __future__ import annotations

import csv
import json
import os
import subprocess
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
import run_production_transition_study as production
from challenge_metrics import bootstrap_regression_report


STUDY_NAME = "production_retrained_ten_rule_cft_ds_gcae_v1"
STUDY = ROOT / "results" / STUDY_NAME
RUNNER = ROOT / "scripts" / "run_graph_ca_visual_prototype.py"
BLIND_CACHE = ROOT / "tmp" / "strange_matter_graph_ca_graphs_with_blind.pkl"
TEST_CSV = ROOT / "data" / "openadmet-cyp-challenge-2026" / "cyp-challenge-TEST-BLINDED.csv"
RULES = (
    "gated_residual", "delayed_memory", "inertial_reaction_diffusion",
    "kuramoto_sakaguchi", "fitzhugh_nagumo", "activator_inhibitor",
    "coupled_map", "damped_symplectic", "gray_scott",
    "conservative_graph_flux",
)
SCALES = ("original", "multiscale")
SPLIT_SEEDS = (731041, 982451)
TRAINING_SEEDS = (5101, 6203)
FOLDS = tuple(range(5))
ENDPOINTS = ensemble.ENDPOINTS


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields=None) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(meta.serialise(payload), indent=2) + "\n", encoding="utf-8")


def progress(stage: str, **details) -> None:
    payload = {"stage": stage, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **details}
    write_json(STUDY / "progress.json", payload)


def config_for(rule: str, scale: str) -> dict:
    preferred = ROOT / "results" / f"production_{rule}_{scale}_ensemble_v2" / "study_summary.json"
    if scale == "original":
        preferred = ROOT / "results" / f"production_{rule}_ensemble_v1" / "study_summary.json"
    if preferred.exists():
        return json.loads(preferred.read_text(encoding="utf-8"))["winner"]
    base = ROOT / "results" / f"production_{rule}_challenge_aligned_v5" / "study_summary.json"
    config = json.loads(base.read_text(encoding="utf-8"))["winner"]
    if scale == "multiscale":
        config = {**config, "trajectory_pooling": "multiscale", "ridge_mode": "shared"}
    return config


def label(repeat: int, scale: str, rule: str, fold: int) -> str:
    return f"repeat_{repeat}_{scale}_{rule}_seed_{TRAINING_SEEDS[repeat]}_fold_{fold}"


def train_path(repeat: int, scale: str, rule: str, fold: int) -> Path:
    return STUDY / "runs" / label(repeat, scale, rule, fold)


def train_members(worker: Path) -> None:
    jobs = [(repeat, scale, rule, fold)
            for repeat in range(len(SPLIT_SEEDS))
            for scale in SCALES for rule in RULES for fold in FOLDS]
    durations = []
    for index, (repeat, scale, rule, fold) in enumerate(jobs, 1):
        run_dir = train_path(repeat, scale, rule, fold)
        if (run_dir / "metrics.json").exists():
            continue
        estimate = (float(np.mean(durations)) * (len(jobs) - index + 1)
                    if durations else None)
        progress("graph_ca_retraining", completed=index - 1, total=len(jobs),
                 repeat=repeat, split_seed=SPLIT_SEEDS[repeat], scale=scale,
                 rule=rule, fold=fold, estimated_remaining_seconds=estimate)
        os.environ["SME_CV_SPLIT_SEED"] = str(SPLIT_SEEDS[repeat])
        os.environ["SME_EVALUATE_RESERVED_HOLDOUT"] = "1"
        started = time.time()
        production.run_fit(
            rule, STUDY_NAME, label(repeat, scale, rule, fold),
            config_for(rule, scale), TRAINING_SEEDS[repeat],
            15, 5, 999999, 999999, "cuda", worker,
            analyse=False, cv_fold=fold, cv_folds=len(FOLDS),
        )
        durations.append(time.time() - started)


def prediction_map(path: Path, split: str) -> dict:
    rows = [row for row in read_csv(path) if row["split"] == split]
    return {(row["molecule_id"], row["cyp_target"]): row for row in rows}


def repeat_oof(repeat: int) -> tuple[list[dict], np.ndarray]:
    by_expert = {}
    for scale in SCALES:
        for rule in RULES:
            values = {}
            for fold in FOLDS:
                rows = prediction_map(
                    train_path(repeat, scale, rule, fold) / "validation_predictions.csv",
                    "validation",
                )
                for key, row in rows.items():
                    values[(fold, *key)] = row
            by_expert[(scale, rule)] = values
    keys = sorted(set.intersection(*(set(rows) for rows in by_expert.values())))
    rows, matrix = [], []
    for key in keys:
        row = dict(by_expert[(SCALES[0], RULES[0])][key])
        row["fold"] = key[0]
        row["repeat"] = repeat
        rows.append(row)
        matrix.append([float(by_expert[(scale, rule)][key]["predicted_pic50"])
                       for scale in SCALES for rule in RULES])
    return rows, np.asarray(matrix)


def averaged_oof(repeats: list[tuple[list[dict], np.ndarray]]) -> tuple[list[dict], np.ndarray]:
    reference_rows, _ = repeats[0]
    matrices = []
    for rows, matrix in repeats:
        lookup = {(row["molecule_id"], row["cyp_target"]): values
                  for row, values in zip(rows, matrix)}
        matrices.append(np.asarray([lookup[(row["molecule_id"], row["cyp_target"])]
                                    for row in reference_rows]))
    return reference_rows, np.mean(np.stack(matrices), axis=0)


def reserved_holdout() -> tuple[list[dict], np.ndarray]:
    by_expert = {}
    for scale in SCALES:
        for rule in RULES:
            accumulated = {}
            for repeat in range(len(SPLIT_SEEDS)):
                for fold in FOLDS:
                    rows = prediction_map(
                        train_path(repeat, scale, rule, fold) / "validation_predictions.csv",
                        "reserved_holdout",
                    )
                    for key, row in rows.items():
                        entry = accumulated.setdefault(key, {"row": row, "values": []})
                        entry["values"].append(float(row["predicted_pic50"]))
            by_expert[(scale, rule)] = accumulated
    keys = sorted(set.intersection(*(set(rows) for rows in by_expert.values())))
    rows = [dict(by_expert[(SCALES[0], RULES[0])][key]["row"]) for key in keys]
    matrix = np.asarray([[np.mean(by_expert[(scale, rule)][key]["values"])
                          for scale in SCALES for rule in RULES] for key in keys])
    return rows, matrix


def blind_member(worker: Path, repeat: int, scale: str, rule: str, fold: int) -> Path:
    name = label(repeat, scale, rule, fold)
    output = STUDY / "blind_members" / name / "blinded_test_predictions.csv"
    if output.exists():
        return output
    env = os.environ.copy()
    env.update({
        "SME_GRAPH_CACHE": str(BLIND_CACHE), "SME_INCLUDE_BLIND": "1",
        "SME_CHECKPOINT": str(train_path(repeat, scale, rule, fold) / "model.pt"),
        "SME_RUN_NAME": f"{STUDY_NAME}/blind_members/{name}", "SME_DEVICE": "cuda",
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / "console.log").open("w", encoding="utf-8") as log:
        subprocess.run([str(worker), str(RUNNER), "predict"], cwd=ROOT, env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    return output


def generate_blind(worker: Path, models: dict, feature_names: list[str]) -> None:
    jobs = [(repeat, scale, rule, fold)
            for repeat in range(len(SPLIT_SEEDS)) for scale in SCALES
            for rule in RULES for fold in FOLDS]
    paths = {}
    for index, job in enumerate(jobs, 1):
        progress("blind_member_inference", completed=index - 1, total=len(jobs),
                 member=label(*job))
        paths[job] = blind_member(worker, *job)
    predictions = {job: {(row["molecule_id"], row["cyp_target"]):
                         float(row["predicted_pic50"]) for row in read_csv(path)}
                   for job, path in paths.items()}
    test_rows = read_csv(TEST_CSV)
    forbidden = {"experimental_pic50", "credible_interval_low", "credible_interval_high"}
    if forbidden.intersection(test_rows[0]):
        raise RuntimeError("Blind input contains forbidden label columns")
    long_rows, wide_rows = [], []
    for test_row in test_rows:
        molecule, smiles = test_row["Molecule_Name"], test_row["SMILES"]
        wide = {"SMILES": smiles, "Molecule_Name": molecule}
        for endpoint in ENDPOINTS:
            key = (molecule, endpoint)
            features = np.asarray([
                np.mean([predictions[(repeat, scale, rule, fold)][key]
                         for repeat in range(len(SPLIT_SEEDS)) for fold in FOLDS])
                for scale in SCALES for rule in RULES
            ])
            prediction = float(meta.predict_with_models(
                [{"cyp_target": endpoint}], features[None, :], models
            )[0])
            wide[f"{endpoint}_pIC50_direct_inhibition"] = prediction
            detail = {"molecule_id": molecule, "smiles": smiles,
                      "cyp_target": endpoint, "predicted_pic50": prediction}
            detail.update({name: float(value) for name, value in zip(feature_names, features)})
            long_rows.append(detail)
        wide_rows.append(wide)
    fields = ["SMILES", "Molecule_Name"] + [
        f"{endpoint}_pIC50_direct_inhibition" for endpoint in ENDPOINTS]
    write_csv(STUDY / "ten_rule_cft_ds_gcae_submission.csv", wide_rows, fields)
    write_csv(STUDY / "blind_predictions_long.csv", long_rows)


def main() -> None:
    worker = Path(os.environ.get(
        "SME_PYTHON", "C:/Users/Anthony/anaconda3/envs/strange-matter-gpu/python.exe"
    ))
    STUDY.mkdir(parents=True, exist_ok=True)
    started = time.time()
    train_members(worker)
    progress("assembling_repeated_oof")
    repeats = [repeat_oof(repeat) for repeat in range(len(SPLIT_SEEDS))]
    repeat_scores = {}
    for repeat, (rows, matrix) in enumerate(repeats):
        prediction, parameters = meta.cross_fitted_predictions(rows, matrix)
        point, rmse, by_cyp = ensemble.point_metrics(rows, prediction)
        repeat_scores[str(repeat)] = {"split_seed": SPLIT_SEEDS[repeat],
                                      "ma_st_rae": point, "rmse": rmse,
                                      "st_rae_by_cyp": by_cyp,
                                      "parameters": parameters}
    oof_rows, oof_matrix = averaged_oof(repeats)
    models = meta.final_models(oof_rows, oof_matrix)
    holdout_rows, holdout_matrix = reserved_holdout()
    holdout_prediction = meta.predict_with_models(holdout_rows, holdout_matrix, models)
    point, rmse, by_cyp = ensemble.point_metrics(holdout_rows, holdout_prediction)
    true, low, high = meta.endpoint_arrays(holdout_rows)
    endpoint_indices = np.asarray([ENDPOINTS.index(row["cyp_target"])
                                   for row in holdout_rows])
    bootstrap = bootstrap_regression_report(
        true, holdout_prediction, low, high, endpoint_indices, ENDPOINTS, 1000, 0
    )
    feature_names = [f"{scale}_{rule}" for scale in SCALES for rule in RULES]
    validation_rows = []
    for row, features, prediction in zip(holdout_rows, holdout_matrix, holdout_prediction):
        output = dict(row)
        output.update({name: float(value) for name, value in zip(feature_names, features)})
        output["predicted_pic50"] = float(prediction)
        validation_rows.append(output)
    write_csv(STUDY / "reserved_holdout_predictions.csv", validation_rows)
    summary = {
        "study": STUDY_NAME,
        "architecture": "retrained_ten_rule_cross_fitted_target_specific_dual_scale_gcae",
        "rules": RULES, "scales": SCALES, "split_seeds": SPLIT_SEEDS,
        "training_seeds": TRAINING_SEEDS, "folds": len(FOLDS),
        "fresh_graph_ca_training_runs": len(SPLIT_SEEDS) * len(SCALES) * len(RULES) * len(FOLDS),
        "feature_names": feature_names, "repeat_oof_scores": repeat_scores,
        "final_models": models, "reserved_holdout_point_ma_st_rae": point,
        "reserved_holdout_rmse": rmse, "reserved_holdout_st_rae_by_cyp": by_cyp,
        "reserved_holdout_bootstrap_metrics": bootstrap, "blind_labels_used": False,
        "elapsed_seconds_before_blind": time.time() - started,
    }
    write_json(STUDY / "study_summary.json", summary)
    progress("generating_blind_predictions", validation_ma_st_rae=point)
    generate_blind(worker, models, feature_names)
    summary["elapsed_seconds"] = time.time() - started
    summary["blind_prediction_rows"] = 3000
    write_json(STUDY / "study_summary.json", summary)
    progress("complete", summary=summary)


if __name__ == "__main__":
    main()
