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

STUDY_NAME = os.environ.get(
    "SME_CYP_CAMPAIGN_NAME", "production_cv_cyp_specialist_gca_v1"
)
STUDY = ROOT / "results" / STUDY_NAME
ENDPOINT_ALIGNED = os.environ.get("SME_ENDPOINT_ALIGNED", "0") == "1"
INTERVAL_REFINEMENT = os.environ.get("SME_INTERVAL_REFINEMENT", "0") == "1"
INTERVAL_BETAS = (0.0, 0.25, 0.5, 0.75)
PARENT_ENDPOINT_STUDY = (
    ROOT / "results" / "production_endpoint_aligned_cv_cyp_gca_v1"
)
RULES = previous.RULES
ENDPOINTS = ensemble.ENDPOINTS
SCREEN_FOLDS = (0, 1)
FOLDS = tuple(range(5))
SEEDS = (7103, 8209)
PENALTIES = meta.RIDGE_PENALTIES
RUNNER = ROOT / "scripts" / "run_graph_ca_visual_prototype.py"
BLIND_CACHE = ROOT / "tmp" / "strange_matter_graph_ca_graphs_with_blind.pkl"
TEST_CSV = ROOT / "data" / "openadmet-cyp-challenge-2026" / "cyp-challenge-TEST-BLINDED.csv"


def campaign_method() -> str:
    if INTERVAL_REFINEMENT:
        return "CIA-EA-CV-CYP-GCA"
    return "EA-CV-CYP-GCA" if ENDPOINT_ALIGNED else "CV-CYP-GCA"


def campaign_architecture() -> str:
    if INTERVAL_REFINEMENT:
        return "credible_interval_aligned_endpoint_graph_ca"
    return ("endpoint_aligned_cross_validated_cyp_specialist_graph_ca"
            if ENDPOINT_ALIGNED else
            "cross_validated_cyp_specialist_graph_ca")


def campaign_submission_name() -> str:
    if INTERVAL_REFINEMENT:
        return "credible_interval_aligned_ea_cv_cyp_gca_submission.csv"
    return ("endpoint_aligned_cv_cyp_gca_submission.csv"
            if ENDPOINT_ALIGNED else "cv_cyp_gca_submission.csv")


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
    config = deepcopy(config)
    if ENDPOINT_ALIGNED:
        config["specialist_objective"] = "endpoint_only"
    label = run_label(stage, endpoint, rule, variant, seed, fold)
    return production.run_fit(
        rule, STUDY_NAME, label, config, seed, epochs, patience,
        999999, 999999, "cuda", worker, analyse=False,
        cv_fold=fold, cv_folds=len(FOLDS), active_cyp=endpoint,
    )


def screen(worker: Path) -> dict:
    if INTERVAL_REFINEMENT:
        return screen_interval_refinement(worker)
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


def screen_interval_refinement(worker: Path) -> dict:
    """Tune smooth credible-interval loss around the selected EA specialists."""
    parent = json.loads(
        (PARENT_ENDPOINT_STUDY / "screening_summary.json").read_text(
            encoding="utf-8"
        )
    )
    jobs = [
        (endpoint, candidate, beta, fold)
        for endpoint in ENDPOINTS
        for candidate in parent[endpoint]
        for beta in INTERVAL_BETAS
        for fold in SCREEN_FOLDS
    ]
    scores = {
        (endpoint, candidate["rule"], beta): []
        for endpoint in ENDPOINTS
        for candidate in parent[endpoint]
        for beta in INTERVAL_BETAS
    }
    durations = []
    for number, (endpoint, candidate, beta, fold) in enumerate(jobs, 1):
        config = deepcopy(candidate["config"])
        if beta > 0:
            config.update({
                "loss_mode": "hybrid_interval",
                "interval_loss_beta": beta,
                "interval_temperature": 0.05,
            })
        else:
            config.update({"loss_mode": "mse", "interval_loss_beta": 0.0})
        variant = f"{candidate['variant']}_loss_{int(beta * 100):02d}"
        progress(
            "interval_loss_screen", completed=number - 1, total=len(jobs),
            endpoint=endpoint, rule=candidate["rule"], beta=beta, fold=fold,
            estimated_remaining_seconds=(
                np.mean(durations) * (len(jobs) - number + 1)
                if durations else None
            ),
        )
        started = time.time()
        metrics = fit(
            worker, "screen", endpoint, candidate["rule"], variant,
            config, 6101, fold, 22, 6,
        )
        durations.append(time.time() - started)
        scores[(endpoint, candidate["rule"], beta)].append(float(
            metrics.get("restored_validation_point_ma_st_rae", np.inf)
        ))

    selected = {}
    for endpoint in ENDPOINTS:
        selected[endpoint] = []
        for candidate in parent[endpoint]:
            choices = []
            for beta in INTERVAL_BETAS:
                values = scores[(endpoint, candidate["rule"], beta)]
                choices.append((float(np.mean(values)), beta, values))
            score, beta, values = min(choices)
            config = deepcopy(candidate["config"])
            if beta > 0:
                config.update({
                    "loss_mode": "hybrid_interval",
                    "interval_loss_beta": beta,
                    "interval_temperature": 0.05,
                })
            else:
                config.update({"loss_mode": "mse", "interval_loss_beta": 0.0})
            selected[endpoint].append({
                "rule": candidate["rule"],
                "variant": f"{candidate['variant']}_loss_{int(beta * 100):02d}",
                "screen_ma_st_rae": score,
                "fold_scores": values,
                "interval_loss_beta": beta,
                "config": config,
            })
        selected[endpoint].sort(key=lambda item: item["screen_ma_st_rae"])
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
    filename = ("checkpoint_evaluation_predictions.csv"
                if split == "reserved_holdout" else "validation_predictions.csv")
    return [row for row in read_csv(run_path(label) / filename)
            if row["split"] == split and row["cyp_target"] == endpoint]


def evaluate_holdouts(worker: Path, selected: dict) -> None:
    jobs = [(endpoint, candidate, seed, fold)
            for endpoint, candidates in selected.items() for candidate in candidates
            for seed in SEEDS for fold in FOLDS]
    for number, (endpoint, candidate, seed, fold) in enumerate(jobs, 1):
        label = run_label("confirm", endpoint, candidate["rule"], candidate["variant"], seed, fold)
        run_dir = run_path(label)
        output = run_dir / "checkpoint_evaluation_predictions.csv"
        if output.exists():
            continue
        progress("checkpoint_holdout_evaluation", completed=number - 1,
                 total=len(jobs), endpoint=endpoint, rule=candidate["rule"],
                 seed=seed, fold=fold)
        env = os.environ.copy()
        env.update({
            "SME_GRAPH_CACHE": str(production.GRAPH_CACHE),
            "SME_CHECKPOINT": str(run_dir / "model.pt"),
            "SME_RUN_NAME": f"{STUDY_NAME}/runs/{label}",
            "SME_DEVICE": "cuda", "SME_EVALUATE_CHECKPOINT": "1",
            "SME_ACTIVE_CYP": endpoint, "SME_CV_FOLD": str(fold),
            "SME_CV_FOLDS": str(len(FOLDS)),
        })
        with (run_dir / "checkpoint_evaluation.log").open("w", encoding="utf-8") as log:
            subprocess.run([str(worker), str(RUNNER), "train"], cwd=ROOT, env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)


def matrices(selected: dict, split: str):
    all_rows, all_matrix = [], []
    for endpoint in ENDPOINTS:
        rules = selected[endpoint]
        maps = []
        reference_rows = {}
        for candidate_index, candidate in enumerate(rules):
            values = {}
            for seed in SEEDS:
                for fold in FOLDS:
                    candidate_rows = prediction_rows(
                        endpoint, candidate, seed, fold, split
                    )
                    for row in candidate_rows:
                        key = ((int(row.get("fold", fold)) if split == "validation" else 0),
                               row["molecule_id"], endpoint)
                        values.setdefault(key, []).append(float(row["predicted_pic50"]))
                        if candidate_index == 0:
                            reference_rows.setdefault(key, row)
            maps.append(values)
        keys = sorted(set.intersection(*(set(mapping) for mapping in maps)))
        for key in keys:
            row = dict(reference_rows[key]); row["fold"] = key[0]
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
                    state = fit_ridge_stable(
                        x[train][:, columns], y[train], penalty
                    )
                    prediction[held] = meta.predict_ridge(x[held][:, columns], state)
                score = float(soft_threshold_rae(y, prediction, low, high))
                if score < best["score"]:
                    best = {"score": score, "columns": list(columns),
                            "penalty": float(penalty)}
    best["ridge"] = fit_ridge_stable(
        x[:, best["columns"]], y, best["penalty"]
    )
    return best


def fit_ridge_stable(x: np.ndarray, y: np.ndarray, penalty: float) -> dict:
    """Fit the small stacking ridge system through PyTorch's stable solver.

    The Windows NumPy LAPACK loader can terminate inside ``numpy.linalg.solve``
    after a long CUDA campaign.  This equivalent CPU float64 solve avoids that
    platform failure while retaining the same standardized ridge objective.
    """
    import torch

    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    standardized = (x - mean) / scale
    target_mean = float(y.mean())
    design = torch.as_tensor(standardized, dtype=torch.float64)
    target = torch.as_tensor(y - target_mean, dtype=torch.float64)
    identity = torch.eye(design.shape[1], dtype=torch.float64)
    coefficients = torch.linalg.solve(
        design.T @ design + float(penalty) * identity,
        design.T @ target,
    ).cpu().numpy()
    return {
        "feature_mean": mean,
        "feature_scale": scale,
        "coefficients": coefficients,
        "intercept": target_mean,
        "penalty": float(penalty),
    }


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
        "method": campaign_method(),
        "architecture": campaign_architecture(),
        "specialisation_stage": "nonlinear_ca_backpropagation_and_ridge_readout",
        "screened_rules_per_endpoint": len(RULES), "selected_candidates": selected,
        "final_models": models, "reserved_holdout_point_ma_st_rae": point,
        "reserved_holdout_rmse": rmse, "reserved_holdout_st_rae_by_cyp": by_cyp,
        "reserved_holdout_bootstrap_metrics": bootstrap,
        "reserved_holdout_used_for_selection": False, "blind_labels_loaded": False,
    }
    write_json(STUDY / "study_summary.json", summary)
    return summary


def blind_member(worker: Path, endpoint: str, candidate: dict,
                 seed: int, fold: int) -> Path:
    label = run_label("confirm", endpoint, candidate["rule"],
                      candidate["variant"], seed, fold)
    output_dir = STUDY / "blind_members" / label
    output = output_dir / "blinded_test_predictions.csv"
    if output.exists():
        return output
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "SME_GRAPH_CACHE": str(BLIND_CACHE), "SME_INCLUDE_BLIND": "1",
        "SME_CHECKPOINT": str(run_path(label) / "model.pt"),
        "SME_RUN_NAME": f"{STUDY_NAME}/blind_members/{label}",
        "SME_DEVICE": "cuda", "SME_ACTIVE_CYP": endpoint,
    })
    with (output_dir / "console.log").open("w", encoding="utf-8") as log:
        subprocess.run([str(worker), str(RUNNER), "predict"], cwd=ROOT, env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    return output


def generate_blind(worker: Path, selected: dict, summary: dict) -> None:
    required = []
    for endpoint in ENDPOINTS:
        columns = summary["final_models"][endpoint]["columns"]
        for column in columns:
            candidate = selected[endpoint][column]
            for seed in SEEDS:
                for fold in FOLDS:
                    required.append((endpoint, column, candidate, seed, fold))
    member_maps = {}
    for number, (endpoint, column, candidate, seed, fold) in enumerate(required, 1):
        progress("blind_checkpoint_inference", completed=number - 1,
                 total=len(required), endpoint=endpoint,
                 rule=candidate["rule"], seed=seed, fold=fold)
        path = blind_member(worker, endpoint, candidate, seed, fold)
        rows = [row for row in read_csv(path) if row["cyp_target"] == endpoint]
        member_maps[(endpoint, column, seed, fold)] = {
            row["molecule_id"]: float(row["predicted_pic50"]) for row in rows
        }

    test_rows = read_csv(TEST_CSV)
    forbidden = {"experimental_pic50", "credible_interval_low", "credible_interval_high"}
    if test_rows and forbidden.intersection(test_rows[0]):
        raise RuntimeError("Blind input contains forbidden label columns")
    wide_rows, long_rows = [], []
    for source in test_rows:
        molecule = source["Molecule_Name"]
        wide = {"SMILES": source["SMILES"], "Molecule_Name": molecule}
        for endpoint in ENDPOINTS:
            model = summary["final_models"][endpoint]
            features = []
            detail = {"molecule_id": molecule, "smiles": source["SMILES"],
                      "cyp_target": endpoint}
            for column in model["columns"]:
                values = [member_maps[(endpoint, column, seed, fold)][molecule]
                          for seed in SEEDS for fold in FOLDS]
                average = float(np.mean(values))
                features.append(average)
                rule = selected[endpoint][column]["rule"]
                detail[f"mean_{rule}"] = average
            prediction = float(meta.predict_ridge(
                np.asarray(features, dtype=float)[None, :], model["ridge"])[0])
            if not np.isfinite(prediction):
                raise RuntimeError(f"Non-finite blind prediction for {molecule}/{endpoint}")
            wide[f"{endpoint}_pIC50_direct_inhibition"] = prediction
            detail["predicted_pic50"] = prediction
            long_rows.append(detail)
        wide_rows.append(wide)
    fields = ["SMILES", "Molecule_Name"] + [
        f"{endpoint}_pIC50_direct_inhibition" for endpoint in ENDPOINTS]
    submission_name = campaign_submission_name()
    submission = STUDY / submission_name
    write_csv(submission, wide_rows, fields)
    long_fields = list(dict.fromkeys(
        key for row in long_rows for key in row.keys()
    ))
    write_csv(STUDY / "blind_predictions_long.csv", long_rows, long_fields)
    manifest = {
        "method": campaign_method(),
        "submission": str(submission.relative_to(ROOT)),
        "rows": len(wide_rows), "columns": fields,
        "finite_predictions": int(len(wide_rows) * len(ENDPOINTS)),
        "expected_predictions": 750 * 4,
        "unique_molecule_names": len({row["Molecule_Name"] for row in wide_rows}),
        "labels_loaded": False, "schema_valid": len(wide_rows) == 750,
        "selected_rules": {endpoint: summary["final_models"][endpoint]["rules"]
                           for endpoint in ENDPOINTS},
    }
    if not manifest["schema_valid"] or manifest["unique_molecule_names"] != 750:
        raise RuntimeError(f"Invalid blind submission manifest: {manifest}")
    write_json(STUDY / "inference_manifest.json", manifest)


def main() -> None:
    worker = Path(os.environ.get(
        "SME_PYTHON", "C:/Users/Anthony/anaconda3/envs/strange-matter-gpu/python.exe"))
    STUDY.mkdir(parents=True, exist_ok=True)
    production.ensure_graph_cache(worker)
    screening_path = STUDY / "screening_summary.json"
    selected = (json.loads(screening_path.read_text(encoding="utf-8"))
                if screening_path.exists() else screen(worker))
    confirm(worker, selected)
    evaluate_holdouts(worker, selected)
    summary_path = STUDY / "study_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        progress("assembling_validation")
        summary = assemble(selected)
    generate_blind(worker, selected, summary)
    progress("complete", summary=summary,
             submission=campaign_submission_name())
    print(json.dumps(meta.serialise(summary), indent=2))


if __name__ == "__main__":
    main()
