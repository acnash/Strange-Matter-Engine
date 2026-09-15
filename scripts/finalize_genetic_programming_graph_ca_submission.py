#!/usr/bin/env python3
"""Evaluate and package the genetic-programming CYP3A4 Graph-CA submission.

The established CIA-EA-CV-CYP-GCA predictions are retained for CYP1A2,
CYP2C9, and CYP2D6. The frozen genetic-programming CYP3A4 winner is averaged
across five scaffold folds and two seeds. The sealed holdout is used only for
this final assessment and was excluded from program evolution and selection.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from challenge_metrics import bootstrap_regression_report, macro_soft_threshold_rae


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "results" / "production_genetic_programming_graph_ca_cyp3a4_v1"
BASELINE = ROOT / "results" / "production_credible_interval_aligned_ea_cv_cyp_gca_v1"
RUNNER = ROOT / "scripts" / "run_graph_ca_visual_prototype.py"
PYTHON = Path(r"C:\Users\Anthony\anaconda3\envs\strange-matter-gpu\python.exe")
GRAPH_CACHE = ROOT / "tmp" / "strange_matter_graph_ca_graphs.pkl"
BLIND_CACHE = ROOT / "tmp" / "strange_matter_graph_ca_graphs_with_blind.pkl"
ENDPOINTS = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")
FOLDS = tuple(range(5))
SEEDS = (1701, 4211)
CANDIDATE = "g00_c02"
PROGRAM = {
    "op": "mean",
    "left": {"op": "reaction"},
    "right": {"op": "neighbour_delta"},
}
METHOD = "GP-CIA-EA-CV-CYP-GCA"
SUBMISSION_NAME = "GP-CIA-EA-CV-CYP-GCA_OpenADMET_submission.csv"

FIXED_ENVIRONMENT = {
    "SME_ACTIVE_CYP": "CYP3A4",
    "SME_SPECIALIST_OBJECTIVE": "endpoint_only",
    "SME_TRAINING_ALGORITHM": "backprop",
    "SME_CA_RULE": "genetic_program",
    "SME_GP_PROGRAM_JSON": json.dumps(PROGRAM, separators=(",", ":")),
    "SME_GENERATIONS": "128",
    "SME_HIDDEN_CHANNELS": "24",
    "SME_ATOM_FEATURE_PROFILE": "comprehensive",
    "SME_TRAJECTORY_POOLING": "multiscale",
    "SME_DEGREE_NORMALIZATION_POWER": "1.0",
    "SME_DYNAMIC_OBSERVABLES": "1",
    "SME_MULTISCALE_TRANSITION_ENERGY": "1",
    "SME_CHEMICAL_FEATURE_GATING": "0",
    "SME_INITIAL_STATE_ANCHOR": "0",
    "SME_CHANNEL_ADAPTIVE_TIMESCALE": "0",
    "SME_MULTILAG_RECURRENCE_SIGNATURE": "0",
    "SME_TEMPORAL_EXTREMA_SIGNATURE": "0",
    "SME_DIRECTIONAL_FLUX_SIGNATURE": "0",
    "SME_CA_LR": "0.003",
    "SME_RIDGE": "0.01",
    "SME_CA_L2": "0.0001",
    "SME_GRAD_CLIP": "0.5",
    "SME_UPDATE_SCALE": "0.08",
    "SME_INIT_SCALE": "0.5",
    "SME_INITIAL_NOISE": "0.005",
    "SME_SUPPORT_FRACTION": "0.75",
    "SME_BATCH_MOLECULES": "64",
    "SME_BOND_TEMPERATURE": "1.0",
    "SME_DYN_A": "0.5",
    "SME_DYN_B": "0.2",
    "SME_DYN_C": "0.8",
    "SME_DYN_D": "0.15",
    "SME_RIDGE_MODE": "shared",
    "SME_CV_FOLDS": "5",
    "SME_CV_SPLIT_SEED": "260822",
}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def member_dir(kind: str, fold: int, seed: int) -> Path:
    return CAMPAIGN / kind / CANDIDATE / f"fold_{fold}_seed_{seed}"


def checkpoint(fold: int, seed: int) -> Path:
    return CAMPAIGN / "confirmation" / CANDIDATE / f"fold_{fold}_seed_{seed}" / "model.pt"


def worker_environment(kind: str, fold: int, seed: int) -> dict[str, str]:
    output = member_dir(kind, fold, seed)
    environment = os.environ.copy()
    environment.update(FIXED_ENVIRONMENT)
    environment.update({
        "SME_CHECKPOINT": str(checkpoint(fold, seed)),
        "SME_RUN_NAME": output.relative_to(ROOT / "results").as_posix(),
        "SME_DEVICE": "cuda",
        "SME_SEED": str(seed),
    })
    if kind == "sealed_members":
        environment.update({
            "SME_GRAPH_CACHE": str(GRAPH_CACHE),
            "SME_EVALUATE_CHECKPOINT": "1",
            "SME_CV_FOLD": str(fold),
            "SME_INCLUDE_BLIND": "0",
        })
    else:
        environment.update({
            "SME_GRAPH_CACHE": str(BLIND_CACHE),
            "SME_INCLUDE_BLIND": "1",
        })
    return environment


def run_member(kind: str, fold: int, seed: int) -> Path:
    output = member_dir(kind, fold, seed)
    filename = (
        "checkpoint_evaluation_predictions.csv"
        if kind == "sealed_members"
        else "blinded_test_predictions.csv"
    )
    product = output / filename
    if product.exists():
        return product
    output.mkdir(parents=True, exist_ok=True)
    phase = "train" if kind == "sealed_members" else "predict"
    with (output / "console.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            [str(PYTHON), str(RUNNER), phase],
            cwd=ROOT,
            env=worker_environment(kind, fold, seed),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return product


def run_members(kind: str) -> list[Path]:
    jobs = [(fold, seed) for fold in FOLDS for seed in SEEDS]
    products: list[Path] = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {
            pool.submit(run_member, kind, fold, seed): (fold, seed)
            for fold, seed in jobs
        }
        for future in as_completed(futures):
            products.append(future.result())
    return products


def averaged_predictions(paths: list[Path], split: str) -> dict[str, float]:
    predictions: dict[str, list[float]] = {}
    for path in paths:
        for row in read_csv(path):
            if row["cyp_target"] != "CYP3A4":
                continue
            if split == "sealed" and row.get("split") != "reserved_holdout":
                continue
            predictions.setdefault(row["molecule_id"], []).append(
                float(row["predicted_pic50"])
            )
    if not predictions or any(len(values) != 10 for values in predictions.values()):
        raise RuntimeError("Every CYP3A4 molecule must have ten ensemble predictions")
    return {key: float(np.mean(values)) for key, values in predictions.items()}


def sealed_assessment(gp_predictions: dict[str, float]) -> dict:
    rows = read_csv(BASELINE / "reserved_holdout_predictions.csv")
    output = []
    for row in rows:
        item = dict(row)
        if item["cyp_target"] == "CYP3A4":
            item["predicted_pic50"] = gp_predictions[item["molecule_id"]]
            item["selected_rules"] = "genetic_program_g00_c02_ten_checkpoint_mean"
            prediction = float(item["predicted_pic50"])
            low = float(item["credible_interval_low"])
            high = float(item["credible_interval_high"])
            item["soft_threshold_absolute_error"] = max(
                low - prediction, prediction - high, 0.0
            )
            item["residual"] = prediction - float(item["experimental_pic50"])
        output.append(item)
    write_csv(CAMPAIGN / "reserved_holdout_predictions.csv", output)

    y = np.asarray([float(row["experimental_pic50"]) for row in output])
    pred = np.asarray([float(row["predicted_pic50"]) for row in output])
    low = np.asarray([float(row["credible_interval_low"]) for row in output])
    high = np.asarray([float(row["credible_interval_high"]) for row in output])
    endpoint_indices = np.asarray([ENDPOINTS.index(row["cyp_target"]) for row in output])
    point, per_endpoint = macro_soft_threshold_rae(
        y, pred, low, high, endpoint_indices, ENDPOINTS
    )
    bootstrap = bootstrap_regression_report(
        y, pred, low, high, endpoint_indices, ENDPOINTS, 1000, 0
    )
    return {
        "method": METHOD,
        "composition": {
            "CYP1A2": "CIA-EA-CV-CYP-GCA",
            "CYP2C9": "CIA-EA-CV-CYP-GCA",
            "CYP2D6": "CIA-EA-CV-CYP-GCA",
            "CYP3A4": "genetic-programming g00_c02, ten-checkpoint mean",
        },
        "genetic_program": PROGRAM,
        "reserved_holdout_point_ma_st_rae": float(point),
        "reserved_holdout_rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
        "reserved_holdout_st_rae_by_cyp": per_endpoint,
        "reserved_holdout_bootstrap_metrics": bootstrap,
        "reserved_holdout_used_for_selection": False,
        "blind_labels_loaded": False,
    }


def blind_submission(gp_predictions: dict[str, float], summary: dict) -> Path:
    baseline_rows = read_csv(
        BASELINE / "credible_interval_aligned_ea_cv_cyp_gca_submission.csv"
    )
    fieldnames = ["SMILES", "Molecule_Name"] + [
        f"{endpoint}_pIC50_direct_inhibition" for endpoint in ENDPOINTS
    ]
    for row in baseline_rows:
        row["CYP3A4_pIC50_direct_inhibition"] = gp_predictions[row["Molecule_Name"]]
    submission = CAMPAIGN / SUBMISSION_NAME
    write_csv(submission, baseline_rows, fieldnames)
    write_csv(ROOT / "output" / SUBMISSION_NAME, baseline_rows, fieldnames)

    long_rows = []
    for row in baseline_rows:
        for endpoint in ENDPOINTS:
            long_rows.append({
                "molecule_id": row["Molecule_Name"],
                "smiles": row["SMILES"],
                "cyp_target": endpoint,
                "predicted_pic50": row[f"{endpoint}_pIC50_direct_inhibition"],
                "source": (
                    "genetic_program_g00_c02_ten_checkpoint_mean"
                    if endpoint == "CYP3A4"
                    else "CIA-EA-CV-CYP-GCA"
                ),
            })
    write_csv(CAMPAIGN / "blind_predictions_long.csv", long_rows)
    manifest = {
        "method": METHOD,
        "submission": str(submission.relative_to(ROOT)),
        "rows": len(baseline_rows),
        "columns": fieldnames,
        "finite_predictions": int(sum(
            np.isfinite(float(row[f"{endpoint}_pIC50_direct_inhibition"]))
            for row in baseline_rows
            for endpoint in ENDPOINTS
        )),
        "expected_predictions": 750 * 4,
        "unique_molecule_names": len({row["Molecule_Name"] for row in baseline_rows}),
        "labels_loaded": False,
        "schema_valid": len(baseline_rows) == 750,
        "sealed_point_ma_st_rae": summary["reserved_holdout_point_ma_st_rae"],
    }
    if (
        not manifest["schema_valid"]
        or manifest["unique_molecule_names"] != 750
        or manifest["finite_predictions"] != 3000
    ):
        raise RuntimeError(f"Invalid submission manifest: {manifest}")
    write_json(CAMPAIGN / "inference_manifest.json", manifest)
    return submission


def main() -> None:
    sealed = averaged_predictions(run_members("sealed_members"), "sealed")
    summary = sealed_assessment(sealed)
    write_json(CAMPAIGN / "study_summary.json", summary)
    blind = averaged_predictions(run_members("blind_members"), "blind")
    submission = blind_submission(blind, summary)
    print(json.dumps({"summary": summary, "submission": str(submission)}, indent=2))


if __name__ == "__main__":
    main()
