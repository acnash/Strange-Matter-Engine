#!/usr/bin/env python3
"""Run the frozen DS-GCAE composite on the blinded challenge molecules."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from runtime_device import python_executable, requested_device


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_graph_ca_visual_prototype.py"
TEST_CSV = ROOT / "data" / "openadmet-cyp-challenge-2026" / "cyp-challenge-TEST-BLINDED.csv"
STUDY = ROOT / "results" / "production_dual_scale_graph_ca_ensemble_v1"
CACHE = ROOT / "tmp" / "strange_matter_graph_ca_graphs_with_blind.pkl"
RULES = (
    "gated_residual", "delayed_memory", "inertial_reaction_diffusion",
    "kuramoto_sakaguchi", "fitzhugh_nagumo",
)
SEEDS = (1701, 2909, 4211)
ENDPOINTS = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fieldnames=None):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or rows[0].keys())
        writer.writeheader(); writer.writerows(rows)


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def checkpoint(expert, rule, seed):
    if expert == "original":
        return (ROOT / "results" / f"production_{rule}_ensemble_v1" /
                "runs" / "final_model" / "model.pt")
    label = "final_model" if seed == 1701 else f"final_seed_{seed}"
    return (ROOT / "results" / f"production_{rule}_multiscale_ensemble_v2" /
            "runs" / label / "model.pt")


def run_member(worker, report_python, device, expert, rule, seed):
    label = f"{expert}_{rule}_seed_{seed}"
    run_name = f"{STUDY.name}/inference_members/{label}"
    output = ROOT / "results" / run_name / "blinded_test_predictions.csv"
    if output.exists():
        return output
    model_path = checkpoint(expert, rule, seed)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    env = os.environ.copy()
    env.update({
        "SME_GRAPH_CACHE": str(CACHE),
        "SME_INCLUDE_BLIND": "1",
        "SME_CHECKPOINT": str(model_path),
        "SME_RUN_NAME": run_name,
        "SME_DEVICE": device,
    })
    log = output.parent / "console.log"
    output.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        subprocess.run(
            [str(worker), str(RUNNER), "predict"], cwd=ROOT, env=env,
            stdout=handle, stderr=subprocess.STDOUT, check=True,
        )
    return output


def keyed_predictions(path):
    rows = read_csv(path)
    return {
        (row["molecule_id"], row["cyp_target"]): float(row["predicted_pic50"])
        for row in rows
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--python", dest="python_path", default=None)
    parser.add_argument("--report-python", default=None)
    args = parser.parse_args()
    device = requested_device(args.device)
    worker = python_executable(args.python_path)
    report_python = python_executable(args.report_python or worker)
    STUDY.mkdir(parents=True, exist_ok=True)
    progress = STUDY / "progress.json"
    started = time.time()

    if not CACHE.exists():
        env = os.environ.copy()
        env.update({"SME_GRAPH_CACHE": str(CACHE), "SME_INCLUDE_BLIND": "1"})
        subprocess.run(
            [str(report_python), str(RUNNER), "prepare"], cwd=ROOT, env=env, check=True
        )

    member_paths = {}
    jobs = [("original", rule, 1701) for rule in RULES]
    jobs.extend(("multiscale", rule, seed) for rule in RULES for seed in SEEDS)
    for index, (expert, rule, seed) in enumerate(jobs, start=1):
        write_json(progress, {
            "stage": "member_inference", "completed": index - 1,
            "total": len(jobs), "expert": expert, "rule": rule, "seed": seed,
        })
        member_paths[(expert, rule, seed)] = run_member(
            worker, report_python, device, expert, rule, seed
        )

    test_rows = read_csv(TEST_CSV)
    if len(test_rows) != 750:
        raise ValueError(f"Expected 750 blinded molecules, found {len(test_rows)}")
    member_predictions = {
        key: keyed_predictions(path) for key, path in member_paths.items()
    }
    expected_keys = {
        (row["Molecule_Name"], endpoint) for row in test_rows for endpoint in ENDPOINTS
    }
    for key, predictions in member_predictions.items():
        if set(predictions) != expected_keys:
            raise ValueError(f"Prediction keys do not align for {key}")

    blend_summary = json.loads((
        ROOT / "results" / "production_endpoint_selective_ca_ensemble_v1" /
        "study_summary.json"
    ).read_text())
    alpha = float(blend_summary["final_parameters"]["global"])
    original_weights = np.asarray(blend_summary["base_weights"]["global"], dtype=float)
    multiscale_weights = np.asarray(
        blend_summary["multiscale_weights"]["global"], dtype=float
    )
    long_rows = []
    wide_rows = []
    for test_row in test_rows:
        molecule = test_row["Molecule_Name"]
        wide = {"SMILES": test_row["SMILES"], "Molecule_Name": molecule}
        for endpoint in ENDPOINTS:
            key = (molecule, endpoint)
            original_values = np.asarray([
                member_predictions[("original", rule, 1701)][key] for rule in RULES
            ])
            multiscale_rule_values = np.asarray([
                np.mean([member_predictions[("multiscale", rule, seed)][key]
                         for seed in SEEDS])
                for rule in RULES
            ])
            original_prediction = float(original_weights @ original_values)
            multiscale_prediction = float(multiscale_weights @ multiscale_rule_values)
            prediction = float((1.0 - alpha) * original_prediction
                               + alpha * multiscale_prediction)
            if not np.isfinite(prediction):
                raise ValueError(f"Non-finite prediction for {key}")
            wide[f"{endpoint}_pIC50_direct_inhibition"] = prediction
            long_rows.append({
                "molecule_id": molecule,
                "smiles": test_row["SMILES"],
                "cyp_target": endpoint,
                "original_ensemble_prediction": original_prediction,
                "multiscale_ensemble_prediction": multiscale_prediction,
                "ds_gcae_predicted_pic50": prediction,
            })
        wide_rows.append(wide)

    submission_columns = ["SMILES", "Molecule_Name"] + [
        f"{endpoint}_pIC50_direct_inhibition" for endpoint in ENDPOINTS
    ]
    write_csv(STUDY / "ds_gcae_blinded_predictions_long.csv", long_rows)
    write_csv(STUDY / "ds_gcae_submission.csv", wide_rows, submission_columns)
    manifest = {
        "model": "Dual-Scale Graph Cellular Automata Ensemble",
        "abbreviation": "DS-GCAE",
        "version": "v1",
        "original_ensemble_weight": 1.0 - alpha,
        "multiscale_ensemble_weight": alpha,
        "original_rule_weights": original_weights.tolist(),
        "multiscale_rule_weights": multiscale_weights.tolist(),
        "multiscale_seeds": list(SEEDS),
        "member_checkpoints": {"|".join(map(str, key)): str(checkpoint(*key).relative_to(ROOT))
                               for key in member_paths},
        "test_molecules": len(test_rows),
        "prediction_rows": len(long_rows),
        "labels_loaded": False,
        "schema_validated": True,
        "all_predictions_finite": True,
        "elapsed_seconds": time.time() - started,
        "device": device,
    }
    write_json(STUDY / "inference_manifest.json", manifest)
    write_json(progress, {"stage": "complete", "manifest": manifest})
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
