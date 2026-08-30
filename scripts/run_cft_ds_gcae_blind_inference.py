#!/usr/bin/env python3
"""Apply the frozen target-specific dual-scale Graph-CA ridge stack to blind predictions."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "production_dual_scale_graph_ca_ensemble_v1"
STUDY = ROOT / "results" / "production_cross_fitted_target_calibrated_gcae_v1"
TEST_CSV = ROOT / "data" / "openadmet-cyp-challenge-2026" / "cyp-challenge-TEST-BLINDED.csv"
RULES = (
    "gated_residual", "delayed_memory", "inertial_reaction_diffusion",
    "kuramoto_sakaguchi", "fitzhugh_nagumo",
)
SEEDS = (1701, 2909, 4211)
ENDPOINTS = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields=None) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def member_file(expert: str, rule: str, seed: int) -> Path:
    label = f"{expert}_{rule}_seed_{seed}"
    return SOURCE / "inference_members" / label / "blinded_test_predictions.csv"


def keyed(path: Path) -> dict[tuple[str, str], float]:
    if not path.exists():
        raise FileNotFoundError(path)
    return {(row["molecule_id"], row["cyp_target"]): float(row["predicted_pic50"])
            for row in read_csv(path)}


def predict(features: np.ndarray, model: dict) -> float:
    ridge = model["ridge"]
    mean = np.asarray(ridge["feature_mean"], dtype=float)
    scale = np.asarray(ridge["feature_scale"], dtype=float)
    coefficients = np.asarray(ridge["coefficients"], dtype=float)
    raw = float(ridge["intercept"] + ((features - mean) / scale) @ coefficients)
    calibration = model["calibration"]
    return float(calibration["slope"] * raw + calibration["intercept"])


def main() -> None:
    started = time.time()
    summary = json.loads((STUDY / "study_summary.json").read_text(encoding="utf-8"))
    models = summary["final_models"]
    test_rows = read_csv(TEST_CSV)
    forbidden = {"experimental_pic50", "credible_interval_low", "credible_interval_high"}
    if forbidden.intersection(test_rows[0]):
        raise RuntimeError("Blind input unexpectedly contains label columns")
    if len(test_rows) != 750:
        raise ValueError(f"Expected 750 blinded molecules, found {len(test_rows)}")

    members = {}
    for rule in RULES:
        members[("original", rule, 1701)] = keyed(member_file("original", rule, 1701))
        for seed in SEEDS:
            members[("multiscale", rule, seed)] = keyed(
                member_file("multiscale", rule, seed)
            )

    expected = {(row["Molecule_Name"], endpoint)
                for row in test_rows for endpoint in ENDPOINTS}
    for name, values in members.items():
        if set(values) != expected:
            raise RuntimeError(f"Blind keys do not align for {name}")

    long_rows, submission_rows = [], []
    for row in test_rows:
        molecule, smiles = row["Molecule_Name"], row["SMILES"]
        submission = {"SMILES": smiles, "Molecule_Name": molecule}
        for endpoint in ENDPOINTS:
            key = (molecule, endpoint)
            original = [members[("original", rule, 1701)][key] for rule in RULES]
            multiscale = [np.mean([members[("multiscale", rule, seed)][key]
                                   for seed in SEEDS]) for rule in RULES]
            features = np.asarray(original + multiscale, dtype=float)
            prediction = predict(features, models[endpoint])
            if not np.isfinite(prediction):
                raise RuntimeError(f"Non-finite blind prediction for {key}")
            submission[f"{endpoint}_pIC50_direct_inhibition"] = prediction
            detail = {"molecule_id": molecule, "smiles": smiles,
                      "cyp_target": endpoint, "predicted_pic50": prediction}
            detail.update({f"original_{rule}": value
                           for rule, value in zip(RULES, original)})
            detail.update({f"multiscale_{rule}": value
                           for rule, value in zip(RULES, multiscale)})
            long_rows.append(detail)
        submission_rows.append(submission)

    columns = ["SMILES", "Molecule_Name"] + [
        f"{endpoint}_pIC50_direct_inhibition" for endpoint in ENDPOINTS
    ]
    write_csv(STUDY / "cft_ds_gcae_blinded_predictions_long.csv", long_rows)
    write_csv(STUDY / "cft_ds_gcae_submission.csv", submission_rows, columns)
    manifest = {
        "model": "Cross-Fitted Target-Specific Dual-Scale Graph Cellular Automata Ensemble",
        "abbreviation": "CFT-DS-GCAE",
        "version": "v1",
        "source_member_predictions": str(SOURCE.relative_to(ROOT)),
        "feature_order": summary["feature_names"],
        "target_specific_models": models,
        "test_molecules": len(test_rows),
        "prediction_rows": len(long_rows),
        "labels_loaded": False,
        "schema_validated": True,
        "all_predictions_finite": True,
        "elapsed_seconds": time.time() - started,
    }
    write_json(STUDY / "inference_manifest.json", manifest)
    write_json(STUDY / "progress.json", {"stage": "complete", "manifest": manifest})
    print(json.dumps({key: value for key, value in manifest.items()
                      if key != "target_specific_models"}, indent=2))


if __name__ == "__main__":
    main()
