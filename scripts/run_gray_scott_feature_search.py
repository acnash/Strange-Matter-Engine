#!/usr/bin/env python3
"""Thorough, resumable Gray-Scott search over atom chemistry and CA controls."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import subprocess
import time
from pathlib import Path

try:
    from run_production_transition_study import run_fit, score
    from runtime_device import python_executable, requested_device
except ModuleNotFoundError:
    from scripts.run_production_transition_study import run_fit, score
    from scripts.runtime_device import python_executable, requested_device


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_graph_ca_visual_prototype.py"
SEARCH_SEED = 240826
SEARCH_VERSION = "gray_scott_atom_features_v1"
STUDY_NAME = f"production_gray_scott_{SEARCH_VERSION}"
FEATURE_PROFILES = (
    "baseline", "periodic", "valence", "electronic", "ring_geometry",
    "periodic_valence", "periodic_electronic", "valence_electronic",
    "comprehensive",
)
BASELINE_WINNER = {
    "generations": 125, "hidden_channels": 16, "ca_lr": 3e-3,
    "ridge": 1e-1, "ca_l2": 1e-5, "gradient_clip": 0.5,
    "batch_molecules": 128, "update_scale": 0.25, "init_scale": 1.5,
    "initial_noise": 0.005, "support_fraction": 0.6,
    "bond_temperature": 1.0, "dyn_a": 0.5, "dyn_b": 0.2,
    "dyn_c": 0.8, "dyn_d": 0.15, "atom_feature_profile": "baseline",
}


def candidates(count: int = 72) -> list[dict]:
    """Create a balanced chemistry-first design around the prior winner."""
    rng = random.Random(SEARCH_SEED)
    chosen = [dict(BASELINE_WINNER)]
    for profile in FEATURE_PROFILES[1:]:
        candidate = dict(BASELINE_WINNER)
        candidate["atom_feature_profile"] = profile
        chosen.append(candidate)
    while len(chosen) < count:
        config = {
            "generations": rng.choice((32, 64, 125, 250)),
            "hidden_channels": rng.choice((8, 16, 24)),
            "ca_lr": rng.choice((3e-4, 1e-3, 3e-3)),
            "ridge": rng.choice((1e-2, 3e-2, 1e-1, 3e-1, 1.0)),
            "ca_l2": rng.choice((1e-6, 1e-5, 1e-4)),
            "gradient_clip": rng.choice((0.5, 1.0, 2.0)),
            "batch_molecules": rng.choice((64, 128)),
            "update_scale": rng.choice((0.15, 0.25, 0.4)),
            "init_scale": rng.choice((0.5, 1.0, 1.5)),
            "initial_noise": rng.choice((0.0, 0.005, 0.01)),
            "support_fraction": rng.choice((0.6, 0.75, 0.85)),
            "bond_temperature": rng.choice((0.5, 1.0, 2.0)),
            "dyn_a": rng.choice((0.2, 0.5, 0.8)),
            "dyn_b": rng.choice((0.2, 0.5, 0.8)),
            "dyn_c": rng.choice((0.2, 0.5, 0.8)),
            "dyn_d": rng.choice((0.05, 0.15, 0.3)),
            "atom_feature_profile": rng.choice(FEATURE_PROFILES),
        }
        if config not in chosen:
            chosen.append(config)
    return chosen


def write_progress(study_dir: Path, payload: dict) -> None:
    (study_dir / "progress.json").write_text(json.dumps(payload, indent=2) + "\n")


def prepare_cache(worker_python: Path, cache: Path) -> None:
    if cache.exists():
        return
    env = os.environ.copy()
    env.update({"SME_GRAPH_CACHE": str(cache), "SME_INCLUDE_BLIND": "0"})
    subprocess.run([str(worker_python), str(RUNNER), "prepare"], cwd=ROOT,
                   env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--python", dest="python_path", default=None)
    args = parser.parse_args()
    device = requested_device(args.device)
    worker_python = python_executable(args.python_path)
    study_dir = ROOT / "results" / STUDY_NAME
    study_dir.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "tmp" / "gray_scott_atom_feature_graphs.pkl"
    prepare_cache(worker_python, cache)
    os.environ["SME_GRAPH_CACHE"] = str(cache)
    started = time.time()
    design = candidates()
    (study_dir / "search_space.json").write_text(json.dumps({
        "search_seed": SEARCH_SEED, "search_version": SEARCH_VERSION,
        "feature_profiles": list(FEATURE_PROFILES),
        "sampled_configurations": design,
    }, indent=2) + "\n")

    stage1 = []
    for index, config in enumerate(design, 1):
        metrics = run_fit("gray_scott", STUDY_NAME, f"stage1_{index:03d}",
                          config, 1701, 4, 2, 800, 250, device, worker_python)
        stage1.append((config, metrics))
        write_progress(study_dir, {"stage": "stage1", "completed": index,
            "total": len(design), "best_rmse": min(score(m) for _, m in stage1),
            "best_feature_profile": min(stage1, key=lambda x: score(x[1]))[0]["atom_feature_profile"]})
    promoted = sorted(stage1, key=lambda item: score(item[1]))[:16]

    stage2 = []
    for index, (config, _) in enumerate(promoted, 1):
        metrics = run_fit("gray_scott", STUDY_NAME, f"stage2_{index:02d}",
                          config, 1701, 8, 4, 2000, 500, device, worker_python)
        stage2.append((config, metrics))
        write_progress(study_dir, {"stage": "stage2", "completed": index,
            "total": len(promoted), "best_rmse": min(score(m) for _, m in stage2),
            "best_feature_profile": min(stage2, key=lambda x: score(x[1]))[0]["atom_feature_profile"]})
    finalists = sorted(stage2, key=lambda item: score(item[1]))[:4]

    seeds = (1701, 2909, 4211)
    confirmations = []
    for config_index, (config, _) in enumerate(finalists, 1):
        metrics_list = [run_fit(
            "gray_scott", STUDY_NAME, f"confirm_{config_index:02d}_seed_{seed}",
            config, seed, 12, 5, 999999, 999999, device, worker_python,
        ) for seed in seeds]
        values = [score(metrics) for metrics in metrics_list]
        mean_rmse = statistics.fmean(values)
        seed_sd = statistics.pstdev(values)
        robust_score = mean_rmse + 0.25 * seed_sd
        confirmations.append((config, metrics_list, mean_rmse, seed_sd, robust_score))
        write_progress(study_dir, {"stage": "confirmation", "completed": config_index,
            "total": len(finalists), "latest_mean_rmse": mean_rmse,
            "latest_seed_sd": seed_sd, "latest_robust_score": robust_score,
            "latest_feature_profile": config["atom_feature_profile"]})
    viable = [item for item in confirmations if math.isfinite(item[4])]
    if not viable:
        raise RuntimeError("Every Gray-Scott confirmation candidate failed")
    winner, winner_metrics, winner_mean, winner_sd, winner_score = min(
        viable, key=lambda item: item[4]
    )
    final_metrics = run_fit("gray_scott", STUDY_NAME, "final_model", winner,
                            1701, 18, 6, 999999, 999999, device,
                            worker_python, analyse=True)

    rows = []
    for stage, collection in (("stage1", stage1), ("stage2", stage2)):
        for config, metrics in collection:
            rows.append({"stage": stage, "seed": metrics["seed"],
                         "validation_rmse": score(metrics), **config})
    for config, metrics_list, _, _, _ in confirmations:
        for metrics in metrics_list:
            rows.append({"stage": "confirmation", "seed": metrics["seed"],
                         "validation_rmse": score(metrics), **config})
    with (study_dir / "all_trials.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    summary = {
        "study": STUDY_NAME, "rule": "gray_scott",
        "search_seed": SEARCH_SEED, "search_version": SEARCH_VERSION,
        "elapsed_seconds": time.time() - started, "winner": winner,
        "winner_confirmation_mean_rmse": winner_mean,
        "winner_confirmation_seed_sd": winner_sd,
        "winner_selection_score": winner_score,
        "confirmation_seeds": list(seeds), "requested_device": device,
        "worker_python": str(worker_python), "final_metrics": final_metrics,
        "baseline_final_validation_rmse": 0.867433488368988,
        "blind_data_used": False,
    }
    (study_dir / "study_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_progress(study_dir, {"stage": "complete", "summary": summary})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
