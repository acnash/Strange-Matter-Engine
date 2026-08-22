#!/usr/bin/env python3
"""Resumable staged production search for one graph-CA transition rule."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import random
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(r"C:\Users\Anthony\anaconda3\envs\strange-matter-gpu\python.exe")
RUNNER = ROOT / "scripts" / "run_graph_ca_visual_prototype.py"
SEARCH_SEED = 260822
SEARCH_VERSION = "conditioned_v2"


def search_space():
    values = itertools.product(
        (16, 32, 64, 125, 250, 500),
        (4, 8, 16),
        (3e-4, 1e-3, 3e-3),
        (1e-2, 1e-1, 1.0),
        (1e-6, 1e-5, 1e-4),
        (0.5, 1.0, 2.0),
        (64, 128),
    )
    return [dict(zip(
        ("generations", "hidden_channels", "ca_lr", "ridge", "ca_l2",
         "gradient_clip", "batch_molecules"), row
    )) for row in values]


def sampled_candidates(count=32):
    default = dict(generations=64, hidden_channels=8, ca_lr=1e-3,
                   ridge=1e-1, ca_l2=1e-5, gradient_clip=1.0,
                   batch_molecules=64)
    space = search_space()
    rng = random.Random(SEARCH_SEED)
    rng.shuffle(space)
    chosen = [default]
    for config in space:
        if config not in chosen:
            chosen.append(config)
        if len(chosen) == count:
            break
    return chosen


def metric_path(run_name):
    return ROOT / "results" / run_name / "metrics.json"


def run_fit(rule, study_name, label, config, seed, epochs, patience,
            fit_limit, validation_limit, analyse=False):
    run_name = f"{study_name}/runs/{label}"
    metrics_file = metric_path(run_name)
    if metrics_file.exists():
        return json.loads(metrics_file.read_text())
    env = os.environ.copy()
    env.update({
        "SME_CA_RULE": rule,
        "SME_GENERATIONS": str(config["generations"]),
        "SME_HIDDEN_CHANNELS": str(config["hidden_channels"]),
        "SME_CA_LR": str(config["ca_lr"]),
        "SME_RIDGE": str(config["ridge"]),
        "SME_CA_L2": str(config["ca_l2"]),
        "SME_GRAD_CLIP": str(config["gradient_clip"]),
        "SME_BATCH_MOLECULES": str(config["batch_molecules"]),
        "SME_RUN_NAME": run_name,
        "SME_DEVICE": "cuda",
        "SME_SEED": str(seed),
        "SME_MAX_EPOCHS": str(epochs),
        "SME_PATIENCE": str(patience),
        "SME_MIN_DELTA": "0.003",
        "SME_TUNING_ONLY": "1",
        "SME_TUNING_FIT_MOLECULES": str(fit_limit),
        "SME_TUNING_VAL_MOLECULES": str(validation_limit),
        "SME_ANALYSE_VALIDATION": "1" if analyse else "0",
        "SME_PERTURBATION_CASES": "0" if analyse else "20",
    })
    run_dir = metrics_file.parent
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "console.log"
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            [str(PYTHON), str(RUNNER), "train"], cwd=ROOT, env=env,
            stdout=log, stderr=subprocess.STDOUT, text=True,
        )
    if process.returncode:
        raise RuntimeError(f"Run {label} failed; inspect {log_path}")
    metrics = json.loads(metrics_file.read_text())
    metrics["wall_seconds"] = time.time() - started
    metrics["study_config"] = config
    history_path = metrics_file.parent / "training_history.csv"
    with history_path.open(newline="") as handle:
        history = list(csv.DictReader(handle))
    metrics["stability"] = {
        "maximum_query_train_rmse": max(float(r["train_rmse"]) for r in history),
        "maximum_raw_gradient_norm": max(float(r["mean_raw_gradient_norm"]) for r in history),
    }
    metrics["stability"]["passed"] = (
        metrics["stability"]["maximum_query_train_rmse"] < 20.0 and
        metrics["stability"]["maximum_raw_gradient_norm"] < 1e7
    )
    metrics_file.write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def score(metrics):
    if not metrics.get("stability", {}).get("passed", True):
        return float("inf")
    value = metrics.get("restored_validation_rmse", float("inf"))
    return value if value == value else float("inf")


def write_progress(study_dir, payload):
    (study_dir / "progress.json").write_text(json.dumps(payload, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rule", required=True,
                        choices=("gated_residual", "inertial_reaction_diffusion"))
    args = parser.parse_args()
    rule = args.rule
    short_name = "gated_residual" if rule == "gated_residual" else "inertial_reaction_diffusion"
    study_name = f"production_{short_name}_{SEARCH_VERSION}"
    study_dir = ROOT / "results" / study_name
    study_dir.mkdir(parents=True, exist_ok=True)
    study_started = time.time()
    candidates = sampled_candidates()
    (study_dir / "search_space.json").write_text(json.dumps({
        "search_seed": SEARCH_SEED,
        "search_version": SEARCH_VERSION,
        "sampled_configurations": candidates,
    }, indent=2) + "\n")

    stage1 = []
    for index, config in enumerate(candidates, 1):
        metrics = run_fit(rule, study_name, f"stage1_{index:02d}", config,
                          1701, 3, 2, 600, 200)
        stage1.append((config, metrics))
        write_progress(study_dir, {"stage": "stage1", "completed": index,
                                   "total": len(candidates),
                                   "best_rmse": min(score(m) for _, m in stage1)})
    promoted8 = sorted(stage1, key=lambda item: score(item[1]))[:8]

    stage2 = []
    for index, (config, _) in enumerate(promoted8, 1):
        metrics = run_fit(rule, study_name, f"stage2_{index:02d}", config,
                          1701, 5, 3, 2000, 500)
        stage2.append((config, metrics))
        write_progress(study_dir, {"stage": "stage2", "completed": index,
                                   "total": len(promoted8),
                                   "best_rmse": min(score(m) for _, m in stage2)})
    promoted3 = sorted(stage2, key=lambda item: score(item[1]))[:3]

    confirmations = []
    seeds = (1701, 2909, 4211)
    for config_index, (config, _) in enumerate(promoted3, 1):
        config_metrics = []
        for seed in seeds:
            metrics = run_fit(
                rule, study_name, f"confirm_{config_index:02d}_seed_{seed}",
                config, seed, 10, 4, 999999, 999999,
            )
            config_metrics.append(metrics)
        mean_rmse = sum(score(m) for m in config_metrics) / len(config_metrics)
        confirmations.append((config, config_metrics, mean_rmse))
        write_progress(study_dir, {"stage": "confirmation", "completed": config_index,
                                   "total": len(promoted3), "latest_mean_rmse": mean_rmse})
    winner, winner_metrics, winner_mean = min(confirmations, key=lambda item: item[2])

    final_metrics = run_fit(rule, study_name, "final_model", winner, 1701,
                            15, 5, 999999, 999999, analyse=True)
    rows = []
    for stage_name, collection in (("stage1", stage1), ("stage2", stage2)):
        for config, metrics in collection:
            rows.append({"stage": stage_name, "seed": metrics["seed"],
                         "validation_rmse": score(metrics), **config})
    for config, metrics_list, _ in confirmations:
        for metrics in metrics_list:
            rows.append({"stage": "confirmation", "seed": metrics["seed"],
                         "validation_rmse": score(metrics), **config})
    with (study_dir / "all_trials.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    summary = {
        "study": study_name,
        "rule": rule,
        "search_seed": SEARCH_SEED,
        "search_version": SEARCH_VERSION,
        "elapsed_seconds": time.time() - study_started,
        "winner": winner,
        "winner_confirmation_mean_rmse": winner_mean,
        "confirmation_seeds": list(seeds),
        "final_metrics": final_metrics,
        "blind_data_used": False,
    }
    (study_dir / "study_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_progress(study_dir, {"stage": "complete", "summary": summary})
    report_python = Path(os.environ.get("SME_REPORT_PYTHON", str(PYTHON)))
    subprocess.run([str(report_python), str(ROOT / "scripts" / "build_production_report.py"),
                    "--study", study_name], cwd=ROOT, check=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
