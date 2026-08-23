#!/usr/bin/env python3
"""Resumable staged production search for one graph-CA transition rule."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import time
from pathlib import Path

from runtime_device import python_executable, requested_device


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_graph_ca_visual_prototype.py"
SEARCH_SEED = 260822
SEARCH_VERSION = "enhanced_v3"
RULES = (
    "gated_residual",
    "inertial_reaction_diffusion",
    "activator_inhibitor",
    "coupled_map",
    "damped_symplectic",
)


def sampled_candidates(count=36):
    default = dict(generations=64, hidden_channels=8, ca_lr=1e-3,
                   ridge=1e-1, ca_l2=1e-5, gradient_clip=1.0,
                   batch_molecules=64, update_scale=0.25, init_scale=1.0,
                   initial_noise=0.0, support_fraction=0.75,
                   bond_temperature=1.0, dyn_a=0.5, dyn_b=0.5,
                   dyn_c=0.5, dyn_d=0.2)
    rng = random.Random(SEARCH_SEED)
    chosen = [default]
    while len(chosen) < count:
        config = {
            "generations": rng.choice((16, 32, 64, 125, 250, 500)),
            "hidden_channels": rng.choice((4, 8, 16)),
            "ca_lr": rng.choice((3e-4, 1e-3, 3e-3)),
            "ridge": rng.choice((1e-2, 1e-1, 1.0)),
            "ca_l2": rng.choice((1e-6, 1e-5, 1e-4)),
            "gradient_clip": rng.choice((0.5, 1.0, 2.0)),
            "batch_molecules": rng.choice((64, 128)),
            "update_scale": rng.choice((0.08, 0.15, 0.25, 0.4)),
            "init_scale": rng.choice((0.5, 1.0, 1.5)),
            "initial_noise": rng.choice((0.0, 0.005, 0.01)),
            "support_fraction": rng.choice((0.6, 0.75, 0.85)),
            "bond_temperature": rng.choice((0.5, 1.0, 2.0)),
            "dyn_a": rng.choice((0.2, 0.5, 0.8)),
            "dyn_b": rng.choice((0.2, 0.5, 0.8)),
            "dyn_c": rng.choice((0.2, 0.5, 0.8)),
            "dyn_d": rng.choice((0.05, 0.15, 0.3)),
        }
        if config not in chosen:
            chosen.append(config)
    return chosen


def metric_path(run_name):
    return ROOT / "results" / run_name / "metrics.json"


def run_fit(rule, study_name, label, config, seed, epochs, patience,
            fit_limit, validation_limit, device="auto", python_path=None,
            analyse=False):
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
        "SME_UPDATE_SCALE": str(config["update_scale"]),
        "SME_INIT_SCALE": str(config["init_scale"]),
        "SME_INITIAL_NOISE": str(config["initial_noise"]),
        "SME_SUPPORT_FRACTION": str(config["support_fraction"]),
        "SME_BOND_TEMPERATURE": str(config["bond_temperature"]),
        "SME_DYN_A": str(config["dyn_a"]),
        "SME_DYN_B": str(config["dyn_b"]),
        "SME_DYN_C": str(config["dyn_c"]),
        "SME_DYN_D": str(config["dyn_d"]),
        "SME_RUN_NAME": run_name,
        "SME_DEVICE": requested_device(device),
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
            [str(python_executable(python_path)), str(RUNNER), "train"],
            cwd=ROOT, env=env,
            stdout=log, stderr=subprocess.STDOUT, text=True,
        )
    if process.returncode:
        metrics = {
            "rule": rule,
            "seed": seed,
            "restored_validation_rmse": float("inf"),
            "wall_seconds": time.time() - started,
            "study_config": config,
            "stability": {"passed": False},
            "failure": f"Run failed with exit code {process.returncode}; inspect {log_path.name}",
        }
        metrics_file.write_text(json.dumps(metrics, indent=2) + "\n")
        return metrics
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
    parser = argparse.ArgumentParser(
        description="Run a resumable Graph-CA production search on CPU or CUDA."
    )
    parser.add_argument("--rule", required=True, choices=RULES)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto",
        help="auto selects CUDA when available and CPU otherwise (default: auto)",
    )
    parser.add_argument(
        "--python", dest="python_path", default=None,
        help="worker Python executable; defaults to SME_PYTHON or this interpreter",
    )
    parser.add_argument(
        "--report-python", default=None,
        help="report Python executable; defaults to SME_REPORT_PYTHON or worker Python",
    )
    args = parser.parse_args()
    rule = args.rule
    device = requested_device(args.device)
    worker_python = python_executable(args.python_path)
    short_name = rule
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
                          1701, 3, 2, 600, 200, device, worker_python)
        stage1.append((config, metrics))
        write_progress(study_dir, {"stage": "stage1", "completed": index,
                                   "total": len(candidates),
                                   "best_rmse": min(score(m) for _, m in stage1)})
    promoted10 = sorted(stage1, key=lambda item: score(item[1]))[:10]

    stage2 = []
    for index, (config, _) in enumerate(promoted10, 1):
        metrics = run_fit(rule, study_name, f"stage2_{index:02d}", config,
                          1701, 5, 3, 2000, 500, device, worker_python)
        stage2.append((config, metrics))
        write_progress(study_dir, {"stage": "stage2", "completed": index,
                                   "total": len(promoted10),
                                   "best_rmse": min(score(m) for _, m in stage2)})
    promoted3 = sorted(stage2, key=lambda item: score(item[1]))[:3]

    confirmations = []
    seeds = (1701, 2909, 4211)
    for config_index, (config, _) in enumerate(promoted3, 1):
        config_metrics = []
        for seed in seeds:
            metrics = run_fit(
                rule, study_name, f"confirm_{config_index:02d}_seed_{seed}",
                config, seed, 10, 4, 999999, 999999, device, worker_python,
            )
            config_metrics.append(metrics)
        mean_rmse = sum(score(m) for m in config_metrics) / len(config_metrics)
        variance = sum((score(m) - mean_rmse) ** 2 for m in config_metrics) / len(config_metrics)
        seed_sd = variance ** 0.5
        robust_score = mean_rmse + 0.25 * seed_sd
        confirmations.append((config, config_metrics, mean_rmse, seed_sd, robust_score))
        write_progress(study_dir, {"stage": "confirmation", "completed": config_index,
                                   "total": len(promoted3), "latest_mean_rmse": mean_rmse,
                                   "latest_seed_sd": seed_sd,
                                   "latest_robust_score": robust_score})
    winner, winner_metrics, winner_mean, winner_sd, winner_score = min(
        confirmations, key=lambda item: item[4]
    )

    final_metrics = run_fit(rule, study_name, "final_model", winner, 1701,
                            15, 5, 999999, 999999, device, worker_python,
                            analyse=True)
    rows = []
    for stage_name, collection in (("stage1", stage1), ("stage2", stage2)):
        for config, metrics in collection:
            rows.append({"stage": stage_name, "seed": metrics["seed"],
                         "validation_rmse": score(metrics), **config})
    for config, metrics_list, _, _, _ in confirmations:
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
        "winner_confirmation_seed_sd": winner_sd,
        "winner_selection_score": winner_score,
        "confirmation_seeds": list(seeds),
        "requested_device": device,
        "worker_python": str(worker_python),
        "final_metrics": final_metrics,
        "blind_data_used": False,
    }
    (study_dir / "study_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_progress(study_dir, {"stage": "complete", "summary": summary})
    if os.environ.get("SME_SKIP_REPORT", "0") != "1":
        report_python = python_executable(
            args.report_python or os.environ.get("SME_REPORT_PYTHON")
            or str(worker_python)
        )
        subprocess.run([str(report_python), str(ROOT / "scripts" / "build_production_report.py"),
                        "--study", study_name], cwd=ROOT, check=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
