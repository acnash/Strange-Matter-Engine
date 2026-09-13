#!/usr/bin/env python3
"""Tune the ES optimizer while holding the CYP3A4 Graph-CA fixed."""

from __future__ import annotations

import csv
import itertools
import json
import math
import os
import random
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CAMPAIGN_NAME = "production_es_optimizer_tuning_fixed_graph_ca_v1"
CAMPAIGN = RESULTS / CAMPAIGN_NAME
PYTHON = Path(r"C:\Users\Anthony\anaconda3\envs\strange-matter-gpu\python.exe")
WORKER = ROOT / "scripts" / "run_graph_ca_visual_prototype.py"
PARALLEL_WORKERS = int(os.environ.get("SME_ES_CAMPAIGN_WORKERS", "5"))
SCREEN_CONFIGS = int(os.environ.get("SME_ES_SCREEN_CONFIGS", "32"))
CONFIRM_CONFIGS = int(os.environ.get("SME_ES_CONFIRM_CONFIGS", "5"))
PROGRESS_LOCK = threading.Lock()


FIXED_GRAPH_CA = {
    "SME_CA_RULE": "damped_symplectic",
    "SME_ACTIVE_CYP": "CYP3A4",
    "SME_SPECIALIST_OBJECTIVE": "endpoint_only",
    "SME_GENERATIONS": "32",
    "SME_HIDDEN_CHANNELS": "16",
    "SME_RIDGE": "0.1",
    "SME_CA_L2": "0.000001",
    "SME_UPDATE_SCALE": "0.25",
    "SME_INIT_SCALE": "1.5",
    "SME_INITIAL_NOISE": "0.0",
    "SME_SUPPORT_FRACTION": "0.6",
    "SME_BOND_TEMPERATURE": "1.0",
    "SME_DYN_A": "0.5",
    "SME_DYN_B": "0.2",
    "SME_DYN_C": "0.2",
    "SME_DYN_D": "0.15",
    "SME_ATOM_FEATURE_PROFILE": "periodic_electronic",
    "SME_TRAJECTORY_POOLING": "multiscale",
    "SME_RIDGE_MODE": "shared",
}

STOPPING_PROFILES = {
    "responsive": {"epochs": 40, "patience": 8, "min_delta": 0.003},
    "balanced": {"epochs": 60, "patience": 14, "min_delta": 0.002},
    "patient": {"epochs": 80, "patience": 20, "min_delta": 0.002},
    "fine": {"epochs": 100, "patience": 25, "min_delta": 0.001},
}


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    temporary.replace(path)


def optimizer_design() -> list[dict]:
    baseline = {
        "population": 64,
        "sigma": 0.02,
        "learning_rate": 0.003,
        "batch_molecules": 1024,
        "batch_composition": "random",
        "stopping_profile": "patient",
    }
    levels = {
        "population": (32, 64, 96, 128),
        "sigma": (0.005, 0.01, 0.02, 0.04, 0.08),
        "learning_rate": (0.0005, 0.001, 0.003, 0.006, 0.01),
        "batch_molecules": (256, 512, 1024, 1600),
        "batch_composition": ("random", "activity_stratified"),
        "stopping_profile": tuple(STOPPING_PROFILES),
    }

    def balanced_sequence(values, count, seed):
        sequence = list(values) * math.ceil(count / len(values))
        sequence = sequence[:count]
        random.Random(seed).shuffle(sequence)
        return sequence

    sequences = {
        key: balanced_sequence(values, SCREEN_CONFIGS, 260913 + 101 * offset)
        for offset, (key, values) in enumerate(levels.items())
    }
    design = [
        {key: sequences[key][index] for key in levels}
        for index in range(SCREEN_CONFIGS)
    ]
    design[0] = baseline
    used = set()
    all_combinations = list(itertools.product(*(levels[key] for key in levels)))
    replacement_rng = random.Random(913260)
    replacement_rng.shuffle(all_combinations)
    for index, config in enumerate(design):
        signature = tuple(config[key] for key in levels)
        if signature in used:
            signature = next(item for item in all_combinations if item not in used)
            design[index] = dict(zip(levels, signature))
        used.add(signature)
    for index, config in enumerate(design):
        config["config_id"] = f"config_{index:02d}"
    return design


def result_directory(stage: str, config_id: str, fold: int, seed: int) -> Path:
    return CAMPAIGN / stage / config_id / f"fold_{fold}_seed_{seed}"


def run_name(output: Path) -> str:
    return output.relative_to(RESULTS).as_posix()


def read_metrics(output: Path):
    metrics_path = output / "metrics.json"
    if not metrics_path.exists():
        return None
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def worker_task(stage: str, config: dict, fold: int, seed: int) -> dict:
    output = result_directory(stage, config["config_id"], fold, seed)
    output.mkdir(parents=True, exist_ok=True)
    existing = read_metrics(output)
    if existing is not None:
        return {
            "stage": stage, "config_id": config["config_id"], "fold": fold,
            "seed": seed, "status": "complete", "resumed": True,
            "best_validation_ma_st_rae": existing["best_validation_ma_st_rae"],
            "best_validation_rmse": existing["best_validation_rmse"],
            "epochs_run": existing["epochs_run"],
        }

    stopping = STOPPING_PROFILES[config["stopping_profile"]]
    environment = os.environ.copy()
    environment.update(FIXED_GRAPH_CA)
    environment.update({
        "SME_TRAINING_ALGORITHM": "evolution_strategy",
        "SME_RUN_NAME": run_name(output),
        "SME_DEVICE": "cuda",
        "SME_CV_FOLD": str(fold),
        "SME_CV_FOLDS": "5",
        "SME_CV_SPLIT_SEED": "260822",
        "SME_TUNING_ONLY": "1",
        "SME_TUNING_FIT_MOLECULES": "999999",
        "SME_TUNING_VAL_MOLECULES": "999999",
        "SME_SEED": str(seed),
        "SME_MAX_EPOCHS": str(stopping["epochs"]),
        "SME_PATIENCE": str(stopping["patience"]),
        "SME_MIN_DELTA": str(stopping["min_delta"]),
        "SME_ES_POPULATION": str(config["population"]),
        "SME_ES_SIGMA": str(config["sigma"]),
        "SME_ES_LR": str(config["learning_rate"]),
        "SME_ES_BATCH_MOLECULES": str(config["batch_molecules"]),
        "SME_ES_BATCH_COMPOSITION": config["batch_composition"],
    })
    manifest = {
        "stage": stage,
        "config": config,
        "fixed_graph_ca": FIXED_GRAPH_CA,
        "fold": fold,
        "seed": seed,
        "blind_labels_loaded": False,
        "sealed_holdout_used_for_selection": False,
    }
    atomic_json(output / "run_manifest.json", manifest)
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    started = time.time()
    with (output / "campaign_stdout.log").open("w", encoding="utf-8") as stdout, \
            (output / "campaign_stderr.log").open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(
            [str(PYTHON), str(WORKER), "train"], cwd=ROOT, env=environment,
            stdout=stdout, stderr=stderr, creationflags=creation_flags,
            check=False,
        )
    metrics = read_metrics(output)
    result = {
        "stage": stage, "config_id": config["config_id"], "fold": fold,
        "seed": seed, "status": "complete" if completed.returncode == 0 else "failed",
        "return_code": completed.returncode, "wall_seconds": time.time() - started,
        "resumed": False,
    }
    if metrics is not None:
        result.update({
            "best_validation_ma_st_rae": metrics["best_validation_ma_st_rae"],
            "best_validation_rmse": metrics["best_validation_rmse"],
            "epochs_run": metrics["epochs_run"],
        })
    return result


def run_tasks(stage: str, tasks: list[tuple[dict, int, int]], progress: dict) -> list[dict]:
    results = []
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        futures = {
            executor.submit(worker_task, stage, config, fold, seed):
            (config["config_id"], fold, seed)
            for config, fold, seed in tasks
        }
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:  # Preserve campaign state for diagnosis.
                config_id, fold, seed = futures[future]
                result = {
                    "stage": stage, "config_id": config_id, "fold": fold,
                    "seed": seed, "status": "failed", "error": repr(exc),
                }
            results.append(result)
            with PROGRESS_LOCK:
                progress["completed_runs"] = progress.get("completed_runs", 0) + 1
                progress["last_result"] = result
                progress["updated_at_unix"] = time.time()
                atomic_json(CAMPAIGN / "progress.json", progress)
            print(json.dumps(result, allow_nan=True), flush=True)
    return results


def rank_screen(configs: list[dict], results: list[dict]) -> list[dict]:
    ranked = []
    for config in configs:
        observations = [
            row for row in results
            if row["config_id"] == config["config_id"] and row["status"] == "complete"
        ]
        if len(observations) != 2:
            continue
        ranked.append({
            **config,
            "screen_mean_ma_st_rae": sum(
                row["best_validation_ma_st_rae"] for row in observations
            ) / len(observations),
            "screen_mean_rmse": sum(
                row["best_validation_rmse"] for row in observations
            ) / len(observations),
            "screen_runs": observations,
        })
    ranked.sort(key=lambda row: (row["screen_mean_ma_st_rae"], row["screen_mean_rmse"]))
    return ranked


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def main() -> None:
    if not PYTHON.exists():
        raise FileNotFoundError(PYTHON)
    CAMPAIGN.mkdir(parents=True, exist_ok=True)
    configs = optimizer_design()
    atomic_json(CAMPAIGN / "optimizer_design.json", {
        "campaign": CAMPAIGN_NAME,
        "fixed_graph_ca": FIXED_GRAPH_CA,
        "screen_folds": [0, 1],
        "confirmation_folds": [0, 1, 2, 3, 4],
        "confirmation_seeds": [1701, 4211],
        "parallel_cuda_workers": PARALLEL_WORKERS,
        "configs": configs,
    })
    progress = {
        "campaign": CAMPAIGN_NAME,
        "stage": "screen",
        "completed_runs": 0,
        "total_planned_runs": len(configs) * 2 + CONFIRM_CONFIGS * 10,
        "blind_labels_loaded": False,
        "sealed_holdout_used_for_selection": False,
        "started_at_unix": time.time(),
    }
    atomic_json(CAMPAIGN / "progress.json", progress)

    screen_tasks = [
        (config, fold, 1701) for config in configs for fold in (0, 1)
    ]
    screen_results = run_tasks("screen", screen_tasks, progress)
    ranked = rank_screen(configs, screen_results)
    if len(ranked) < CONFIRM_CONFIGS:
        raise RuntimeError("Too few complete screen configurations for confirmation")
    atomic_json(CAMPAIGN / "screen_ranking.json", ranked)
    write_csv(
        CAMPAIGN / "screen_ranking.csv", ranked,
        ["config_id", "population", "sigma", "learning_rate",
         "batch_molecules", "batch_composition", "stopping_profile",
         "screen_mean_ma_st_rae", "screen_mean_rmse"],
    )

    finalists = ranked[:CONFIRM_CONFIGS]
    progress["stage"] = "confirmation"
    progress["finalist_config_ids"] = [row["config_id"] for row in finalists]
    atomic_json(CAMPAIGN / "progress.json", progress)
    confirmation_tasks = [
        (config, fold, seed)
        for config in finalists
        for fold in range(5)
        for seed in (1701, 4211)
    ]
    confirmation_results = run_tasks(
        "confirmation", confirmation_tasks, progress
    )
    confirmed = []
    for config in finalists:
        observations = [
            row for row in confirmation_results
            if row["config_id"] == config["config_id"] and row["status"] == "complete"
        ]
        if len(observations) != 10:
            continue
        confirmed.append({
            **{key: config[key] for key in (
                "config_id", "population", "sigma", "learning_rate",
                "batch_molecules", "batch_composition", "stopping_profile",
            )},
            "confirmation_mean_ma_st_rae": sum(
                row["best_validation_ma_st_rae"] for row in observations
            ) / len(observations),
            "confirmation_mean_rmse": sum(
                row["best_validation_rmse"] for row in observations
            ) / len(observations),
            "confirmation_runs": observations,
        })
    confirmed.sort(key=lambda row: (
        row["confirmation_mean_ma_st_rae"], row["confirmation_mean_rmse"]
    ))
    atomic_json(CAMPAIGN / "confirmation_ranking.json", confirmed)
    write_csv(
        CAMPAIGN / "confirmation_ranking.csv", confirmed,
        ["config_id", "population", "sigma", "learning_rate",
         "batch_molecules", "batch_composition", "stopping_profile",
         "confirmation_mean_ma_st_rae", "confirmation_mean_rmse"],
    )
    progress["stage"] = "complete"
    progress["completed_at_unix"] = time.time()
    progress["winner"] = confirmed[0] if confirmed else None
    atomic_json(CAMPAIGN / "progress.json", progress)
    print(json.dumps({"campaign_complete": True, "winner": progress["winner"]},
                     allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
