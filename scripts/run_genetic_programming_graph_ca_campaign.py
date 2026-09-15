#!/usr/bin/env python3
"""Full genetic-programming search over CYP3A4 Graph-CA update equations."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DEFAULT_CAMPAIGN = "production_genetic_programming_graph_ca_cyp3a4_v1"
PYTHON = Path(r"C:\Users\Anthony\anaconda3\envs\strange-matter-gpu\python.exe")
WORKER = ROOT / "scripts" / "run_graph_ca_visual_prototype.py"
LOCK = threading.Lock()

TERMINALS = ("reaction", "state", "neighbour_delta", "memory_delta")
UNARY = ("neg", "tanh", "sin", "scale")
BINARY = ("add", "sub", "mul", "mean")
SCALES = (-1.0, -0.5, 0.25, 0.5, 1.0)
MAX_DEPTH = 4

FIXED_TRAINING = {
    "SME_ACTIVE_CYP": "CYP3A4",
    "SME_SPECIALIST_OBJECTIVE": "endpoint_only",
    "SME_TRAINING_ALGORITHM": "backprop",
    "SME_CA_RULE": "genetic_program",
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
    "SME_TUNING_ONLY": "1",
    "SME_EVALUATE_RESERVED_HOLDOUT": "0",
}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def terminal(name: str) -> dict:
    return {"op": name}


def random_tree(rng: random.Random, depth: int = 0, full: bool = False) -> dict:
    if depth >= MAX_DEPTH or (depth > 0 and not full and rng.random() < 0.30):
        return terminal(rng.choice(TERMINALS))
    family = rng.choice(("unary", "binary"))
    if family == "unary":
        op = rng.choice(UNARY)
        node = {"op": op, "arg": random_tree(rng, depth + 1, full)}
        if op == "scale":
            node["value"] = rng.choice(SCALES)
        return node
    return {
        "op": rng.choice(BINARY),
        "left": random_tree(rng, depth + 1, full),
        "right": random_tree(rng, depth + 1, full),
    }


def tree_signature(tree: dict) -> str:
    return json.dumps(tree, sort_keys=True, separators=(",", ":"))


def tree_depth(tree: dict) -> int:
    if tree["op"] in TERMINALS:
        return 0
    if tree["op"] in UNARY:
        return 1 + tree_depth(tree["arg"])
    return 1 + max(tree_depth(tree["left"]), tree_depth(tree["right"]))


def tree_nodes(tree: dict) -> int:
    if tree["op"] in TERMINALS:
        return 1
    if tree["op"] in UNARY:
        return 1 + tree_nodes(tree["arg"])
    return 1 + tree_nodes(tree["left"]) + tree_nodes(tree["right"])


def paths(tree: dict, prefix: tuple = ()) -> list[tuple]:
    found = [prefix]
    if tree["op"] in UNARY:
        found.extend(paths(tree["arg"], prefix + ("arg",)))
    elif tree["op"] in BINARY:
        found.extend(paths(tree["left"], prefix + ("left",)))
        found.extend(paths(tree["right"], prefix + ("right",)))
    return found


def subtree(tree: dict, path: tuple) -> dict:
    node = tree
    for part in path:
        node = node[part]
    return node


def replace_subtree(tree: dict, path: tuple, replacement: dict) -> dict:
    result = deepcopy(tree)
    if not path:
        return deepcopy(replacement)
    parent = result
    for part in path[:-1]:
        parent = parent[part]
    parent[path[-1]] = deepcopy(replacement)
    return result


def crossover(left: dict, right: dict, rng: random.Random) -> dict:
    for _ in range(100):
        child = replace_subtree(
            left, rng.choice(paths(left)), subtree(right, rng.choice(paths(right)))
        )
        if tree_depth(child) <= MAX_DEPTH:
            return child
    return deepcopy(left)


def mutate(tree: dict, rng: random.Random) -> dict:
    for _ in range(100):
        path = rng.choice(paths(tree))
        remaining = MAX_DEPTH - len(path)
        replacement = random_tree(rng, MAX_DEPTH - max(0, remaining), False)
        child = replace_subtree(tree, path, replacement)
        if tree_depth(child) <= MAX_DEPTH:
            return child
    return deepcopy(tree)


def seeded_programs() -> list[dict]:
    r, n, m, h = map(terminal, ("reaction", "neighbour_delta", "memory_delta", "state"))
    return [
        r,
        {"op": "add", "left": r, "right": n},
        {"op": "mean", "left": r, "right": n},
        {"op": "add", "left": r, "right": {"op": "scale", "value": 0.5, "arg": m}},
        {"op": "tanh", "arg": {"op": "add", "left": r, "right": n}},
        {"op": "sub", "left": r, "right": h},
        {"op": "sin", "arg": {"op": "add", "left": r, "right": m}},
    ]


def candidate(candidate_id: str, generation: int, program: dict,
              parents: list[str], operator: str) -> dict:
    return {
        "candidate_id": candidate_id,
        "generation": generation,
        "program": program,
        "program_nodes": tree_nodes(program),
        "program_depth": tree_depth(program),
        "parents": parents,
        "operator": operator,
    }


def initial_population(size: int, rng: random.Random) -> list[dict]:
    programs = seeded_programs()
    signatures = {tree_signature(program) for program in programs}
    while len(programs) < size:
        program = random_tree(rng, full=(len(programs) % 2 == 0))
        signature = tree_signature(program)
        if signature not in signatures:
            signatures.add(signature)
            programs.append(program)
    return [candidate(f"g00_c{index:02d}", 0, program, [],
                      "canonical" if index == 0 else "seeded_or_random")
            for index, program in enumerate(programs[:size])]


def tournament(ranked: list[dict], rng: random.Random) -> dict:
    return min(rng.sample(ranked, min(3, len(ranked))),
               key=lambda row: (row["selection_score"], row["screen_mean_rmse"]))


def breed(generation: int, size: int, ranked: list[dict], used: set[str],
          rng: random.Random, mutation_probability: float) -> list[dict]:
    population = []
    attempts = 0
    while len(population) < size:
        attempts += 1
        if attempts > size * 2000:
            raise RuntimeError("Unable to generate a unique GP population")
        left, right = tournament(ranked, rng), tournament(ranked, rng)
        program = crossover(left["program"], right["program"], rng)
        operator = "subtree_crossover"
        if rng.random() < mutation_probability:
            program = mutate(program, rng)
            operator += "_and_mutation"
        signature = tree_signature(program)
        if signature in used:
            continue
        used.add(signature)
        population.append(candidate(
            f"g{generation:02d}_c{len(population):02d}", generation, program,
            [left["candidate_id"], right["candidate_id"]], operator,
        ))
    return population


def result_dir(campaign: Path, stage: str, item: dict, fold: int, seed: int) -> Path:
    return campaign / stage / item["candidate_id"] / f"fold_{fold}_seed_{seed}"


def read_metrics(output: Path) -> dict | None:
    path = output / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def worker_task(campaign: Path, stage: str, item: dict, fold: int, seed: int,
                epochs: int, patience: int, smoke: bool) -> dict:
    output = result_dir(campaign, stage, item, fold, seed)
    output.mkdir(parents=True, exist_ok=True)
    metrics = read_metrics(output)
    if metrics:
        return {"stage": stage, "candidate_id": item["candidate_id"],
                "fold": fold, "seed": seed, "status": "complete", "resumed": True,
                "best_validation_ma_st_rae": metrics["best_validation_ma_st_rae"],
                "best_validation_rmse": metrics["best_validation_rmse"],
                "epochs_run": metrics["epochs_run"]}
    environment = os.environ.copy()
    environment.update(FIXED_TRAINING)
    environment.update({
        "SME_GP_PROGRAM_JSON": json.dumps(item["program"], separators=(",", ":")),
        "SME_RUN_NAME": output.relative_to(RESULTS).as_posix(),
        "SME_DEVICE": "cuda", "SME_CV_FOLD": str(fold), "SME_SEED": str(seed),
        "SME_MAX_EPOCHS": str(epochs), "SME_PATIENCE": str(patience),
        "SME_MIN_DELTA": "0.002",
        "SME_TUNING_FIT_MOLECULES": "128" if smoke else "999999",
        "SME_TUNING_VAL_MOLECULES": "64" if smoke else "999999",
    })
    atomic_json(output / "run_manifest.json", {
        "stage": stage, "candidate": item, "fold": fold, "seed": seed,
        "fixed_training": FIXED_TRAINING, "sealed_holdout_used_for_selection": False,
        "blind_labels_loaded": False,
    })
    started = time.time()
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with (output / "campaign_stdout.log").open("w", encoding="utf-8") as stdout, \
            (output / "campaign_stderr.log").open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(
            [str(PYTHON), str(WORKER), "train"], cwd=ROOT, env=environment,
            stdout=stdout, stderr=stderr, creationflags=flags, check=False,
        )
    metrics = read_metrics(output)
    result = {"stage": stage, "candidate_id": item["candidate_id"],
              "fold": fold, "seed": seed,
              "status": "complete" if completed.returncode == 0 and metrics else "failed",
              "return_code": completed.returncode, "wall_seconds": time.time() - started,
              "resumed": False}
    if metrics:
        result.update({"best_validation_ma_st_rae": metrics["best_validation_ma_st_rae"],
                       "best_validation_rmse": metrics["best_validation_rmse"],
                       "epochs_run": metrics["epochs_run"]})
    return result


def run_tasks(campaign: Path, stage: str, tasks: list[tuple], progress: dict,
              workers: int, epochs: int, patience: int, smoke: bool) -> list[dict]:
    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(worker_task, campaign, stage, item, fold, seed,
                                   epochs, patience, smoke): (item, fold, seed)
                   for item, fold, seed in tasks}
        for future in as_completed(futures):
            item, fold, seed = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"stage": stage, "candidate_id": item["candidate_id"],
                          "fold": fold, "seed": seed, "status": "failed",
                          "error": repr(exc)}
            results.append(result)
            with LOCK:
                progress["completed_runs"] += 1
                progress["last_result"] = result
                progress["updated_at_unix"] = time.time()
                atomic_json(campaign / "progress.json", progress)
            print(json.dumps(result, allow_nan=False), flush=True)
    return results


def rank(population: list[dict], results: list[dict], expected: int) -> list[dict]:
    ranked = []
    for item in population:
        rows = [row for row in results if row["candidate_id"] == item["candidate_id"]
                and row["status"] == "complete"]
        if len(rows) != expected:
            continue
        mean_score = sum(row["best_validation_ma_st_rae"] for row in rows) / len(rows)
        ranked.append({**item,
                       "screen_mean_ma_st_rae": mean_score,
                       "screen_mean_rmse": sum(row["best_validation_rmse"] for row in rows) / len(rows),
                       "selection_score": mean_score + 0.0005 * item["program_nodes"],
                       "screen_runs": rows})
    ranked.sort(key=lambda row: (row["selection_score"], row["screen_mean_rmse"]))
    return ranked


def write_csv(path: Path, rows: list[dict], prefix: str) -> None:
    fields = ["candidate_id", "generation", "program", "program_nodes", "program_depth",
              f"{prefix}_mean_ma_st_rae", f"{prefix}_mean_rmse"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({"candidate_id": row["candidate_id"],
                             "generation": row["generation"],
                             "program": tree_signature(row["program"]),
                             "program_nodes": row["program_nodes"],
                             "program_depth": row["program_depth"],
                             f"{prefix}_mean_ma_st_rae": row[f"{prefix}_mean_ma_st_rae"],
                             f"{prefix}_mean_rmse": row[f"{prefix}_mean_rmse"]})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-name", default=DEFAULT_CAMPAIGN)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    campaign = RESULTS / args.campaign_name
    campaign.mkdir(parents=True, exist_ok=True)
    population_size = 2 if args.smoke else int(os.environ.get("SME_GP_POPULATION", "24"))
    generations = 1 if args.smoke else int(os.environ.get("SME_GP_GENERATIONS", "4"))
    confirm_count = 1 if args.smoke else int(os.environ.get("SME_GP_CONFIRM", "5"))
    workers = 1 if args.smoke else int(os.environ.get("SME_GP_WORKERS", "5"))
    screen_folds = (0,) if args.smoke else (0, 1)
    confirmation_folds = (0,) if args.smoke else tuple(range(5))
    confirmation_seeds = (4211,) if args.smoke else (1701, 4211)
    screen_epochs = 1 if args.smoke else int(os.environ.get("SME_GP_SCREEN_EPOCHS", "35"))
    confirmation_epochs = 1 if args.smoke else int(os.environ.get("SME_GP_CONFIRM_EPOCHS", "80"))
    mutation_probability = float(os.environ.get("SME_GP_MUTATION", "0.35"))
    rng = random.Random(260914)
    total = population_size * generations * len(screen_folds) + confirm_count * len(confirmation_folds) * len(confirmation_seeds)
    atomic_json(campaign / "genetic_programming_design.json", {
        "campaign": args.campaign_name, "scope": "CYP3A4",
        "search": "genetic_programming_of_cellular_update_equations",
        "population_size": population_size, "genetic_generations": generations,
        "mutation_probability": mutation_probability, "selection": "tournament_3",
        "crossover": "subtree", "maximum_tree_depth": MAX_DEPTH,
        "parsimony_penalty_per_node": 0.0005,
        "terminals": list(TERMINALS), "unary_primitives": list(UNARY),
        "binary_primitives": list(BINARY), "scales": list(SCALES),
        "screen_folds": list(screen_folds), "confirmation_folds": list(confirmation_folds),
        "confirmation_seeds": list(confirmation_seeds), "confirmation_candidates": confirm_count,
        "parallel_cuda_workers": workers, "screen_epochs": screen_epochs,
        "confirmation_epochs": confirmation_epochs, "fixed_training": FIXED_TRAINING,
        "total_planned_runs": total, "sealed_holdout_used_for_selection": False,
        "blind_labels_loaded": False,
    })
    progress = {"campaign": args.campaign_name, "stage": "genetic_program_screen",
                "genetic_generation": 0, "completed_runs": 0,
                "total_planned_runs": total, "started_at_unix": time.time(),
                "sealed_holdout_used_for_selection": False, "blind_labels_loaded": False}
    atomic_json(campaign / "progress.json", progress)
    population = initial_population(population_size, rng)
    used = {tree_signature(item["program"]) for item in population}
    global_ranked = []
    for generation in range(generations):
        progress["genetic_generation"] = generation
        progress["current_candidate_ids"] = [item["candidate_id"] for item in population]
        atomic_json(campaign / "progress.json", progress)
        atomic_json(campaign / f"generation_{generation:02d}_population.json", population)
        tasks = [(item, fold, 1701) for item in population for fold in screen_folds]
        results = run_tasks(campaign, "screen", tasks, progress, workers,
                            screen_epochs, min(8, screen_epochs), args.smoke)
        ranked = rank(population, results, len(screen_folds))
        if len(ranked) < max(1, population_size // 2):
            raise RuntimeError("Too few successful programs to continue")
        atomic_json(campaign / f"generation_{generation:02d}_ranking.json", ranked)
        global_ranked.extend(ranked)
        global_ranked.sort(key=lambda row: (row["selection_score"], row["screen_mean_rmse"]))
        progress["generation_leader"] = ranked[0]
        progress["global_leader"] = global_ranked[0]
        atomic_json(campaign / "progress.json", progress)
        if generation + 1 < generations:
            population = breed(generation + 1, population_size, ranked, used, rng,
                               mutation_probability)
    atomic_json(campaign / "screen_ranking.json", global_ranked)
    write_csv(campaign / "screen_ranking.csv", global_ranked, "screen")
    canonical = next(row for row in global_ranked
                     if tree_signature(row["program"]) == tree_signature(terminal("reaction")))
    finalists = global_ranked[:confirm_count]
    if canonical["candidate_id"] not in {row["candidate_id"] for row in finalists}:
        finalists = [*global_ranked[:confirm_count - 1], canonical]
    progress["stage"] = "confirmation"
    progress["finalist_candidate_ids"] = [item["candidate_id"] for item in finalists]
    atomic_json(campaign / "progress.json", progress)
    confirmation_tasks = [(item, fold, seed) for item in finalists
                          for fold in confirmation_folds for seed in confirmation_seeds]
    confirmation_results = run_tasks(campaign, "confirmation", confirmation_tasks,
                                     progress, workers, confirmation_epochs,
                                     min(20, confirmation_epochs), args.smoke)
    expected = len(confirmation_folds) * len(confirmation_seeds)
    confirmed = []
    for item in finalists:
        rows = [row for row in confirmation_results
                if row["candidate_id"] == item["candidate_id"] and row["status"] == "complete"]
        if len(rows) != expected:
            continue
        confirmed.append({**{key: item[key] for key in ("candidate_id", "generation", "program",
                                                         "program_nodes", "program_depth", "parents", "operator")},
                          "confirmation_mean_ma_st_rae": sum(row["best_validation_ma_st_rae"] for row in rows) / len(rows),
                          "confirmation_mean_rmse": sum(row["best_validation_rmse"] for row in rows) / len(rows),
                          "confirmation_runs": rows})
    confirmed.sort(key=lambda row: (row["confirmation_mean_ma_st_rae"],
                                    row["confirmation_mean_rmse"]))
    atomic_json(campaign / "confirmation_ranking.json", confirmed)
    write_csv(campaign / "confirmation_ranking.csv", confirmed, "confirmation")
    progress["stage"] = "complete"
    progress["completed_at_unix"] = time.time()
    progress["winner"] = confirmed[0] if confirmed else None
    atomic_json(campaign / "progress.json", progress)
    print(json.dumps({"campaign_complete": True, "winner": progress["winner"]}), flush=True)


if __name__ == "__main__":
    main()
