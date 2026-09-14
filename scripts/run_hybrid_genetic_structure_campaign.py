#!/usr/bin/env python3
"""Hybrid genetic structure search for the CYP3A4 encoded Graph-CA.

A genetic algorithm evolves discrete Graph-CA structures. Every candidate is
trained with the same bounded backpropagation budget and uses the exact
differentiable ridge readout implemented by run_graph_ca_visual_prototype.py.
"""

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
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DEFAULT_CAMPAIGN = "production_hybrid_genetic_structure_search_cyp3a4_v1"
PYTHON = Path(r"C:\Users\Anthony\anaconda3\envs\strange-matter-gpu\python.exe")
WORKER = ROOT / "scripts" / "run_graph_ca_visual_prototype.py"
PROGRESS_LOCK = threading.Lock()


FIXED_TRAINING = {
    "SME_ACTIVE_CYP": "CYP3A4",
    "SME_SPECIALIST_OBJECTIVE": "endpoint_only",
    "SME_TRAINING_ALGORITHM": "backprop",
    "SME_CA_LR": "0.003",
    "SME_BATCH_MOLECULES": "64",
    "SME_RIDGE_MODE": "shared",
    "SME_CV_FOLDS": "5",
    "SME_CV_SPLIT_SEED": "260822",
    "SME_TUNING_ONLY": "1",
    "SME_EVALUATE_RESERVED_HOLDOUT": "0",
}

# Continuous numerical settings follow the established endpoint-specific rule
# profiles. The GA changes structure, not the numerical optimizer settings.
RULE_PROFILES = {
    "damped_symplectic": {
        "SME_RIDGE": "0.1", "SME_CA_L2": "0.000001",
        "SME_GRAD_CLIP": "1.0", "SME_UPDATE_SCALE": "0.25",
        "SME_INIT_SCALE": "1.5", "SME_INITIAL_NOISE": "0.0",
        "SME_SUPPORT_FRACTION": "0.6", "SME_BOND_TEMPERATURE": "1.0",
        "SME_DYN_A": "0.5", "SME_DYN_B": "0.2",
        "SME_DYN_C": "0.2", "SME_DYN_D": "0.15",
    },
    "fitzhugh_nagumo": {
        "SME_RIDGE": "0.01", "SME_CA_L2": "0.0001",
        "SME_GRAD_CLIP": "0.5", "SME_UPDATE_SCALE": "0.08",
        "SME_INIT_SCALE": "0.5", "SME_INITIAL_NOISE": "0.005",
        "SME_SUPPORT_FRACTION": "0.75", "SME_BOND_TEMPERATURE": "1.0",
        "SME_DYN_A": "0.5", "SME_DYN_B": "0.2",
        "SME_DYN_C": "0.8", "SME_DYN_D": "0.15",
    },
    "delayed_memory": {
        "SME_RIDGE": "0.1", "SME_CA_L2": "0.000001",
        "SME_GRAD_CLIP": "2.0", "SME_UPDATE_SCALE": "0.08",
        "SME_INIT_SCALE": "0.5", "SME_INITIAL_NOISE": "0.0",
        "SME_SUPPORT_FRACTION": "0.6", "SME_BOND_TEMPERATURE": "2.0",
        "SME_DYN_A": "0.2", "SME_DYN_B": "0.2",
        "SME_DYN_C": "0.8", "SME_DYN_D": "0.15",
    },
}

GENE_SPACE = {
    "rule": tuple(RULE_PROFILES),
    "generations": (16, 32, 64, 128),
    "hidden_channels": (8, 16, 24),
    "atom_feature_profile": (
        "periodic_electronic", "electronic_local",
        "periodic_valence", "comprehensive",
    ),
    "trajectory_pooling": ("legacy", "multiscale", "temporal_attention"),
    "degree_normalization_power": (0.5, 1.0),
    "chemical_feature_gating": (False, True),
    "initial_state_anchor": (False, True),
    "dynamic_observables": (False, True),
    "multiscale_transition_energy": (False, True),
    "channel_adaptive_timescale": (False, True),
    "multilag_recurrence_signature": (False, True),
    "temporal_extrema_signature": (False, True),
    "directional_flux_signature": (False, True),
}

CANONICAL_GENOME = {
    "rule": "damped_symplectic",
    "generations": 32,
    "hidden_channels": 16,
    "atom_feature_profile": "periodic_electronic",
    "trajectory_pooling": "multiscale",
    "degree_normalization_power": 1.0,
    "chemical_feature_gating": False,
    "initial_state_anchor": False,
    "dynamic_observables": False,
    "multiscale_transition_energy": False,
    "channel_adaptive_timescale": False,
    "multilag_recurrence_signature": False,
    "temporal_extrema_signature": False,
    "directional_flux_signature": False,
}

SEEDED_VARIANTS = (
    {"initial_state_anchor": True},
    {"dynamic_observables": True},
    {"multiscale_transition_energy": True},
    {"channel_adaptive_timescale": True},
    {"multilag_recurrence_signature": True},
    {"temporal_extrema_signature": True},
    {"directional_flux_signature": True},
    {"chemical_feature_gating": True},
    {"rule": "fitzhugh_nagumo", "generations": 128, "hidden_channels": 8},
    {"rule": "delayed_memory", "generations": 64, "hidden_channels": 16},
)

BOOLEAN_OBSERVABLE_GENES = (
    "dynamic_observables", "multiscale_transition_energy",
    "multilag_recurrence_signature", "temporal_extrema_signature",
    "directional_flux_signature",
)


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def genome_signature(genome: dict) -> str:
    return json.dumps(genome, sort_keys=True, separators=(",", ":"))


def normalize_genome(genome: dict, rng: random.Random) -> dict:
    normalized = {key: genome[key] for key in GENE_SPACE}
    active = [key for key in BOOLEAN_OBSERVABLE_GENES if normalized[key]]
    while len(active) > 3:
        key = rng.choice(active)
        normalized[key] = False
        active.remove(key)
    return normalized


def random_genome(rng: random.Random) -> dict:
    return normalize_genome(
        {key: rng.choice(values) for key, values in GENE_SPACE.items()}, rng
    )


def candidate(candidate_id: str, generation: int, genome: dict,
              parents: list[str], operator: str) -> dict:
    return {
        "candidate_id": candidate_id,
        "generation": generation,
        "genome": genome,
        "parents": parents,
        "operator": operator,
    }


def initial_population(size: int, rng: random.Random) -> list[dict]:
    genomes = [dict(CANONICAL_GENOME)]
    for changes in SEEDED_VARIANTS:
        genome = dict(CANONICAL_GENOME)
        genome.update(changes)
        genomes.append(normalize_genome(genome, rng))
    signatures = set()
    unique = []
    for genome in genomes:
        signature = genome_signature(genome)
        if signature not in signatures:
            signatures.add(signature)
            unique.append(genome)
    while len(unique) < size:
        genome = random_genome(rng)
        signature = genome_signature(genome)
        if signature not in signatures:
            signatures.add(signature)
            unique.append(genome)
    return [
        candidate(f"g00_c{index:02d}", 0, genome, [],
                  "canonical" if index == 0 else "seeded_or_random")
        for index, genome in enumerate(unique[:size])
    ]


def tournament(ranked: list[dict], rng: random.Random, size: int = 3) -> dict:
    contestants = rng.sample(ranked, min(size, len(ranked)))
    return min(contestants, key=lambda row: (
        row["screen_mean_ma_st_rae"], row["screen_mean_rmse"]
    ))


def crossover(left: dict, right: dict, rng: random.Random) -> dict:
    return {
        key: (left[key] if rng.random() < 0.5 else right[key])
        for key in GENE_SPACE
    }


def mutate(genome: dict, rng: random.Random, probability: float) -> dict:
    changed = dict(genome)
    mutated = False
    for key, levels in GENE_SPACE.items():
        if rng.random() < probability:
            alternatives = [value for value in levels if value != changed[key]]
            changed[key] = rng.choice(alternatives)
            mutated = True
    if not mutated:
        key = rng.choice(tuple(GENE_SPACE))
        alternatives = [value for value in GENE_SPACE[key] if value != changed[key]]
        changed[key] = rng.choice(alternatives)
    return normalize_genome(changed, rng)


def breed_population(generation: int, size: int, ranked: list[dict],
                     used_signatures: set[str], rng: random.Random,
                     mutation_probability: float) -> list[dict]:
    offspring = []
    attempts = 0
    while len(offspring) < size:
        attempts += 1
        if attempts > size * 1000:
            raise RuntimeError("Unable to generate a unique genetic population")
        left = tournament(ranked, rng)
        right = tournament(ranked, rng)
        genome = mutate(
            crossover(left["genome"], right["genome"], rng),
            rng, mutation_probability,
        )
        signature = genome_signature(genome)
        if signature in used_signatures:
            continue
        used_signatures.add(signature)
        index = len(offspring)
        offspring.append(candidate(
            f"g{generation:02d}_c{index:02d}", generation, genome,
            [left["candidate_id"], right["candidate_id"]],
            "uniform_crossover_gene_mutation",
        ))
    return offspring


def genome_environment(genome: dict) -> dict[str, str]:
    environment = dict(RULE_PROFILES[genome["rule"]])
    environment.update({
        "SME_CA_RULE": str(genome["rule"]),
        "SME_GENERATIONS": str(genome["generations"]),
        "SME_HIDDEN_CHANNELS": str(genome["hidden_channels"]),
        "SME_ATOM_FEATURE_PROFILE": str(genome["atom_feature_profile"]),
        "SME_TRAJECTORY_POOLING": str(genome["trajectory_pooling"]),
        "SME_DEGREE_NORMALIZATION_POWER": str(genome["degree_normalization_power"]),
        "SME_CHEMICAL_FEATURE_GATING": str(int(genome["chemical_feature_gating"])),
        "SME_INITIAL_STATE_ANCHOR": str(int(genome["initial_state_anchor"])),
        "SME_DYNAMIC_OBSERVABLES": str(int(genome["dynamic_observables"])),
        "SME_MULTISCALE_TRANSITION_ENERGY": str(int(genome["multiscale_transition_energy"])),
        "SME_CHANNEL_ADAPTIVE_TIMESCALE": str(int(genome["channel_adaptive_timescale"])),
        "SME_MULTILAG_RECURRENCE_SIGNATURE": str(int(genome["multilag_recurrence_signature"])),
        "SME_TEMPORAL_EXTREMA_SIGNATURE": str(int(genome["temporal_extrema_signature"])),
        "SME_DIRECTIONAL_FLUX_SIGNATURE": str(int(genome["directional_flux_signature"])),
    })
    return environment


def result_directory(campaign: Path, stage: str, candidate_id: str,
                     fold: int, seed: int) -> Path:
    return campaign / stage / candidate_id / f"fold_{fold}_seed_{seed}"


def read_metrics(output: Path):
    path = output / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def worker_task(campaign: Path, stage: str, item: dict, fold: int, seed: int,
                epochs: int, patience: int, min_delta: float,
                smoke: bool) -> dict:
    output = result_directory(campaign, stage, item["candidate_id"], fold, seed)
    output.mkdir(parents=True, exist_ok=True)
    metrics = read_metrics(output)
    if metrics is not None:
        return {
            "stage": stage, "candidate_id": item["candidate_id"],
            "fold": fold, "seed": seed, "status": "complete", "resumed": True,
            "best_validation_ma_st_rae": metrics["best_validation_ma_st_rae"],
            "best_validation_rmse": metrics["best_validation_rmse"],
            "epochs_run": metrics["epochs_run"],
        }

    environment = os.environ.copy()
    environment.update(FIXED_TRAINING)
    environment.update(genome_environment(item["genome"]))
    environment.update({
        "SME_RUN_NAME": output.relative_to(RESULTS).as_posix(),
        "SME_DEVICE": "cuda",
        "SME_CV_FOLD": str(fold),
        "SME_SEED": str(seed),
        "SME_MAX_EPOCHS": str(epochs),
        "SME_PATIENCE": str(patience),
        "SME_MIN_DELTA": str(min_delta),
        "SME_TUNING_FIT_MOLECULES": "128" if smoke else "999999",
        "SME_TUNING_VAL_MOLECULES": "64" if smoke else "999999",
    })
    atomic_json(output / "run_manifest.json", {
        "stage": stage, "candidate": item, "fold": fold, "seed": seed,
        "training": FIXED_TRAINING, "rule_profile": RULE_PROFILES[item["genome"]["rule"]],
        "epochs": epochs, "patience": patience, "min_delta": min_delta,
        "blind_labels_loaded": False, "sealed_holdout_used_for_selection": False,
    })
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    started = time.time()
    with (output / "campaign_stdout.log").open("w", encoding="utf-8") as stdout, \
            (output / "campaign_stderr.log").open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(
            [str(PYTHON), str(WORKER), "train"], cwd=ROOT, env=environment,
            stdout=stdout, stderr=stderr, creationflags=creation_flags, check=False,
        )
    metrics = read_metrics(output)
    result = {
        "stage": stage, "candidate_id": item["candidate_id"],
        "fold": fold, "seed": seed,
        "status": "complete" if completed.returncode == 0 and metrics else "failed",
        "return_code": completed.returncode, "wall_seconds": time.time() - started,
        "resumed": False,
    }
    if metrics:
        result.update({
            "best_validation_ma_st_rae": metrics["best_validation_ma_st_rae"],
            "best_validation_rmse": metrics["best_validation_rmse"],
            "epochs_run": metrics["epochs_run"],
        })
    return result


def run_tasks(campaign: Path, stage: str, tasks: list[tuple], progress: dict,
              workers: int, epochs: int, patience: int, min_delta: float,
              smoke: bool) -> list[dict]:
    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                worker_task, campaign, stage, item, fold, seed,
                epochs, patience, min_delta, smoke,
            ): (item["candidate_id"], fold, seed)
            for item, fold, seed in tasks
        }
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:
                candidate_id, fold, seed = futures[future]
                result = {
                    "stage": stage, "candidate_id": candidate_id,
                    "fold": fold, "seed": seed, "status": "failed",
                    "error": repr(exc),
                }
            results.append(result)
            with PROGRESS_LOCK:
                progress["completed_runs"] += 1
                progress["last_result"] = result
                progress["updated_at_unix"] = time.time()
                atomic_json(campaign / "progress.json", progress)
            print(json.dumps(result, allow_nan=False), flush=True)
    return results


def rank_candidates(population: list[dict], results: list[dict],
                    expected_runs: int) -> list[dict]:
    ranked = []
    for item in population:
        observations = [
            row for row in results
            if row["candidate_id"] == item["candidate_id"]
            and row["status"] == "complete"
        ]
        if len(observations) != expected_runs:
            continue
        ranked.append({
            **item,
            "screen_mean_ma_st_rae": sum(
                row["best_validation_ma_st_rae"] for row in observations
            ) / len(observations),
            "screen_mean_rmse": sum(
                row["best_validation_rmse"] for row in observations
            ) / len(observations),
            "screen_runs": observations,
        })
    ranked.sort(key=lambda row: (
        row["screen_mean_ma_st_rae"], row["screen_mean_rmse"]
    ))
    return ranked


def write_flat_csv(path: Path, rows: list[dict], score_prefix: str) -> None:
    fields = ["candidate_id", "generation", *GENE_SPACE,
              f"{score_prefix}_mean_ma_st_rae", f"{score_prefix}_mean_rmse"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flattened = {
                "candidate_id": row["candidate_id"],
                "generation": row["generation"],
                **row["genome"],
                f"{score_prefix}_mean_ma_st_rae": row[f"{score_prefix}_mean_ma_st_rae"],
                f"{score_prefix}_mean_rmse": row[f"{score_prefix}_mean_rmse"],
            }
            writer.writerow(flattened)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-name", default=DEFAULT_CAMPAIGN)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not PYTHON.exists():
        raise FileNotFoundError(PYTHON)
    campaign = RESULTS / args.campaign_name
    campaign.mkdir(parents=True, exist_ok=True)

    population_size = 2 if args.smoke else int(os.environ.get("SME_GA_POPULATION", "24"))
    generations = 1 if args.smoke else int(os.environ.get("SME_GA_GENERATIONS", "4"))
    confirm_count = 1 if args.smoke else int(os.environ.get("SME_GA_CONFIRM", "5"))
    workers = 1 if args.smoke else int(os.environ.get("SME_GA_WORKERS", "5"))
    screen_folds = (0,) if args.smoke else (0, 1)
    confirmation_folds = (0,) if args.smoke else tuple(range(5))
    confirmation_seeds = (4211,) if args.smoke else (1701, 4211)
    screen_epochs = 1 if args.smoke else int(os.environ.get("SME_GA_SCREEN_EPOCHS", "35"))
    confirmation_epochs = 1 if args.smoke else int(os.environ.get("SME_GA_CONFIRM_EPOCHS", "80"))
    mutation_probability = float(os.environ.get("SME_GA_MUTATION", "0.25"))
    rng = random.Random(260914)

    total_planned = (
        population_size * generations * len(screen_folds)
        + confirm_count * len(confirmation_folds) * len(confirmation_seeds)
    )
    design = {
        "campaign": args.campaign_name,
        "scope": "CYP3A4",
        "search": "hybrid_genetic_structure_search",
        "population_size": population_size,
        "genetic_generations": generations,
        "mutation_probability_per_gene": mutation_probability,
        "selection": "tournament_3",
        "crossover": "uniform",
        "screen_folds": list(screen_folds),
        "confirmation_folds": list(confirmation_folds),
        "confirmation_seeds": list(confirmation_seeds),
        "confirmation_candidates": confirm_count,
        "parallel_cuda_workers": workers,
        "screen_epochs": screen_epochs,
        "confirmation_epochs": confirmation_epochs,
        "gene_space": {key: list(values) for key, values in GENE_SPACE.items()},
        "canonical_genome": CANONICAL_GENOME,
        "fixed_training": FIXED_TRAINING,
        "rule_profiles": RULE_PROFILES,
        "sealed_holdout_used_for_selection": False,
        "blind_labels_loaded": False,
        "total_planned_runs": total_planned,
    }
    atomic_json(campaign / "genetic_design.json", design)
    progress = {
        "campaign": args.campaign_name, "stage": "genetic_screen",
        "genetic_generation": 0, "completed_runs": 0,
        "total_planned_runs": total_planned,
        "blind_labels_loaded": False,
        "sealed_holdout_used_for_selection": False,
        "started_at_unix": time.time(),
    }
    atomic_json(campaign / "progress.json", progress)

    population = initial_population(population_size, rng)
    used_signatures = {genome_signature(item["genome"]) for item in population}
    global_ranked = []
    for generation in range(generations):
        progress["genetic_generation"] = generation
        progress["current_candidate_ids"] = [item["candidate_id"] for item in population]
        atomic_json(campaign / "progress.json", progress)
        atomic_json(campaign / f"generation_{generation:02d}_population.json", population)
        tasks = [
            (item, fold, 1701) for item in population for fold in screen_folds
        ]
        results = run_tasks(
            campaign, "screen", tasks, progress, workers,
            screen_epochs, min(8, screen_epochs), 0.002, args.smoke,
        )
        ranked = rank_candidates(population, results, len(screen_folds))
        if len(ranked) < max(1, population_size // 2):
            raise RuntimeError("Too few complete candidates to continue genetic search")
        atomic_json(campaign / f"generation_{generation:02d}_ranking.json", ranked)
        global_ranked.extend(ranked)
        global_ranked.sort(key=lambda row: (
            row["screen_mean_ma_st_rae"], row["screen_mean_rmse"]
        ))
        progress["generation_leader"] = ranked[0]
        progress["global_leader"] = global_ranked[0]
        atomic_json(campaign / "progress.json", progress)
        if generation + 1 < generations:
            population = breed_population(
                generation + 1, population_size, ranked, used_signatures,
                rng, mutation_probability,
            )

    atomic_json(campaign / "screen_ranking.json", global_ranked)
    write_flat_csv(campaign / "screen_ranking.csv", global_ranked, "screen")

    canonical = next(
        row for row in global_ranked
        if genome_signature(row["genome"]) == genome_signature(CANONICAL_GENOME)
    )
    finalists = global_ranked[:confirm_count]
    if canonical["candidate_id"] not in {row["candidate_id"] for row in finalists}:
        finalists = [*global_ranked[:max(0, confirm_count - 1)], canonical]
    progress["stage"] = "confirmation"
    progress["finalist_candidate_ids"] = [row["candidate_id"] for row in finalists]
    atomic_json(campaign / "progress.json", progress)
    confirmation_tasks = [
        (item, fold, seed) for item in finalists
        for fold in confirmation_folds for seed in confirmation_seeds
    ]
    confirmation_results = run_tasks(
        campaign, "confirmation", confirmation_tasks, progress, workers,
        confirmation_epochs, min(20, confirmation_epochs), 0.002, args.smoke,
    )
    expected_confirmation = len(confirmation_folds) * len(confirmation_seeds)
    confirmed = []
    for item in finalists:
        observations = [
            row for row in confirmation_results
            if row["candidate_id"] == item["candidate_id"]
            and row["status"] == "complete"
        ]
        if len(observations) != expected_confirmation:
            continue
        confirmed.append({
            "candidate_id": item["candidate_id"],
            "generation": item["generation"],
            "genome": item["genome"],
            "parents": item["parents"],
            "operator": item["operator"],
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
    atomic_json(campaign / "confirmation_ranking.json", confirmed)
    write_flat_csv(campaign / "confirmation_ranking.csv", confirmed, "confirmation")
    progress["stage"] = "complete"
    progress["completed_at_unix"] = time.time()
    progress["winner"] = confirmed[0] if confirmed else None
    atomic_json(campaign / "progress.json", progress)
    print(json.dumps({"campaign_complete": True, "winner": progress["winner"]}), flush=True)


if __name__ == "__main__":
    main()
