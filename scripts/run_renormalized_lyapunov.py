#!/usr/bin/env python3
"""Benettin-style renormalized Lyapunov analysis for frozen Graph-CA states."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd


def _circular_difference(torch, a, b):
    phase_delta = math.pi * (a - b)
    return torch.atan2(torch.sin(phase_delta), torch.cos(phase_delta)) / math.pi


def _wrap(torch, h):
    phase = math.pi * h
    return torch.atan2(torch.sin(phase), torch.cos(phase)) / math.pi


def _one_run(*, model, rec, cyp, torch, device, burn_in, measured_generations,
             interval, epsilon, seed):
    with torch.no_grad():
        reference = model.initial_node_state(rec)
        for _ in range(burn_in):
            reference = model.kuramoto_step(rec, cyp, reference)
        generator = torch.Generator(device=device).manual_seed(seed)
        direction = torch.randn(reference.shape, generator=generator, device=device)
        direction /= torch.linalg.vector_norm(direction).clamp_min(1e-30)
        companion = _wrap(torch, reference + epsilon * direction)
        local_exponents, pre_rescale_norms = [], []
        completed = 0
        while completed < measured_generations:
            block = min(interval, measured_generations - completed)
            for _ in range(block):
                reference = model.kuramoto_step(rec, cyp, reference)
                companion = model.kuramoto_step(rec, cyp, companion)
            difference = _circular_difference(torch, companion, reference)
            norm = torch.linalg.vector_norm(difference).clamp_min(1e-30)
            local_exponents.append(float(torch.log(norm / epsilon) / block))
            pre_rescale_norms.append(float(norm))
            direction = difference / norm
            companion = _wrap(torch, reference + epsilon * direction)
            completed += block
    local = np.asarray(local_exponents, dtype=np.float64)
    cumulative = np.cumsum(local * interval) / (np.arange(len(local)) + 1) / interval
    return local.astype(np.float32), cumulative.astype(np.float32), np.asarray(pre_rescale_norms, dtype=np.float32)


def run_renormalized_campaign(*, model, data, device, torch_module,
                              selected_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if os.environ.get("SME_RENORMALIZED_DTYPE", "float32") == "float64":
        model.double()
    selected = pd.read_csv(selected_path)
    requested = os.environ.get("SME_RENORMALIZED_CYPS", "CYP1A2,CYP2C9").split(",")
    selected = selected[selected.cyp_target.isin(requested)].copy()
    burn_in = int(os.environ.get("SME_RENORMALIZED_BURN_IN", "1000"))
    measured = int(os.environ.get("SME_RENORMALIZED_GENERATIONS", "4000"))
    interval = int(os.environ.get("SME_RENORMALIZED_INTERVAL", "10"))
    repeats = int(os.environ.get("SME_RENORMALIZED_REPEATS", "8"))
    epsilons = [float(value) for value in
                os.environ.get("SME_RENORMALIZED_EPSILONS", "1e-4,1e-5,1e-6").split(",")]
    started = time.perf_counter()
    rows = []
    for case_number, row in enumerate(selected.itertuples(), start=1):
        rec = data["train"][int(row.training_index)]
        if rec["name"] != row.molecule_id:
            raise RuntimeError("Graph cache and selected candidate ordering differ")
        for epsilon in epsilons:
            case_local, case_cumulative, case_norms = [], [], []
            for repeat in range(repeats):
                local, cumulative, norms = _one_run(
                    model=model, rec=rec, cyp=int(row.cyp_index), torch=torch_module,
                    device=device, burn_in=burn_in, measured_generations=measured,
                    interval=interval, epsilon=epsilon,
                    seed=88421 + 1009 * case_number + 7919 * repeat,
                )
                case_local.append(local); case_cumulative.append(cumulative); case_norms.append(norms)
                rows.append({
                    "molecule_id": row.molecule_id, "cyp_target": row.cyp_target,
                    "training_index": int(row.training_index), "epsilon": epsilon,
                    "repeat": repeat + 1, "renormalized_lyapunov": float(cumulative[-1]),
                    "positive_block_fraction": float(np.mean(local > 0)),
                    "local_exponent_mean": float(local.mean()),
                    "local_exponent_std": float(local.std()),
                })
                print(json.dumps({"case": case_number, "molecule_id": row.molecule_id,
                                  "epsilon": epsilon, "repeat": repeat + 1,
                                  "lyapunov": float(cumulative[-1])}), flush=True)
            stem = f"{case_number:02d}_{row.molecule_id}_{row.cyp_target}_eps_{epsilon:.0e}"
            np.savez_compressed(
                output_dir / f"{stem}.npz",
                local_exponents=np.stack(case_local),
                cumulative_exponents=np.stack(case_cumulative),
                pre_rescale_norms=np.stack(case_norms),
                interval=np.asarray(interval), epsilon=np.asarray(epsilon),
                burn_in=np.asarray(burn_in), measured_generations=np.asarray(measured),
                molecule_id=np.asarray(row.molecule_id), cyp_target=np.asarray(row.cyp_target),
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "renormalized_lyapunov_runs.csv", index=False)
    summary = frame.groupby(["molecule_id", "cyp_target", "epsilon"]).agg(
        repeats=("repeat", "size"), mean_lyapunov=("renormalized_lyapunov", "mean"),
        std_lyapunov=("renormalized_lyapunov", "std"),
        minimum_lyapunov=("renormalized_lyapunov", "min"),
        maximum_lyapunov=("renormalized_lyapunov", "max"),
        positive_fraction=("renormalized_lyapunov", lambda values: float(np.mean(values > 0))),
        mean_positive_block_fraction=("positive_block_fraction", "mean"),
    ).reset_index()
    summary.to_csv(output_dir / "renormalized_lyapunov_summary.csv", index=False)
    metadata = {
        "method": "Benettin repeated renormalization with circular phase distance",
        "device": str(device), "cases": len(selected), "burn_in": burn_in,
        "measured_generations": measured, "renormalization_interval": interval,
        "epsilons": epsilons, "repeats_per_epsilon": repeats,
        "total_runs": len(frame), "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


def _spectrum_run(*, model, rec, cyp, torch, device, burn_in,
                  measured_generations, interval, epsilon, vectors, seed):
    with torch.no_grad():
        reference = model.initial_node_state(rec)
        for _ in range(burn_in):
            reference = model.kuramoto_step(rec, cyp, reference)
        dimension = reference.numel()
        if vectors >= dimension:
            raise ValueError("Lyapunov vector count must be smaller than state dimension")
        generator = torch.Generator(device=device).manual_seed(seed)
        random_matrix = torch.randn(
            (dimension, vectors), generator=generator, device=device,
            dtype=reference.dtype,
        )
        basis, _ = torch.linalg.qr(random_matrix, mode="reduced")
        companions = [
            _wrap(torch, reference + epsilon * basis[:, index].reshape_as(reference))
            for index in range(vectors)
        ]
        local_spectra = []
        completed = 0
        while completed < measured_generations:
            block = min(interval, measured_generations - completed)
            for _ in range(block):
                reference = model.kuramoto_step(rec, cyp, reference)
                companions = [
                    model.kuramoto_step(rec, cyp, state) for state in companions
                ]
            differences = torch.stack([
                _circular_difference(torch, state, reference).reshape(-1)
                for state in companions
            ], dim=1)
            basis, upper = torch.linalg.qr(differences, mode="reduced")
            expansion = torch.abs(torch.diagonal(upper)).clamp_min(1e-300)
            local_spectra.append(torch.log(expansion / epsilon) / block)
            companions = [
                _wrap(torch, reference + epsilon * basis[:, index].reshape_as(reference))
                for index in range(vectors)
            ]
            completed += block
    local = torch.stack(local_spectra).cpu().numpy()
    cumulative = (
        np.cumsum(local * interval, axis=0)
        / ((np.arange(len(local)) + 1)[:, None] * interval)
    )
    return local.astype(np.float64), cumulative.astype(np.float64)


def run_lyapunov_spectrum_campaign(*, model, data, device, torch_module,
                                   selected_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.double()
    selected = pd.read_csv(selected_path)
    requested = os.environ.get("SME_RENORMALIZED_CYPS", "CYP1A2,CYP2C9").split(",")
    selected = selected[selected.cyp_target.isin(requested)].copy()
    burn_in = int(os.environ.get("SME_SPECTRUM_BURN_IN", "1000"))
    measured = int(os.environ.get("SME_SPECTRUM_GENERATIONS", "4000"))
    intervals = [int(value) for value in
                 os.environ.get("SME_SPECTRUM_INTERVALS", "5,10,20").split(",")]
    epsilon = float(os.environ.get("SME_SPECTRUM_EPSILON", "1e-7"))
    vectors = int(os.environ.get("SME_SPECTRUM_VECTORS", "8"))
    repeats = int(os.environ.get("SME_SPECTRUM_REPEATS", "2"))
    started = time.perf_counter()
    rows = []
    for case_number, row in enumerate(selected.itertuples(), start=1):
        rec = data["train"][int(row.training_index)]
        for interval in intervals:
            for repeat in range(repeats):
                local, cumulative = _spectrum_run(
                    model=model, rec=rec, cyp=int(row.cyp_index), torch=torch_module,
                    device=device, burn_in=burn_in, measured_generations=measured,
                    interval=interval, epsilon=epsilon, vectors=vectors,
                    seed=44119 + case_number * 1009 + interval * 7919 + repeat * 65537,
                )
                final = cumulative[-1]
                stem = (f"{case_number:02d}_{row.molecule_id}_{row.cyp_target}"
                        f"_interval_{interval:02d}_repeat_{repeat + 1:02d}")
                np.savez_compressed(
                    output_dir / f"{stem}.npz", local_spectra=local,
                    cumulative_spectra=cumulative, interval=np.asarray(interval),
                    epsilon=np.asarray(epsilon), vectors=np.asarray(vectors),
                    burn_in=np.asarray(burn_in), measured_generations=np.asarray(measured),
                )
                for spectrum_index, exponent in enumerate(final, start=1):
                    rows.append({
                        "molecule_id": row.molecule_id, "cyp_target": row.cyp_target,
                        "interval": interval, "repeat": repeat + 1,
                        "spectrum_index": spectrum_index,
                        "lyapunov_exponent": exponent,
                    })
                print(json.dumps({
                    "molecule_id": row.molecule_id, "interval": interval,
                    "repeat": repeat + 1, "largest": float(final[0]),
                    "positive_exponents": int(np.sum(final > 0)),
                }), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "lyapunov_spectrum_runs.csv", index=False)
    summary = frame.groupby(
        ["molecule_id", "cyp_target", "interval", "spectrum_index"]
    ).agg(
        mean_exponent=("lyapunov_exponent", "mean"),
        std_exponent=("lyapunov_exponent", "std"),
        minimum_exponent=("lyapunov_exponent", "min"),
        maximum_exponent=("lyapunov_exponent", "max"),
    ).reset_index()
    summary.to_csv(output_dir / "lyapunov_spectrum_summary.csv", index=False)
    metadata = {
        "method": "float64 Benettin QR Lyapunov spectrum with circular phase distance",
        "cases": len(selected), "burn_in": burn_in,
        "measured_generations": measured, "intervals": intervals,
        "epsilon": epsilon, "vectors": vectors, "repeats": repeats,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)
