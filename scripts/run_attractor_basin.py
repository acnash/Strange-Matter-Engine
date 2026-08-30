#!/usr/bin/env python3
"""Test whether displaced Graph-CA states approach a common chaotic invariant set."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd


def _wrap(torch, state):
    phase = math.pi * state
    return torch.atan2(torch.sin(phase), torch.cos(phase)) / math.pi


def _embed(torch, states):
    phase = math.pi * states
    return torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1).flatten(1)


def _evolve(*, model, rec, cyp, initial, generations, stride, torch):
    states = [initial.detach().cpu()]
    current = initial
    with torch.no_grad():
        for generation in range(1, generations + 1):
            current = model.kuramoto_step(rec, cyp, current)
            if generation % stride == 0:
                states.append(current.detach().cpu())
    return torch.stack(states)


def _sliced_distance(torch, a, b, projections):
    count = min(len(a), len(b))
    projected_a = torch.sort(a @ projections, dim=0).values[:count]
    projected_b = torch.sort(b @ projections, dim=0).values[:count]
    return float(torch.mean(torch.abs(projected_a - projected_b)))


def _nearest_cloud_distance(torch, query, cloud, batch=128):
    distances = []
    scale = math.sqrt(query.shape[1])
    for start in range(0, len(query), batch):
        values = torch.cdist(query[start:start + batch], cloud)
        distances.append(values.min(dim=1).values / scale)
    return float(torch.cat(distances).mean())


def _spectral_entropy(values):
    centered = values - values.mean()
    power = np.abs(np.fft.rfft(centered)) ** 2
    probability = power[1:] / max(float(power[1:].sum()), 1e-30)
    return float(-np.sum(probability * np.log(probability.clip(1e-30))) /
                 np.log(max(2, len(probability))))


def run_attractor_basin_campaign(*, model, data, device, torch_module,
                                 selected_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.double()
    selected = pd.read_csv(selected_path)
    requested = os.environ.get("SME_RENORMALIZED_CYPS", "CYP1A2,CYP2C9").split(",")
    selected = selected[selected.cyp_target.isin(requested)].copy()
    burn_in = int(os.environ.get("SME_BASIN_BURN_IN", "1000"))
    generations = int(os.environ.get("SME_BASIN_GENERATIONS", "6000"))
    stride = int(os.environ.get("SME_BASIN_STRIDE", "10"))
    late_start = int(os.environ.get("SME_BASIN_LATE_START", "3000")) // stride
    early_end = int(os.environ.get("SME_BASIN_EARLY_END", "1000")) // stride
    radii = [float(value) for value in
             os.environ.get("SME_BASIN_RADII", "0.1,0.5,1.0,2.0").split(",")]
    repeats = int(os.environ.get("SME_BASIN_REPEATS", "8"))
    started = time.perf_counter()
    rows = []
    for case_number, row in enumerate(selected.itertuples(), start=1):
        rec = data["train"][int(row.training_index)]
        with torch_module.no_grad():
            base = model.initial_node_state(rec)
            for _ in range(burn_in):
                base = model.kuramoto_step(rec, int(row.cyp_index), base)
        reference = _evolve(
            model=model, rec=rec, cyp=int(row.cyp_index), initial=base,
            generations=generations, stride=stride, torch=torch_module,
        )
        reference_device = reference.to(device)
        reference_embedding = _embed(torch_module, reference_device)
        reference_late = reference_embedding[late_start:]
        half = len(reference_late) // 2
        generator = torch_module.Generator(device=device).manual_seed(78191 + case_number)
        projection_count = 32
        projections = torch_module.randn(
            (reference_embedding.shape[1], projection_count), generator=generator,
            device=device, dtype=reference_embedding.dtype,
        )
        projections /= torch_module.linalg.vector_norm(
            projections, dim=0, keepdim=True
        ).clamp_min(1e-30)
        baseline_sliced = _sliced_distance(
            torch_module, reference_late[:half], reference_late[-half:], projections
        )
        archived = [reference.numpy().astype(np.float32)]
        for radius in radii:
            for repeat in range(repeats):
                direction = torch_module.randn(
                    base.shape, generator=generator, device=device, dtype=base.dtype
                )
                direction /= torch_module.linalg.vector_norm(direction).clamp_min(1e-30)
                initial = _wrap(torch_module, base + radius * direction)
                trajectory = _evolve(
                    model=model, rec=rec, cyp=int(row.cyp_index), initial=initial,
                    generations=generations, stride=stride, torch=torch_module,
                )
                archived.append(trajectory.numpy().astype(np.float32))
                trajectory_device = trajectory.to(device)
                embedding = _embed(torch_module, trajectory_device)
                early = embedding[:early_end]
                late = embedding[late_start:]
                early_sliced = _sliced_distance(
                    torch_module, early, reference_late[:len(early)], projections
                )
                late_sliced = _sliced_distance(
                    torch_module, late, reference_late, projections
                )
                early_cloud = _nearest_cloud_distance(
                    torch_module, early, reference_late
                )
                late_cloud = _nearest_cloud_distance(
                    torch_module, late, reference_late
                )
                phase = math.pi * trajectory_device
                coherence = torch_module.abs(torch_module.mean(
                    torch_module.complex(torch_module.cos(phase), torch_module.sin(phase)),
                    dim=(1, 2),
                )).cpu().numpy()
                rows.append({
                    "molecule_id": row.molecule_id, "cyp_target": row.cyp_target,
                    "radius": radius, "repeat": repeat + 1,
                    "maximum_absolute_state": float(torch_module.max(torch_module.abs(trajectory_device))),
                    "early_sliced_distance": early_sliced,
                    "late_sliced_distance": late_sliced,
                    "sliced_distance_ratio": late_sliced / max(early_sliced, 1e-30),
                    "baseline_sliced_distance": baseline_sliced,
                    "late_to_baseline_ratio": late_sliced / max(baseline_sliced, 1e-30),
                    "early_cloud_distance": early_cloud,
                    "late_cloud_distance": late_cloud,
                    "cloud_distance_ratio": late_cloud / max(early_cloud, 1e-30),
                    "late_coherence_mean": float(coherence[late_start:].mean()),
                    "late_coherence_std": float(coherence[late_start:].std()),
                    "late_coherence_spectral_entropy": _spectral_entropy(coherence[late_start:]),
                })
                print(json.dumps({
                    "molecule_id": row.molecule_id, "radius": radius,
                    "repeat": repeat + 1, "sliced_ratio": rows[-1]["sliced_distance_ratio"],
                    "cloud_ratio": rows[-1]["cloud_distance_ratio"],
                }), flush=True)
        np.savez_compressed(
            output_dir / f"{case_number:02d}_{row.molecule_id}_{row.cyp_target}.npz",
            trajectories=np.stack(archived), radii=np.asarray(radii),
            repeats=np.asarray(repeats), stride=np.asarray(stride),
            burn_in=np.asarray(burn_in), generations=np.asarray(generations),
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "basin_runs.csv", index=False)
    summary = frame.groupby(["molecule_id", "cyp_target", "radius"]).agg(
        repeats=("repeat", "size"),
        mean_sliced_ratio=("sliced_distance_ratio", "mean"),
        maximum_sliced_ratio=("sliced_distance_ratio", "max"),
        mean_late_to_baseline=("late_to_baseline_ratio", "mean"),
        mean_cloud_ratio=("cloud_distance_ratio", "mean"),
        maximum_cloud_ratio=("cloud_distance_ratio", "max"),
        bounded_fraction=("maximum_absolute_state", lambda values: float(np.mean(values <= 1.000001))),
    ).reset_index()
    summary["all_runs_approach_distribution"] = summary.maximum_sliced_ratio < 1.0
    summary["all_runs_approach_cloud"] = summary.maximum_cloud_ratio < 1.0
    summary.to_csv(output_dir / "basin_summary.csv", index=False)
    metadata = {
        "method": "float64 multi-radius invariant-set basin test",
        "cases": len(selected), "burn_in": burn_in, "generations": generations,
        "stride": stride, "radii": radii, "repeats_per_radius": repeats,
        "total_displaced_trajectories": len(frame),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)
