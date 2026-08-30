#!/usr/bin/env python3
"""Analyse frozen 5,000-generation Graph-CA trajectories for attractor evidence.

This is a post-training analysis. It retains compact, plot-ready dynamical arrays
and uses the complete atom-by-channel state, rather than molecule means alone.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    import torch
except ModuleNotFoundError:  # Plot-only mode can use the bundled numeric runtime.
    torch = None


RULES = (
    "gated_residual", "delayed_memory", "inertial_reaction_diffusion",
    "kuramoto_sakaguchi", "fitzhugh_nagumo",
)
INK, WHITE, CYAN, MAGENTA, LIME = "#070914", "#DCE6F2", "#00E5FF", "#FF1493", "#A6FF00"


def pca_coordinates(x: np.ndarray, components: int = 6) -> tuple[np.ndarray, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
    tensor = tensor - tensor.mean(dim=0, keepdim=True)
    count = min(components, tensor.shape[0] - 1, tensor.shape[1])
    u, s, _ = torch.pca_lowrank(tensor, q=count, center=False, niter=4)
    coords = u[:, :count] * s[:count]
    variance = s.square()
    explained = variance / torch.sum(tensor.square()).clamp_min(1e-30)
    return coords.cpu().numpy().astype(np.float32), explained.cpu().numpy().astype(np.float32)


def linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean(); y = y - y.mean()
    return float(np.sum(x * y) / max(float(np.sum(x * x)), 1e-30))


def recurrence_analysis(coords: np.ndarray, samples: int = 1000,
                        theiler: int = 10) -> tuple[dict, np.ndarray]:
    indices = np.linspace(0, len(coords) - 1, min(samples, len(coords)), dtype=int)
    z = coords[indices, :min(6, coords.shape[1])].astype(np.float64)
    z /= np.std(z, axis=0, keepdims=True).clip(1e-12)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z_tensor = torch.as_tensor(z, dtype=torch.float32, device=device)
    distances_tensor = torch.cdist(z_tensor, z_tensor)
    ii, jj = np.triu_indices(len(z), k=theiler + 1)
    upper = distances_tensor[
        torch.as_tensor(ii, device=device), torch.as_tensor(jj, device=device)
    ]
    threshold = float(torch.quantile(upper, 0.05).cpu())
    recurrence = (distances_tensor <= threshold).cpu().numpy()
    del distances_tensor, z_tensor, upper
    near_diagonal = np.abs(np.arange(len(z))[:, None] - np.arange(len(z))[None, :]) <= theiler
    recurrence[near_diagonal] = False
    recurrence_ratio = float(recurrence.sum() / max(1, (~near_diagonal).sum()))

    diagonal_lengths = []
    vertical_lengths = []
    for offset in range(-len(z) + 1, len(z)):
        values = np.diagonal(recurrence, offset=offset)
        padded = np.r_[False, values, False].astype(np.int8)
        runs = np.flatnonzero(np.diff(padded))
        diagonal_lengths.extend((runs[1::2] - runs[::2]).tolist())
    for column in range(len(z)):
        padded = np.r_[False, recurrence[:, column], False].astype(np.int8)
        runs = np.flatnonzero(np.diff(padded))
        vertical_lengths.extend((runs[1::2] - runs[::2]).tolist())
    diagonal_lengths = np.asarray(diagonal_lengths, dtype=int)
    vertical_lengths = np.asarray(vertical_lengths, dtype=int)
    recurrent_points = max(1, int(recurrence.sum()))
    determinism = float(diagonal_lengths[diagonal_lengths >= 2].sum() / recurrent_points)
    laminarity = float(vertical_lengths[vertical_lengths >= 2].sum() / recurrent_points)
    return {
        "recurrence_threshold": threshold,
        "recurrence_rate": recurrence_ratio,
        "determinism": determinism,
        "laminarity": laminarity,
        "longest_diagonal": int(diagonal_lengths.max(initial=0)),
        "mean_diagonal": float(diagonal_lengths[diagonal_lengths >= 2].mean())
        if np.any(diagonal_lengths >= 2) else 0.0,
    }, recurrence.astype(np.uint8)


def correlation_dimension(coords: np.ndarray, seed: int) -> tuple[float, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    count = min(250_000, len(coords) * 80)
    i = rng.integers(0, len(coords), count)
    j = rng.integers(0, len(coords), count)
    keep = np.abs(i - j) > 20
    delta = coords[i[keep], :6] - coords[j[keep], :6]
    distances = np.sqrt(np.sum(delta * delta, axis=1))
    positive = distances[distances > 1e-12]
    if len(positive) < 100:
        return 0.0, np.asarray([]), np.asarray([])
    radii = np.quantile(positive, np.linspace(0.01, 0.35, 18))
    correlation = np.asarray([(positive <= radius).mean() for radius in radii])
    usable = (correlation > 0.02) & (correlation < 0.25)
    dimension = linear_slope(np.log(radii[usable]), np.log(correlation[usable]))
    return dimension, radii.astype(np.float32), correlation.astype(np.float32)


def rosenstein_divergence(coords: np.ndarray, horizon: int = 120,
                          theiler: int = 80) -> tuple[float, np.ndarray]:
    z = coords[:, :min(6, coords.shape[1])].astype(np.float64)
    z /= np.std(z, axis=0, keepdims=True).clip(1e-12)
    anchors = np.linspace(0, len(z) - horizon - 1, min(450, len(z) - horizon), dtype=int)
    curves = []
    for anchor in anchors:
        distances = np.sqrt(np.sum((z - z[anchor]) ** 2, axis=1))
        lo, hi = max(0, anchor - theiler), min(len(z), anchor + theiler + 1)
        distances[lo:hi] = np.inf
        distances[len(z) - horizon:] = np.inf
        neighbour = int(np.argmin(distances))
        if not np.isfinite(distances[neighbour]):
            continue
        delta = z[anchor:anchor + horizon] - z[neighbour:neighbour + horizon]
        separation = np.sqrt(np.sum(delta * delta, axis=1))
        curves.append(np.log(np.maximum(separation, 1e-12)))
    mean_curve = np.mean(curves, axis=0) if curves else np.full(horizon, np.nan)
    fit_end = min(40, horizon)
    slope = linear_slope(np.arange(2, fit_end), mean_curve[2:fit_end])
    return slope, mean_curve.astype(np.float32)


def spectrum(x: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
    centered = x - x.mean(axis=0, keepdims=True)
    power = np.abs(np.fft.rfft(centered, axis=0)) ** 2
    total = power[1:].sum(axis=1)
    probability = total / max(float(total.sum()), 1e-30)
    entropy = float(-np.sum(probability * np.log(probability.clip(1e-30))) /
                    np.log(max(2, len(probability))))
    order = np.argsort(total)[::-1][:8] + 1
    periods = len(x) / order
    frequencies = np.fft.rfftfreq(len(x))[1:]
    return {
        "spectral_entropy": entropy,
        "spectral_concentration": float(probability.max(initial=0.0)),
        "dominant_period": float(periods[0]),
        "secondary_period": float(periods[1]) if len(periods) > 1 else np.nan,
    }, frequencies.astype(np.float32), probability.astype(np.float32)


def classify(row: dict) -> str:
    if row["late_motion"] < 1e-8:
        return "point-attractor candidate"
    if row["recurrence_determinism"] > 0.8 and row["spectral_concentration"] > 0.35:
        return "periodic candidate"
    if row["recurrence_determinism"] > 0.65 and row["spectral_entropy"] < 0.65:
        return "quasiperiodic candidate"
    if row["rosenstein_slope"] > 0.005 and 1.1 < row["correlation_dimension"] < 6.0:
        return "chaotic or complex-transient candidate"
    if row["late_motion"] < 1e-4:
        return "contracting long transient"
    return "persistent unresolved dynamics"


def make_figures(frame: pd.DataFrame, output: Path) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.style.use("dark_background")
    plt.rcParams.update({"figure.facecolor": INK, "axes.facecolor": INK,
                         "savefig.facecolor": INK, "text.color": WHITE})
    fig, ax = plt.subplots(figsize=(11, 7))
    scatter = ax.scatter(frame.correlation_dimension, frame.rosenstein_slope,
                         c=frame.spectral_entropy, cmap="cool", s=90,
                         edgecolors=WHITE, linewidths=.35)
    for row in frame.itertuples():
        if row.transition_rule == "kuramoto_sakaguchi":
            ax.annotate(row.molecule_id, (row.correlation_dimension, row.rosenstein_slope),
                        fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.axhline(0, color=LIME, lw=1, alpha=.7)
    ax.set(xlabel="Estimated correlation dimension", ylabel="Rosenstein divergence slope",
           title="Long-horizon Graph-CA attractor screen")
    ax.grid(alpha=.12); fig.colorbar(scatter, ax=ax, label="Spectral entropy")
    fig.tight_layout(); fig.savefig(figures / "01_attractor_screen.png", dpi=220); plt.close(fig)

    fig, axes = plt.subplots(4, 5, figsize=(16, 12), constrained_layout=True)
    for ax, row in zip(axes.flat, frame.itertuples()):
        data = np.load(output / "case_data" / f"{row.case_id}.npz")
        coords = data["pca_coordinates"]
        ax.plot(coords[:, 0], coords[:, 1], color=CYAN, lw=.35, alpha=.65)
        ax.scatter(coords[-1, 0], coords[-1, 1], s=12, color=MAGENTA)
        ax.set_title(f"{row.transition_rule}\n{row.molecule_id} / {row.cyp_target}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("PCA projections of generations 1,000 to 5,000")
    fig.savefig(figures / "02_phase_portraits.png", dpi=190); plt.close(fig)

    fig, axes = plt.subplots(4, 5, figsize=(16, 12), constrained_layout=True)
    for ax, row in zip(axes.flat, frame.itertuples()):
        recurrence = np.load(output / "case_data" / f"{row.case_id}.npz")["recurrence_matrix"]
        ax.imshow(recurrence, cmap="magma", origin="lower", interpolation="nearest")
        ax.set_title(f"{row.visual_rank}: {row.molecule_id}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Downsampled full-state recurrence plots")
    fig.savefig(figures / "03_recurrence_gallery.png", dpi=190); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("results/ds_gcae_1000_generation_pymol"))
    parser.add_argument("--output", type=Path, default=Path("results/long_horizon_attractor_campaign_v1"))
    parser.add_argument("--burn-in", type=int, default=1000)
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "case_data").mkdir(exist_ok=True)
    if args.plots_only:
        frame = pd.read_csv(args.output / "attractor_screen.csv")
        make_figures(frame, args.output)
        metadata = {"source": str(args.source), "cases": len(frame), "generations": 5000,
                    "burn_in": args.burn_in,
                    "claim_status": "screening only; candidates require perturbation replication"}
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        return
    manifest = pd.read_csv(args.source / "trajectory_manifest.csv")
    rows = []
    started = time.perf_counter()
    for visual in manifest.itertuples():
        trajectory_root = args.output / "base_trajectories" / visual.transition_rule
        candidates = pd.read_csv(trajectory_root / "selected_candidates.csv")
        case_number = int(candidates.index[(candidates.molecule_id == visual.molecule_id) &
                                           (candidates.cyp_target == visual.cyp_target)][0]) + 1
        archive = trajectory_root / "node_trajectories" / f"case_{case_number:03d}.npz"
        trajectory = np.load(archive)["trajectory"].astype(np.float64)
        late = trajectory[args.burn_in:].reshape(len(trajectory) - args.burn_in, -1)
        if visual.transition_rule == "kuramoto_sakaguchi":
            phase = np.pi * late
            late = np.concatenate((np.sin(phase), np.cos(phase)), axis=1)
        print(json.dumps({"case": len(rows) + 1, "stage": "pca", "frames": len(trajectory)}), flush=True)
        coords, explained = pca_coordinates(late)
        print(json.dumps({"case": len(rows) + 1, "stage": "recurrence"}), flush=True)
        recurrence_metrics, recurrence = recurrence_analysis(coords)
        print(json.dumps({"case": len(rows) + 1, "stage": "dimension"}), flush=True)
        dimension, radii, correlation = correlation_dimension(coords, 7300 + int(visual.visual_rank))
        print(json.dumps({"case": len(rows) + 1, "stage": "divergence"}), flush=True)
        divergence_slope, divergence_curve = rosenstein_divergence(coords)
        spectral_metrics, frequencies, spectral_power = spectrum(late)
        steps = np.sqrt(np.mean(np.diff(late, axis=0) ** 2, axis=1))
        case_id = f"{int(visual.visual_rank):02d}_{visual.transition_rule}_{visual.molecule_id}_{visual.cyp_target}"
        row = {
            "case_id": case_id, "visual_rank": int(visual.visual_rank),
            "transition_rule": visual.transition_rule, "molecule_id": visual.molecule_id,
            "cyp_target": visual.cyp_target, "atoms": trajectory.shape[1],
            "channels": trajectory.shape[2], "frames": trajectory.shape[0],
            "burn_in": args.burn_in, "late_motion": float(steps.mean()),
            "late_amplitude": float(np.sqrt(np.mean(np.var(late, axis=0)))),
            "pca_variance_3d": float(explained[:3].sum()),
            "correlation_dimension": dimension, "rosenstein_slope": divergence_slope,
            **{f"recurrence_{key}": value for key, value in recurrence_metrics.items()},
            **spectral_metrics,
        }
        row["classification"] = classify(row)
        rows.append(row)
        np.savez_compressed(
            args.output / "case_data" / f"{case_id}.npz",
            pca_coordinates=coords, pca_explained_variance=explained,
            recurrence_matrix=recurrence, correlation_radii=radii,
            correlation_integral=correlation, rosenstein_log_divergence=divergence_curve,
            frequencies=frequencies, spectral_power=spectral_power,
            step_energy=steps.astype(np.float32), mean_state=trajectory[args.burn_in:].mean(axis=1).astype(np.float32),
        )
        print(json.dumps({"case": len(rows), "total": len(manifest), "id": case_id,
                          "classification": row["classification"]}), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "attractor_screen.csv", index=False)
    make_figures(frame, args.output)
    metadata = {"source": str(args.source), "cases": len(frame), "generations": 5000,
                "burn_in": args.burn_in, "elapsed_seconds": time.perf_counter() - started,
                "claim_status": "screening only; candidates require perturbation replication"}
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
