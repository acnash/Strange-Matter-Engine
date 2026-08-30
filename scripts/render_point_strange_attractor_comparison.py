#!/usr/bin/env python3
"""Render matched 0-to-5,000-generation point and strange attractor panels."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Line3DCollection


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "long_horizon_attractor_campaign_v1" / "base_trajectories"
POINT = BASE / "gated_residual" / "node_trajectories" / "case_001.npz"
STRANGE = BASE / "kuramoto_sakaguchi" / "node_trajectories" / "case_002.npz"
OUTPUT = ROOT / "results" / "long_horizon_attractor_campaign_v1" / "figures"


def project(path: Path, early_fit: bool, circular: bool = False) -> np.ndarray:
    trajectory = np.load(path)["trajectory"].astype(np.float64)
    flattened = trajectory.reshape(len(trajectory), -1)
    if circular:
        phase = np.pi * flattened
        flattened = np.concatenate((np.sin(phase), np.cos(phase)), axis=1)
    fit = flattened[:500] if early_fit else flattened
    device = "cpu"
    fit_tensor = torch.as_tensor(fit, dtype=torch.float32, device=device)
    full_tensor = torch.as_tensor(flattened, dtype=torch.float32, device=device)
    mean = fit_tensor.mean(dim=0, keepdim=True)
    _, _, components = torch.pca_lowrank(fit_tensor - mean, q=3, center=False)
    coordinates = ((full_tensor - mean) @ components).numpy().astype(np.float64)
    if early_fit:
        terminal = flattened[-500:].mean(axis=0)
        terminal_tensor = torch.as_tensor(terminal[None, :], dtype=torch.float32,
                                          device=device)
        terminal_coordinate = ((terminal_tensor - mean) @ components).numpy()[0]
        coordinates -= terminal_coordinate
        scale = np.std(coordinates[:500], axis=0).clip(1e-12)
    else:
        coordinates -= np.median(coordinates, axis=0, keepdims=True)
        scale = np.std(coordinates, axis=0).clip(1e-12)
    return coordinates / scale


def equal_limits(coordinates: np.ndarray):
    low = coordinates.min(axis=0)
    high = coordinates.max(axis=0)
    centre = (low + high) / 2
    radius = .55 * float(np.max(high - low))
    return [(value - radius, value + radius) for value in centre]


def add_panel(ax, coordinates, title, subtitle, view):
    generations = np.arange(len(coordinates))
    segments = np.stack((coordinates[:-1], coordinates[1:]), axis=1)
    cmap = plt.colormaps["viridis"]
    norm = colors.PowerNorm(gamma=.35, vmin=0, vmax=5000)
    shades = cmap(norm(generations[:-1])); shades[:, 3] = .80
    ax.add_collection3d(Line3DCollection(segments, colors=shades, linewidths=1.0))
    for setter, limit in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim),
                             equal_limits(coordinates)):
        setter(*limit)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(*view)
    ax.scatter(*coordinates[0], s=55, marker="o", color=cmap(0), edgecolor="black",
               linewidth=.7, depthshade=False, label="Generation 0")
    ax.scatter(*coordinates[-1], s=105, marker="*", color=cmap(1), edgecolor="black",
               linewidth=.7, depthshade=False, label="Generation 5,000")
    ax.set_xlabel("Dynamical PC1", labelpad=7)
    ax.set_ylabel("Dynamical PC2", labelpad=7)
    ax.set_zlabel("Dynamical PC3", labelpad=7)
    ax.tick_params(labelsize=7, colors="#333333")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 1)); axis.pane.set_edgecolor("#CCCCCC")
        axis._axinfo["grid"]["color"] = (.86, .86, .86, 1)
        axis._axinfo["grid"]["linewidth"] = .5
    ax.text2D(.5, .965, title, transform=ax.transAxes, ha="center", va="top",
              fontsize=14, fontweight="bold", color="#111111")
    ax.text2D(.5, .91, subtitle, transform=ax.transAxes, ha="center", va="top",
              fontsize=8.5, color="#444444")
    ax.legend(loc="upper left", bbox_to_anchor=(.01, .91), frameon=False, fontsize=8)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    point = project(POINT, early_fit=True)
    strange = project(STRANGE, early_fit=False, circular=True)
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white", "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(18, 9.2), facecolor="white")
    left = fig.add_subplot(121, projection="3d")
    right = fig.add_subplot(122, projection="3d")
    add_panel(left, point, "A  |  Trajectory 1: Point Attractor",
              "Gated residual · OCNT-2328519 · CYP1A2", (22, 42))
    add_panel(right, strange, "B  |  Trajectory 7: Strange Attractor",
              "Kuramoto–Sakaguchi · OCNT-0494110 · CYP2C9", (23, 178))
    norm = colors.PowerNorm(gamma=.35, vmin=0, vmax=5000)
    scalar = ScalarMappable(norm=norm, cmap=plt.colormaps["viridis"]); scalar.set_array([])
    colourbar = fig.colorbar(scalar, ax=[left, right], fraction=.022, pad=.035, shrink=.79)
    colourbar.set_label("Cellular-Automata Generation", labelpad=10)
    colourbar.set_ticks([0, 25, 50, 100, 250, 500, 1000, 2500, 5000])
    fig.suptitle("Contrasting Long-Horizon Graph Cellular-Automata Dynamics",
                 fontsize=17, fontweight="bold", y=.985)
    fig.text(.5, .025,
             "Both panels show the complete 0–5,000-generation atom-by-channel trajectory using the same time-colour mapping.",
             ha="center", fontsize=9, color="#444444")
    fig.subplots_adjust(left=.015, right=.91, bottom=.07, top=.91, wspace=.02)
    png = OUTPUT / "14_point_and_strange_attractor_comparison.png"
    pdf = OUTPUT / "14_point_and_strange_attractor_comparison.pdf"
    fig.savefig(png, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(png); print(pdf)


if __name__ == "__main__":
    main()
