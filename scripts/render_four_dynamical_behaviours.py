#!/usr/bin/env python3
"""Render matched Graph-CA behaviour figures and cyberpunk animations."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Line3DCollection


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "long_horizon_attractor_campaign_v1" / "base_trajectories"
FIGURES = ROOT / "results" / "long_horizon_attractor_campaign_v1" / "figures"
VIDEOS = ROOT / "results" / "long_horizon_attractor_campaign_v1" / "videos"
FFMPEG = Path(r"C:\ffmpeg\bin\ffmpeg.exe")
INK, WHITE, CYAN, MAGENTA, LIME = "#070914", "#DCE6F2", "#00E5FF", "#FF1493", "#A6FF00"

SOURCES = {
    "point": BASE / "gated_residual" / "node_trajectories" / "case_001.npz",
    "strange": BASE / "kuramoto_sakaguchi" / "node_trajectories" / "case_002.npz",
    "complex": BASE / "kuramoto_sakaguchi" / "node_trajectories" / "case_003.npz",
    "oscillator": BASE / "coupled_map" / "node_trajectories" / "case_004.npz",
}


def pca_project(path: Path, fit_slice=slice(None), circular: bool = False,
                terminal_center: bool = False) -> np.ndarray:
    torch.manual_seed(1701)
    trajectory = np.load(path)["trajectory"].astype(np.float64)
    flattened = trajectory.reshape(len(trajectory), -1)
    if circular:
        phase = np.pi * flattened
        flattened = np.concatenate((np.sin(phase), np.cos(phase)), axis=1)
    fit = flattened[fit_slice]
    fit_tensor = torch.as_tensor(fit, dtype=torch.float32)
    full_tensor = torch.as_tensor(flattened, dtype=torch.float32)
    mean = fit_tensor.mean(dim=0, keepdim=True)
    _, _, components = torch.pca_lowrank(fit_tensor - mean, q=3, center=False)
    coordinates = ((full_tensor - mean) @ components).numpy().astype(np.float64)
    if terminal_center:
        terminal = torch.as_tensor(flattened[-500:].mean(axis=0, keepdims=True),
                                   dtype=torch.float32)
        coordinates -= ((terminal - mean) @ components).numpy()[0]
    else:
        coordinates -= np.median(coordinates[fit_slice], axis=0, keepdims=True)
    scale = np.std(coordinates[fit_slice], axis=0).clip(1e-12)
    return coordinates / scale


def oscillator_delay_project(path: Path, burn_in: int = 1000) -> np.ndarray:
    """Delay-embed the leading late-orbit full-state component."""
    torch.manual_seed(1701)
    trajectory = np.load(path)["trajectory"].astype(np.float64)
    flattened = trajectory.reshape(len(trajectory), -1)
    fit_tensor = torch.as_tensor(flattened[burn_in:], dtype=torch.float32)
    full_tensor = torch.as_tensor(flattened, dtype=torch.float32)
    mean = fit_tensor.mean(dim=0, keepdim=True)
    _, _, component = torch.pca_lowrank(fit_tensor - mean, q=1, center=False)
    score = ((full_tensor - mean) @ component).numpy()[:, 0].astype(np.float64)
    coordinates = np.column_stack((score, np.roll(score, 1), np.roll(score, 2)))
    coordinates[0, 1:] = score[0]
    coordinates[1, 2] = score[0]
    centre = np.median(coordinates[burn_in:], axis=0, keepdims=True)
    scale = np.std(coordinates[burn_in:], axis=0).clip(1e-12)
    return (coordinates - centre) / scale


def limits(coordinates: np.ndarray, window=slice(None), quantile: float = 0.0):
    values = coordinates[window]
    if quantile:
        low = np.quantile(values, quantile, axis=0)
        high = np.quantile(values, 1 - quantile, axis=0)
    else:
        low, high = values.min(axis=0), values.max(axis=0)
    centre = (low + high) / 2
    radius = .55 * float(np.max(high - low))
    return [(float(value - radius), float(value + radius)) for value in centre]


def add_static_panel(ax, coordinates, title, subtitle, view, window=slice(None),
                     line_width=1.0, phase_markers: bool = False):
    selected = coordinates[window]
    indices = np.arange(len(coordinates))[window]
    segments = np.stack((selected[:-1], selected[1:]), axis=1)
    norm = colors.PowerNorm(gamma=.35, vmin=0, vmax=5000)
    shades = plt.colormaps["viridis"](norm(indices[:-1])); shades[:, 3] = .80
    ax.add_collection3d(Line3DCollection(segments, colors=shades, linewidths=line_width))
    for setter, bound in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), limits(coordinates, window)):
        setter(*bound)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(*view)
    ax.scatter(*selected[0], s=38, marker="o", color=plt.colormaps["viridis"](norm(indices[0])),
               edgecolor="black", linewidth=.6, depthshade=False,
               label=f"Generation {indices[0]:,}")
    ax.scatter(*selected[-1], s=82, marker="*", color=plt.colormaps["viridis"](1),
               edgecolor="black", linewidth=.6, depthshade=False,
               label="Generation 5,000")
    if phase_markers:
        late = coordinates[-1000:]
        phase_a = late[::2].mean(axis=0); phase_b = late[1::2].mean(axis=0)
        ax.scatter(*phase_a, s=72, marker="o", color=MAGENTA, edgecolor="black",
                   linewidth=.6, depthshade=False, label="Alternating state A")
        ax.scatter(*phase_b, s=72, marker="o", color=CYAN, edgecolor="black",
                   linewidth=.6, depthshade=False, label="Alternating state B")
    ax.set_xlabel("Dynamical PC1", labelpad=4); ax.set_ylabel("Dynamical PC2", labelpad=4)
    ax.set_zlabel("Dynamical PC3", labelpad=1); ax.tick_params(labelsize=6)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 1)); axis.pane.set_edgecolor("#CCCCCC")
        axis._axinfo["grid"]["color"] = (.87, .87, .87, 1)
        axis._axinfo["grid"]["linewidth"] = .45
    ax.text2D(.5, .99, title, transform=ax.transAxes, ha="center", va="top",
              fontsize=12.5, fontweight="bold")
    ax.text2D(.5, .935, subtitle, transform=ax.transAxes, ha="center", va="top",
              fontsize=7.5, color="#444444")
    ax.legend(loc="upper left", bbox_to_anchor=(.01, .89), frameon=False, fontsize=7)


def render_individual_static(key, coordinates, title, subtitle, view, window,
                             phase_markers: bool = False):
    fig = plt.figure(figsize=(12, 10), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    add_static_panel(ax, coordinates, title, subtitle, view, window,
                     .4 if phase_markers else 1.25, phase_markers)
    norm = colors.PowerNorm(gamma=.35, vmin=0, vmax=5000)
    scalar = ScalarMappable(norm=norm, cmap=plt.colormaps["viridis"]); scalar.set_array([])
    colourbar = fig.colorbar(scalar, ax=ax, fraction=.032, pad=.07, shrink=.72)
    colourbar.set_label("Cellular-Automata Generation", labelpad=9)
    colourbar.set_ticks([0, 25, 50, 100, 250, 500, 1000, 2500, 5000])
    fig.subplots_adjust(left=.02, right=.91, bottom=.03, top=.94)
    png = FIGURES / f"{key}.png"; pdf = FIGURES / f"{key}.pdf"
    fig.savefig(png, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white"); plt.close(fig)
    return png, pdf


def render_video(key, coordinates, title, subtitle, metric_lines, view,
                 zoom_window=slice(None)):
    mpl.rcParams["animation.ffmpeg_path"] = str(FFMPEG)
    frames, fps = 900, 30
    frame_indices = np.linspace(1, len(coordinates) - 1, frames, dtype=int)
    full_limits = limits(coordinates)
    target_limits = limits(coordinates, zoom_window, quantile=.002)
    zoom = zoom_window != slice(None)
    cmap = LinearSegmentedColormap.from_list("cyber_behaviour", [MAGENTA, CYAN, LIME])
    fig = plt.figure(figsize=(16, 9), facecolor=INK)
    ax = fig.add_subplot(111, projection="3d", facecolor=INK)
    fig.subplots_adjust(left=.01, right=.99, bottom=.01, top=.94)
    ax.set_box_aspect((1, 1, 1)); ax.grid(False); ax.set_axis_off()
    ax.set_title(title, color=WHITE, fontsize=19, pad=10)
    ax.text2D(.5, .965, subtitle, transform=ax.transAxes, ha="center", va="top",
              color=CYAN, fontsize=11)
    history, = ax.plot([], [], [], color=CYAN, lw=.48, alpha=.18)
    trail = Line3DCollection(np.zeros((1, 2, 3)), linewidths=2.1,
                             colors=[(0, 0, 0, 0)])
    ax.add_collection3d(trail)
    head = ax.scatter([], [], [], s=58, color=LIME, edgecolors=WHITE,
                      linewidths=.7, depthshade=False)
    status = ax.text2D(.035, .055, "", transform=ax.transAxes, color=WHITE,
                       fontsize=11, family="monospace", alpha=.92)
    note_text = ("3D delay embedding of the leading full-state component"
                 if zoom else "3D PCA of the complete atom × channel state")
    note = ax.text2D(.965, .055, note_text,
                     transform=ax.transAxes, ha="right", color=WHITE, fontsize=9, alpha=.62)

    def update(frame):
        index = int(frame_indices[frame])
        history_start = 1000 if zoom and index >= 1000 else 0
        visible = coordinates[history_start:index + 1]
        history.set_data_3d(visible[:, 0], visible[:, 1], visible[:, 2])
        start = max(0, index - 260); recent = coordinates[start:index + 1]
        if len(recent) > 1:
            pieces = np.stack((recent[:-1], recent[1:]), axis=1)
            shades = cmap(np.linspace(0, 1, len(pieces))); shades[:, 3] = np.linspace(.08, .98, len(pieces))
            trail.set_segments(pieces); trail.set_color(shades)
        point = coordinates[index]; head._offsets3d = ([point[0]], [point[1]], [point[2]])
        status.set_text(f"generation  {index:5d}\n{metric_lines}")
        blend = min(1.0, max(0.0, (frame - 120) / 240)) if zoom else 0.0
        bounds = [[(1 - blend) * a + blend * b for a, b in zip(full, target)]
                  for full, target in zip(full_limits, target_limits)]
        for setter, bound in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), bounds): setter(*bound)
        ax.view_init(elev=view[0] + 3 * np.sin(2 * np.pi * frame / frames),
                     azim=view[1] + 85 * frame / frames)
        return history, trail, head, status, note

    update(frames - 1)
    poster = VIDEOS / f"{key}.png"; video = VIDEOS / f"{key}.mp4"
    fig.savefig(poster, dpi=120, facecolor=INK)
    animation = FuncAnimation(fig, update, frames=frames, interval=1000 / fps,
                              blit=False, repeat=False)
    writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=8000,
                          extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    animation.save(video, writer=writer, dpi=120, savefig_kwargs={"facecolor": INK})
    plt.close(fig)
    return video, poster


def main():
    FIGURES.mkdir(parents=True, exist_ok=True); VIDEOS.mkdir(parents=True, exist_ok=True)
    if os.environ.get("SME_BEHAVIOUR_RENDER_ONLY") == "oscillator_video":
        oscillator = oscillator_delay_project(SOURCES["oscillator"])
        for output in render_video(
            "trajectory_coupled_map_period2_oscillator_candidate", oscillator,
            "Graph-CA Period-2 Oscillator Candidate",
            "Coupled-map molecular information dynamics",
            "dominant period  ≈ 2 generations\nspectral concentration  0.802",
            (24, 152), slice(1000, None)):
            print(output)
        return
    coordinates = {
        "point": pca_project(SOURCES["point"], slice(0, 500), terminal_center=True),
        "strange": pca_project(SOURCES["strange"], circular=True),
        "complex": pca_project(SOURCES["complex"], circular=True),
        "oscillator": oscillator_delay_project(SOURCES["oscillator"]),
    }
    outputs = []
    outputs += render_individual_static(
        "15_kuramoto_persistent_complex_candidate", coordinates["complex"],
        "Trajectory: Persistent or Complex Candidate",
        "Kuramoto–Sakaguchi · OCNT-0495493 · CYP2D6", (24, 138), slice(None))
    outputs += render_individual_static(
        "16_coupled_map_oscillator_candidate", coordinates["oscillator"],
        "Trajectory: Period-2 Oscillator Candidate",
        "Coupled map · OCNT-0495275 · CYP3A4 · mature delay embedding", (24, 152),
        slice(1000, None), True)

    fig = plt.figure(figsize=(18, 16.2), facecolor="white")
    panels = [fig.add_subplot(221, projection="3d"), fig.add_subplot(222, projection="3d"),
              fig.add_subplot(223, projection="3d"), fig.add_subplot(224, projection="3d")]
    add_static_panel(panels[0], coordinates["point"], "A  |  Point Attractor",
                     "Gated residual · OCNT-2328519 · CYP1A2", (22, 42), slice(None))
    add_static_panel(panels[1], coordinates["strange"], "B  |  Strange Attractor",
                     "Kuramoto–Sakaguchi · OCNT-0494110 · CYP2C9", (25, 165), slice(None))
    add_static_panel(panels[2], coordinates["complex"], "C  |  Persistent or Complex Candidate",
                     "Kuramoto–Sakaguchi · OCNT-0495493 · CYP2D6", (24, 138), slice(None))
    add_static_panel(panels[3], coordinates["oscillator"], "D  |  Period-2 Oscillator Candidate",
                     "Coupled map · OCNT-0495275 · CYP3A4 · mature delay embedding",
                     (24, 152), slice(1000, None), .35, True)
    norm = colors.PowerNorm(gamma=.35, vmin=0, vmax=5000)
    scalar = ScalarMappable(norm=norm, cmap=plt.colormaps["viridis"]); scalar.set_array([])
    colourbar = fig.colorbar(scalar, ax=panels, fraction=.018, pad=.055, shrink=.76)
    colourbar.set_label("Cellular-Automata Generation", labelpad=9)
    colourbar.set_ticks([0, 25, 50, 100, 250, 500, 1000, 2500, 5000])
    fig.suptitle("Four Dynamical Regimes in Graph Cellular Automata",
                 fontsize=18, fontweight="bold", y=.985)
    fig.subplots_adjust(left=.02, right=.84, bottom=.025, top=.95, hspace=.02, wspace=.01)
    four_png = FIGURES / "17_four_graph_ca_dynamical_behaviours.png"
    four_pdf = FIGURES / "17_four_graph_ca_dynamical_behaviours.pdf"
    fig.savefig(four_png, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(four_pdf, bbox_inches="tight", facecolor="white"); plt.close(fig)
    outputs += [four_png, four_pdf]

    if os.environ.get("SME_BEHAVIOUR_RENDER_ONLY") == "figures":
        for output in outputs: print(output)
        return

    outputs += render_video(
        "trajectory_kuramoto_persistent_complex_candidate", coordinates["complex"],
        "Graph-CA Persistent or Complex Candidate",
        "Kuramoto–Sakaguchi molecular information dynamics",
        "spectral entropy  0.605\ndominant period  ≈ 572 generations",
        (24, 138))
    outputs += render_video(
        "trajectory_coupled_map_period2_oscillator_candidate", coordinates["oscillator"],
        "Graph-CA Period-2 Oscillator Candidate",
        "Coupled-map molecular information dynamics",
        "dominant period  ≈ 2 generations\nspectral concentration  0.802",
        (24, 152), slice(1000, None))
    for output in outputs: print(output)


if __name__ == "__main__":
    main()
