#!/usr/bin/env python3
"""Render a Lorenz-style 3D animation of a Graph-CA strange attractor candidate."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Line3DCollection


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "results" / "long_horizon_attractor_campaign_v1" / "case_data" /
          "07_kuramoto_sakaguchi_OCNT-0494110_CYP2C9.npz")
OUTPUT = ROOT / "results" / "long_horizon_attractor_campaign_v1" / "videos"
FFMPEG = Path(r"C:\ffmpeg\bin\ffmpeg.exe")
INK, WHITE, CYAN, MAGENTA, LIME = "#070914", "#DCE6F2", "#00E5FF", "#FF1493", "#A6FF00"


def equal_limits(coordinates: np.ndarray):
    low = np.quantile(coordinates, 0.005, axis=0)
    high = np.quantile(coordinates, 0.995, axis=0)
    center = (low + high) / 2
    radius = 0.56 * float(np.max(high - low))
    return [(float(value - radius), float(value + radius)) for value in center]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path,
                        default=OUTPUT / "trajectory_07_hyperchaotic_strange_attractor.mp4")
    parser.add_argument("--frames", type=int, default=1350)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--trail", type=int, default=260)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mpl.rcParams["animation.ffmpeg_path"] = str(FFMPEG)
    data = np.load(args.source)
    coordinates = data["pca_coordinates"][:, :3].astype(np.float64)
    coordinates -= np.median(coordinates, axis=0, keepdims=True)
    scale = np.std(coordinates, axis=0, keepdims=True).clip(1e-12)
    coordinates /= scale
    limits = equal_limits(coordinates)
    frame_indices = np.linspace(1, len(coordinates) - 1, args.frames, dtype=int)

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 9), facecolor=INK)
    ax = fig.add_subplot(111, projection="3d", facecolor=INK)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.94)
    for axis, limit in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), limits):
        axis(*limit)
    ax.set_box_aspect((1, 1, 1))
    ax.grid(False)
    ax.set_axis_off()
    ax.set_title("Trajectory 7  |  Graph-CA Hyperchaotic Strange Attractor",
                 color=WHITE, fontsize=19, pad=10)
    ax.text2D(.5, .965, "Kuramoto–Sakaguchi molecular information dynamics",
              transform=ax.transAxes, ha="center", va="top", color=CYAN,
              fontsize=11, alpha=.9)
    status = ax.text2D(.035, .055, "", transform=ax.transAxes, color=WHITE,
                       fontsize=11, family="monospace", alpha=.9)
    note = ax.text2D(.965, .055,
                     "3D PCA of the full atom × channel state\nGeneration is the time coordinate",
                     transform=ax.transAxes, ha="right", color=WHITE,
                     fontsize=9, alpha=.62)

    history, = ax.plot([], [], [], color=CYAN, lw=.48, alpha=.18)
    trail_collection = Line3DCollection(np.zeros((1, 2, 3)), linewidths=2.0,
                                        colors=[(0, 0, 0, 0)])
    ax.add_collection3d(trail_collection)
    head = ax.scatter([], [], [], s=58, color=LIME, edgecolors=WHITE,
                      linewidths=.7, depthshade=False)
    cmap = LinearSegmentedColormap.from_list("sme_attractor", [MAGENTA, CYAN, LIME])

    def update(frame_number: int):
        index = int(frame_indices[frame_number])
        visible = coordinates[:index + 1]
        history.set_data_3d(visible[:, 0], visible[:, 1], visible[:, 2])
        start = max(0, index - args.trail)
        recent = coordinates[start:index + 1]
        if len(recent) > 1:
            segments = np.stack((recent[:-1], recent[1:]), axis=1)
            colours = cmap(np.linspace(0, 1, len(segments)))
            colours[:, 3] = np.linspace(.08, .98, len(segments))
            trail_collection.set_segments(segments)
            trail_collection.set_color(colours)
        point = coordinates[index]
        head._offsets3d = ([point[0]], [point[1]], [point[2]])
        generation = 1000 + index
        status.set_text(
            f"generation  {generation:5d}\n"
            f"λ₁ ≈ +0.0118 / generation\n"
            f"8 leading Lyapunov exponents > 0"
        )
        ax.view_init(elev=23 + 4 * np.sin(2 * np.pi * frame_number / args.frames),
                     azim=28 + 150 * frame_number / args.frames, roll=0)
        return history, trail_collection, head, status, note

    update(args.frames - 1)
    poster = args.output.with_suffix(".png")
    fig.savefig(poster, dpi=args.dpi, facecolor=INK)
    animation = FuncAnimation(fig, update, frames=args.frames, interval=1000 / args.fps,
                              blit=False, repeat=False)
    writer = FFMpegWriter(
        fps=args.fps, codec="libx264", bitrate=8000,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        metadata={"title": "Trajectory 7 Graph-CA Hyperchaotic Strange Attractor",
                  "artist": "Strange Matter Engine"},
    )
    animation.save(args.output, writer=writer, dpi=args.dpi)
    plt.close(fig)
    print(args.output)
    print(poster)


if __name__ == "__main__":
    main()
