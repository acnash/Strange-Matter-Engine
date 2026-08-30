#!/usr/bin/env python3
"""Render a contracting Graph-CA trajectory as a static figure and cyberpunk movie."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "results" / "long_horizon_attractor_campaign_v1" /
          "base_trajectories" / "gated_residual" / "node_trajectories" /
          "case_001.npz")
FIGURES = ROOT / "results" / "long_horizon_attractor_campaign_v1" / "figures"
VIDEOS = ROOT / "results" / "long_horizon_attractor_campaign_v1" / "videos"
FFMPEG = Path(r"C:\ffmpeg\bin\ffmpeg.exe")
INK, WHITE, CYAN, MAGENTA, LIME = "#070914", "#DCE6F2", "#00E5FF", "#FF1493", "#A6FF00"


def coordinates_and_distance():
    data = np.load(SOURCE)
    trajectory = data["trajectory"].astype(np.float64)
    flattened = trajectory.reshape(len(trajectory), -1)
    terminal = flattened[-500:].mean(axis=0)
    distance = np.linalg.norm(flattened - terminal, axis=1)
    fit_count = min(500, len(flattened))
    pca = PCA(n_components=3).fit(flattened[:fit_count])
    coordinates = pca.transform(flattened)
    terminal_coordinate = pca.transform(terminal[None, :])[0]
    coordinates -= terminal_coordinate
    scale = np.std(coordinates[:fit_count], axis=0).clip(1e-12)
    coordinates /= scale
    return data, coordinates, distance, pca.explained_variance_ratio_


def limits(coordinates):
    low = coordinates.min(axis=0); high = coordinates.max(axis=0)
    margin = .08 * np.maximum(high - low, .1)
    return list(zip(low - margin, high + margin))


def render_static(data, coordinates, distance, explained):
    generations = np.arange(len(coordinates))
    segments = np.stack((coordinates[:-1], coordinates[1:]), axis=1)
    cmap = plt.colormaps["viridis"]
    norm = colors.PowerNorm(gamma=.35, vmin=0, vmax=generations[-1])
    segment_colours = cmap(norm(generations[:-1])); segment_colours[:, 3] = .82
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white", "text.color": "#111111",
                         "axes.labelcolor": "#111111", "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(12, 10)); ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(Line3DCollection(segments, colors=segment_colours, linewidths=1.55))
    for setter, limit in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), limits(coordinates)):
        setter(*limit)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=22, azim=42)
    ax.scatter(*coordinates[0], s=78, marker="o", color=cmap(0), edgecolor="black",
               linewidth=.8, depthshade=False, label="Generation 0")
    ax.scatter(*coordinates[-1], s=145, marker="*", color=cmap(1.0), edgecolor="black",
               linewidth=.8, depthshade=False, label="Generation 5,000")
    ax.set_xlabel("Dynamical PC1", labelpad=10); ax.set_ylabel("Dynamical PC2", labelpad=10)
    ax.set_zlabel("Dynamical PC3", labelpad=10); ax.tick_params(colors="#333333", labelsize=8)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 1)); axis.pane.set_edgecolor("#CCCCCC")
        axis._axinfo["grid"]["color"] = (0.86, 0.86, 0.86, 1)
        axis._axinfo["grid"]["linewidth"] = .55
    ax.set_title("Trajectory 1: Convergence to a Point Attractor",
                 fontsize=16, fontweight="bold", pad=18)
    ax.text2D(.5, .965, "Gated-residual dynamics of OCNT-2328519 conditioned on CYP1A2",
              transform=ax.transAxes, ha="center", va="top", fontsize=10, color="#444444")
    ax.legend(loc="upper left", bbox_to_anchor=(.01, .93), frameon=False)
    scalar = ScalarMappable(norm=norm, cmap=cmap); scalar.set_array([])
    colourbar = fig.colorbar(scalar, ax=ax, fraction=.032, pad=.07, shrink=.72)
    colourbar.set_label("Cellular-Automata Generation", labelpad=10)
    colourbar.set_ticks([0, 25, 50, 100, 250, 500, 1000, 2500, 5000])
    fig.text(.5, .025,
             f"The full-state distance to the terminal state falls from {distance[0]:.2f} to {distance[100]:.4f} by generation 100; "
             f"the first three PCs explain {100 * explained.sum():.1f}% of early-trajectory variance.",
             ha="center", fontsize=9, color="#444444")
    fig.subplots_adjust(left=.02, right=.91, bottom=.07, top=.93)
    png = FIGURES / "13_trajectory_01_point_attractor_convergence.png"
    pdf = FIGURES / "13_trajectory_01_point_attractor_convergence.pdf"
    fig.savefig(png, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white"); plt.close(fig)
    return png, pdf


def render_video(coordinates, distance):
    mpl.rcParams["animation.ffmpeg_path"] = str(FFMPEG)
    mpl.rcParams["figure.facecolor"] = INK
    mpl.rcParams["savefig.facecolor"] = INK
    cmap = LinearSegmentedColormap.from_list("cyber_convergence", [MAGENTA, CYAN, LIME])
    frames = 900; fps = 30
    progress = np.linspace(0, 1, frames)
    frame_indices = np.rint(
        (np.exp(7 * progress) - 1) / (np.exp(7) - 1) * (len(coordinates) - 1)
    ).astype(int)
    figure = plt.figure(figsize=(16, 9), facecolor=INK)
    figure.patch.set_facecolor(INK)
    ax = figure.add_subplot(111, projection="3d", facecolor=INK)
    figure.subplots_adjust(left=.01, right=.99, bottom=.01, top=.94)
    for setter, limit in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), limits(coordinates)):
        setter(*limit)
    ax.set_box_aspect((1, 1, 1)); ax.grid(False); ax.set_axis_off()
    ax.set_title("Trajectory 1  |  Graph-CA Point-Attractor Convergence",
                 color=WHITE, fontsize=19, pad=10)
    ax.text2D(.5, .965, "Gated-residual molecular information dynamics",
              transform=ax.transAxes, ha="center", va="top", color=CYAN, fontsize=11)
    history, = ax.plot([], [], [], color=CYAN, lw=.65, alpha=.28)
    trail = Line3DCollection(np.zeros((1, 2, 3)), linewidths=2.4,
                             colors=[(0, 0, 0, 0)])
    ax.add_collection3d(trail)
    head = ax.scatter([], [], [], s=62, color=LIME, edgecolors=WHITE,
                      linewidths=.7, depthshade=False)
    target = ax.scatter(*coordinates[-1], s=210, marker="*", color=MAGENTA,
                        edgecolors=WHITE, linewidths=.9, alpha=.9, depthshade=False)
    status = ax.text2D(.035, .06, "", transform=ax.transAxes, color=WHITE,
                       fontsize=11, family="monospace", alpha=.92)
    note = ax.text2D(.965, .055,
                     "3D PCA of the complete atom × channel state\nAnimation time is expanded during early convergence",
                     transform=ax.transAxes, ha="right", color=WHITE, fontsize=9, alpha=.65)

    def update(frame):
        index = int(frame_indices[frame]); visible = coordinates[:index + 1]
        history.set_data_3d(visible[:, 0], visible[:, 1], visible[:, 2])
        start = max(0, index - max(20, index // 8)); recent = coordinates[start:index + 1]
        if len(recent) > 1:
            pieces = np.stack((recent[:-1], recent[1:]), axis=1)
            colours = cmap(np.linspace(0, 1, len(pieces)))
            colours[:, 3] = np.linspace(.15, 1, len(pieces))
            trail.set_segments(pieces); trail.set_color(colours)
        point = coordinates[index]; head._offsets3d = ([point[0]], [point[1]], [point[2]])
        status.set_text(f"generation  {index:5d}\nfull-state distance to attractor  {distance[index]:.6f}\nperturbation slope  −0.0272 / generation")
        ax.view_init(elev=22 + 3 * np.sin(2 * np.pi * frame / frames),
                     azim=42 + 80 * frame / frames, roll=0)
        return history, trail, head, target, status, note

    update(frames - 1)
    poster = VIDEOS / "trajectory_01_point_attractor_convergence.png"
    video = VIDEOS / "trajectory_01_point_attractor_convergence.mp4"
    figure.savefig(poster, dpi=120, facecolor=INK)
    animation = FuncAnimation(figure, update, frames=frames, interval=1000 / fps,
                              blit=False, repeat=False)
    writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=8000,
                          extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
                          metadata={"title": "Graph-CA point-attractor convergence",
                                    "artist": "Strange Matter Engine"})
    animation.save(video, writer=writer, dpi=120,
                   savefig_kwargs={"facecolor": INK}); plt.close(figure)
    return video, poster


def main():
    FIGURES.mkdir(parents=True, exist_ok=True); VIDEOS.mkdir(parents=True, exist_ok=True)
    data, coordinates, distance, explained = coordinates_and_distance()
    for path in (*render_static(data, coordinates, distance, explained),
                 *render_video(coordinates, distance)):
        print(path)


if __name__ == "__main__":
    main()
