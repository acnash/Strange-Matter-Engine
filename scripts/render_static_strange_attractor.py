#!/usr/bin/env python3
"""Render the complete trajectory 7 attractor with a colourblind-safe time gradient."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Line3DCollection


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "results" / "long_horizon_attractor_campaign_v1" / "case_data" /
          "07_kuramoto_sakaguchi_OCNT-0494110_CYP2C9.npz")
OUTPUT = ROOT / "results" / "long_horizon_attractor_campaign_v1" / "figures"


def equal_limits(coordinates):
    low = np.quantile(coordinates, 0.0025, axis=0)
    high = np.quantile(coordinates, 0.9975, axis=0)
    centre = (low + high) / 2
    radius = 0.55 * float(np.max(high - low))
    return [(value - radius, value + radius) for value in centre]


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = np.load(SOURCE)
    coordinates = data["pca_coordinates"][:, :3].astype(np.float64)
    coordinates -= np.median(coordinates, axis=0, keepdims=True)
    coordinates /= np.std(coordinates, axis=0, keepdims=True).clip(1e-12)
    generations = np.arange(1000, 1000 + len(coordinates))
    segments = np.stack((coordinates[:-1], coordinates[1:]), axis=1)

    cmap = plt.colormaps["viridis"]
    norm = colors.Normalize(vmin=generations[0], vmax=generations[-1])
    segment_colours = cmap(norm(generations[:-1]))
    segment_colours[:, 3] = 0.78

    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "text.color": "#111111",
        "axes.labelcolor": "#111111", "font.family": "DejaVu Sans",
    })
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    collection = Line3DCollection(segments, colors=segment_colours,
                                  linewidths=0.78)
    ax.add_collection3d(collection)
    for setter, limit in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim),
                             equal_limits(coordinates)):
        setter(*limit)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=23, azim=178)
    ax.scatter(*coordinates[0], s=75, marker="o", color=cmap(0.0),
               edgecolor="black", linewidth=0.8, depthshade=False, label="Generation 1,000")
    ax.scatter(*coordinates[-1], s=135, marker="*", color=cmap(1.0),
               edgecolor="black", linewidth=0.8, depthshade=False, label="Generation 5,000")
    ax.set_xlabel("Dynamical PC1", labelpad=10)
    ax.set_ylabel("Dynamical PC2", labelpad=10)
    ax.set_zlabel("Dynamical PC3", labelpad=10)
    ax.tick_params(colors="#333333", labelsize=8)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 1))
        axis.pane.set_edgecolor("#CCCCCC")
        axis._axinfo["grid"]["color"] = (0.86, 0.86, 0.86, 1)
        axis._axinfo["grid"]["linewidth"] = 0.55
    ax.set_title("Trajectory 7: Graph-CA Hyperchaotic Strange Attractor",
                 fontsize=16, fontweight="bold", pad=18)
    ax.text2D(.5, .965,
              "Complete post-burn-in orbit of OCNT-0494110 conditioned on CYP2C9",
              transform=ax.transAxes, ha="center", va="top", fontsize=10,
              color="#444444")
    ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.93), frameon=False)
    scalar = ScalarMappable(norm=norm, cmap=cmap); scalar.set_array([])
    colourbar = fig.colorbar(scalar, ax=ax, fraction=.032, pad=.07, shrink=.72)
    colourbar.set_label("Cellular-Automata Generation", labelpad=10)
    colourbar.outline.set_edgecolor("#777777")
    fig.text(.5, .025,
             "Colour records computational time from violet-blue to yellow; coordinates are a 3D PCA projection of the full atom-by-channel state.",
             ha="center", fontsize=9, color="#444444")
    fig.subplots_adjust(left=.02, right=.91, bottom=.07, top=.93)

    png = OUTPUT / "12_trajectory_07_complete_attractor_time_gradient.png"
    pdf = OUTPUT / "12_trajectory_07_complete_attractor_time_gradient.pdf"
    fig.savefig(png, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
