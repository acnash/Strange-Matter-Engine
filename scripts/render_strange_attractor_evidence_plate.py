#!/usr/bin/env python3
"""Render the complementary evidence supporting the strange-attractor claim."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "results" / "long_horizon_attractor_campaign_v1"
OUTPUT = CAMPAIGN / "figures"
BLUE, ORANGE, GREY, BLACK = "#0072B2", "#D55E00", "#6B7280", "#171717"
MOLECULES = {
    "OCNT-0494110": ("Trajectory 7 · OCNT-0494110 · CYP2C9", BLUE),
    "OCNT-2328784": ("Trajectory 8 · OCNT-2328784 · CYP1A2", ORANGE),
}


def style_axis(ax, letter, title):
    ax.set_title(f"{letter}  |  {title}", loc="left", fontsize=12,
                 fontweight="bold", pad=10)
    ax.grid(color="#D9D9D9", linewidth=.65, alpha=.75)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


def main():
    renorm = pd.read_csv(CAMPAIGN / "renormalized_lyapunov" /
                         "renormalized_lyapunov_summary.csv")
    spectrum = pd.read_csv(CAMPAIGN / "lyapunov_spectrum_float64" /
                           "lyapunov_spectrum_summary.csv")
    basin = pd.read_csv(CAMPAIGN / "attractor_basin_float64" / "basin_runs.csv")

    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white", "font.family": "DejaVu Sans",
                         "text.color": BLACK, "axes.labelcolor": BLACK})
    fig, axes = plt.subplots(2, 2, figsize=(15.2, 11.2), facecolor="white")

    ax = axes[0, 0]
    for molecule, (label, colour) in MOLECULES.items():
        group = renorm[renorm.molecule_id == molecule].sort_values("epsilon")
        ax.errorbar(group.epsilon, group.mean_lyapunov, yerr=group.std_lyapunov,
                    marker="o", ms=6, capsize=3, lw=1.8, color=colour, label=label)
    ax.axhline(0, color=BLACK, lw=1)
    ax.set_xscale("log"); ax.set_xlabel("Perturbation magnitude, ε")
    ax.set_ylabel("Renormalized largest Lyapunov exponent")
    style_axis(ax, "A", "Continually regenerated divergence")
    ax.legend(frameon=False, fontsize=8, loc="best")

    ax = axes[0, 1]
    chosen = spectrum[spectrum.interval == 10]
    offsets = {"OCNT-0494110": -.08, "OCNT-2328784": .08}
    for molecule, (label, colour) in MOLECULES.items():
        group = chosen[chosen.molecule_id == molecule].sort_values("spectrum_index")
        ax.errorbar(group.spectrum_index + offsets[molecule], group.mean_exponent,
                    yerr=group.std_exponent, marker="o", ms=5, capsize=2.5,
                    lw=1.6, color=colour, label=label)
    ax.axhline(0, color=BLACK, lw=1)
    ax.set_xticks(range(1, 9)); ax.set_xlabel("Lyapunov-spectrum index")
    ax.set_ylabel("Lyapunov exponent per generation")
    style_axis(ax, "B", "Eight positive Lyapunov exponents")

    ax = axes[1, 0]
    largest = spectrum[spectrum.spectrum_index == 1]
    for molecule, (label, colour) in MOLECULES.items():
        group = largest[largest.molecule_id == molecule].sort_values("interval")
        ax.errorbar(group.interval, group.mean_exponent, yerr=group.std_exponent,
                    marker="o", ms=6, capsize=3, lw=1.8, color=colour, label=label)
    ax.axhline(0, color=BLACK, lw=1)
    ax.set_xticks([5, 10, 20]); ax.set_xlabel("Renormalization interval (generations)")
    ax.set_ylabel("Largest Lyapunov exponent")
    style_axis(ax, "C", "Exponent stability across calculation intervals")

    ax = axes[1, 1]
    radius_positions = np.arange(4)
    radii = sorted(basin.radius.unique())
    width = .16
    rng = np.random.default_rng(44)
    for molecule_index, (molecule, (label, colour)) in enumerate(MOLECULES.items()):
        for index, radius in enumerate(radii):
            values = basin[(basin.molecule_id == molecule) &
                           (basin.radius == radius)].sliced_distance_ratio.to_numpy()
            x = radius_positions[index] + (molecule_index - .5) * .28
            jitter = rng.uniform(-.045, .045, size=len(values))
            ax.scatter(np.full(len(values), x) + jitter, values, s=22, alpha=.75,
                       color=colour, edgecolor="white", linewidth=.35)
            ax.plot([x - width / 2, x + width / 2], [values.mean(), values.mean()],
                    color=BLACK, lw=2)
    ax.axhline(1, color=BLACK, lw=1, ls="--", label="No contraction")
    ax.set_xticks(radius_positions, [str(value) for value in radii])
    ax.set_xlabel("Initial-condition perturbation radius")
    ax.set_ylabel("Late-to-early distribution distance ratio")
    ax.set_ylim(0, 1.12)
    style_axis(ax, "D", "Bounded basin return across 64 replicated starts")
    handles = [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=colour,
                          markeredgecolor="none", label=label)
               for label, colour in MOLECULES.values()]
    handles.append(plt.Line2D([0], [0], color=BLACK, ls="--", label="No contraction"))
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower right")

    fig.suptitle("Independent Evidence for Graph-CA Strange Attractors",
                 fontsize=17, fontweight="bold", y=.985)
    fig.text(.5, .018,
             "Positive divergence is regenerated after repeated renormalization, spans the leading spectrum, remains stable across numerical intervals, and occurs within a bounded attracting basin.",
             ha="center", fontsize=9, color="#444444")
    fig.subplots_adjust(left=.08, right=.98, bottom=.075, top=.93, hspace=.30, wspace=.23)
    png = OUTPUT / "18_strange_attractor_evidence_plate.png"
    pdf = OUTPUT / "18_strange_attractor_evidence_plate.pdf"
    fig.savefig(png, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(png); print(pdf)


if __name__ == "__main__":
    main()
