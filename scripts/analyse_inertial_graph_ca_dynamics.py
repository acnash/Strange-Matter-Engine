#!/usr/bin/env python3
"""Create dynamical diagnostic figures for the 100-generation prototype."""

from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "graph_ca_inertial_100_prototype"
FIG = OUT / "figures"
CYAN, MAGENTA, LIME, VIOLET = "#00e5ff", "#ff1493", "#a6ff00", "#6c4cff"


def main():
    plt.style.use("dark_background")
    scores = pd.read_csv(OUT / "trajectory_novelty_scores.csv")
    selected = scores.selected_for_visualisation.astype(str).str.lower().eq("true")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(scores.recurrence_ratio, scores.late_motion, s=9, c=VIOLET,
               alpha=0.22, label="All blinded trajectories")
    ax.scatter(scores.loc[selected, "recurrence_ratio"], scores.loc[selected, "late_motion"],
               s=55, c=MAGENTA, edgecolors=CYAN, linewidths=0.8, label="Selected 20")
    ax.set(xlabel="Approximate recurrence ratio (smaller is closer)",
           ylabel="Mean late step size", title="Dynamical screening landscape")
    ax.grid(alpha=0.15); ax.legend(); fig.tight_layout()
    fig.savefig(FIG / "05_dynamical_screening.png", dpi=180); plt.close(fig)

    with (OUT / "selected_trajectories.pkl").open("rb") as handle:
        records = pickle.load(handle)
    ranked = sorted(records, key=lambda r: r["late_motion"], reverse=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    colours = (LIME, CYAN, MAGENTA, VIOLET, "#ff9f1c")
    for record, colour in zip(ranked[:5], colours):
        mean_state = record["trajectory"].mean(axis=1)
        step = np.linalg.norm(np.diff(mean_state, axis=0), axis=1)
        ax.plot(np.arange(1, len(step) + 1), step, color=colour, lw=2,
                label=f'{record["molecule_id"]} / {record["cyp_target"]}')
    ax.set_yscale("log")
    ax.set(xlabel="Generation", ylabel="Molecular mean-state step size (log scale)",
           title="Five most persistent selected trajectories")
    ax.grid(alpha=0.15); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(FIG / "06_persistent_transients.png", dpi=180); plt.close(fig)


if __name__ == "__main__":
    main()
