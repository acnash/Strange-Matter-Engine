#!/usr/bin/env python3
"""Comparative plots for the tuned 200-generation Graph-CA prototype."""

from pathlib import Path
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TARGET_RUN = os.environ.get("SME_ANALYSIS_RUN", "graph_ca_inertial_200_tuned_prototype")
TUNING_PREFIX = os.environ.get("SME_TUNING_PREFIX", "graph_ca_inertial_200_tuning")
OUT = ROOT / "results" / TARGET_RUN
FIG = OUT / "figures"
P1 = ROOT / "results" / "graph_ca_visual_prototype"
P2 = ROOT / "results" / "graph_ca_inertial_100_prototype"
P3 = ROOT / "results" / "graph_ca_inertial_200_tuned_prototype"
CYAN, MAGENTA, LIME, VIOLET, ORANGE = "#00e5ff", "#ff1493", "#a6ff00", "#6c4cff", "#ff9f1c"


def rmse(frame, prediction="predicted_pic50"):
    return float(np.sqrt(np.mean((frame[prediction] - frame.experimental_pic50) ** 2)))


def main():
    plt.style.use("dark_background")
    FIG.mkdir(parents=True, exist_ok=True)
    target_metrics = json.loads((OUT / "metrics.json").read_text())
    generations = target_metrics["generations"]
    target_predictions = pd.read_csv(OUT / "validation_predictions.csv")
    validation_only = target_predictions[target_predictions.split == "validation"].copy()
    validation_only.to_csv(OUT / "validation_set_predictions.csv", index=False)
    runs = [("Prototype 1\n16-gen gated", P1),
            ("Prototype 2\n100-gen inertial", P2)]
    if OUT.resolve() != P3.resolve():
        runs.append(("Prototype 3\n200-gen tuned", P3))
        target_number = 4
    else:
        target_number = 3
    runs.append((f"Prototype {target_number}\n{generations}-gen tuned", OUT))
    metrics = [json.loads((path / "metrics.json").read_text()) for _, path in runs]

    tuning = []
    for idx in range(1, 7):
        path = ROOT / "results" / f"{TUNING_PREFIX}_{idx:02d}" / "metrics.json"
        item = json.loads(path.read_text()); item["candidate"] = idx; tuning.append(item)
    fig, ax = plt.subplots(figsize=(9, 5))
    values = [x["restored_validation_rmse"] for x in tuning]
    bars = ax.bar([f"C{x['candidate']}" for x in tuning], values,
                  color=[MAGENTA if i == int(np.argmin(values)) else VIOLET for i in range(6)])
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set(ylim=(min(values) - .02, max(values) + .03), ylabel="Tuning-subset RMSE (pIC50)",
           title="Multi-fidelity hyperparameter screen")
    ax.grid(axis="y", alpha=.15); fig.tight_layout(); fig.savefig(FIG / "05_hyperparameter_screen.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    vals = [m["restored_validation_rmse"] for m in metrics]
    bars = ax.bar([x[0] for x in runs], vals,
                  color=[CYAN, VIOLET, ORANGE, MAGENTA][-len(runs):])
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set(ylabel="Grouped-validation RMSE (pIC50)", title="Prototype comparison",
           ylim=(0, max(vals) * 1.18)); ax.grid(axis="y", alpha=.15)
    fig.tight_layout(); fig.savefig(FIG / "06_prototype_rmse_comparison.png", dpi=180); plt.close(fig)

    per_cyp = []
    for label, path in runs:
        frame = pd.read_csv(path / "validation_predictions.csv")
        frame = frame[frame.split == "validation"]
        for cyp, group in frame.groupby("cyp_target"):
            per_cyp.append({"prototype": label.split("\n")[0], "cyp": cyp, "rmse": rmse(group)})
    p = pd.DataFrame(per_cyp).pivot(index="cyp", columns="prototype", values="rmse")
    ax = p.plot.bar(figsize=(10, 5), color=[CYAN, VIOLET, ORANGE, MAGENTA][:len(runs)])
    ax.set(ylabel="Grouped-validation RMSE (pIC50)", title="Validation error by CYP")
    ax.tick_params(axis="x", rotation=0); ax.grid(axis="y", alpha=.15); ax.figure.tight_layout()
    ax.figure.savefig(FIG / "07_per_cyp_rmse.png", dpi=180); plt.close(ax.figure)

    p1 = pd.read_csv(P1 / "blinded_test_predictions.csv")
    p3 = pd.read_csv(OUT / "blinded_test_predictions.csv")
    joined = p1.merge(p3, on=["molecule_id", "cyp_target"], suffixes=("_p1", "_p3"))
    fig, ax = plt.subplots(figsize=(6, 6))
    for cyp, colour in zip(sorted(joined.cyp_target.unique()), [CYAN, LIME, MAGENTA, ORANGE]):
        q = joined[joined.cyp_target == cyp]
        ax.scatter(q.predicted_pic50_p1, q.predicted_pic50_p3, s=10, alpha=.35, c=colour, label=cyp)
    lo = min(joined.predicted_pic50_p1.min(), joined.predicted_pic50_p3.min())
    hi = max(joined.predicted_pic50_p1.max(), joined.predicted_pic50_p3.max())
    ax.plot([lo, hi], [lo, hi], "--", c="white"); ax.set(xlabel="Prototype 1 prediction",
        ylabel=f"Prototype {target_number} prediction", title="Blinded prediction comparison")
    ax.legend(); ax.grid(alpha=.15); fig.tight_layout(); fig.savefig(FIG / "08_blinded_prediction_comparison.png", dpi=180); plt.close(fig)

    scores = pd.read_csv(OUT / "trajectory_novelty_scores.csv")
    selected = scores.selected_for_visualisation.astype(str).str.lower().eq("true")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(scores.recurrence_ratio, scores.late_motion, s=9, c=VIOLET, alpha=.22, label="All 3,000")
    ax.scatter(scores.loc[selected, "recurrence_ratio"], scores.loc[selected, "late_motion"],
               s=55, c=MAGENTA, edgecolors=CYAN, linewidths=.8, label="Selected 20")
    ax.set(xlabel="Approximate recurrence ratio", ylabel="Mean late step size",
           title=f"{generations}-generation dynamical screening")
    ax.grid(alpha=.15); ax.legend(); fig.tight_layout(); fig.savefig(FIG / "09_dynamical_screening.png", dpi=180); plt.close(fig)

    with (OUT / "selected_trajectories.pkl").open("rb") as handle:
        records = pickle.load(handle)
    ranked = sorted(records, key=lambda r: r["late_motion"], reverse=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    for record, colour in zip(ranked[:5], [LIME, CYAN, MAGENTA, VIOLET, ORANGE]):
        mean_state = record["trajectory"].mean(axis=1)
        step = np.linalg.norm(np.diff(mean_state, axis=0), axis=1)
        ax.plot(np.arange(1, len(step) + 1), step, c=colour, lw=2,
                label=f'{record["molecule_id"]} / {record["cyp_target"]}')
    ax.set_yscale("log"); ax.set(xlabel="Generation", ylabel="Mean-state step size (log scale)",
        title="Five most persistent selected trajectories")
    ax.grid(alpha=.15); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIG / "10_persistent_transients.png", dpi=180); plt.close(fig)

    perturb = pd.read_csv(OUT / "perturbation_analysis.csv").sort_values("finite_time_lyapunov")
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = perturb.molecule_id + " / " + perturb.cyp_target
    colours = [MAGENTA if x > 0 else CYAN for x in perturb.finite_time_lyapunov]
    ax.barh(labels, perturb.finite_time_lyapunov, color=colours)
    ax.axvline(0, c="white", ls="--"); ax.set(xlabel="Finite-time local divergence slope",
        title="Perturbation response over generations 1–30")
    ax.grid(axis="x", alpha=.15); fig.tight_layout(); fig.savefig(FIG / "11_finite_time_lyapunov.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, record, colour in zip(axes.flat, ranked[:4], [LIME, CYAN, MAGENTA, ORANGE]):
        mean_state = record["trajectory"].mean(axis=1)
        centred = mean_state - mean_state.mean(0)
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        xy = centred @ vt[:2].T
        ax.plot(xy[:, 0], xy[:, 1], c=colour, lw=1.6)
        ax.scatter(xy[0, 0], xy[0, 1], c=CYAN, s=35, label="start")
        ax.scatter(xy[-1, 0], xy[-1, 1], c=MAGENTA, s=35, label="end")
        ax.set(title=f'{record["molecule_id"]} / {record["cyp_target"]}', xlabel="PC1", ylabel="PC2")
        ax.grid(alpha=.12)
    axes.flat[0].legend(); fig.suptitle("State-space projections of persistent trajectories")
    fig.tight_layout(); fig.savefig(FIG / "12_state_space_projections.png", dpi=180); plt.close(fig)


if __name__ == "__main__":
    main()
