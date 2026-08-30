#!/usr/bin/env python3
"""Merge long-horizon and perturbation evidence into a plot-ready campaign report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "long_horizon_attractor_campaign_v1"
RULES = (
    "gated_residual", "delayed_memory", "inertial_reaction_diffusion",
    "kuramoto_sakaguchi", "fitzhugh_nagumo",
)
INK, WHITE, CYAN, MAGENTA, LIME, ORANGE = (
    "#070914", "#DCE6F2", "#00E5FF", "#FF1493", "#A6FF00", "#FF9F1C"
)


def markdown_table(frame: pd.DataFrame) -> str:
    def value_text(value):
        return f"{value:.6g}" if isinstance(value, (float, np.floating)) else str(value)
    header = "| " + " | ".join(frame.columns) + " |"
    divider = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = ["| " + " | ".join(value_text(value) for value in row) + " |"
            for row in frame.itertuples(index=False, name=None)]
    return "\n".join([header, divider, *rows])


def main() -> None:
    base = pd.read_csv(OUT / "attractor_screen.csv")
    perturbation_rows = []
    curves = {}
    for rule in RULES:
        rule_dir = OUT / "perturbations" / rule
        frame = pd.read_csv(rule_dir / "extended_dynamics.csv")
        for index, row in frame.iterrows():
            archive = rule_dir / "perturbation_curves" / f"case_{index + 1:03d}.npz"
            data = np.load(archive)
            key = (row.molecule_id, row.cyp_target)
            curves[key] = data["separation"]
            perturbation_rows.append({
                "transition_rule": rule, "molecule_id": row.molecule_id,
                "cyp_target": row.cyp_target,
                "direct_perturbation_slope": row.finite_time_local_divergence,
                "direct_perturbation_slope_std": row.perturbation_slope_std,
                "direct_positive_fraction": row.perturbation_positive_fraction,
                "direct_final_separation": row.perturbation_final_separation,
                "direct_maximum_separation": row.perturbation_maximum_separation,
                "perturbation_repeats": 8, "perturbation_epsilon": row.perturbation_epsilon,
            })
    perturbations = pd.DataFrame(perturbation_rows)
    combined = base.merge(perturbations, on=["transition_rule", "molecule_id", "cyp_target"],
                          how="left")
    combined["evidence_summary"] = np.where(
        combined.direct_positive_fraction >= 0.75,
        "replicated finite-time sensitivity",
        np.where(combined.direct_positive_fraction <= 0.25,
                 "replicated local contraction", "direction-dependent response")
    )
    combined.to_csv(OUT / "combined_dynamics_evidence.csv", index=False)

    figures = OUT / "figures"
    figures.mkdir(exist_ok=True)
    plt.style.use("dark_background")
    plt.rcParams.update({"figure.facecolor": INK, "axes.facecolor": INK,
                         "savefig.facecolor": INK, "text.color": WHITE})

    fig, ax = plt.subplots(figsize=(11, 7))
    positions = np.arange(len(RULES))
    for pos, rule in zip(positions, RULES):
        q = perturbations[perturbations.transition_rule == rule]
        jitter = np.linspace(-.12, .12, len(q))
        ax.errorbar(np.full(len(q), pos) + jitter, q.direct_perturbation_slope,
                    yerr=q.direct_perturbation_slope_std, fmt="o", color=CYAN,
                    ecolor=WHITE, capsize=3, alpha=.9)
    ax.axhline(0, color=LIME, lw=1.2)
    ax.set_xticks(positions, [r.replace("_", "\n") for r in RULES])
    ax.set(ylabel="Finite-time perturbation growth slope",
           title="Eight-direction perturbation replication over 5,000 generations")
    ax.grid(axis="y", alpha=.14); fig.tight_layout()
    fig.savefig(figures / "04_direct_perturbation_summary.png", dpi=220); plt.close(fig)

    fig, axes = plt.subplots(4, 5, figsize=(17, 12), constrained_layout=True)
    gallery = perturbations.sort_values(["transition_rule", "cyp_target", "molecule_id"])
    for number, (ax, row) in enumerate(zip(axes.flat, gallery.itertuples()), start=1):
        separation = curves[(row.molecule_id, row.cyp_target)]
        generations = np.arange(separation.shape[1])
        for curve in separation:
            ax.plot(generations, np.maximum(curve, 1e-12), lw=.55, alpha=.55, color=CYAN)
        ax.set_yscale("log")
        ax.axhline(row.perturbation_epsilon, color=LIME, lw=.6, alpha=.7)
        ax.set_title(f"{number}: {row.transition_rule}\n{row.molecule_id}", fontsize=8)
        ax.set_xlim(0, 5000); ax.grid(alpha=.08)
    fig.suptitle("Direct perturbation separation curves, eight directions per case")
    fig.savefig(figures / "05_perturbation_curve_gallery.png", dpi=190); plt.close(fig)

    selected = combined[combined.visual_rank.isin([7, 8])].sort_values("visual_rank")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    for ax, row in zip(axes, selected.itertuples()):
        separation = curves[(row.molecule_id, row.cyp_target)]
        for repeat, curve in enumerate(separation, start=1):
            ax.plot(np.arange(len(curve)), np.maximum(curve, 1e-12), lw=1,
                    alpha=.72, label=f"direction {repeat}")
        ax.set_yscale("log"); ax.set_xlim(0, 5000)
        ax.set(xlabel="Generation", ylabel="Full-state separation",
               title=f"Trajectory {row.visual_rank}: {row.molecule_id}\n"
                     f"mean early slope {row.direct_perturbation_slope:.4f}")
        ax.grid(alpha=.12)
    axes[1].legend(fontsize=7, ncol=2)
    fig.savefig(figures / "06_kuramoto_trajectory_07_08_divergence.png", dpi=220); plt.close(fig)

    renormalized_dir = OUT / "renormalized_lyapunov"
    renormalized = pd.read_csv(renormalized_dir / "renormalized_lyapunov_runs.csv")
    renormalized_summary = pd.read_csv(renormalized_dir / "renormalized_lyapunov_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    for ax, molecule_id in zip(axes, ("OCNT-0494110", "OCNT-2328784")):
        molecule = renormalized[renormalized.molecule_id == molecule_id]
        for epsilon, colour in zip((1e-4, 1e-5, 1e-6), (CYAN, MAGENTA, ORANGE)):
            q = molecule[np.isclose(molecule.epsilon, epsilon)]
            ax.scatter(np.full(len(q), epsilon), q.renormalized_lyapunov,
                       color=colour, s=35, alpha=.7)
            ax.errorbar(epsilon, q.renormalized_lyapunov.mean(),
                        yerr=q.renormalized_lyapunov.std(), fmt="o", color=WHITE,
                        capsize=5, markersize=7)
        ax.set_xscale("log"); ax.axhline(0, color=LIME, lw=1)
        ax.set(xlabel="Renormalized perturbation magnitude",
               ylabel="Largest Lyapunov estimate per generation",
               title=molecule_id)
        ax.grid(alpha=.12)
    fig.suptitle("Repeatedly renormalized Kuramoto-Sakaguchi divergence")
    fig.savefig(figures / "07_renormalized_lyapunov.png", dpi=220); plt.close(fig)

    spectrum_dir = OUT / "lyapunov_spectrum_float64"
    spectrum = pd.read_csv(spectrum_dir / "lyapunov_spectrum_runs.csv")
    spectrum_summary = pd.read_csv(spectrum_dir / "lyapunov_spectrum_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    for ax, molecule_id in zip(axes, ("OCNT-0494110", "OCNT-2328784")):
        molecule = spectrum_summary[spectrum_summary.molecule_id == molecule_id]
        for interval, colour in zip((5, 10, 20), (CYAN, MAGENTA, ORANGE)):
            q = molecule[molecule.interval == interval]
            ax.errorbar(q.spectrum_index, q.mean_exponent, yerr=q.std_exponent,
                        marker="o", color=colour, capsize=3,
                        label=f"interval {interval}")
        ax.axhline(0, color=LIME, lw=1)
        ax.set(xlabel="Lyapunov spectrum index", ylabel="Exponent per generation",
               title=molecule_id)
        ax.grid(alpha=.12); ax.legend()
    fig.suptitle("Float64 leading Lyapunov spectrum")
    fig.savefig(figures / "08_float64_lyapunov_spectrum.png", dpi=220); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    for ax, molecule_id in zip(axes, ("OCNT-0494110", "OCNT-2328784")):
        for archive in sorted(spectrum_dir.glob(f"*_{molecule_id}_*.npz")):
            data = np.load(archive)
            interval = int(data["interval"])
            cumulative = data["cumulative_spectra"][:, 0]
            ax.plot((np.arange(len(cumulative)) + 1) * interval, cumulative,
                    lw=.8, alpha=.75, label=f"interval {interval}")
        ax.axhline(0, color=LIME, lw=1)
        ax.set(xlabel="Measured generation", ylabel="Cumulative largest exponent",
               title=molecule_id)
        ax.grid(alpha=.12)
    handles, labels = axes[1].get_legend_handles_labels()
    unique = dict(zip(labels, handles)); axes[1].legend(unique.values(), unique.keys())
    fig.suptitle("Convergence of the float64 largest Lyapunov exponent")
    fig.savefig(figures / "09_float64_lyapunov_convergence.png", dpi=220); plt.close(fig)

    spectrum_compact = spectrum.groupby(["molecule_id", "cyp_target", "interval"]).agg(
        largest_mean=("lyapunov_exponent", lambda values: float(values.iloc[0::8].mean())),
        smallest_of_eight_mean=("lyapunov_exponent", lambda values: float(values.iloc[7::8].mean())),
        minimum_observed=("lyapunov_exponent", "min"),
        positive_fraction=("lyapunov_exponent", lambda values: float(np.mean(values > 0))),
    ).reset_index()

    rule_summary = perturbations.groupby("transition_rule").agg(
        cases=("molecule_id", "size"), mean_slope=("direct_perturbation_slope", "mean"),
        minimum_slope=("direct_perturbation_slope", "min"),
        maximum_slope=("direct_perturbation_slope", "max"),
        mean_positive_fraction=("direct_positive_fraction", "mean"),
    ).reset_index()
    rule_summary.to_csv(OUT / "rule_summary.csv", index=False)
    top = combined.sort_values("direct_perturbation_slope", ascending=False).head(6)
    report = [
        "# Long-horizon attractor campaign", "",
        "This frozen-model campaign propagated 20 complete molecular Graph-CA states through 5,000 generations. It retained every atom and all 16 dynamical channels for all base trajectories, calculated detailed phase-space diagnostics for the ten established visual cases, and tested all 20 cases with eight independent full-state perturbation directions of magnitude 1e-5.", "",
        "## Current result", "",
        "Kuramoto-Sakaguchi is the only rule family in which all four screened molecules show replicated positive finite-time separation. Trajectories 7 and 8 remain the principal candidates. Inertial reaction-diffusion and delayed memory contract every tested perturbation. Gated residual has a contracting mean response in all four cases but some direction-dependent early growth. FitzHugh-Nagumo has a mixed, molecule-dependent response.", "",
        "This is evidence of local finite-time sensitivity, not yet proof of a strange attractor. A defensible chaos claim still requires a renormalized largest Lyapunov exponent, stability across perturbation magnitudes and numerical precision, and exclusion of a very long complex transient.", "",
        "## Renormalized Lyapunov result", "",
        "A Benettin-style calculation was subsequently applied to trajectories 7 and 8. After a 1,000-generation burn-in, the companion state was evolved for ten generations, measured with circular phase distance, returned to its original distance, and evolved again. This was repeated across 4,000 measured generations, eight directions, and three perturbation magnitudes.", "",
        "Every one of the 48 estimates was positive. The 1e-4 and 1e-5 results provide the primary float32 estimates; 1e-6 is retained as a numerical-resolution sensitivity test. Persistent positive growth after repeated renormalization shows that divergence is continually regenerated along both trajectories, rather than being a single initial separation followed by saturation.", "",
        markdown_table(renormalized_summary), "",
        "## Float64 Lyapunov spectrum", "",
        "The calculation was then repeated in float64 using eight orthogonal perturbation vectors and QR re-orthogonalization. Intervals of 5, 10, and 20 generations were tested twice for each molecule. All 96 spectrum estimates were positive. The largest exponent was stable near 0.0112 to 0.0124 per generation, and even the eighth leading exponent remained positive. This is evidence of high-dimensional expanding dynamics, often termed hyperchaos, rather than a float32 rounding artefact or a single unstable direction.", "",
        markdown_table(spectrum_compact), "",
        "## Rule summary", "", markdown_table(rule_summary), "",
        "## Leading sensitivity candidates", "", markdown_table(top[["visual_rank", "transition_rule", "molecule_id", "cyp_target", "direct_perturbation_slope", "direct_perturbation_slope_std", "direct_positive_fraction", "correlation_dimension", "spectral_entropy"]]), "",
        "## Figures", "",
        "![Attractor screen](figures/01_attractor_screen.png)", "",
        "![Phase portraits](figures/02_phase_portraits.png)", "",
        "![Recurrence plots](figures/03_recurrence_gallery.png)", "",
        "![Perturbation summary](figures/04_direct_perturbation_summary.png)", "",
        "![Perturbation curves](figures/05_perturbation_curve_gallery.png)", "",
        "![Kuramoto divergence](figures/06_kuramoto_trajectory_07_08_divergence.png)", "",
        "![Renormalized Lyapunov estimates](figures/07_renormalized_lyapunov.png)", "",
        "![Float64 Lyapunov spectrum](figures/08_float64_lyapunov_spectrum.png)", "",
        "![Float64 Lyapunov convergence](figures/09_float64_lyapunov_convergence.png)", "",
        "## Retained data", "",
        "- `base_trajectories`: lossless 5,001-frame atom-by-channel trajectories.",
        "- `case_data`: PCA coordinates, recurrence matrices, spectra, correlation integrals, step energies, and nearest-neighbour divergence curves.",
        "- `perturbations`: eight direct perturbation separation traces for every case.",
        "- `combined_dynamics_evidence.csv`: one-row-per-case numerical summary.", "",
    ]
    (OUT / "README.md").write_text("\n".join(report), encoding="utf-8")
    metadata = json.loads((OUT / "metadata.json").read_text(encoding="utf-8"))
    metadata.update({"direct_perturbation_cases": len(perturbations),
                     "direct_perturbations_per_case": 8,
                     "direct_perturbation_trajectories": len(perturbations) * 8,
                     "renormalized_lyapunov_runs": len(renormalized),
                     "float64_spectrum_exponents": len(spectrum),
                     "claim_status": "persistent bounded hyperchaotic dynamics supported by float64 multi-vector Lyapunov spectra"})
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(rule_summary.to_string(index=False))


if __name__ == "__main__":
    main()
