#!/usr/bin/env python3
"""Render publication-style structure–dynamics figures from completed tables."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "structure_dynamics_publication_v1"
FIG = OUT / "figures"
BLUE, RED, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#555555"


def label(value):
    replacements = {
        "3d": "3D", "tpsa": "TPSA", "hba": "HBA", "hbd": "HBD",
        "r2": "R²", "cyp": "CYP",
    }
    words = str(value).replace("_", " ").split()
    return " ".join(replacements.get(word.lower(), word.capitalize()) for word in words)


def target_label(value):
    text = str(value).replace("_", " ")
    for source, replacement in (("SINGLE", "Single"), ("DOUBLE", "Double"),
                                ("TRIPLE", "Triple"), ("AROMATIC", "Aromatic"),
                                ("single", "Single"), ("double", "Double"),
                                ("triple", "Triple"), ("aromatic", "Aromatic")):
        text = text.replace(source, replacement)
    return text[:1].upper() + text[1:]


def style_axis(ax):
    ax.set_facecolor("white")
    ax.tick_params(colors="black", labelsize=9)
    for spine in ax.spines.values(): spine.set_color("#333333")
    ax.grid(axis="x", color="#DDDDDD", linewidth=.7, alpha=.8)
    ax.set_axisbelow(True)


def panel_correlations(ax, bootstrap):
    top = bootstrap.head(15).sort_values("spearman_rho")
    xerr = np.vstack((top.spearman_rho - top.cluster_bootstrap_low,
                      top.cluster_bootstrap_high - top.spearman_rho))
    ax.errorbar(top.spearman_rho, range(len(top)), xerr=xerr, fmt="o",
                color=BLUE, ecolor=GREY, capsize=2)
    ax.set_yticks(range(len(top)), [label(x) for x in top.feature])
    ax.axvline(0, color="black", alpha=.5, linewidth=1)
    ax.set_xlabel("Spearman Correlation With Largest Lyapunov Exponent")
    ax.set_title("A  Structural Associations\n95% Scaffold-Cluster Bootstrap Intervals",
                 loc="left", fontweight="bold")
    style_axis(ax)


def panel_importance(ax, importance):
    top = importance.head(15).sort_values("mean_importance")
    ax.barh([label(x) for x in top.feature], top.mean_importance,
            xerr=top.std_importance, color=BLUE, alpha=.85, capsize=2)
    ax.axvline(0, color="black", alpha=.5, linewidth=1)
    ax.set_xlabel("Held-Out MAE Increase After Permutation")
    ax.set_title("B  Scaffold-Held-Out Descriptor Importance",
                 loc="left", fontweight="bold")
    style_axis(ax)


def panel_strongest(ax, interventions):
    effects = interventions[interventions.intervention != "baseline"].copy()
    strongest = effects.reindex(
        effects.lyapunov_change.abs().sort_values(ascending=False).index
    ).head(20)
    labels = [f"{m.replace('OCNT-', '')} | {label(kind)} | {target_label(target)}"
              for m, kind, target in zip(strongest.molecule_id,
                                         strongest.intervention, strongest.target)]
    colours = np.where(strongest.lyapunov_change > 0, RED, BLUE)
    ax.barh(range(len(strongest)), strongest.lyapunov_change, color=colours, alpha=.88)
    ax.set_yticks(range(len(strongest)), labels); ax.axvline(0, color="black", alpha=.6)
    ax.invert_yaxis(); ax.set_xlabel("Change in Largest Lyapunov Exponent per Generation")
    ax.set_title("C  Strongest Causal Structural Interventions",
                 loc="left", fontweight="bold")
    ax.legend(handles=[Patch(facecolor=RED, label="Increased Divergence"),
                       Patch(facecolor=BLUE, label="Reduced Divergence")],
              loc="lower right", frameon=False, fontsize=8)
    style_axis(ax)


def panel_families(ax, interventions):
    effects = interventions[interventions.intervention != "baseline"].copy()
    molecules = list(effects.groupby(["molecule_id", "cyp_target"]).groups)
    families = sorted(effects.intervention.unique())
    positions, data, colours, tick_labels = [], [], [], []
    cursor = 1
    palette = (BLUE, GREEN)
    for molecule_index, (molecule, cyp) in enumerate(molecules):
        group = effects[(effects.molecule_id == molecule) & (effects.cyp_target == cyp)]
        for family in families:
            values = group[group.intervention == family].lyapunov_change
            if len(values):
                positions.append(cursor); data.append(values); colours.append(palette[molecule_index])
                tick_labels.append(f"{molecule.replace('OCNT-', '')}\n{label(family)}")
                cursor += 1
        cursor += 1
    boxes = ax.boxplot(data, positions=positions, patch_artist=True, widths=.7,
                       medianprops={"color": "black"})
    for patch, colour in zip(boxes["boxes"], colours):
        patch.set_facecolor(colour); patch.set_alpha(.72)
    ax.set_xticks(positions, tick_labels, rotation=45, ha="right", fontsize=8)
    ax.axhline(0, color="black", alpha=.6)
    ax.set_ylabel("Change in Largest Lyapunov Exponent per Generation")
    ax.set_title("D  Intervention Families for Trajectories 7 and 8",
                 loc="left", fontweight="bold")
    ax.grid(axis="y", color="#DDDDDD", linewidth=.7); ax.set_axisbelow(True)
    ax.set_facecolor("white")
    for spine in ax.spines.values(): spine.set_color("#333333")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    bootstrap = pd.read_csv(OUT / "scaffold_bootstrap_correlations.csv")
    importance = pd.read_csv(OUT / "descriptor_permutation_importance.csv")
    interventions = pd.read_csv(OUT / "causal_interventions_with_effects.csv")
    plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white",
                         "axes.labelcolor": "black", "text.color": "black",
                         "font.family": "DejaVu Sans"})
    panels = [
        (panel_correlations, bootstrap, "01_structure_correlations_publication.png", (10, 8)),
        (panel_importance, importance, "02_scaffold_permutation_importance_publication.png", (10, 8)),
        (panel_strongest, interventions, "03_causal_intervention_effects_publication.png", (13, 9)),
        (panel_families, interventions, "04_intervention_families_publication.png", (13, 8)),
    ]
    for function, data, filename, size in panels:
        fig, ax = plt.subplots(figsize=size); function(ax, data)
        fig.tight_layout(); fig.savefig(FIG / filename, dpi=300, bbox_inches="tight"); plt.close(fig)

    fig = plt.figure(figsize=(22, 18), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1, 1.15), height_ratios=(1, 1.05),
                           wspace=.28, hspace=.18)
    panel_correlations(fig.add_subplot(grid[0, 0]), bootstrap)
    panel_importance(fig.add_subplot(grid[0, 1]), importance)
    panel_strongest(fig.add_subplot(grid[1, 0]), interventions)
    panel_families(fig.add_subplot(grid[1, 1]), interventions)
    fig.suptitle("Molecular Structure Governs Emergent Graph-CA Dynamics",
                 fontsize=22, fontweight="bold")
    fig.savefig(FIG / "structure_dynamics_publication_composite.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / "structure_dynamics_publication_composite.pdf",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(FIG / "structure_dynamics_publication_composite.png")


if __name__ == "__main__":
    main()
