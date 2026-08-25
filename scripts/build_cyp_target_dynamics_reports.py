#!/usr/bin/env python3
"""Build one CYP-target dynamical-regime PDF for each production rule."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)
from scipy.stats import f_oneway
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, silhouette_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = ROOT / "output" / "pdf"
ANALYSIS = RESULTS / "cyp_target_dynamical_regimes_v1"
TRAIN_CSV = ROOT / "data" / "openadmet-cyp-challenge-2026" / "cyp-challenge-TRAIN_inhibition.csv"

FEATURES = ["recurrence_ratio", "late_motion", "spectral_entropy"]
FEATURE_LABELS = {
    "recurrence_ratio": "Recurrence ratio",
    "late_motion": "Late motion",
    "spectral_entropy": "Spectral entropy",
}
TARGETS = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
TARGET_COLOURS = {
    "CYP1A2": "#27E1FF",
    "CYP2C9": "#FF3CAC",
    "CYP2D6": "#B6FF3B",
    "CYP3A4": "#FF9F1C",
}
INK, PANEL, CYAN, MAGENTA, LIME, ORANGE, WHITE, GREY = (
    "#070914", "#11152A", "#27E1FF", "#FF3CAC", "#B6FF3B", "#FF9F1C",
    "#DCE6F2", "#65758B",
)


def pretty_rule(rule: str) -> str:
    names = {
        "gray_scott": "Gray-Scott",
        "coupled_map": "Coupled map",
        "gated_residual": "Gated residual",
        "inertial_reaction_diffusion": "Inertial reaction-diffusion",
        "activator_inhibitor": "Activator-inhibitor",
        "damped_symplectic": "Damped symplectic",
        "fitzhugh_nagumo": "FitzHugh-Nagumo",
        "kuramoto_sakaguchi": "Kuramoto-Sakaguchi",
        "conservative_graph_flux": "Conservative graph flux",
        "delayed_memory": "Delayed memory",
    }
    return names.get(rule, rule.replace("_", " ").title())


def set_plot_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": INK, "axes.facecolor": INK, "savefig.facecolor": INK,
        "axes.edgecolor": GREY, "axes.labelcolor": WHITE, "axes.titlecolor": WHITE,
        "xtick.color": WHITE, "ytick.color": WHITE, "text.color": WHITE,
        "grid.color": GREY, "font.family": "DejaVu Sans", "font.size": 9,
    })


def standardize(frame: pd.DataFrame, columns=FEATURES) -> np.ndarray:
    return StandardScaler().fit_transform(frame[list(columns)].to_numpy(float))


def eta_squared(values: np.ndarray, labels: np.ndarray) -> float:
    grand = float(np.mean(values))
    ss_total = float(np.sum((values - grand) ** 2))
    if ss_total == 0:
        return 0.0
    ss_between = sum(np.sum(labels == target) *
                     (float(np.mean(values[labels == target])) - grand) ** 2
                     for target in np.unique(labels))
    return float(ss_between / ss_total)


def pseudo_f(x: np.ndarray, labels: np.ndarray) -> float:
    overall = x.mean(axis=0)
    groups = np.unique(labels)
    between = sum(np.sum(labels == g) * np.sum((x[labels == g].mean(axis=0) - overall) ** 2)
                  for g in groups)
    within = sum(np.sum((x[labels == g] - x[labels == g].mean(axis=0)) ** 2)
                 for g in groups)
    return float((between / (len(groups) - 1)) /
                 max(within / (len(x) - len(groups)), 1e-12))


def permutation_p(x: np.ndarray, labels: np.ndarray, seed=260825, permutations=499) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    observed = pseudo_f(x, labels)
    exceed = sum(pseudo_f(x, rng.permutation(labels)) >= observed for _ in range(permutations))
    return observed, (exceed + 1) / (permutations + 1)


def cross_validated_accuracy(x: np.ndarray, labels: np.ndarray,
                             groups: np.ndarray) -> tuple[float, float]:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=260825)
    scores = []
    for train, test in splitter.split(x, labels, groups):
        model = make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=2000, class_weight="balanced"
        ))
        model.fit(x[train], labels[train])
        scores.append(balanced_accuracy_score(labels[test], model.predict(x[test])))
    return float(np.mean(scores)), float(np.std(scores))


def confidence_ellipse(ax, x: np.ndarray, y: np.ndarray, colour: str) -> None:
    if len(x) < 3:
        return
    covariance = np.cov(x, y)
    values, vectors = np.linalg.eigh(covariance)
    order = values.argsort()[::-1]
    values, vectors = values[order], vectors[:, order]
    angle = math.degrees(math.atan2(vectors[1, 0], vectors[0, 0]))
    width, height = 2 * 1.52 * np.sqrt(np.maximum(values, 0))  # approximately 80%
    ax.add_patch(Ellipse((x.mean(), y.mean()), width, height, angle=angle,
                         facecolor=colour, edgecolor=colour, alpha=.12, lw=2))
    ax.scatter([x.mean()], [y.mean()], c=colour, marker="X", s=85,
               edgecolor=INK, linewidth=.8, zorder=5)


def molecule_scaffolds() -> dict[str, str]:
    source = pd.read_csv(TRAIN_CSV, usecols=["Molecule_Name", "SMILES"])
    mapping = {}
    for row in source.itertuples(index=False):
        mol = Chem.MolFromSmiles(row.SMILES)
        if mol is None:
            mapping[str(row.Molecule_Name)] = f"INVALID::{row.Molecule_Name}"
            continue
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        mapping[str(row.Molecule_Name)] = scaffold or f"ACYCLIC::{Chem.MolToSmiles(mol)}"
    return mapping


def adjusted_within_molecule(frame: pd.DataFrame) -> pd.DataFrame:
    counts = frame.groupby("molecule_id").size()
    q = frame[frame.molecule_id.isin(counts[counts >= 2].index)].copy()
    for feature in FEATURES:
        centered_x = q[feature] - q.groupby("molecule_id")[feature].transform("mean")
        centered_y = (q.experimental_pic50 -
                      q.groupby("molecule_id").experimental_pic50.transform("mean"))
        denominator = float(np.dot(centered_y, centered_y))
        beta = float(np.dot(centered_y, centered_x) / denominator) if denominator else 0.0
        q[f"adjusted_{feature}"] = centered_x - beta * centered_y
    return q


def calculate_statistics(frame: pd.DataFrame, scaffold_map: dict[str, str]) -> dict:
    frame = frame.dropna(subset=FEATURES + ["cyp_target", "experimental_pic50"]).copy()
    frame["scaffold"] = frame.molecule_id.astype(str).map(scaffold_map).fillna("UNKNOWN")
    x = standardize(frame)
    labels = frame.cyp_target.to_numpy()
    sil = silhouette_score(x, labels)
    p_f, p_value = permutation_p(x, labels)
    molecule_acc = cross_validated_accuracy(frame[FEATURES].to_numpy(), labels,
                                            frame.molecule_id.to_numpy())[0:2]
    scaffold_acc = cross_validated_accuracy(frame[FEATURES].to_numpy(), labels,
                                            frame.scaffold.to_numpy())[0:2]
    adjusted = adjusted_within_molecule(frame)
    adjusted_cols = [f"adjusted_{f}" for f in FEATURES]
    if len(adjusted) > 20 and all(adjusted[c].std() > 0 for c in adjusted_cols):
        adjusted_x = StandardScaler().fit_transform(adjusted[adjusted_cols])
        adjusted_labels = adjusted.cyp_target.to_numpy()
        adjusted_sil = silhouette_score(adjusted_x, adjusted_labels)
        adjusted_f, adjusted_p = permutation_p(adjusted_x, adjusted_labels, seed=260826)
    else:
        adjusted_sil, adjusted_f, adjusted_p = float("nan"), float("nan"), float("nan")
    centroids = pd.DataFrame(x, columns=FEATURES).assign(cyp_target=labels).groupby("cyp_target").mean()
    distance = pd.DataFrame(index=TARGETS, columns=TARGETS, dtype=float)
    for a in TARGETS:
        for b in TARGETS:
            distance.loc[a, b] = float(np.linalg.norm(centroids.loc[a] - centroids.loc[b]))
    return {
        "n_observations": len(frame),
        "n_molecules": int(frame.molecule_id.nunique()),
        "n_scaffolds": int(frame.scaffold.nunique()),
        "target_counts": frame.cyp_target.value_counts().reindex(TARGETS).fillna(0).astype(int).to_dict(),
        "silhouette": float(sil), "pseudo_f": p_f, "permutation_p": p_value,
        "molecule_grouped_balanced_accuracy_mean": molecule_acc[0],
        "molecule_grouped_balanced_accuracy_sd": molecule_acc[1],
        "scaffold_grouped_balanced_accuracy_mean": scaffold_acc[0],
        "scaffold_grouped_balanced_accuracy_sd": scaffold_acc[1],
        "eta_squared": {f: eta_squared(frame[f].to_numpy(), labels) for f in FEATURES},
        "adjusted_n_observations": len(adjusted),
        "adjusted_n_molecules": int(adjusted.molecule_id.nunique()),
        "adjusted_silhouette": float(adjusted_sil),
        "adjusted_pseudo_f": float(adjusted_f),
        "adjusted_permutation_p": float(adjusted_p),
        "centroid_distance": distance.to_dict(),
    }


def scatter_figure(frame: pd.DataFrame, path: Path, rule_name: str) -> None:
    pairs = [("recurrence_ratio", "late_motion"),
             ("recurrence_ratio", "spectral_entropy"),
             ("spectral_entropy", "late_motion")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, (xcol, ycol) in zip(axes, pairs):
        for target in TARGETS:
            q = frame[frame.cyp_target == target]
            colour = TARGET_COLOURS[target]
            ax.scatter(q[xcol], q[ycol], s=11, alpha=.34, c=colour,
                       label=target, edgecolors="none")
            confidence_ellipse(ax, q[xcol].to_numpy(), q[ycol].to_numpy(), colour)
        ax.set_xlabel(FEATURE_LABELS[xcol]); ax.set_ylabel(FEATURE_LABELS[ycol])
        ax.grid(alpha=.15)
        if ycol == "late_motion" and (frame[ycol] > 0).all():
            ax.set_yscale("log")
    axes[0].legend(frameon=False, labelcolor=WHITE, fontsize=8)
    fig.suptitle(f"{rule_name}: CYP-conditioned dynamical geometry", fontsize=16, color=CYAN)
    fig.tight_layout(rect=(0, 0, 1, .94)); fig.savefig(path, dpi=220); plt.close(fig)


def distribution_figure(frame: pd.DataFrame, path: Path, rule_name: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, feature in zip(axes, FEATURES):
        groups = [frame.loc[frame.cyp_target == t, feature].to_numpy() for t in TARGETS]
        parts = ax.violinplot(groups, showmeans=False, showmedians=True, widths=.82)
        for body, target in zip(parts["bodies"], TARGETS):
            body.set_facecolor(TARGET_COLOURS[target]); body.set_edgecolor(TARGET_COLOURS[target]); body.set_alpha(.55)
        for key in ("cmedians", "cbars", "cmins", "cmaxes"):
            parts[key].set_color(WHITE); parts[key].set_linewidth(.8)
        ax.set_xticks(range(1, 5), TARGETS, rotation=24)
        ax.set_ylabel(FEATURE_LABELS[feature]); ax.grid(axis="y", alpha=.15)
        if feature == "late_motion" and (frame[feature] > 0).all(): ax.set_yscale("log")
    fig.suptitle(f"{rule_name}: target-wise dynamical distributions", fontsize=16, color=MAGENTA)
    fig.tight_layout(rect=(0, 0, 1, .94)); fig.savefig(path, dpi=220); plt.close(fig)


def centroid_figure(frame: pd.DataFrame, path: Path, rule_name: str) -> None:
    x = standardize(frame); labels = frame.cyp_target.to_numpy()
    centroids = np.vstack([x[labels == t].mean(axis=0) for t in TARGETS])
    distances = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=2)
    pca = PCA(n_components=2).fit_transform(x)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    heat = axes[0].imshow(centroids, cmap="cool", aspect="auto", vmin=-2, vmax=2)
    axes[0].set_xticks(range(3), ["Recurrence", "Late motion", "Entropy"], rotation=20)
    axes[0].set_yticks(range(4), TARGETS); axes[0].set_title("Standardized target centroids")
    fig.colorbar(heat, ax=axes[0], fraction=.046, label="z score")
    dist = axes[1].imshow(distances, cmap="magma", aspect="equal")
    axes[1].set_xticks(range(4), TARGETS, rotation=25); axes[1].set_yticks(range(4), TARGETS)
    axes[1].set_title("Centroid distances")
    for i in range(4):
        for j in range(4): axes[1].text(j, i, f"{distances[i,j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(dist, ax=axes[1], fraction=.046)
    for target in TARGETS:
        q = labels == target
        axes[2].scatter(pca[q, 0], pca[q, 1], s=10, alpha=.35,
                        c=TARGET_COLOURS[target], label=target, edgecolors="none")
        confidence_ellipse(axes[2], pca[q, 0], pca[q, 1], TARGET_COLOURS[target])
    axes[2].set(xlabel="Dynamical PC1", ylabel="Dynamical PC2", title="Three-feature PCA")
    axes[2].grid(alpha=.15); axes[2].legend(frameon=False, fontsize=7, labelcolor=WHITE)
    fig.suptitle(f"{rule_name}: target centres and multivariate separation", fontsize=16, color=LIME)
    fig.tight_layout(rect=(0, 0, 1, .94)); fig.savefig(path, dpi=220); plt.close(fig)


def adjusted_figure(frame: pd.DataFrame, path: Path, rule_name: str) -> None:
    q = adjusted_within_molecule(frame)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, feature in zip(axes, FEATURES):
        col = f"adjusted_{feature}"
        for target in TARGETS:
            values = q.loc[q.cyp_target == target, col]
            ax.hist(values, bins=25, density=True, histtype="step", lw=1.8,
                    color=TARGET_COLOURS[target], label=target, alpha=.9)
        ax.set_xlabel(f"Adjusted {FEATURE_LABELS[feature].lower()}")
        ax.set_ylabel("Density"); ax.grid(alpha=.15)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=WHITE)
    fig.suptitle(f"{rule_name}: within-molecule, pIC50-adjusted target signal", fontsize=15, color=ORANGE)
    fig.tight_layout(rect=(0, 0, 1, .94)); fig.savefig(path, dpi=220); plt.close(fig)


def paragraph_styles():
    styles = getSampleStyleSheet()
    title = ParagraphStyle("CyberTitle", parent=styles["Title"], textColor=colors.HexColor(CYAN),
                           alignment=TA_CENTER, fontSize=22, leading=27, spaceAfter=8)
    heading = ParagraphStyle("CyberHeading", parent=styles["Heading2"],
                             textColor=colors.HexColor(MAGENTA), spaceBefore=7, spaceAfter=6)
    body = ParagraphStyle("CyberBody", parent=styles["BodyText"], textColor=colors.HexColor("#182033"),
                          fontSize=9.3, leading=12.5)
    small = ParagraphStyle("CyberSmall", parent=body, fontSize=8.2, leading=10.5)
    return title, heading, body, small


def styled_table(rows, widths):
    table = Table(rows, colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(PANEL)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor(WHITE)),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), .35, colors.HexColor(GREY)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EEF3F8")]),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
    ]))
    return table


def target_legend_table(counts):
    rows = [["Target", "Observations", "Colour"]] + [[t, str(counts[t]), ""] for t in TARGETS]
    table = styled_table(rows, (55*mm, 55*mm, 50*mm))
    for row, target in enumerate(TARGETS, 1):
        table.setStyle(TableStyle([
            ("BACKGROUND", (2, row), (2, row), colors.HexColor(TARGET_COLOURS[target])),
        ]))
    return table


def page_footer(canvas, document):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor(GREY)); canvas.setLineWidth(.35)
    canvas.line(15*mm, 9*mm, A4[0] - 15*mm, 9*mm)
    canvas.setFont("Helvetica", 7.5); canvas.setFillColor(colors.HexColor(GREY))
    canvas.drawString(15*mm, 5.5*mm, "Strange Matter Engine | CYP-target dynamical regimes")
    canvas.drawRightString(A4[0] - 15*mm, 5.5*mm, f"Page {document.page}")
    canvas.restoreState()


def build_pdf(rule: str, study_dir: Path, frame: pd.DataFrame, stats: dict) -> Path:
    rule_name = pretty_rule(rule)
    fig_dir = ANALYSIS / rule / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    scatter_figure(frame, fig_dir / "01_target_geometry.png", rule_name)
    distribution_figure(frame, fig_dir / "02_distributions.png", rule_name)
    centroid_figure(frame, fig_dir / "03_centroids.png", rule_name)
    adjusted_figure(frame, fig_dir / "04_adjusted.png", rule_name)
    output = OUTPUT / f"{rule}_cyp_target_dynamical_regimes_v1.pdf"
    title, heading, body, small = paragraph_styles()
    doc = SimpleDocTemplate(str(output), pagesize=A4, leftMargin=15*mm, rightMargin=15*mm,
                            topMargin=13*mm, bottomMargin=13*mm,
                            title=f"{rule_name} CYP target dynamical regimes")
    counts = stats["target_counts"]
    p_text = "< 0.002" if stats["permutation_p"] <= .002 else f"= {stats['permutation_p']:.3f}"
    story = [
        Paragraph("Strange Matter Engine", title),
        Paragraph(f"{rule_name}: CYP-target dynamical regimes", heading),
        Paragraph("OpenADMET direct-inhibition validation trajectories | target-conditioned Graph Cellular Automaton", body),
        Spacer(1, 3*mm),
        styled_table([
            ["Quantity", "Result"],
            ["Trajectory observations", f"{stats['n_observations']:,}"],
            ["Distinct molecules", f"{stats['n_molecules']:,}"],
            ["Distinct Murcko scaffold groups", f"{stats['n_scaffolds']:,}"],
            ["CYP-label silhouette score", f"{stats['silhouette']:.3f}"],
            ["Multivariate permutation test", f"pseudo-F {stats['pseudo_f']:.1f}; p {p_text}"],
            ["Molecule-grouped target accuracy", f"{stats['molecule_grouped_balanced_accuracy_mean']:.3f} +/- {stats['molecule_grouped_balanced_accuracy_sd']:.3f}"],
            ["Scaffold-grouped target accuracy", f"{stats['scaffold_grouped_balanced_accuracy_mean']:.3f} +/- {stats['scaffold_grouped_balanced_accuracy_sd']:.3f}"],
        ], (78*mm, 82*mm)),
        Spacer(1, 3*mm),
        Paragraph("What the scores mean", heading),
        Paragraph("A silhouette score near 1 indicates compact, well-separated CYP-labelled regions; a value near 0 indicates overlap. Balanced accuracy is measured against a 0.25 four-target chance level. Molecule-grouped folds prevent the same molecule entering both sides of a classification test. Scaffold-grouped folds additionally separate related chemical frameworks.", body),
        Spacer(1, 2*mm),
        target_legend_table(counts),
        PageBreak(), Paragraph("Target-coloured dynamical geometry", heading),
        Paragraph("Each point is one molecule-target validation observation. Translucent ellipses enclose approximately 80% of each target's bivariate Gaussian geometry; X markers denote target centroids.", body),
        Image(str(fig_dir / "01_target_geometry.png"), width=180*mm, height=58*mm),
        Spacer(1, 3*mm),
        Paragraph("Target-specific effect sizes", heading),
        styled_table([["Dynamical quantity", "Variance associated with CYP target"]] +
                     [[FEATURE_LABELS[f], f"{100*stats['eta_squared'][f]:.1f}%"] for f in FEATURES],
                     (85*mm, 75*mm)),
        Paragraph("These eta-squared values quantify association, not causation. The CYP identity is supplied to the shared model as context, so strong separation means the learned dynamics use that context to create distinct regimes.", small),
        PageBreak(), Paragraph("Distributions and target centroids", heading),
        Image(str(fig_dir / "02_distributions.png"), width=180*mm, height=58*mm),
        Spacer(1, 2*mm),
        Image(str(fig_dir / "03_centroids.png"), width=180*mm, height=58*mm),
        Paragraph("The centroid heatmap expresses every target mean in standard-deviation units. The distance matrix measures separation between those three-dimensional centres. PCA rotates the same three standardized variables into the two directions containing the most variance; it does not add information.", small),
        PageBreak(), Paragraph("Does the target signal survive molecular and potency controls?", heading),
        Paragraph(f"This stricter analysis retained {stats['adjusted_n_observations']:,} observations from {stats['adjusted_n_molecules']:,} molecules having at least two CYP measurements. Each dynamical feature was centred within molecule, then its linear association with within-molecule pIC50 was removed. The remaining signal cannot be explained by a molecule's overall dynamical baseline or by a linear within-molecule potency difference.", body),
        Image(str(fig_dir / "04_adjusted.png"), width=180*mm, height=58*mm),
        styled_table([
            ["Adjusted analysis", "Result"],
            ["CYP-label silhouette", f"{stats['adjusted_silhouette']:.3f}"],
            ["Permutation pseudo-F", f"{stats['adjusted_pseudo_f']:.1f}"],
            ["Permutation p value", f"{stats['adjusted_permutation_p']:.3f}"],
        ], (85*mm, 75*mm)),
        PageBreak(), Paragraph("Scientific interpretation", heading),
        Paragraph("The analysis tests whether the trained Graph Cellular Automaton produces distinct trajectory-summary regimes for CYP1A2, CYP2C9, CYP2D6 and CYP3A4. Clear separation is evidence of target-conditioned internal dynamics. It does not by itself demonstrate four physical protein conformational states: the simulated states belong to the molecular graph model, and CYP identity is an explicit model input.", body),
        Spacer(1, 3*mm), Paragraph("Recommended reading order", heading),
        Paragraph("1. Inspect recurrence ratio versus late motion for visibly separated target regions. 2. Check spectral entropy to see whether the target also changes temporal complexity. 3. Compare target centroids and their distances. 4. Use scaffold-grouped accuracy to judge whether separation generalizes beyond related chemistry. 5. Use the within-molecule, pIC50-adjusted analysis to determine whether target structure remains after stringent controls.", body),
        Spacer(1, 3*mm), Paragraph("Methods", heading),
        Paragraph("The three trajectory summaries were standardized before multivariate calculations. Silhouette score used known CYP labels rather than an unsupervised cluster assignment. Multivariate pseudo-F significance used 499 label permutations. Logistic target classification used five stratified group folds and class-balanced multinomial regression. Murcko scaffolds were calculated from the standardized challenge SMILES. Confidence ellipses are descriptive Gaussian approximations. No model was retrained and no trajectory was recomputed.", small),
    ]
    doc.build(story, onFirstPage=page_footer, onLaterPages=page_footer)
    return output


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True); ANALYSIS.mkdir(parents=True, exist_ok=True)
    set_plot_style(); scaffold_map = molecule_scaffolds(); summaries = []
    studies = sorted(RESULTS.glob("production_*_challenge_aligned_v5"))
    if len(studies) != 10:
        raise RuntimeError(f"Expected 10 challenge-aligned studies; found {len(studies)}")
    for study in studies:
        summary = json.loads((study / "study_summary.json").read_text())
        rule = summary["rule"]
        dynamics_path = study / "runs" / "final_model" / "validation_dynamics.csv"
        frame = pd.read_csv(dynamics_path).dropna(subset=FEATURES + ["cyp_target", "experimental_pic50"])
        stats = calculate_statistics(frame, scaffold_map)
        rule_dir = ANALYSIS / rule; rule_dir.mkdir(parents=True, exist_ok=True)
        (rule_dir / "statistics.json").write_text(json.dumps(stats, indent=2) + "\n")
        pdf = build_pdf(rule, study, frame, stats)
        summaries.append({"rule": rule, "pdf": str(pdf), **{k: v for k, v in stats.items()
                                                           if not isinstance(v, dict)}})
        print(pdf, flush=True)
    pd.DataFrame(summaries).to_csv(ANALYSIS / "summary.csv", index=False)


if __name__ == "__main__":
    main()
