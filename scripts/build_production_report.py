#!/usr/bin/env python3
"""Build figures and a PDF report for a completed transition-rule study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)


ROOT = Path(__file__).resolve().parents[1]
INK, PANEL, CYAN, MAGENTA, LIME, WHITE, GREY = (
    "#070914", "#11152A", "#27E1FF", "#FF3CAC", "#F9F871", "#DCE6F2", "#65758B"
)


def figure_style():
    plt.rcParams.update({"figure.facecolor": INK, "axes.facecolor": INK,
                         "axes.edgecolor": GREY, "axes.labelcolor": WHITE,
                         "xtick.color": WHITE, "ytick.color": WHITE,
                         "text.color": WHITE, "grid.color": GREY,
                         "font.size": 10})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True)
    args = parser.parse_args()
    study_dir = ROOT / "results" / args.study
    summary = json.loads((study_dir / "study_summary.json").read_text())
    final_dir = study_dir / "runs" / "final_model"
    trials = pd.read_csv(study_dir / "all_trials.csv")
    predictions = pd.read_csv(final_dir / "validation_predictions.csv")
    history = pd.read_csv(final_dir / "training_history.csv")
    dynamics = pd.read_csv(final_dir / "validation_dynamics.csv")
    perturbations = pd.read_csv(final_dir / "validation_perturbations.csv")
    figures = study_dir / "figures"; figures.mkdir(exist_ok=True)
    figure_style()

    fig, ax = plt.subplots(figsize=(8, 5))
    for stage, colour in (("stage1", CYAN), ("stage2", MAGENTA), ("confirmation", LIME)):
        q = trials[trials.stage == stage]
        ax.scatter(q.generations, q.validation_rmse, c=colour, alpha=.7, label=stage)
    ax.set(xscale="log", xlabel="CA generations", ylabel="Validation RMSE (pIC50)",
           title="Hyperparameter search by trajectory length")
    ax.grid(alpha=.18); ax.legend(); fig.subplots_adjust(left=.12, right=.97, bottom=.13, top=.90)
    fig.savefig(figures / "01_search.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    q = predictions[predictions.split == "validation"]
    ax.scatter(q.experimental_pic50, q.predicted_pic50, s=12, c=CYAN, alpha=.45)
    lo = min(q.experimental_pic50.min(), q.predicted_pic50.min())
    hi = max(q.experimental_pic50.max(), q.predicted_pic50.max())
    ax.plot([lo, hi], [lo, hi], "--", c=LIME)
    ax.set(xlabel="Experimental pIC50", ylabel="Predicted pIC50",
           title="Grouped-validation predictions"); ax.grid(alpha=.18)
    fig.subplots_adjust(left=.14, right=.97, bottom=.12, top=.91); fig.savefig(figures / "02_validation.png", dpi=200); plt.close(fig)

    per_cyp = q.groupby("cyp_target").apply(
        lambda x: np.sqrt(np.mean((x.predicted_pic50 - x.experimental_pic50) ** 2)),
        include_groups=False,
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(per_cyp.index, per_cyp.values, color=(CYAN, MAGENTA, LIME, "#7A5CFA"))
    ax.set(ylabel="Validation RMSE (pIC50)", title="Per-CYP predictive performance")
    ax.grid(axis="y", alpha=.18); fig.subplots_adjust(left=.12, right=.97, bottom=.14, top=.88)
    fig.savefig(figures / "03_per_cyp.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history.epoch, history.train_rmse, c=CYAN, label="query training")
    ax.plot(history.epoch, history.validation_rmse, c=MAGENTA, label="validation")
    ax.set(xlabel="Epoch", ylabel="RMSE (pIC50)", title="Final-model learning curve")
    ax.grid(alpha=.18); ax.legend(); fig.subplots_adjust(left=.12, right=.97, bottom=.14, top=.88)
    fig.savefig(figures / "04_learning.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(dynamics.recurrence_ratio, dynamics.late_motion,
                         c=dynamics.spectral_entropy, cmap="cool", s=16, alpha=.7)
    ax.set(xlabel="Recurrence ratio", ylabel="Late motion",
           title="Validation dynamical regimes"); ax.set_yscale("log")
    fig.colorbar(scatter, ax=ax, label="Spectral entropy")
    ax.grid(alpha=.15); fig.subplots_adjust(left=.12, right=.97, bottom=.14, top=.88)
    fig.savefig(figures / "05_dynamics.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if len(perturbations):
        colours = [LIME if x > 0 else MAGENTA for x in perturbations.finite_time_lyapunov]
        ax.bar(np.arange(len(perturbations)), perturbations.finite_time_lyapunov, color=colours)
        ax.axhline(0, c=WHITE, lw=1)
    else:
        ax.text(.5, .5, "Perturbation confirmation deferred\nafter native CUDA failure",
                ha="center", va="center", transform=ax.transAxes, color=MAGENTA, fontsize=14)
    ax.set(xlabel="Selected validation case",
        ylabel="Finite-time divergence slope", title="Perturbation-response screening")
    ax.grid(axis="y", alpha=.18); fig.subplots_adjust(left=.12, right=.97, bottom=.14, top=.88)
    fig.savefig(figures / "06_perturbations.png", dpi=200); plt.close(fig)

    output_dir = ROOT / "output" / "pdf"; output_dir.mkdir(parents=True, exist_ok=True)
    version_suffix = (f"_{summary['search_version']}"
                      if summary.get("search_version") == "enhanced_v3" else "")
    pdf_path = output_dir / f"{summary['rule']}{version_suffix}_production_report.pdf"
    styles = getSampleStyleSheet()
    title = ParagraphStyle("CyberTitle", parent=styles["Title"], textColor=colors.HexColor(CYAN),
                           alignment=TA_CENTER, fontSize=23, leading=27, spaceAfter=12)
    heading = ParagraphStyle("CyberHeading", parent=styles["Heading2"],
                             textColor=colors.HexColor(MAGENTA), spaceBefore=8, spaceAfter=7)
    body = ParagraphStyle("CyberBody", parent=styles["BodyText"], textColor=colors.HexColor("#182033"),
                          fontSize=9.5, leading=13)
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, rightMargin=16*mm, leftMargin=16*mm,
                            topMargin=14*mm, bottomMargin=14*mm,
                            title=f"{summary['rule']} production study")
    winner = summary["winner"]; metrics = summary["final_metrics"]
    story = [Paragraph("Strange Matter Engine", title),
             Paragraph(f"Production study: {summary['rule']}", heading),
             Paragraph("Direct-inhibition pIC50 | grouped validation | blinded set excluded", body),
             Spacer(1, 5*mm)]
    table_data = [["Quantity", "Result"],
                  ["Validation RMSE", f"{metrics['restored_validation_rmse']:.4f} pIC50"],
                  ["Fit RMSE", f"{metrics['restored_fit_rmse']:.4f} pIC50"],
                  ["Generations", str(winner["generations"])],
                  ["Dynamical channels", str(winner["hidden_channels"])],
                  ["CA learning rate", str(winner["ca_lr"])],
                  ["Ridge penalty", str(winner["ridge"])],
                  ["CA L2", str(winner["ca_l2"])],
                  ["Gradient clip", str(winner["gradient_clip"])],
                  ["Confirmation mean RMSE", f"{summary['winner_confirmation_mean_rmse']:.4f}"],
                  ["Confirmation seed SD", f"{summary.get('winner_confirmation_seed_sd', 0.0):.4f}"]]
    table = Table(table_data, colWidths=(70*mm, 85*mm))
    table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor(PANEL)),
                               ("TEXTCOLOR", (0,0), (-1,0), colors.HexColor(WHITE)),
                               ("GRID", (0,0), (-1,-1), .35, colors.HexColor(GREY)),
                               ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
                               ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
                               ("ROWBACKGROUNDS", (0,1), (-1,-1),
                                [colors.white, colors.HexColor("#EEF3F8")])]))
    enhanced = (f"Enhanced CA controls: update scale {winner.get('update_scale')}, initial-state scale "
                f"{winner.get('init_scale')}, support fraction {winner.get('support_fraction')}, "
                f"bond temperature {winner.get('bond_temperature')}. Rule dynamics: "
                f"A={winner.get('dyn_a')}, B={winner.get('dyn_b')}, "
                f"C={winner.get('dyn_c')}, D={winner.get('dyn_d')}.")
    story += [table, Spacer(1, 2*mm), Paragraph(enhanced, body), Spacer(1, 3*mm),
              Image(str(figures / "01_search.png"), width=170*mm, height=106*mm),
              PageBreak(), Paragraph("Predictive validation", heading),
              Paragraph("Model selection used grouped-validation RMSE only. Dynamical-interest scores did not influence promotion.", body),
              Image(str(figures / "02_validation.png"), width=124*mm, height=124*mm),
              Image(str(figures / "03_per_cyp.png"), width=170*mm, height=96*mm),
              PageBreak(), Paragraph("Optimization", heading),
              Paragraph("Ridge coefficients and the unpenalized intercept were solved from permitted fitting fingerprints. Query error differentiated through the solve into the graph CA; Adam updated only CA parameters. Bond-conditioned messages, cosine learning-rate decay, and multitime trajectory statistics were shared across all five rules.", body),
              Image(str(figures / "04_learning.png"), width=170*mm, height=96*mm),
              PageBreak(), Paragraph("Emergent dynamics", heading),
              Paragraph("The validation trajectories were screened for convergence, recurrence, persistent motion, spectral complexity, and perturbation sensitivity. Labels ending in candidate require longer, renormalized and seed-repeated confirmation before any attractor or chaos claim.", body),
              Image(str(figures / "05_dynamics.png"), width=170*mm, height=106*mm),
              Image(str(figures / "06_perturbations.png"), width=170*mm, height=96*mm),
              Paragraph("Scientific limitations", heading),
              Paragraph("This report describes one transition-rule study under the declared grouped split. Hyperparameter selection and early stopping used labelled training data only. The blinded challenge set was not loaded or predicted. Finite-time positive slopes are screening signals rather than proof of a strange attractor or sustained chaos.", body)]
    doc.build(story)
    print(pdf_path)


if __name__ == "__main__":
    main()
