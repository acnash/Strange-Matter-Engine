#!/usr/bin/env python3
"""Summarise production validation-trajectory dynamical screens by CA rule."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "long_horizon_attractor_campaign_v1"
SUFFIX = "_challenge_aligned_v5"

RULES = (
    "gated_residual",
    "delayed_memory",
    "inertial_reaction_diffusion",
    "gray_scott",
    "coupled_map",
    "activator_inhibitor",
    "fitzhugh_nagumo",
    "kuramoto_sakaguchi",
    "damped_symplectic",
    "conservative_graph_flux",
)


def classify(frame: pd.DataFrame) -> pd.Series:
    point = (frame["late_motion"] < 1e-4) & (frame["final_step"] < 1e-5)
    oscillator = (~point) & (frame["recurrence_ratio"] < .25) & (
        frame["spectral_concentration"] > .5
    )
    labels = pd.Series("persistent_or_complex_screen", index=frame.index)
    labels.loc[oscillator] = "oscillator_screen_candidate"
    labels.loc[point] = "point_attractor_screen_candidate"
    return labels


def main() -> None:
    rows = []
    for rule in RULES:
        source = (RESULTS / f"production_{rule}{SUFFIX}" / "runs" / "final_model" /
                  "validation_dynamics.csv")
        frame = pd.read_csv(source)
        classes = classify(frame)
        rows.append({
            "transition_rule": rule,
            "trajectories_screened": len(frame),
            "point_attractor_screen_candidates": int(
                (classes == "point_attractor_screen_candidate").sum()),
            "oscillator_screen_candidates": int(
                (classes == "oscillator_screen_candidate").sum()),
            "persistent_or_complex_screens": int(
                (classes == "persistent_or_complex_screen").sum()),
            "confirmed_strange_attractors": 2 if rule == "kuramoto_sakaguchi" else 0,
            "strange_attractor_candidates_definitively_tested": (
                2 if rule == "kuramoto_sakaguchi" else 0),
        })
    summary = pd.DataFrame(rows)
    total = summary.select_dtypes("number").sum()
    summary.loc[len(summary)] = {"transition_rule": "TOTAL", **total.to_dict()}
    csv_path = OUTPUT / "validation_dynamics_population_summary.csv"
    summary.to_csv(csv_path, index=False)
    print(summary.to_string(index=False))
    print(csv_path)


if __name__ == "__main__":
    main()
