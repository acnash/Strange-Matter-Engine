"""Build the compact, public data bundle used by the Molab notebook."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "data"


def main() -> None:
    OUT.mkdir(exist_ok=True)

    assay_path = (
        ROOT
        / "data"
        / "openadmet-cyp-challenge-2026"
        / "cyp-challenge-TRAIN_inhibition.csv"
    )
    dynamics_path = (
        ROOT
        / "results"
        / "structure_dynamics_publication_v1"
        / "structure_dynamics_population.csv"
    )

    assay = pd.read_csv(assay_path)
    dynamics = pd.read_csv(dynamics_path)
    merged = dynamics.merge(
        assay,
        left_on="molecule_id",
        right_on="Molecule_Name",
        how="left",
        validate="many_to_one",
    )

    def endpoint_value(row: pd.Series, suffix: str = "") -> float:
        column = f"{row['cyp_target']}_pIC50_direct_inhibition{suffix}"
        return row[column]

    merged["measured_pic50"] = merged.apply(endpoint_value, axis=1)
    merged["measured_pic50_conf_low"] = merged.apply(
        endpoint_value, axis=1, suffix="_conf_low"
    )
    merged["measured_pic50_conf_high"] = merged.apply(
        endpoint_value, axis=1, suffix="_conf_high"
    )
    merged["measured_pic50_std"] = merged.apply(
        endpoint_value, axis=1, suffix="_std"
    )

    assay_columns = [
        column
        for column in merged.columns
        if "_pIC50_direct_inhibition" in column
    ]
    output_columns = [
        column
        for column in dynamics.columns
        if column not in {"training_index", "cyp_index"}
    ] + [
        "measured_pic50",
        "measured_pic50_conf_low",
        "measured_pic50_conf_high",
        "measured_pic50_std",
    ]
    merged[output_columns].to_csv(OUT / "cyp_dynamics_cohort.csv", index=False)

    source_files = {
        "dynamical_regime_summary.csv": ROOT
        / "results"
        / "cyp_target_dynamical_regimes_v1"
        / "summary.csv",
        "descriptor_correlations.csv": ROOT
        / "results"
        / "structure_dynamics_publication_v1"
        / "descriptor_correlations.csv",
        "causal_interventions.csv": ROOT
        / "results"
        / "structure_dynamics_publication_v1"
        / "causal_interventions_with_effects.csv",
        "attractor_screen.csv": ROOT
        / "results"
        / "long_horizon_attractor_campaign_v1"
        / "attractor_screen.csv",
    }
    for destination, source in source_files.items():
        shutil.copyfile(source, OUT / destination)

    provenance = {
        "assay_source": str(assay_path.relative_to(ROOT)).replace("\\", "/"),
        "dynamics_source": str(dynamics_path.relative_to(ROOT)).replace("\\", "/"),
        "rows": int(len(merged)),
        "unique_molecules": int(merged["molecule_id"].nunique()),
        "isoforms": sorted(merged["cyp_target"].unique().tolist()),
        "assay_columns_excluded_from_compact_table": assay_columns,
        "interpretation_boundary": (
            "Graph-state trajectories are learned computational dynamics and are "
            "not physical molecular-dynamics trajectories."
        ),
    }
    (OUT / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
