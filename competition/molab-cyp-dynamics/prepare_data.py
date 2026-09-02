"""Build the compact, public data bundle used by the Molab notebook."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "data"


CASCADE_CASES = (
    "01_gated-residual_OCNT-2328519_CYP1A2",
    "05_inertial-reaction-diffusion_OCNT-2312382_CYP1A2",
    "07_kuramoto-sakaguchi_OCNT-0494110_CYP2C9",
    "10_fitzhugh-nagumo_OCNT-2309705_CYP3A4",
)


def parse_pdb(path: Path) -> tuple[np.ndarray, list[str]]:
    coordinates, elements = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            coordinates.append(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            )
            elements.append((line[76:78].strip() or line[12:14].strip()).title())
    xyz = np.asarray(coordinates, dtype=np.float32)
    xyz -= xyz.mean(axis=0)
    return xyz, elements


def export_trajectory_examples() -> None:
    visual_root = ROOT / "results" / "ds_gcae_1000_generation_pymol"
    attractor_root = ROOT / "results" / "long_horizon_attractor_campaign_v1"
    screen = pd.read_csv(attractor_root / "attractor_screen.csv").set_index("case_id")

    cascade_rows, phase_rows = [], []
    for stem in CASCADE_CASES:
        rank = int(stem.split("_", 1)[0])
        case_id = screen.index[screen["visual_rank"] == rank][0]
        metadata = screen.loc[case_id]
        xyz, elements = parse_pdb(visual_root / "structures" / f"{stem}.pdb")
        values = np.load(visual_root / "display_values" / f"{stem}.npz")[
            "display_values"
        ]
        for generation in range(0, 1001, 10):
            angle = generation * 2.0 * np.pi * 5.25 / 1000.0
            cosine, sine = np.cos(angle), np.sin(angle)
            rotated_x = xyz[:, 0] * cosine + xyz[:, 2] * sine
            rotated_z = -xyz[:, 0] * sine + xyz[:, 2] * cosine
            vertical_step = max(np.ptp(xyz[:, [0, 2]]) / 2.0 * 0.0105, 0.035)
            for atom, element in enumerate(elements):
                cascade_rows.append(
                    {
                        "case_id": case_id,
                        "generation": generation,
                        "atom": atom,
                        "element": element,
                        "x": rotated_x[atom],
                        "y": xyz[atom, 1] - generation * vertical_step,
                        "z": rotated_z[atom],
                        "activity": values[generation, atom],
                    }
                )

        phase = np.load(attractor_root / "case_data" / f"{case_id}.npz")[
            "pca_coordinates"
        ]
        for offset in range(0, len(phase), 4):
            phase_rows.append(
                {
                    "case_id": case_id,
                    "generation": 1000 + offset,
                    "pc1": phase[offset, 0],
                    "pc2": phase[offset, 1],
                    "pc3": phase[offset, 2],
                }
            )

    pd.DataFrame(cascade_rows).to_csv(
        OUT / "molecular_cascades.csv.gz", index=False, compression="gzip"
    )
    pd.DataFrame(phase_rows).to_csv(
        OUT / "phase_trajectories.csv.gz", index=False, compression="gzip"
    )


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

    export_trajectory_examples()

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
