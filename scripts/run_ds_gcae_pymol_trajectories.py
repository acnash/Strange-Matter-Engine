#!/usr/bin/env python3
"""Extend ten frozen DS-GCAE member trajectories and build self-playing PyMOL visuals."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_graph_ca_visual_prototype.py"
OUT = ROOT / "results" / "ds_gcae_1000_generation_pymol"
CACHE = ROOT / "tmp" / "strange_matter_graph_ca_graphs_with_blind.pkl"
RULES = (
    "gated_residual",
    "delayed_memory",
    "inertial_reaction_diffusion",
    "kuramoto_sakaguchi",
    "fitzhugh_nagumo",
)
CYPS = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")


def checkpoint(rule: str) -> Path:
    return ROOT / "results" / f"production_{rule}_multiscale_ensemble_v2" / "runs" / "final_model" / "model.pt"


def screening(rule: str) -> Path:
    return ROOT / "results" / f"production_{rule}_multiscale_ensemble_v2" / "runs" / "final_model" / "validation_dynamics.csv"


def run_extensions(python: Path, device: str, generations: int, burn_in: int) -> None:
    for index, rule in enumerate(RULES, start=1):
        rule_out = OUT / "extended" / rule
        metadata = rule_out / "metadata.json"
        if (rule_out / "node_trajectories" / "case_004.npz").exists():
            build_candidate_metrics(rule, burn_in)
            print(json.dumps({"rule": rule, "stage": "reuse", "index": index, "total": len(RULES)}), flush=True)
            continue
        env = os.environ.copy()
        env.update({
            "SME_DEVICE": device,
            "SME_GRAPH_CACHE": str(CACHE),
            "SME_CHECKPOINT": str(checkpoint(rule)),
            "SME_SCREENING_CSV": str(screening(rule)),
            "SME_EXTENDED_OUTPUT": str(rule_out),
            "SME_EXTENDED_GENERATIONS": str(generations),
            "SME_EXTENDED_CANDIDATES": "4",
            "SME_EXTENDED_BURN_IN": str(burn_in),
            "SME_EXTENDED_SAVE_NODE_TRAJECTORIES": "1",
            "SME_EXTENDED_SKIP_PERTURBATION": "1",
            "SME_EXTENDED_TRAJECTORY_ONLY": "1",
            "MPLCONFIGDIR": str(ROOT / "tmp" / "matplotlib"),
        })
        print(json.dumps({"rule": rule, "stage": "start", "index": index, "total": len(RULES)}), flush=True)
        completed = subprocess.run([str(python), str(RUNNER), "extended-dynamics"], cwd=ROOT, env=env)
        archives = list((rule_out / "node_trajectories").glob("case_*.npz"))
        if completed.returncode and len(archives) != 4:
            raise subprocess.CalledProcessError(completed.returncode, completed.args)
        build_candidate_metrics(rule, burn_in)


def trajectory_metrics(trajectory: np.ndarray, burn_in: int) -> dict:
    flattened = trajectory.reshape(trajectory.shape[0], -1)
    late = flattened[burn_in:]
    steps = np.sqrt(np.mean(np.diff(late, axis=0) ** 2, axis=1))
    mean_step = float(np.mean(steps)) + 1e-15
    maximum_lag = min(128, max(2, len(late) // 4))
    lag_values = np.arange(2, maximum_lag + 1)
    distances = np.asarray([
        np.mean(np.sqrt(np.mean((late[lag:] - late[:-lag]) ** 2, axis=1)))
        for lag in lag_values
    ])
    ratios = distances / (lag_values * mean_step + 1e-15)
    best = int(np.argmin(ratios))
    centered = late - late.mean(axis=0, keepdims=True)
    power = np.abs(np.fft.rfft(centered, axis=0)) ** 2
    total_power = power[1:].sum(axis=1)
    probability = total_power / max(float(total_power.sum()), 1e-30)
    entropy = float(-np.sum(probability * np.log(np.maximum(probability, 1e-30)))
                    / np.log(max(2, len(probability))))
    return {
        "late_motion_5000": mean_step,
        "final_step_5000": float(steps[-1]),
        "late_amplitude_5000": float(np.sqrt(np.mean(np.var(late, axis=0)))),
        "recurrence_ratio_5000": float(ratios[best]),
        "recurrence_lag_5000": int(lag_values[best]),
        "recurrence_distance_5000": float(distances[best]),
        "spectral_entropy_5000": entropy,
        "spectral_concentration_5000": float(np.max(probability)),
    }


def build_candidate_metrics(rule: str, burn_in: int) -> None:
    rule_out = OUT / "extended" / rule
    selected = pd.read_csv(rule_out / "selected_candidates.csv")
    rows = []
    for number, row in enumerate(selected.itertuples(), start=1):
        trajectory = np.load(rule_out / "node_trajectories" / f"case_{number:03d}.npz")["trajectory"]
        metrics = trajectory_metrics(trajectory, burn_in)
        classification = (
            f"recurrent_period_{metrics['recurrence_lag_5000']}_candidate"
            if metrics["recurrence_ratio_5000"] < 0.1
            else "extended_transient_or_unresolved"
        )
        rows.append({**row._asdict(), **metrics,
                     "extended_screening_classification": classification})
    pd.DataFrame(rows).to_csv(rule_out / "visual_candidate_metrics.csv", index=False)


def select_cases() -> pd.DataFrame:
    selected = []
    used_molecules: set[str] = set()
    for rule in RULES:
        frame = pd.read_csv(OUT / "extended" / rule / "visual_candidate_metrics.csv")
        frame["interest"] = (
            frame["late_motion_5000"].rank(pct=True)
            + (1.0 - frame["recurrence_ratio_5000"].rank(pct=True))
            + frame["spectral_entropy_5000"].rank(pct=True)
            + frame["late_amplitude_5000"].rank(pct=True)
        )
        frame = frame.sort_values("interest", ascending=False)
        chosen = []
        for row in frame.itertuples():
            if row.molecule_id in used_molecules:
                continue
            chosen.append(row)
            used_molecules.add(row.molecule_id)
            if len(chosen) == 2:
                break
        if len(chosen) < 2:
            for row in frame.itertuples():
                if any(existing.molecule_id == row.molecule_id for existing in chosen):
                    continue
                chosen.append(row)
                if len(chosen) == 2:
                    break
        for row in chosen:
            item = row._asdict()
            item["rule"] = rule
            selected.append(item)
    return pd.DataFrame(selected)


def embed_molecule(smiles: str, seed: int) -> Chem.Mol:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")
    molecule = Chem.AddHs(molecule)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    if AllChem.EmbedMolecule(molecule, params) != 0:
        AllChem.Compute2DCoords(molecule)
    else:
        try:
            AllChem.MMFFOptimizeMolecule(molecule, maxIters=500)
        except Exception:
            pass
    return molecule


def write_static_pdb(molecule: Chem.Mol, path: Path) -> None:
    Chem.MolToPDBFile(molecule, str(path))


def display_values(trajectory: np.ndarray, molecule: Chem.Mol) -> tuple[np.ndarray, float, float]:
    heavy_count = int(trajectory.shape[1])
    activity = np.linalg.norm(trajectory, axis=2)
    lo, hi = np.quantile(activity, [0.01, 0.99])
    scaled_heavy = np.clip(100.0 * (activity - lo) / max(float(hi - lo), 1e-8), 0.0, 100.0)
    values = np.zeros((trajectory.shape[0], molecule.GetNumAtoms()), dtype=np.float32)
    values[:, :heavy_count] = scaled_heavy
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() == 1:
            values[:, atom.GetIdx()] = scaled_heavy[:, atom.GetNeighbors()[0].GetIdx()]
    return values, float(lo), float(hi)


def build_pymol(generations: int) -> None:
    if not CACHE.exists():
        raise FileNotFoundError(CACHE)
    with CACHE.open("rb") as handle:
        data = pickle.load(handle)
    chosen = select_cases()
    structure_dir = OUT / "structures"
    value_dir = OUT / "display_values"
    structure_dir.mkdir(parents=True, exist_ok=True)
    value_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    embedded_values = {}
    pml = [
        "reinitialize",
        "bg_color black",
        "set antialias, 2",
        "set ray_opaque_background, off",
        "set stick_radius, 0.16",
        "set sphere_scale, 0.28",
        "set_color cyber_lime, [0.65, 1.00, 0.10]",
    ]
    for visual_index, row in enumerate(chosen.itertuples(), start=1):
        rule_dir = OUT / "extended" / row.rule
        selected_candidates = pd.read_csv(rule_dir / "selected_candidates.csv")
        case_number = int(selected_candidates.index[selected_candidates.molecule_id == row.molecule_id][0]) + 1
        archive = rule_dir / "node_trajectories" / f"case_{case_number:03d}.npz"
        trajectory = np.load(archive)["trajectory"]
        rec = data["train"][int(row.training_index)]
        if rec["name"] != row.molecule_id:
            raise RuntimeError("Selected case and graph cache are misaligned")
        molecule = embed_molecule(rec["canonical_smiles"], 1701 + visual_index)
        safe_rule = row.rule.replace("_", "-")
        stem = f"{visual_index:02d}_{safe_rule}_{row.molecule_id}_{row.cyp_target}"
        obj = f"traj_{visual_index:02d}_{row.rule}_{row.cyp_target}"
        pdb_path = structure_dir / f"{stem}.pdb"
        write_static_pdb(molecule, pdb_path)
        values, lo, hi = display_values(trajectory, molecule)
        np.savez_compressed(value_dir / f"{stem}.npz", display_values=values, trajectory=trajectory)
        embedded_values[obj] = values.round(3).tolist()
        relative_pdb = pdb_path.relative_to(OUT).as_posix()
        pml.extend([
            f'load "{pdb_path.as_posix()}", {obj}',
            f"hide everything, {obj}",
            f"show sticks, {obj}",
            f"show spheres, {obj} and not elem H",
            f"hide sticks, {obj} and elem H",
            f"disable {obj}",
        ])
        manifest_rows.append({
            "visual_rank": visual_index,
            "object": obj,
            "transition_rule": row.rule,
            "molecule_id": row.molecule_id,
            "cyp_target": row.cyp_target,
            "smiles": rec["smiles"],
            "training_index": int(row.training_index),
            "selection_reason": row.selection_reason,
            "screening_classification": row.extended_screening_classification,
            "late_motion": row.late_motion_5000,
            "recurrence_ratio": row.recurrence_ratio_5000,
            "recurrence_lag": int(row.recurrence_lag_5000),
            "spectral_entropy": row.spectral_entropy_5000,
            "activity_scale_low": lo,
            "activity_scale_high": hi,
            "generations": generations,
            "frames": generations + 1,
            "pdb_file": relative_pdb,
        })
    controller_template = (ROOT / "scripts" / "pymol_gca_controller.py").read_text(encoding="utf-8")
    values_path = OUT / "display_values.json.gz"
    with gzip.open(values_path, "wt", encoding="utf-8", compresslevel=9) as handle:
        json.dump(embedded_values, handle, separators=(",", ":"))
    controller = (
        "import gzip\n"
        "import json\n"
        f"GCA_STATE_COUNT = {generations + 1}\n"
        "GCA_HYDROGEN_CODA_STATE = None\n"
        f"with gzip.open(r'{values_path}', 'rt', encoding='utf-8') as _handle:\n"
        "    GCA_DISPLAY_VALUES = json.load(_handle)\n"
        + controller_template
    )
    controller_path = OUT / "ds_gcae_trajectory_controls.py"
    controller_path.write_text(controller, encoding="utf-8")
    first_obj = manifest_rows[0]["object"]
    pml.extend([
        f"enable {first_obj}",
        f"orient {first_obj}",
        "python",
        f"exec(compile(open(r'{controller_path}', encoding='utf-8').read(), r'{controller_path}', 'exec'))",
        "python end",
        "gca_state 1",
        "refresh",
        "gca_play 0.05, 1",
        "# Run: gca_play 0.05, 1",
        "# Stop: gca_stop",
        "# Enable one traj_* object at a time for the clearest movie.",
    ])
    (OUT / "load_ds_gcae_10_trajectories.pml").write_text("\n".join(pml) + "\n", encoding="utf-8")
    with (OUT / "trajectory_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)
    (OUT / "README.md").write_text(f"""# DS-GCAE 1,000-generation PyMOL trajectories

This visual set contains ten validation molecule-CYP trajectories, two from each transition-rule family in DS-GCAE v1. Every trajectory was extended to {generations:,} generations from a frozen seed-1701 multiscale checkpoint. No optimisation, ridge fitting, parameter update, or blinded-test selection occurred.

## Open and play

1. Start PyMOL.
2. Choose **File > Run Script**.
3. Select `load_ds_gcae_10_trajectories.pml` from this directory.
4. The first trajectory is enabled automatically.
5. Playback starts automatically and plays all {generations + 1:,} frames once at 0.05 seconds per frame.
6. Enter `gca_play 0.05, 1` to replay it with one command. Enter `gca_stop` to stop. Use `gca_state 500`, `gca_next`, or `gca_previous` for manual inspection.

Enable one `traj_*` object at a time in the PyMOL object panel. Atom colours encode the within-trajectory magnitude of the learned dynamical state, robustly scaled between its first and ninety-ninth percentiles. Cyan indicates lower magnitude and magenta indicates higher magnitude. Coordinates are display-only RDKit conformers reconstructed from SMILES; the graph CA used molecular connectivity and chemical features rather than 3D coordinates.

The lossless atom-by-channel trajectories and display arrays are retained in `display_values`. Selection metrics and provenance are recorded in `trajectory_manifest.csv`.
""", encoding="utf-8")
    print(json.dumps({"stage": "complete", "visuals": len(manifest_rows), "output": str(OUT)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--generations", type=int, default=1000)
    parser.add_argument("--burn-in", type=int, default=250)
    args = parser.parse_args()
    if args.generations < 2:
        raise ValueError("generations must be at least two")
    if not 0 <= args.burn_in < args.generations:
        raise ValueError("burn-in must be within the trajectory")
    OUT.mkdir(parents=True, exist_ok=True)
    run_extensions(Path(args.python), args.device, args.generations, args.burn_in)
    build_pymol(args.generations)


if __name__ == "__main__":
    main()
