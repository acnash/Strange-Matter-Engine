#!/usr/bin/env python3
"""Population and causal-intervention study of frozen Graph-CA dynamics."""

from __future__ import annotations

import copy
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from run_renormalized_lyapunov import _one_run
except ModuleNotFoundError:
    from scripts.run_renormalized_lyapunov import _one_run


CYPS = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")


def _descriptors(rec):
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors

    mol = Chem.MolFromSmiles(rec["canonical_smiles"])
    atoms = mol.GetNumAtoms(); bonds = mol.GetNumBonds()
    ring_info = mol.GetRingInfo()
    aromatic_atoms = sum(a.GetIsAromatic() for a in mol.GetAtoms())
    hetero = sum(a.GetAtomicNum() not in (1, 6) for a in mol.GetAtoms())
    bond_counts = {"single_bonds": 0, "double_bonds": 0, "triple_bonds": 0,
                   "aromatic_bonds": 0, "ring_bonds": 0, "conjugated_bonds": 0}
    for bond in mol.GetBonds():
        name = str(bond.GetBondType()).lower() + "_bonds"
        if name in bond_counts: bond_counts[name] += 1
        bond_counts["ring_bonds"] += int(bond.IsInRing())
        bond_counts["conjugated_bonds"] += int(bond.GetIsConjugated())
    degrees = np.asarray([a.GetDegree() for a in mol.GetAtoms()], dtype=float)
    adjacency = Chem.GetAdjacencyMatrix(mol).astype(float)
    laplacian = np.diag(adjacency.sum(1)) - adjacency
    adjacency_eigen = np.linalg.eigvalsh(adjacency) if atoms else np.zeros(1)
    laplacian_eigen = np.sort(np.linalg.eigvalsh(laplacian)) if atoms else np.zeros(1)
    distances = Chem.GetDistanceMatrix(mol)
    finite_distances = distances[np.triu_indices(atoms, 1)] if atoms > 1 else np.zeros(1)
    conformer_descriptors = {
        "radius_of_gyration_3d": np.nan, "asphericity_3d": np.nan,
        "eccentricity_3d": np.nan, "inertial_shape_factor_3d": np.nan,
        "spherocity_index_3d": np.nan,
    }
    conformer = Chem.AddHs(Chem.Mol(mol))
    params = AllChem.ETKDGv3(); params.randomSeed = 260830
    if AllChem.EmbedMolecule(conformer, params) == 0:
        try: AllChem.MMFFOptimizeMolecule(conformer, maxIters=200)
        except Exception: pass
        conformer_descriptors = {
            "radius_of_gyration_3d": rdMolDescriptors.CalcRadiusOfGyration(conformer),
            "asphericity_3d": rdMolDescriptors.CalcAsphericity(conformer),
            "eccentricity_3d": rdMolDescriptors.CalcEccentricity(conformer),
            "inertial_shape_factor_3d": rdMolDescriptors.CalcInertialShapeFactor(conformer),
            "spherocity_index_3d": rdMolDescriptors.CalcSpherocityIndex(conformer),
        }
    return {
        "molecule_id": rec["name"], "smiles": rec["canonical_smiles"],
        "scaffold": rec["scaffold"], "heavy_atoms": atoms, "bonds": bonds,
        "molecular_weight": Descriptors.MolWt(mol), "logp": Descriptors.MolLogP(mol),
        "tpsa": rdMolDescriptors.CalcTPSA(mol), "hbd": Lipinski.NumHDonors(mol),
        "hba": Lipinski.NumHAcceptors(mol), "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
        "rings": ring_info.NumRings(), "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "fraction_csp3": rdMolDescriptors.CalcFractionCSP3(mol),
        "aromatic_atom_fraction": aromatic_atoms / max(atoms, 1),
        "heteroatom_fraction": hetero / max(atoms, 1),
        "formal_charge": sum(a.GetFormalCharge() for a in mol.GetAtoms()),
        "cyclomatic_number": bonds - atoms + 1,
        "graph_density": 2 * bonds / max(atoms * (atoms - 1), 1),
        "degree_mean": float(degrees.mean()), "degree_variance": float(degrees.var()),
        "adjacency_spectral_radius": float(max(abs(adjacency_eigen))),
        "algebraic_connectivity": float(laplacian_eigen[1] if atoms > 1 else 0),
        "laplacian_largest": float(laplacian_eigen[-1]),
        "graph_diameter": float(finite_distances.max(initial=0)),
        "mean_shortest_path": float(finite_distances.mean()), **bond_counts,
        **conformer_descriptors,
    }


def _select_population(screening, per_cyp, seed):
    rng = np.random.default_rng(seed)
    selected = []
    score = (screening.late_motion * (1 + screening.spectral_entropy) /
             screening.recurrence_ratio.clip(lower=1e-6))
    screening = screening.assign(dynamic_score=score)
    for cyp in CYPS:
        group = screening[screening.cyp_target == cyp]
        bins = pd.qcut(group.dynamic_score.rank(method="first"), 4, labels=False)
        for _, stratum in group.groupby(bins):
            count = min(int(math.ceil(per_cyp / 4)), len(stratum))
            indices = rng.choice(stratum.index.to_numpy(), count, replace=False)
            selected.extend(indices.tolist())
    return screening.loc[selected].drop_duplicates().sort_values(
        ["cyp_target", "dynamic_score"], ascending=[True, False]
    ).groupby("cyp_target", group_keys=False).head(per_cyp).reset_index(drop=True)


def _lle(model, rec, cyp, torch, device, burn_in, measured, interval, epsilon, repeats, seed):
    values, positive_blocks = [], []
    for repeat in range(repeats):
        local, cumulative, _ = _one_run(
            model=model, rec=rec, cyp=cyp, torch=torch, device=device,
            burn_in=burn_in, measured_generations=measured, interval=interval,
            epsilon=epsilon, seed=seed + 7919 * repeat,
        )
        values.append(float(cumulative[-1])); positive_blocks.append(float(np.mean(local > 0)))
    return values, positive_blocks


def _delete_bond(rec, bond_index):
    changed = copy.deepcopy(rec)
    keep = [i for i in range(len(rec["src"])) if i // 2 != bond_index]
    for key in ("src", "dst", "edge"):
        changed[key] = [rec[key][i] for i in keep]
    return changed


def _alter_bond(rec, bond_index, target_type):
    changed = copy.deepcopy(rec)
    for edge_index in (2 * bond_index, 2 * bond_index + 1):
        feature = list(changed["edge"][edge_index])
        feature[:4] = [float(i == target_type) for i in range(4)]
        feature[4] = float(target_type in (1, 3))
        changed["edge"][edge_index] = feature
    return changed


def _ablate_features(rec, indices):
    changed = copy.deepcopy(rec)
    values = np.asarray(changed["x"], dtype=float)
    values[:, indices] = 0.0
    changed["x"] = values.tolist()
    return changed


def _interventions(model, data, population, atom_names, torch, device, settings, output_dir):
    targets = {("OCNT-0494110", "CYP2C9"), ("OCNT-2328784", "CYP1A2")}
    rows = []
    name_groups = {
        "element_identity": list(range(0, 11)), "charge_aromaticity": [11, 12],
        "hybridisation": [13, 14, 15, 16], "local_valence": [17, 18, 19],
        "donor_acceptor": [20, 21], "chirality": [22, 23, 24],
        "neighbour_chemistry": [25, 26, 27, 28],
    }
    type_names = ("single", "double", "triple", "aromatic")
    for case_no, row in enumerate(population.itertuples(), 1):
        if (row.molecule_id, row.cyp_target) not in targets: continue
        rec = data["train"][int(row.training_index)]; cyp = int(row.cyp_index)
        from rdkit import Chem
        mol = Chem.MolFromSmiles(rec["canonical_smiles"])
        cases = [("baseline", "whole_molecule", -1, rec)]
        for bond in mol.GetBonds():
            b = bond.GetIdx(); label = f"{bond.GetBeginAtomIdx()}-{bond.GetEndAtomIdx()}:{bond.GetBondType()}"
            cases.append(("bond_deletion", label, b, _delete_bond(rec, b)))
            if bond.IsInRing():
                opened = _delete_bond(rec, b)
                opened_x = np.asarray(opened["x"], dtype=float)
                opened_x[:, 19] = 0.0
                opened["x"] = opened_x.tolist()
                cases.append(("ring_opening", label, b, opened))
            current = int(np.argmax(np.asarray(rec["edge"])[2*b, :4]))
            for target_type in range(4):
                if target_type != current:
                    cases.append(("bond_type_change", f"{label}->{type_names[target_type]}", b,
                                  _alter_bond(rec, b, target_type)))
        for group, indices in name_groups.items():
            cases.append(("atom_feature_ablation", group, -1, _ablate_features(rec, indices)))
        for intervention_no, (kind, label, index, changed) in enumerate(cases):
            values, blocks = _lle(model, changed, cyp, torch, device, **settings,
                                  seed=91001 + case_no * 100003 + intervention_no * 101)
            rows.append({"molecule_id": row.molecule_id, "cyp_target": row.cyp_target,
                         "intervention": kind, "target": label, "bond_index": index,
                         "ring_bond": bool(index >= 0 and mol.GetBondWithIdx(index).IsInRing()),
                         "mean_lyapunov": np.mean(values), "std_lyapunov": np.std(values),
                         "positive_block_fraction": np.mean(blocks), "repeats": len(values)})
            print(json.dumps({"phase": "intervention", "molecule": row.molecule_id,
                              "case": intervention_no + 1, "total": len(cases),
                              "kind": kind, "target": label, "lle": np.mean(values)}), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "causal_interventions.csv", index=False)


def run_structure_dynamics_campaign(*, model, data, device, torch_module,
                                    screening_path: Path, output_dir: Path,
                                    atom_feature_names: list[str]):
    output_dir.mkdir(parents=True, exist_ok=True)
    model.double()
    per_cyp = int(os.environ.get("SME_STRUCTURE_CASES_PER_CYP", "64"))
    settings = dict(
        burn_in=int(os.environ.get("SME_STRUCTURE_BURN_IN", "1000")),
        measured=int(os.environ.get("SME_STRUCTURE_GENERATIONS", "2000")),
        interval=int(os.environ.get("SME_STRUCTURE_INTERVAL", "10")),
        epsilon=float(os.environ.get("SME_STRUCTURE_EPSILON", "1e-7")),
        repeats=int(os.environ.get("SME_STRUCTURE_REPEATS", "2")),
    )
    started = time.perf_counter()
    screening = pd.read_csv(screening_path)
    population = _select_population(screening, per_cyp, 260830)
    required = screening[((screening.molecule_id == "OCNT-0494110") & (screening.cyp_target == "CYP2C9")) |
                         ((screening.molecule_id == "OCNT-2328784") & (screening.cyp_target == "CYP1A2"))]
    population = pd.concat((population, required)).drop_duplicates(
        ["molecule_id", "cyp_target"]).reset_index(drop=True)
    population.to_csv(output_dir / "selected_population.csv", index=False)
    descriptor_cache = {}
    rows = []
    for case_no, row in enumerate(population.itertuples(), 1):
        rec = data["train"][int(row.training_index)]
        descriptor_cache.setdefault(rec["name"], _descriptors(rec))
        values, blocks = _lle(model, rec, int(row.cyp_index), torch_module, device,
                              **settings, seed=51001 + case_no * 1009)
        rows.append({**descriptor_cache[rec["name"]], "training_index": int(row.training_index),
                     "cyp_index": int(row.cyp_index), "cyp_target": row.cyp_target,
                     "mean_lyapunov": np.mean(values), "std_lyapunov": np.std(values),
                     "minimum_lyapunov": min(values), "maximum_lyapunov": max(values),
                     "positive_repeat_fraction": np.mean(np.asarray(values) > 0),
                     "positive_block_fraction": np.mean(blocks),
                     "short_dynamic_score": float(row.dynamic_score)})
        print(json.dumps({"phase": "population", "case": case_no,
                          "total": len(population), "molecule": row.molecule_id,
                          "cyp": row.cyp_target, "lle": np.mean(values)}), flush=True)
    pd.DataFrame(rows).to_csv(output_dir / "structure_dynamics_population.csv", index=False)
    _interventions(model, data, population, atom_feature_names, torch_module, device,
                   settings, output_dir)
    metadata = {"method": "frozen Kuramoto-Sakaguchi population and intervention campaign",
                "device": str(device), "population_cases": len(rows), **settings,
                "atom_feature_names": atom_feature_names,
                "elapsed_seconds": time.perf_counter() - started}
    (output_dir / "campaign_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)
