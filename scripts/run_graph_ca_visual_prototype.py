#!/usr/bin/env python3
"""One-attempt OpenADMET graph-CA prototype.

Run in three phases so RDKit and PyTorch never share an OpenMP process:

  python scripts/run_graph_ca_visual_prototype.py prepare
  python scripts/run_graph_ca_visual_prototype.py train
  python scripts/run_graph_ca_visual_prototype.py render

The prepare/render phases require RDKit. The train phase requires PyTorch.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import random
from pathlib import Path

import numpy as np

try:
    from runtime_device import load_checkpoint, resolve_torch_device, runtime_metadata
    from challenge_metrics import (bootstrap_macro_soft_threshold_rae,
                                   bootstrap_regression_report,
                                   macro_soft_threshold_rae)
except ModuleNotFoundError:  # Imported as scripts.run_graph_ca_visual_prototype.
    from scripts.runtime_device import load_checkpoint, resolve_torch_device, runtime_metadata
    from scripts.challenge_metrics import (bootstrap_macro_soft_threshold_rae,
                                           bootstrap_regression_report,
                                           macro_soft_threshold_rae)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "openadmet-cyp-challenge-2026"
TRAIN_CSV = DATA / "cyp-challenge-TRAIN_inhibition.csv"
TEST_CSV = DATA / "cyp-challenge-TEST-BLINDED.csv"
CACHE = Path(os.environ.get(
    "SME_GRAPH_CACHE",
    str(ROOT / "tmp" / "strange_matter_graph_ca_graphs.pkl"),
))
RULE = os.environ.get("SME_CA_RULE", "gated_residual")
GENERATIONS = int(os.environ.get("SME_GENERATIONS", "16"))
RUN_NAME = os.environ.get("SME_RUN_NAME", "graph_ca_visual_prototype")
OUT = ROOT / "results" / RUN_NAME
CA_LR = float(os.environ.get("SME_CA_LR", "1e-3"))
RIDGE_STRENGTH = float(os.environ.get("SME_RIDGE", "1e-3"))
CA_L2 = float(os.environ.get("SME_CA_L2", "1e-5"))
GRAD_CLIP = float(os.environ.get("SME_GRAD_CLIP", "1.0"))
UPDATE_SCALE = float(os.environ.get("SME_UPDATE_SCALE", "0.25"))
INIT_SCALE = float(os.environ.get("SME_INIT_SCALE", "1.0"))
INITIAL_NOISE = float(os.environ.get("SME_INITIAL_NOISE", "0.0"))
SUPPORT_FRACTION = float(os.environ.get("SME_SUPPORT_FRACTION", "0.75"))
BOND_TEMPERATURE = float(os.environ.get("SME_BOND_TEMPERATURE", "1.0"))
DYN_A = float(os.environ.get("SME_DYN_A", "0.5"))
DYN_B = float(os.environ.get("SME_DYN_B", "0.5"))
DYN_C = float(os.environ.get("SME_DYN_C", "0.5"))
DYN_D = float(os.environ.get("SME_DYN_D", "0.5"))
PATIENCE_LIMIT = int(os.environ.get("SME_PATIENCE", "20"))
MIN_DELTA = float(os.environ.get("SME_MIN_DELTA", "0.005"))
TUNING_ONLY = os.environ.get("SME_TUNING_ONLY", "0") == "1"
TRAJECTORY_POOLING = os.environ.get("SME_TRAJECTORY_POOLING", "legacy")
CHEMICAL_FEATURE_GATING = os.environ.get("SME_CHEMICAL_FEATURE_GATING", "0") == "1"
RIDGE_MODE = os.environ.get("SME_RIDGE_MODE", "shared")
LOSS_MODE = os.environ.get("SME_LOSS_MODE", "mse")
INTERVAL_LOSS_BETA = float(os.environ.get("SME_INTERVAL_LOSS_BETA", "0.5"))
INTERVAL_TEMPERATURE = float(os.environ.get("SME_INTERVAL_TEMPERATURE", "0.05"))
PERTURBATION_CONSISTENCY_WEIGHT = float(os.environ.get("SME_PERTURBATION_CONSISTENCY_WEIGHT", "0.0"))
PERTURBATION_CONSISTENCY_EPSILON = float(os.environ.get("SME_PERTURBATION_CONSISTENCY_EPSILON", "0.001"))
INITIAL_STATE_ANCHOR = os.environ.get("SME_INITIAL_STATE_ANCHOR", "0") == "1"
SEED = int(os.environ.get("SME_SEED", "1701"))
CYPS = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")
ACTIVE_CYP = os.environ.get("SME_ACTIVE_CYP", "").strip()
if ACTIVE_CYP and ACTIVE_CYP not in CYPS:
    raise ValueError(f"SME_ACTIVE_CYP must be one of {CYPS}, received {ACTIVE_CYP!r}")
ACTIVE_CYP_INDEX = CYPS.index(ACTIVE_CYP) if ACTIVE_CYP else None
SPECIALIST_OBJECTIVE = os.environ.get("SME_SPECIALIST_OBJECTIVE", "shared")
if SPECIALIST_OBJECTIVE not in {"shared", "endpoint_only", "partial_pool"}:
    raise ValueError(
        "SME_SPECIALIST_OBJECTIVE must be 'shared', 'endpoint_only', or 'partial_pool'"
    )
if SPECIALIST_OBJECTIVE in {"endpoint_only", "partial_pool"} and ACTIVE_CYP_INDEX is None:
    raise ValueError(f"{SPECIALIST_OBJECTIVE} training requires SME_ACTIVE_CYP")
AUXILIARY_ENDPOINT_WEIGHT = float(os.environ.get("SME_AUXILIARY_ENDPOINT_WEIGHT", "0.15"))
if not 0.0 <= AUXILIARY_ENDPOINT_WEIGHT <= 1.0:
    raise ValueError("SME_AUXILIARY_ENDPOINT_WEIGHT must lie in [0, 1]")
LABEL_COLS = tuple(f"{c}_pIC50_direct_inhibition" for c in CYPS)
LABEL_HIGH_COLS = tuple(f"{col}_conf_high" for col in LABEL_COLS)
LABEL_LOW_COLS = tuple(f"{col}_conf_low" for col in LABEL_COLS)

BASELINE_ATOM_FEATURES = (
    "element_H", "element_C", "element_N", "element_O", "element_F",
    "element_P", "element_S", "element_Cl", "element_Br", "element_I",
    "element_other", "formal_charge", "aromatic", "hybrid_sp",
    "hybrid_sp2", "hybrid_sp3", "hybrid_other", "degree", "total_hydrogens",
    "in_ring", "hbond_donor", "hbond_acceptor", "chiral_cw", "chiral_ccw",
    "chiral_other",
)
PERIODIC_ATOM_FEATURES = (
    "atomic_number", "atomic_mass", "covalent_radius", "vdw_radius",
    "outer_electrons", "period", "periodic_group", "atomic_volume",
)
VALENCE_ATOM_FEATURES = (
    "total_valence", "implicit_valence", "heavy_atom_degree",
    "radical_electrons", "absolute_formal_charge",
)
ELECTRONIC_ATOM_FEATURES = (
    "electronegativity", "polarizability", "heteroatom", "halogen",
    "conjugated_bond_fraction", "first_ionization_energy", "electron_affinity",
    "electronic_property_missing",
)
LOCAL_ENVIRONMENT_ATOM_FEATURES = (
    "mean_neighbour_electronegativity", "electronegativity_difference",
    "mean_neighbour_atomic_number", "neighbour_formal_charge",
)
RING_GEOMETRY_ATOM_FEATURES = (
    "ring_count", "ring_size_3", "ring_size_4", "ring_size_5", "ring_size_6",
    "ring_size_7", "ring_size_8_plus",
)
ATOM_FEATURE_NAMES = (BASELINE_ATOM_FEATURES + PERIODIC_ATOM_FEATURES
                      + VALENCE_ATOM_FEATURES + ELECTRONIC_ATOM_FEATURES
                      + RING_GEOMETRY_ATOM_FEATURES
                      + LOCAL_ENVIRONMENT_ATOM_FEATURES)
ATOM_FEATURE_PROFILES = {
    "baseline": BASELINE_ATOM_FEATURES,
    "periodic": BASELINE_ATOM_FEATURES + PERIODIC_ATOM_FEATURES,
    "valence": BASELINE_ATOM_FEATURES + VALENCE_ATOM_FEATURES,
    "electronic": BASELINE_ATOM_FEATURES + ELECTRONIC_ATOM_FEATURES,
    "ring_geometry": BASELINE_ATOM_FEATURES + RING_GEOMETRY_ATOM_FEATURES,
    "local_environment": BASELINE_ATOM_FEATURES + LOCAL_ENVIRONMENT_ATOM_FEATURES,
    "periodic_valence": (BASELINE_ATOM_FEATURES + PERIODIC_ATOM_FEATURES
                         + VALENCE_ATOM_FEATURES),
    "periodic_electronic": (BASELINE_ATOM_FEATURES + PERIODIC_ATOM_FEATURES
                            + ELECTRONIC_ATOM_FEATURES),
    "valence_electronic": (BASELINE_ATOM_FEATURES + VALENCE_ATOM_FEATURES
                           + ELECTRONIC_ATOM_FEATURES),
    "electronic_local": (BASELINE_ATOM_FEATURES + ELECTRONIC_ATOM_FEATURES
                          + LOCAL_ENVIRONMENT_ATOM_FEATURES),
    "comprehensive": ATOM_FEATURE_NAMES,
}


def set_seeds() -> None:
    random.seed(SEED)
    np.random.seed(SEED)


def differentiable_ridge_fit(features, targets, penalty: float):
    """Solve standardized ridge without penalizing the intercept.

    All returned tensors remain connected to ``features`` and ``targets`` so a
    query loss can differentiate through the linear solve.
    """
    import torch

    if features.ndim != 2 or targets.ndim != 1 or len(features) != len(targets):
        raise ValueError("Ridge features must be [n, p] and targets must be [n]")
    if len(features) < 2:
        raise ValueError("Differentiable ridge requires at least two support observations")
    if penalty <= 0:
        raise ValueError("Ridge penalty must be strictly positive")
    feature_mean = features.mean(0)
    feature_scale = features.std(0, unbiased=False).clamp_min(1e-6)
    target_mean = targets.mean()
    standardized = (features - feature_mean) / feature_scale
    centered_target = targets - target_mean
    identity = torch.eye(features.shape[1], dtype=features.dtype, device=features.device)
    system = standardized.T @ standardized + penalty * identity
    rhs = standardized.T @ centered_target
    try:
        coefficients = torch.linalg.solve(system, rhs)
    except torch._C._LinAlgError:
        # Highly correlated trajectory statistics can still make the
        # float32 normal equations numerically singular.  The Hermitian
        # pseudoinverse preserves the same ridge objective and remains
        # differentiable, while allowing an unstable search candidate to be
        # measured rather than aborting the complete production study.
        coefficients = torch.linalg.pinv(system, hermitian=True) @ rhs
    return {
        "coefficients": coefficients,
        "intercept": target_mean,
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
    }


def differentiable_ridge_predict(features, ridge_state):
    standardized = ((features - ridge_state["feature_mean"])
                    / ridge_state["feature_scale"])
    return ridge_state["intercept"] + standardized @ ridge_state["coefficients"]


def atom_features(atom, donor_ids: set[int], acceptor_ids: set[int], mol) -> list[float]:
    from rdkit import Chem

    elements = (1, 6, 7, 8, 9, 15, 16, 17, 35, 53)
    z = atom.GetAtomicNum()
    element = [float(z == e) for e in elements] + [float(z not in elements)]
    hybs = (
        Chem.HybridizationType.SP,
        Chem.HybridizationType.SP2,
        Chem.HybridizationType.SP3,
        Chem.HybridizationType.OTHER,
    )
    hybrid = [float(atom.GetHybridization() == h) for h in hybs]
    chiral = atom.GetChiralTag()
    chirality = [
        float(chiral == Chem.ChiralType.CHI_TETRAHEDRAL_CW),
        float(chiral == Chem.ChiralType.CHI_TETRAHEDRAL_CCW),
        float(chiral not in (Chem.ChiralType.CHI_UNSPECIFIED,
                             Chem.ChiralType.CHI_TETRAHEDRAL_CW,
                             Chem.ChiralType.CHI_TETRAHEDRAL_CCW)),
    ]
    baseline = element + [
        atom.GetFormalCharge() / 3.0,
        float(atom.GetIsAromatic()),
    ] + hybrid + [
        atom.GetDegree() / 4.0,
        atom.GetTotalNumHs() / 4.0,
        float(atom.IsInRing()),
        float(atom.GetIdx() in donor_ids),
        float(atom.GetIdx() in acceptor_ids),
    ] + chirality
    periodic_table = Chem.GetPeriodicTable()
    period_group_volume = {
        1: (1, 1, 14.1), 6: (2, 14, 16.5), 7: (2, 15, 17.3),
        8: (2, 16, 14.0), 9: (2, 17, 17.1), 15: (3, 15, 24.4),
        16: (3, 16, 24.4), 17: (3, 17, 22.7), 35: (4, 17, 27.1),
        53: (5, 17, 32.5),
    }
    period, group, atomic_volume = period_group_volume.get(z, (0, 0, 0.0))
    periodic = [
        z / 100.0,
        periodic_table.GetAtomicWeight(z) / 250.0,
        periodic_table.GetRcovalent(z) / 2.5,
        periodic_table.GetRvdw(z) / 3.0,
        periodic_table.GetNOuterElecs(z) / 8.0,
        period / 7.0,
        group / 18.0,
        atomic_volume / 40.0,
    ]
    explicit_valence = atom.GetValence(Chem.ValenceType.EXPLICIT)
    implicit_valence = atom.GetValence(Chem.ValenceType.IMPLICIT)
    valence = [
        (explicit_valence + implicit_valence) / 8.0,
        implicit_valence / 4.0,
        sum(1 for neighbour in atom.GetNeighbors() if neighbour.GetAtomicNum() > 1) / 4.0,
        atom.GetNumRadicalElectrons() / 2.0,
        abs(atom.GetFormalCharge()) / 3.0,
    ]
    # Pauling electronegativity and approximate atomic polarizability (A^3)
    # for the elements represented explicitly by the baseline encoding.
    electronegativity = {
        1: 2.20, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 15: 2.19,
        16: 2.58, 17: 3.16, 35: 2.96, 53: 2.66,
    }.get(z, 2.5)
    polarizability = {
        1: 0.667, 6: 1.76, 7: 1.10, 8: 0.802, 9: 0.557, 15: 3.63,
        16: 2.90, 17: 2.18, 35: 3.05, 53: 5.35,
    }.get(z, 2.5)
    ionization = {1: 13.60, 6: 11.26, 7: 14.53, 8: 13.62, 9: 17.42,
                  15: 10.49, 16: 10.36, 17: 12.97, 35: 11.81, 53: 10.45}
    affinity = {1: 0.75, 6: 1.26, 7: -0.07, 8: 1.46, 9: 3.40,
                15: 0.75, 16: 2.08, 17: 3.61, 35: 3.36, 53: 3.06}
    electronic_missing = float(z not in ionization)
    bonds = list(atom.GetBonds())
    electronic = [
        electronegativity / 4.0,
        polarizability / 6.0,
        float(z not in (1, 6)),
        float(z in (9, 17, 35, 53)),
        (sum(float(bond.GetIsConjugated()) for bond in bonds) / len(bonds)
         if bonds else 0.0),
        ionization.get(z, 0.0) / 20.0,
        affinity.get(z, 0.0) / 4.0,
        electronic_missing,
    ]
    atom_index = atom.GetIdx()
    ring_sizes = [len(ring) for ring in mol.GetRingInfo().AtomRings()
                  if atom_index in ring]
    ring_geometry = [
        min(len(ring_sizes), 4) / 4.0,
        *[float(size in ring_sizes) for size in (3, 4, 5, 6, 7)],
        float(any(size >= 8 for size in ring_sizes)),
    ]
    neighbours = list(atom.GetNeighbors())
    neighbour_en = [{1: 2.20, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98,
                     15: 2.19, 16: 2.58, 17: 3.16, 35: 2.96, 53: 2.66}.get(
                         neighbour.GetAtomicNum(), 2.5) for neighbour in neighbours]
    mean_neighbour_en = float(np.mean(neighbour_en)) if neighbour_en else electronegativity
    local_environment = [
        mean_neighbour_en / 4.0,
        (electronegativity - mean_neighbour_en) / 4.0,
        (float(np.mean([n.GetAtomicNum() for n in neighbours])) / 100.0
         if neighbours else z / 100.0),
        (float(np.mean([n.GetFormalCharge() for n in neighbours])) / 3.0
         if neighbours else 0.0),
    ]
    return baseline + periodic + valence + electronic + ring_geometry + local_environment


def bond_features(bond) -> list[float]:
    from rdkit import Chem

    bt = bond.GetBondType()
    types = [
        float(bt == Chem.BondType.SINGLE),
        float(bt == Chem.BondType.DOUBLE),
        float(bt == Chem.BondType.TRIPLE),
        float(bt == Chem.BondType.AROMATIC),
    ]
    stereo = bond.GetStereo()
    stereos = [
        float(stereo in (Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOCIS)),
        float(stereo in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOTRANS)),
        float(stereo not in (Chem.BondStereo.STEREONONE,
                             Chem.BondStereo.STEREOANY,
                             Chem.BondStereo.STEREOZ,
                             Chem.BondStereo.STEREOCIS,
                             Chem.BondStereo.STEREOE,
                             Chem.BondStereo.STEREOTRANS)),
    ]
    return types + [float(bond.GetIsConjugated()), float(bond.IsInRing())] + stereos


def standardise(smiles: str):
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    mol = rdMolStandardize.Cleanup(mol)
    mol = rdMolStandardize.LargestFragmentChooser(preferOrganic=True).choose(mol)
    Chem.SanitizeMol(mol)
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    return mol, canonical


def graph_record(name: str, smiles: str, labels=None, label_lows=None,
                 label_highs=None) -> dict | None:
    from rdkit.Chem import Lipinski
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol, canonical = standardise(smiles)
    if mol is None:
        return None
    donor_ids = {idx for match in Lipinski._HDonors(mol) for idx in match}
    acceptor_ids = {idx for match in Lipinski._HAcceptors(mol) for idx in match}
    x = [atom_features(a, donor_ids, acceptor_ids, mol) for a in mol.GetAtoms()]
    src, dst, edges = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        e = bond_features(bond)
        src.extend((i, j)); dst.extend((j, i)); edges.extend((e, e))
    if not edges:
        edges = []
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    if not scaffold:
        scaffold = f"ACYCLIC::{canonical}"
    return {
        "name": str(name), "smiles": str(smiles), "canonical_smiles": canonical,
        "scaffold": scaffold, "x": x, "src": src,
        "dst": dst, "edge": edges,
        "labels": None if labels is None else [float(v) for v in labels],
        "label_conf_low": (None if label_lows is None
                           else [float(v) for v in label_lows]),
        "label_conf_high": (None if label_highs is None
                            else [float(v) for v in label_highs]),
    }


def prepare() -> None:
    import pandas as pd

    set_seeds()
    train_df = pd.read_csv(TRAIN_CSV)
    include_blind = os.environ.get("SME_INCLUDE_BLIND", "1") == "1"
    test_df = pd.read_csv(TEST_CSV) if include_blind else None
    train, test, rejected = [], [], []
    for row in train_df.itertuples(index=False):
        labels = [getattr(row, col) for col in LABEL_COLS]
        label_lows = [getattr(row, col) for col in LABEL_LOW_COLS]
        label_highs = [getattr(row, col) for col in LABEL_HIGH_COLS]
        rec = graph_record(row.Molecule_Name, row.SMILES, labels,
                           label_lows, label_highs)
        (train if rec is not None else rejected).append(rec if rec is not None else row.Molecule_Name)
    if test_df is not None:
        for row in test_df.itertuples(index=False):
            rec = graph_record(row.Molecule_Name, row.SMILES)
            (test if rec is not None else rejected).append(
                rec if rec is not None else row.Molecule_Name
            )

    groups = sorted({r["scaffold"] for r in train})
    rng = random.Random(SEED)
    rng.shuffle(groups)
    validation_groups = set(groups[:max(1, round(0.2 * len(groups)))])
    train_idx = [i for i, r in enumerate(train) if r["scaffold"] not in validation_groups]
    val_idx = [i for i, r in enumerate(train) if r["scaffold"] in validation_groups]
    payload = {"train": train, "test": test, "train_idx": train_idx,
               "val_idx": val_idx, "rejected": rejected, "seed": SEED}
    payload["challenge_metric_schema"] = "challenge_aligned_v2"
    payload["atom_feature_names"] = list(ATOM_FEATURE_NAMES)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(json.dumps({"training_molecules": len(train), "blinded_molecules": len(test),
                      "fit_molecules": len(train_idx), "validation_molecules": len(val_idx),
                      "blinded_data_loaded": include_blind,
                      "rejected": rejected, "cache": str(CACHE)}, indent=2))


def train(extended_dynamics: bool = False) -> None:
    global RULE, GENERATIONS, UPDATE_SCALE, INIT_SCALE, INITIAL_NOISE
    global SUPPORT_FRACTION, BOND_TEMPERATURE, DYN_A, DYN_B, DYN_C, DYN_D
    global TRAJECTORY_POOLING, RIDGE_MODE, CHEMICAL_FEATURE_GATING
    import torch
    from torch import nn

    set_seeds()
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    with CACHE.open("rb") as handle:
        data = pickle.load(handle)
    residual_targets_path = os.environ.get("SME_RESIDUAL_TARGETS")
    residual_mode = bool(residual_targets_path)
    residual_alpha = float(os.environ.get("SME_RESIDUAL_ALPHA", "1.0"))
    if residual_mode:
        with Path(residual_targets_path).open(newline="", encoding="utf-8") as handle:
            residual_rows = list(csv.DictReader(handle))
        residual_lookup = {
            (row["molecule_id"], row["cyp_target"]): row
            for row in residual_rows
        }
        missing_targets = []
        for record in data["train"]:
            original_labels = list(record["labels"])
            record["original_labels"] = original_labels
            record["base_predictions"] = [float("nan")] * len(CYPS)
            residual_labels = list(original_labels)
            for cyp_index, cyp in enumerate(CYPS):
                if not np.isfinite(original_labels[cyp_index]):
                    continue
                row = residual_lookup.get((record["name"], cyp))
                if row is None:
                    missing_targets.append((record["name"], cyp))
                    residual_labels[cyp_index] = float("nan")
                    continue
                base_prediction = float(row["base_prediction"])
                record["base_predictions"][cyp_index] = base_prediction
                residual_labels[cyp_index] = float(row.get(
                    "residual_target",
                    float(original_labels[cyp_index]) - base_prediction,
                ))
            record["labels"] = residual_labels
        if missing_targets:
            raise ValueError(
                f"Residual target table is missing {len(missing_targets)} observed rows; "
                f"first missing key is {missing_targets[0]}"
            )
    device = resolve_torch_device(torch)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cuda.matmul.allow_tf32 = True
    run_runtime = runtime_metadata(torch, device)
    print(json.dumps(run_runtime), flush=True)
    checkpoint = None
    inference_only = os.environ.get("SME_INFERENCE_ONLY", "0") == "1"
    checkpoint_evaluation = os.environ.get("SME_EVALUATE_CHECKPOINT", "0") == "1"
    if extended_dynamics or inference_only or checkpoint_evaluation:
        checkpoint_path = os.environ.get("SME_CHECKPOINT")
        if not checkpoint_path:
            raise ValueError("SME_CHECKPOINT is required for checkpoint inference")
        checkpoint = load_checkpoint(torch, checkpoint_path, device)
        hyperparameters = checkpoint["hyperparameters"]
        RULE = checkpoint["rule"]
        GENERATIONS = (int(os.environ.get("SME_EXTENDED_GENERATIONS", "5000"))
                       if extended_dynamics else int(checkpoint["generations"]))
        UPDATE_SCALE = float(hyperparameters["update_scale"])
        INIT_SCALE = float(hyperparameters["init_scale"])
        INITIAL_NOISE = 0.0
        SUPPORT_FRACTION = float(hyperparameters["support_fraction"])
        BOND_TEMPERATURE = float(hyperparameters["bond_temperature"])
        DYN_A = float(hyperparameters["dyn_a"])
        DYN_B = float(hyperparameters["dyn_b"])
        DYN_C = float(hyperparameters["dyn_c"])
        DYN_D = float(hyperparameters["dyn_d"])
        TRAJECTORY_POOLING = checkpoint.get("trajectory_pooling", "legacy")
        RIDGE_MODE = checkpoint.get("ridge_mode", "shared")
        CHEMICAL_FEATURE_GATING = bool(checkpoint.get("chemical_feature_gating", False))
    feature_profile = (checkpoint.get("atom_feature_profile", "baseline")
                       if checkpoint is not None
                       else os.environ.get("SME_ATOM_FEATURE_PROFILE", "baseline"))
    if feature_profile not in ATOM_FEATURE_PROFILES:
        raise ValueError(f"Unknown atom feature profile: {feature_profile}")
    cached_feature_names = tuple(data.get("atom_feature_names", BASELINE_ATOM_FEATURES))
    requested_feature_names = ATOM_FEATURE_PROFILES[feature_profile]
    missing_features = [name for name in requested_feature_names
                        if name not in cached_feature_names]
    if missing_features:
        raise ValueError(
            f"Graph cache lacks atom features required by {feature_profile}: "
            + ", ".join(missing_features)
        )
    feature_indices = [cached_feature_names.index(name)
                       for name in requested_feature_names]
    for split in ("train", "test"):
        for record in data.get(split, []):
            record["x"] = [[row[index] for index in feature_indices]
                           for row in record["x"]]
    data["selected_atom_feature_names"] = list(requested_feature_names)
    chem_dim = len(data["train"][0]["x"][0])
    bond_dim = 9
    hidden = (int(checkpoint["state_dict"]["init.weight"].shape[0])
              if checkpoint is not None
              else int(os.environ.get("SME_HIDDEN_CHANNELS", "8")))

    class GraphCA(nn.Module):
        def __init__(self):
            super().__init__()
            self.init = nn.Linear(chem_dim, hidden)
            self.self_layer = nn.Linear(hidden, hidden, bias=False)
            self.neighbour = nn.Linear(hidden, hidden, bias=False)
            self.bond = nn.Linear(bond_dim, hidden, bias=False)
            self.bond_gate = nn.Linear(bond_dim, hidden)
            self.chem = nn.Linear(chem_dim, hidden, bias=False)
            self.context = nn.Linear(4, hidden, bias=False)
            self.bias = nn.Parameter(torch.zeros(hidden))
            if CHEMICAL_FEATURE_GATING:
                self.feature_gate_logits = nn.Parameter(torch.full((chem_dim,), 2.0))
            if TRAJECTORY_POOLING == "temporal_attention":
                self.temporal_logits = nn.Parameter(torch.zeros(5))
            if RULE == "gated_residual":
                self.gate = nn.Linear(hidden * 2 + chem_dim + 4, hidden)
            elif RULE == "inertial_reaction_diffusion":
                self.raw_gamma = nn.Parameter(torch.zeros(hidden))
                self.raw_dt = nn.Parameter(torch.zeros(hidden))
                self.raw_diffusion = nn.Parameter(torch.full((hidden,), -1.5))
                self.raw_restoring = nn.Parameter(torch.full((hidden,), -1.5))
            elif RULE == "activator_inhibitor":
                if hidden % 2:
                    raise ValueError("Activator-inhibitor rule requires an even channel count")
                half = hidden // 2
                self.activator_drive = nn.Linear(hidden, half)
                self.inhibitor_drive = nn.Linear(hidden, half)
            elif RULE == "coupled_map":
                self.map_drive = nn.Linear(hidden, hidden)
            elif RULE == "damped_symplectic":
                if hidden % 2:
                    raise ValueError("Damped-symplectic rule requires an even channel count")
                self.force_drive = nn.Linear(hidden, hidden // 2)
            elif RULE == "fitzhugh_nagumo":
                if hidden % 2:
                    raise ValueError("FitzHugh-Nagumo rule requires an even channel count")
                self.excitation_drive = nn.Linear(hidden, hidden // 2)
                self.recovery_drive = nn.Linear(hidden, hidden // 2)
            elif RULE == "gray_scott":
                if hidden % 2:
                    raise ValueError("Gray-Scott rule requires an even channel count")
                self.gray_scott_drive = nn.Linear(hidden, hidden)
            elif RULE == "kuramoto_sakaguchi":
                self.frequency_drive = nn.Linear(hidden, hidden)
            elif RULE == "conservative_graph_flux":
                self.flux_drive = nn.Linear(hidden, hidden, bias=False)
            elif RULE == "delayed_memory":
                self.delayed_drive = nn.Linear(hidden * 2, hidden)
            else:
                raise ValueError(f"Unknown CA rule: {RULE}")

        @staticmethod
        def _graph_mean(values, graph_index, graph_count, atom_counts):
            pooled = torch.zeros((graph_count, values.shape[1]), device=device)
            pooled.index_add_(0, graph_index, values)
            return pooled / atom_counts[:, None]

        @staticmethod
        def _readout_features(fingerprint, endpoint_indices):
            if RIDGE_MODE == "shared":
                return fingerprint
            if RIDGE_MODE != "per_endpoint":
                raise ValueError(f"Unknown ridge mode: {RIDGE_MODE}")
            endpoint_indices = torch.as_tensor(
                endpoint_indices, dtype=torch.long, device=fingerprint.device
            )
            one_hot = torch.nn.functional.one_hot(
                endpoint_indices, num_classes=len(CYPS)
            ).to(fingerprint.dtype)
            blocks = fingerprint[:, None, :] * one_hot[:, :, None]
            return torch.cat((one_hot, blocks.flatten(1)), dim=1)

        def _chemical_features(self, x):
            if not CHEMICAL_FEATURE_GATING:
                return x
            return x * torch.sigmoid(self.feature_gate_logits)[None, :]

        def forward_batch(self, examples, initial_perturbations=None,
                          return_node_trajectory=False):
            """Evaluate independent molecule-CYP graphs as one disconnected graph."""
            graph_count = len(examples)
            xs, srcs, dsts, edges, graph_ids, contexts, atom_counts = [], [], [], [], [], [], []
            offset = 0
            for graph_id, (rec, cyp) in enumerate(examples):
                x_part = torch.as_tensor(rec["x"], device=device)
                atom_count = x_part.shape[0]
                xs.append(x_part)
                graph_ids.append(torch.full((atom_count,), graph_id, dtype=torch.long, device=device))
                context = torch.zeros(4, device=device); context[cyp] = 1.0
                contexts.append(context.expand(atom_count, -1))
                atom_counts.append(atom_count)
                if rec["src"]:
                    srcs.append(torch.as_tensor(rec["src"], device=device) + offset)
                    dsts.append(torch.as_tensor(rec["dst"], device=device) + offset)
                    edges.append(torch.as_tensor(rec["edge"], device=device))
                offset += atom_count
            x = torch.cat(xs)
            x = self._chemical_features(x)
            graph_index = torch.cat(graph_ids)
            c = torch.cat(contexts)
            atom_counts_tensor = torch.tensor(atom_counts, dtype=x.dtype, device=device)
            if srcs:
                src, dst, edge = torch.cat(srcs), torch.cat(dsts), torch.cat(edges)
            else:
                src = dst = torch.empty(0, dtype=torch.long, device=device)
                edge = torch.empty((0, bond_dim), device=device)

            h = torch.tanh(INIT_SCALE * self.init(x))
            if self.training and INITIAL_NOISE > 0:
                h = h + INITIAL_NOISE * torch.randn_like(h)
            if initial_perturbations is not None:
                h = h + torch.cat(initial_perturbations)
            initial_mean = self._graph_mean(h, graph_index, graph_count, atom_counts_tensor)
            initial_second = self._graph_mean(h.square(), graph_index, graph_count, atom_counts_tensor)
            initial_var = (initial_second - initial_mean.square()).clamp_min(0.0)
            velocity = torch.zeros_like(h)
            state_history = [h]
            node_states = [h] if return_node_trajectory else None
            temporal_atom_sum = h.clone()
            graph_mean = self._graph_mean(h, graph_index, graph_count, atom_counts_tensor)
            graph_mean_sum = graph_mean.clone()
            graph_mean_sq_sum = graph_mean.square()
            checkpoint_steps = tuple(max(1, round(GENERATIONS * fraction))
                                     for fraction in (0.125, 0.25, 0.5, 0.75, 1.0))
            checkpoint_summaries = []
            step_energy = torch.zeros_like(h)
            for step_index in range(1, GENERATIONS + 1):
                agg = torch.zeros_like(h)
                neighbour_mean = torch.zeros_like(h)
                degree = torch.zeros((h.shape[0], 1), device=device)
                if src.numel():
                    edge_gate = torch.sigmoid(self.bond_gate(edge) / BOND_TEMPERATURE)
                    msg = edge_gate * self.neighbour(h[src]) + self.bond(edge)
                    agg.index_add_(0, dst, msg)
                    neighbour_mean.index_add_(0, dst, h[src])
                    degree.index_add_(0, dst, torch.ones((dst.numel(), 1), device=device))
                    agg = agg / degree.clamp_min(1.0)
                    neighbour_mean = neighbour_mean / degree.clamp_min(1.0)
                reaction = torch.tanh(self.self_layer(h) + agg + self.chem(x) +
                                      self.context(c) + self.bias)
                if RULE == "gated_residual":
                    alpha = torch.sigmoid(self.gate(torch.cat((h, agg, x, c), dim=1)))
                    scaled_alpha = (UPDATE_SCALE * alpha).clamp(max=1.0)
                    new_h = (1.0 - scaled_alpha) * h + scaled_alpha * reaction
                elif RULE == "inertial_reaction_diffusion":
                    gamma = 0.99 * torch.sigmoid(self.raw_gamma)
                    dt = UPDATE_SCALE * torch.sigmoid(self.raw_dt)
                    diffusion = DYN_A * torch.nn.functional.softplus(self.raw_diffusion)
                    restoring = DYN_B * torch.nn.functional.softplus(self.raw_restoring)
                    force = reaction + diffusion * (neighbour_mean - h) - restoring * h
                    velocity = (0.5 + 0.49 * DYN_C) * gamma * velocity + dt * force
                    new_h = torch.tanh(h + dt * velocity)
                elif RULE == "activator_inhibitor":
                    u, v = h.chunk(2, dim=1)
                    neighbour_u, neighbour_v = neighbour_mean.chunk(2, dim=1)
                    drive_u = torch.tanh(self.activator_drive(reaction))
                    drive_v = torch.tanh(self.inhibitor_drive(reaction))
                    du = u - u.pow(3) / 3.0 - v + DYN_C * drive_u + DYN_A * (neighbour_u - u)
                    dv = DYN_D * (u + (2.0 * DYN_B - 1.0) - (0.5 + DYN_C) * v) + DYN_B * (neighbour_v - v) + 0.1 * drive_v
                    new_h = torch.tanh(torch.cat((u + UPDATE_SCALE * du, v + UPDATE_SCALE * dv), dim=1))
                elif RULE == "coupled_map":
                    q = (h + 1.0) * 0.5
                    r = 2.5 + 1.5 * DYN_A
                    local = r * q * (1.0 - q)
                    conditioned = torch.sigmoid(self.map_drive(reaction))
                    local = (1.0 - 0.25 * DYN_C) * local + 0.25 * DYN_C * conditioned
                    neighbour_q = (neighbour_mean + 1.0) * 0.5
                    neighbour_map = r * neighbour_q * (1.0 - neighbour_q)
                    coupling = min(0.95, UPDATE_SCALE * DYN_B)
                    mapped = (1.0 - coupling) * local + coupling * neighbour_map
                    new_h = 2.0 * mapped.clamp(0.0, 1.0) - 1.0
                elif RULE == "damped_symplectic":
                    q, p = h.chunk(2, dim=1)
                    neighbour_q = neighbour_mean[:, :q.shape[1]]
                    force = (torch.tanh(self.force_drive(reaction))
                             + DYN_A * (neighbour_q - q) - DYN_B * q)
                    damping = 0.8 + 0.199 * DYN_C
                    new_p = damping * p + UPDATE_SCALE * force
                    new_q = torch.tanh(q + UPDATE_SCALE * new_p)
                    new_h = torch.cat((new_q, torch.tanh(new_p)), dim=1)
                elif RULE == "fitzhugh_nagumo":
                    u, v = h.chunk(2, dim=1)
                    neighbour_u, neighbour_v = neighbour_mean.chunk(2, dim=1)
                    stimulus = torch.tanh(self.excitation_drive(reaction))
                    recovery_input = torch.tanh(self.recovery_drive(reaction))
                    epsilon = 0.01 + 0.19 * DYN_D
                    threshold = 0.2 + 0.8 * DYN_B
                    du = (u - u.pow(3) / 3.0 - v + DYN_C * stimulus
                          + DYN_A * (neighbour_u - u))
                    dv = epsilon * (u + threshold - v + 0.1 * recovery_input)
                    dv = dv + 0.25 * DYN_A * (neighbour_v - v)
                    new_h = torch.tanh(torch.cat(
                        (u + UPDATE_SCALE * du, v + UPDATE_SCALE * dv), dim=1
                    ))
                elif RULE == "gray_scott":
                    u_raw, v_raw = h.chunk(2, dim=1)
                    neighbour_u_raw, neighbour_v_raw = neighbour_mean.chunk(2, dim=1)
                    u, v = (u_raw + 1.0) * 0.5, (v_raw + 1.0) * 0.5
                    neighbour_u = (neighbour_u_raw + 1.0) * 0.5
                    neighbour_v = (neighbour_v_raw + 1.0) * 0.5
                    drive_u, drive_v = torch.tanh(self.gray_scott_drive(reaction)).chunk(2, dim=1)
                    diffusion_u = 0.01 + 0.19 * DYN_A
                    diffusion_v = 0.005 + 0.095 * DYN_B
                    feed = 0.01 + 0.07 * DYN_C
                    kill = 0.03 + 0.04 * DYN_D
                    reaction_uv = u * v.square()
                    du = (diffusion_u * (neighbour_u - u) - reaction_uv
                          + feed * (1.0 - u) + 0.02 * drive_u)
                    dv = (diffusion_v * (neighbour_v - v) + reaction_uv
                          - (feed + kill) * v + 0.02 * drive_v)
                    next_u = (u + UPDATE_SCALE * du).clamp(0.0, 1.0)
                    next_v = (v + UPDATE_SCALE * dv).clamp(0.0, 1.0)
                    new_h = 2.0 * torch.cat((next_u, next_v), dim=1) - 1.0
                elif RULE == "kuramoto_sakaguchi":
                    phase = math.pi * h
                    phase_coupling = torch.zeros_like(h)
                    if src.numel():
                        lag = math.pi * (DYN_C - 0.5)
                        phase_messages = edge_gate * torch.sin(phase[src] - phase[dst] - lag)
                        phase_coupling.index_add_(0, dst, phase_messages)
                        phase_coupling = phase_coupling / degree.clamp_min(1.0)
                    natural_frequency = DYN_A * torch.tanh(self.frequency_drive(reaction))
                    next_phase = phase + UPDATE_SCALE * (
                        natural_frequency + DYN_B * phase_coupling
                    )
                    new_h = torch.atan2(torch.sin(next_phase), torch.cos(next_phase)) / math.pi
                elif RULE == "conservative_graph_flux":
                    net_flux = torch.zeros_like(h)
                    if src.numel():
                        # Each undirected bond is stored in both directions.  The odd
                        # flux law therefore produces equal and opposite transfers.
                        directed_flux = edge_gate * torch.tanh(self.flux_drive(h[src] - h[dst]))
                        net_flux.index_add_(0, dst, directed_flux)
                    normalizer = degree.max().clamp_min(1.0)
                    new_h = h + UPDATE_SCALE * DYN_A * net_flux / normalizer
                else:  # delayed_memory
                    delay = max(1, min(len(state_history), round(1 + 15 * DYN_A)))
                    delayed_h = state_history[-delay]
                    delayed_reaction = torch.tanh(self.delayed_drive(torch.cat((reaction, delayed_h), dim=1)))
                    memory_mix = 0.1 + 0.8 * DYN_B
                    damping = 0.05 + 0.45 * DYN_D
                    drive = ((1.0 - memory_mix) * reaction + memory_mix * delayed_reaction
                             + DYN_C * (delayed_h - h))
                    new_h = torch.tanh((1.0 - damping * UPDATE_SCALE) * h
                                       + UPDATE_SCALE * drive)
                step_energy += (new_h - h).square() / float(GENERATIONS)
                h = new_h
                state_history.append(h)
                if return_node_trajectory:
                    node_states.append(h)
                temporal_atom_sum += h
                graph_mean = self._graph_mean(h, graph_index, graph_count, atom_counts_tensor)
                graph_mean_sum += graph_mean
                graph_mean_sq_sum += graph_mean.square()
                if TRAJECTORY_POOLING in {"multiscale", "temporal_attention"} and step_index in checkpoint_steps:
                    checkpoint_summaries.append(graph_mean)

            final_mean = self._graph_mean(h, graph_index, graph_count, atom_counts_tensor)
            final_second = self._graph_mean(h.square(), graph_index, graph_count, atom_counts_tensor)
            final_var = (final_second - final_mean.square()).clamp_min(0.0)
            temporal_mean = self._graph_mean(
                temporal_atom_sum / float(GENERATIONS + 1),
                graph_index, graph_count, atom_counts_tensor,
            )
            series_mean = graph_mean_sum / float(GENERATIONS + 1)
            series_var = (graph_mean_sq_sum / float(GENERATIONS + 1) - series_mean.square()).clamp_min(0.0)
            energy_mean = self._graph_mean(step_energy, graph_index, graph_count, atom_counts_tensor)
            fingerprint = torch.cat((final_mean, final_var, temporal_mean, series_var, energy_mean), dim=1)
            if INITIAL_STATE_ANCHOR:
                fingerprint = torch.cat((fingerprint, initial_mean, initial_var), dim=1)
            if TRAJECTORY_POOLING == "multiscale":
                if len(checkpoint_summaries) != 5:
                    raise RuntimeError("Multiscale checkpoint collection is incomplete")
                fingerprint = torch.cat((fingerprint, *checkpoint_summaries), dim=1)
            elif TRAJECTORY_POOLING == "temporal_attention":
                if len(checkpoint_summaries) != 5:
                    raise RuntimeError("Temporal-attention checkpoints are incomplete")
                checkpoints = torch.stack(checkpoint_summaries, dim=1)
                weights = torch.softmax(self.temporal_logits, dim=0)
                weighted_mean = (checkpoints * weights[None, :, None]).sum(dim=1)
                weighted_var = ((checkpoints - weighted_mean[:, None, :]).square()
                                * weights[None, :, None]).sum(dim=1)
                fingerprint = torch.cat((fingerprint, weighted_mean, weighted_var), dim=1)
            fingerprint = self._readout_features(
                fingerprint, [cyp for _, cyp in examples]
            )
            if return_node_trajectory:
                return fingerprint, torch.stack(node_states), graph_index, atom_counts
            return fingerprint

        def forward_one(self, rec, cyp: int, return_trajectory=False,
                        initial_perturbation=None):
            x = torch.as_tensor(rec["x"], device=device)
            x = self._chemical_features(x)
            src = torch.as_tensor(rec["src"], device=device)
            dst = torch.as_tensor(rec["dst"], device=device)
            edge = torch.as_tensor(rec["edge"], device=device)
            context = torch.zeros(4, device=device); context[cyp] = 1.0
            h = torch.tanh(INIT_SCALE * self.init(x))
            if initial_perturbation is not None:
                h = h + initial_perturbation
            initial_mean = h.mean(0)
            initial_var = h.var(0, unbiased=False)
            velocity = torch.zeros_like(h)
            state_history = [h]
            states = [h]
            means = [h.mean(0)]
            checkpoint_steps = tuple(max(1, round(GENERATIONS * fraction))
                                     for fraction in (0.125, 0.25, 0.5, 0.75, 1.0))
            checkpoint_summaries = []
            step_energy = torch.zeros(hidden, device=device)
            for step_index in range(1, GENERATIONS + 1):
                agg = torch.zeros_like(h)
                neighbour_mean = torch.zeros_like(h)
                degree = torch.zeros((h.shape[0], 1), device=device)
                if src.numel():
                    edge_gate = torch.sigmoid(self.bond_gate(edge) / BOND_TEMPERATURE)
                    msg = edge_gate * self.neighbour(h[src]) + self.bond(edge)
                    agg.index_add_(0, dst, msg)
                    neighbour_mean.index_add_(0, dst, h[src])
                    degree.index_add_(0, dst, torch.ones((dst.numel(), 1), device=device))
                    agg = agg / degree.clamp_min(1.0)
                    neighbour_mean = neighbour_mean / degree.clamp_min(1.0)
                c = context.expand(h.shape[0], -1)
                reaction = torch.tanh(self.self_layer(h) + agg + self.chem(x) +
                                      self.context(c) + self.bias)
                if RULE == "gated_residual":
                    alpha = torch.sigmoid(self.gate(torch.cat((h, agg, x, c), dim=1)))
                    scaled_alpha = (UPDATE_SCALE * alpha).clamp(max=1.0)
                    new_h = (1.0 - scaled_alpha) * h + scaled_alpha * reaction
                elif RULE == "inertial_reaction_diffusion":
                    gamma = 0.99 * torch.sigmoid(self.raw_gamma)
                    dt = UPDATE_SCALE * torch.sigmoid(self.raw_dt)
                    diffusion = DYN_A * torch.nn.functional.softplus(self.raw_diffusion)
                    restoring = DYN_B * torch.nn.functional.softplus(self.raw_restoring)
                    force = reaction + diffusion * (neighbour_mean - h) - restoring * h
                    velocity = (0.5 + 0.49 * DYN_C) * gamma * velocity + dt * force
                    new_h = torch.tanh(h + dt * velocity)
                elif RULE == "activator_inhibitor":
                    u, v = h.chunk(2, dim=1)
                    neighbour_u, neighbour_v = neighbour_mean.chunk(2, dim=1)
                    drive_u = torch.tanh(self.activator_drive(reaction))
                    drive_v = torch.tanh(self.inhibitor_drive(reaction))
                    du = u - u.pow(3) / 3.0 - v + DYN_C * drive_u + DYN_A * (neighbour_u - u)
                    dv = DYN_D * (u + (2.0 * DYN_B - 1.0) - (0.5 + DYN_C) * v) + DYN_B * (neighbour_v - v) + 0.1 * drive_v
                    new_h = torch.tanh(torch.cat((u + UPDATE_SCALE * du, v + UPDATE_SCALE * dv), dim=1))
                elif RULE == "coupled_map":
                    q = (h + 1.0) * 0.5
                    r = 2.5 + 1.5 * DYN_A
                    local = r * q * (1.0 - q)
                    conditioned = torch.sigmoid(self.map_drive(reaction))
                    local = (1.0 - 0.25 * DYN_C) * local + 0.25 * DYN_C * conditioned
                    neighbour_q = (neighbour_mean + 1.0) * 0.5
                    neighbour_map = r * neighbour_q * (1.0 - neighbour_q)
                    coupling = min(0.95, UPDATE_SCALE * DYN_B)
                    mapped = (1.0 - coupling) * local + coupling * neighbour_map
                    new_h = 2.0 * mapped.clamp(0.0, 1.0) - 1.0
                elif RULE == "damped_symplectic":
                    q, p = h.chunk(2, dim=1)
                    neighbour_q = neighbour_mean[:, :q.shape[1]]
                    force = (torch.tanh(self.force_drive(reaction))
                             + DYN_A * (neighbour_q - q) - DYN_B * q)
                    damping = 0.8 + 0.199 * DYN_C
                    new_p = damping * p + UPDATE_SCALE * force
                    new_q = torch.tanh(q + UPDATE_SCALE * new_p)
                    new_h = torch.cat((new_q, torch.tanh(new_p)), dim=1)
                elif RULE == "fitzhugh_nagumo":
                    u, v = h.chunk(2, dim=1)
                    neighbour_u, neighbour_v = neighbour_mean.chunk(2, dim=1)
                    stimulus = torch.tanh(self.excitation_drive(reaction))
                    recovery_input = torch.tanh(self.recovery_drive(reaction))
                    epsilon = 0.01 + 0.19 * DYN_D
                    threshold = 0.2 + 0.8 * DYN_B
                    du = (u - u.pow(3) / 3.0 - v + DYN_C * stimulus
                          + DYN_A * (neighbour_u - u))
                    dv = (epsilon * (u + threshold - v + 0.1 * recovery_input)
                          + 0.25 * DYN_A * (neighbour_v - v))
                    new_h = torch.tanh(torch.cat(
                        (u + UPDATE_SCALE * du, v + UPDATE_SCALE * dv), dim=1
                    ))
                elif RULE == "gray_scott":
                    u_raw, v_raw = h.chunk(2, dim=1)
                    neighbour_u_raw, neighbour_v_raw = neighbour_mean.chunk(2, dim=1)
                    u, v = (u_raw + 1.0) * 0.5, (v_raw + 1.0) * 0.5
                    neighbour_u = (neighbour_u_raw + 1.0) * 0.5
                    neighbour_v = (neighbour_v_raw + 1.0) * 0.5
                    drive_u, drive_v = torch.tanh(self.gray_scott_drive(reaction)).chunk(2, dim=1)
                    diffusion_u = 0.01 + 0.19 * DYN_A
                    diffusion_v = 0.005 + 0.095 * DYN_B
                    feed = 0.01 + 0.07 * DYN_C
                    kill = 0.03 + 0.04 * DYN_D
                    reaction_uv = u * v.square()
                    next_u = (u + UPDATE_SCALE * (diffusion_u * (neighbour_u - u)
                              - reaction_uv + feed * (1.0 - u) + 0.02 * drive_u)).clamp(0.0, 1.0)
                    next_v = (v + UPDATE_SCALE * (diffusion_v * (neighbour_v - v)
                              + reaction_uv - (feed + kill) * v + 0.02 * drive_v)).clamp(0.0, 1.0)
                    new_h = 2.0 * torch.cat((next_u, next_v), dim=1) - 1.0
                elif RULE == "kuramoto_sakaguchi":
                    phase = math.pi * h
                    phase_coupling = torch.zeros_like(h)
                    if src.numel():
                        lag = math.pi * (DYN_C - 0.5)
                        phase_messages = edge_gate * torch.sin(phase[src] - phase[dst] - lag)
                        phase_coupling.index_add_(0, dst, phase_messages)
                        phase_coupling = phase_coupling / degree.clamp_min(1.0)
                    natural_frequency = DYN_A * torch.tanh(self.frequency_drive(reaction))
                    next_phase = phase + UPDATE_SCALE * (natural_frequency + DYN_B * phase_coupling)
                    new_h = torch.atan2(torch.sin(next_phase), torch.cos(next_phase)) / math.pi
                elif RULE == "conservative_graph_flux":
                    net_flux = torch.zeros_like(h)
                    if src.numel():
                        directed_flux = edge_gate * torch.tanh(self.flux_drive(h[src] - h[dst]))
                        net_flux.index_add_(0, dst, directed_flux)
                    new_h = h + UPDATE_SCALE * DYN_A * net_flux / degree.max().clamp_min(1.0)
                else:  # delayed_memory
                    delay = max(1, min(len(state_history), round(1 + 15 * DYN_A)))
                    delayed_h = state_history[-delay]
                    delayed_reaction = torch.tanh(self.delayed_drive(torch.cat((reaction, delayed_h), dim=1)))
                    memory_mix = 0.1 + 0.8 * DYN_B
                    damping = 0.05 + 0.45 * DYN_D
                    drive = ((1.0 - memory_mix) * reaction + memory_mix * delayed_reaction
                             + DYN_C * (delayed_h - h))
                    new_h = torch.tanh((1.0 - damping * UPDATE_SCALE) * h
                                       + UPDATE_SCALE * drive)
                step_energy += ((new_h - h) ** 2).mean(0) / float(GENERATIONS)
                h = new_h
                state_history.append(h)
                states.append(h); means.append(h.mean(0))
                if TRAJECTORY_POOLING in {"multiscale", "temporal_attention"} and step_index in checkpoint_steps:
                    checkpoint_summaries.append(h.mean(0))
            mean_series = torch.stack(means)
            fingerprint = torch.cat((
                h.mean(0), h.var(0, unbiased=False),
                torch.stack(states).mean((0, 1)),
                mean_series.var(0, unbiased=False), step_energy,
            ))
            if INITIAL_STATE_ANCHOR:
                fingerprint = torch.cat((fingerprint, initial_mean, initial_var))
            if TRAJECTORY_POOLING == "multiscale":
                if len(checkpoint_summaries) != 5:
                    raise RuntimeError("Multiscale checkpoint collection is incomplete")
                fingerprint = torch.cat((fingerprint, *checkpoint_summaries))
            elif TRAJECTORY_POOLING == "temporal_attention":
                if len(checkpoint_summaries) != 5:
                    raise RuntimeError("Temporal-attention checkpoints are incomplete")
                checkpoints = torch.stack(checkpoint_summaries)
                weights = torch.softmax(self.temporal_logits, dim=0)
                weighted_mean = (checkpoints * weights[:, None]).sum(dim=0)
                weighted_var = ((checkpoints - weighted_mean[None, :]).square()
                                * weights[:, None]).sum(dim=0)
                fingerprint = torch.cat((fingerprint, weighted_mean, weighted_var))
            fingerprint = self._readout_features(
                fingerprint.unsqueeze(0), [cyp]
            ).squeeze(0)
            pred = fingerprint
            if return_trajectory:
                return pred, torch.stack(states)
            return pred

        def initial_node_state(self, rec):
            """Return the learned initial CA state for dynamical analysis."""
            dtype = next(self.parameters()).dtype
            x = torch.as_tensor(rec["x"], device=device, dtype=dtype)
            x = self._chemical_features(x)
            return torch.tanh(INIT_SCALE * self.init(x))

        def kuramoto_step(self, rec, cyp: int, h):
            """Advance one frozen Kuramoto-Sakaguchi generation from an arbitrary state."""
            if RULE != "kuramoto_sakaguchi":
                raise ValueError("kuramoto_step is only defined for the Kuramoto-Sakaguchi rule")
            x = torch.as_tensor(rec["x"], device=device, dtype=h.dtype)
            x = self._chemical_features(x)
            src = torch.as_tensor(rec["src"], device=device)
            dst = torch.as_tensor(rec["dst"], device=device)
            edge = torch.as_tensor(rec["edge"], device=device, dtype=h.dtype)
            context = torch.zeros(4, device=device, dtype=h.dtype); context[cyp] = 1.0
            agg = torch.zeros_like(h)
            degree = torch.zeros((h.shape[0], 1), device=device, dtype=h.dtype)
            if src.numel():
                edge_gate = torch.sigmoid(self.bond_gate(edge) / BOND_TEMPERATURE)
                msg = edge_gate * self.neighbour(h[src]) + self.bond(edge)
                agg.index_add_(0, dst, msg)
                degree.index_add_(0, dst, torch.ones((dst.numel(), 1), device=device,
                                                     dtype=h.dtype))
                agg = agg / degree.clamp_min(1.0)
            c = context.expand(h.shape[0], -1)
            reaction = torch.tanh(self.self_layer(h) + agg + self.chem(x) +
                                  self.context(c) + self.bias)
            phase = math.pi * h
            phase_coupling = torch.zeros_like(h)
            if src.numel():
                lag = math.pi * (DYN_C - 0.5)
                phase_messages = edge_gate * torch.sin(phase[src] - phase[dst] - lag)
                phase_coupling.index_add_(0, dst, phase_messages)
                phase_coupling = phase_coupling / degree.clamp_min(1.0)
            natural_frequency = DYN_A * torch.tanh(self.frequency_drive(reaction))
            next_phase = phase + UPDATE_SCALE * (natural_frequency + DYN_B * phase_coupling)
            return torch.atan2(torch.sin(next_phase), torch.cos(next_phase)) / math.pi

    model = GraphCA().to(device)
    if inference_only:
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        ridge_state = {
            key: value.to(device) for key, value in checkpoint["ridge_state"].items()
        }
        prediction_rows = []
        examples = [(record, cyp_index)
                    for record in data.get("test", [])
                    for cyp_index in range(len(CYPS))]
        if not examples:
            raise ValueError("Inference cache contains no blinded test molecules")
        with torch.no_grad():
            for start in range(0, len(examples), 128):
                chunk = examples[start:start + 128]
                fingerprints = model.forward_batch(chunk)
                predictions = differentiable_ridge_predict(fingerprints, ridge_state)
                for (record, cyp_index), prediction in zip(chunk, predictions):
                    prediction_rows.append({
                        "molecule_id": record["name"],
                        "smiles": record["smiles"],
                        "canonical_smiles": record["canonical_smiles"],
                        "cyp_target": CYPS[cyp_index],
                        "predicted_pic50": float(prediction),
                    })
        OUT.mkdir(parents=True, exist_ok=True)
        with (OUT / "blinded_test_predictions.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=prediction_rows[0].keys())
            writer.writeheader(); writer.writerows(prediction_rows)
        inference_manifest = {
            "checkpoint": str(Path(os.environ["SME_CHECKPOINT"]).resolve()),
            "rule": RULE,
            "generations": GENERATIONS,
            "trajectory_pooling": TRAJECTORY_POOLING,
            "ridge_mode": RIDGE_MODE,
            "test_molecules": len(data["test"]),
            "predictions": len(prediction_rows),
            "labels_loaded": False,
            "runtime": run_runtime,
        }
        (OUT / "inference_manifest.json").write_text(
            json.dumps(inference_manifest, indent=2) + "\n"
        )
        print(json.dumps(inference_manifest, indent=2))
        return
    if extended_dynamics:
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        try:
            from run_extended_coupled_map_dynamics import run_extended_analysis
        except ModuleNotFoundError:
            from scripts.run_extended_coupled_map_dynamics import run_extended_analysis
        if os.environ.get("SME_SKIP_EXTENDED_ANALYSIS", "0") != "1":
            run_extended_analysis(
                model=model,
                data=data,
                device=device,
                torch_module=torch,
                hidden_channels=hidden,
                generations=GENERATIONS,
                checkpoint_path=Path(os.environ["SME_CHECKPOINT"]),
                screening_path=Path(os.environ["SME_SCREENING_CSV"]),
                output_dir=Path(os.environ["SME_EXTENDED_OUTPUT"]),
                candidate_count=int(os.environ.get("SME_EXTENDED_CANDIDATES", "100")),
                burn_in=int(os.environ.get("SME_EXTENDED_BURN_IN", "1000")),
                rule=RULE,
            )
        if os.environ.get("SME_RENORMALIZED_LYAPUNOV", "0") == "1":
            try:
                from run_renormalized_lyapunov import run_renormalized_campaign
            except ModuleNotFoundError:
                from scripts.run_renormalized_lyapunov import run_renormalized_campaign
            run_renormalized_campaign(
                model=model, data=data, device=device, torch_module=torch,
                selected_path=Path(os.environ["SME_EXTENDED_OUTPUT"]) / "selected_candidates.csv",
                output_dir=Path(os.environ["SME_RENORMALIZED_OUTPUT"]),
            )
        if os.environ.get("SME_LYAPUNOV_SPECTRUM", "0") == "1":
            try:
                from run_renormalized_lyapunov import run_lyapunov_spectrum_campaign
            except ModuleNotFoundError:
                from scripts.run_renormalized_lyapunov import run_lyapunov_spectrum_campaign
            run_lyapunov_spectrum_campaign(
                model=model, data=data, device=device, torch_module=torch,
                selected_path=Path(os.environ["SME_EXTENDED_OUTPUT"]) / "selected_candidates.csv",
                output_dir=Path(os.environ["SME_SPECTRUM_OUTPUT"]),
            )
        if os.environ.get("SME_ATTRACTOR_BASIN", "0") == "1":
            try:
                from run_attractor_basin import run_attractor_basin_campaign
            except ModuleNotFoundError:
                from scripts.run_attractor_basin import run_attractor_basin_campaign
            run_attractor_basin_campaign(
                model=model, data=data, device=device, torch_module=torch,
                selected_path=Path(os.environ["SME_EXTENDED_OUTPUT"]) / "selected_candidates.csv",
                output_dir=Path(os.environ["SME_ATTRACTOR_BASIN_OUTPUT"]),
            )
        if os.environ.get("SME_STRUCTURE_DYNAMICS", "0") == "1":
            try:
                from run_structure_dynamics_campaign import run_structure_dynamics_campaign
            except ModuleNotFoundError:
                from scripts.run_structure_dynamics_campaign import run_structure_dynamics_campaign
            run_structure_dynamics_campaign(
                model=model, data=data, device=device, torch_module=torch,
                screening_path=Path(os.environ["SME_SCREENING_CSV"]),
                output_dir=Path(os.environ["SME_STRUCTURE_DYNAMICS_OUTPUT"]),
                atom_feature_names=list(requested_feature_names),
            )
        return
    if checkpoint_evaluation:
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
    ca_params = list(model.parameters())
    optimizer = torch.optim.Adam(ca_params, lr=CA_LR,
                                 betas=(0.9, 0.999), eps=1e-8)

    def observed_pairs(indices):
        pairs = []
        for i in indices:
            for c, y in enumerate(data["train"][i]["labels"]):
                if (ACTIVE_CYP_INDEX is None or c == ACTIVE_CYP_INDEX) and np.isfinite(y):
                    pairs.append((i, c, float(y)))
        return pairs

    metric_cyps = (ACTIVE_CYP,) if ACTIVE_CYP else CYPS

    def metric_endpoints(endpoints):
        if ACTIVE_CYP_INDEX is None:
            return endpoints
        if not np.all(endpoints == ACTIVE_CYP_INDEX):
            raise RuntimeError("A CYP-specialist run received another endpoint")
        return np.zeros_like(endpoints)

    def complete_per_cyp(values):
        return {name: float(values.get(name, np.nan)) for name in CYPS}

    fit_indices = list(data["train_idx"])
    validation_indices = list(data["val_idx"])
    reserved_validation_indices = list(validation_indices)
    cv_fold_text = os.environ.get("SME_CV_FOLD")
    cv_folds = int(os.environ.get("SME_CV_FOLDS", "5"))
    if cv_fold_text is not None:
        cv_fold = int(cv_fold_text)
        if not 0 <= cv_fold < cv_folds:
            raise ValueError("SME_CV_FOLD must lie in [0, SME_CV_FOLDS)")
        fitting_scaffolds = sorted({data["train"][i]["scaffold"] for i in fit_indices})
        cv_split_seed = int(os.environ.get("SME_CV_SPLIT_SEED", "260822"))
        fold_rng = random.Random(cv_split_seed)
        fold_rng.shuffle(fitting_scaffolds)
        validation_scaffolds = set(fitting_scaffolds[cv_fold::cv_folds])
        pool = fit_indices
        fit_indices = [i for i in pool
                       if data["train"][i]["scaffold"] not in validation_scaffolds]
        validation_indices = [i for i in pool
                              if data["train"][i]["scaffold"] in validation_scaffolds]
    if TUNING_ONLY:
        subset_rng = random.Random(SEED + 991)
        subset_rng.shuffle(fit_indices); subset_rng.shuffle(validation_indices)
        fit_limit = int(os.environ.get("SME_TUNING_FIT_MOLECULES", "600"))
        validation_limit = int(os.environ.get("SME_TUNING_VAL_MOLECULES", "200"))
        fit_indices = fit_indices[:fit_limit]
        validation_indices = validation_indices[:validation_limit]
    fit_pairs = observed_pairs(fit_indices)
    val_pairs = observed_pairs(validation_indices)

    interval_normalizers = {}
    for cyp_index in range(len(CYPS)):
        endpoint_pairs = [
            (i, float(record["labels"][cyp_index]))
            for i in fit_indices
            for record in [data["train"][i]]
            if np.isfinite(record["labels"][cyp_index])
        ]
        if not endpoint_pairs:
            interval_normalizers[cyp_index] = 1.0
            continue
        endpoint_targets = np.asarray([y for _, y in endpoint_pairs], dtype=float)
        baseline = float(endpoint_targets.mean())
        endpoint_errors = []
        for i, _ in endpoint_pairs:
            record = data["train"][i]
            low = float(record["label_conf_low"][cyp_index])
            high = float(record["label_conf_high"][cyp_index])
            endpoint_errors.append(max(low - baseline, baseline - high, 0.0))
        interval_normalizers[cyp_index] = max(float(np.mean(endpoint_errors)), 1e-3)

    def credible_bounds(pairs):
        lows, highs, endpoints = [], [], []
        for i, c, _ in pairs:
            record = data["train"][i]
            if (record.get("label_conf_low") is None
                    or record.get("label_conf_high") is None):
                raise ValueError(
                    "Graph cache lacks challenge credible intervals; rerun prepare"
                )
            lows.append(record["label_conf_low"][c])
            highs.append(record["label_conf_high"][c])
            endpoints.append(c)
        return np.asarray(lows), np.asarray(highs), np.asarray(endpoints)

    def fingerprints_and_targets(pairs):
        fingerprints, targets = [], []
        with torch.no_grad():
            for start in range(0, len(pairs), 64):
                chunk = pairs[start:start + 64]
                fingerprints.append(model.forward_batch([
                    (data["train"][i], c) for i, c, _ in chunk
                ]))
                targets.extend(y for _, _, y in chunk)
        return torch.cat(fingerprints), torch.tensor(targets, dtype=torch.float32, device=device)

    def fitted_ridge_state():
        model.eval()
        features, targets = fingerprints_and_targets(fit_pairs)
        return differentiable_ridge_fit(features, targets, RIDGE_STRENGTH)

    def evaluate(pairs, ridge_state, bootstrap=False):
        model.eval(); ys, ps = [], []
        features, targets = fingerprints_and_targets(pairs)
        with torch.no_grad():
            predictions = differentiable_ridge_predict(features, ridge_state)
        residual_ys = targets.cpu().numpy()
        residual_ps = predictions.cpu().numpy()
        if residual_mode:
            ys = np.asarray([
                data["train"][i]["original_labels"][c] for i, c, _ in pairs
            ], dtype=float)
            base = np.asarray([
                data["train"][i]["base_predictions"][c] for i, c, _ in pairs
            ], dtype=float)
            ps = base + residual_alpha * residual_ps
        else:
            ys, ps = residual_ys, residual_ps
        lows, highs, endpoints = credible_bounds(pairs)
        metric = (bootstrap_macro_soft_threshold_rae if bootstrap
                  else macro_soft_threshold_rae)
        ma_st_rae, per_cyp = metric(
            ys, ps, lows, highs, metric_endpoints(endpoints), metric_cyps
        )
        return (float(np.sqrt(np.mean((ys - ps) ** 2))), ma_st_rae,
                complete_per_cyp(per_cyp), ys, ps)

    if checkpoint_evaluation:
        ridge_state = {
            key: value.to(device) for key, value in checkpoint["ridge_state"].items()
        }
        pairs = observed_pairs(reserved_validation_indices)
        _, _, _, ys, predictions = evaluate(pairs, ridge_state, bootstrap=False)
        rows = []
        for (i, c, _), experimental, prediction in zip(pairs, ys, predictions):
            record = data["train"][i]
            low = record["label_conf_low"][c]
            high = record["label_conf_high"][c]
            rows.append({
                "split": "reserved_holdout", "molecule_id": record["name"],
                "cyp_target": CYPS[c], "experimental_pic50": experimental,
                "predicted_pic50": prediction, "credible_interval_low": low,
                "credible_interval_high": high,
                "soft_threshold_absolute_error": max(low - prediction,
                                                     prediction - high, 0.0),
                "residual": prediction - experimental,
            })
        OUT.mkdir(parents=True, exist_ok=True)
        with (OUT / "checkpoint_evaluation_predictions.csv").open(
                "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader(); writer.writerows(rows)
        print(json.dumps({"checkpoint_evaluation_observations": len(rows),
                          "active_cyp": ACTIVE_CYP or None}, indent=2))
        return

    history, best, best_rmse, best_state, patience = [], math.inf, math.inf, None, 0
    rng = random.Random(SEED)
    max_epochs = int(os.environ.get("SME_MAX_EPOCHS", "200"))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=CA_LR * 0.1
    )
    training_started = __import__("time").perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        molecule_order = list(fit_indices); rng.shuffle(molecule_order)
        total_sq, total_n, raw_norms, clipped_norms = 0.0, 0, [], []
        batch_molecules = int(os.environ.get("SME_BATCH_MOLECULES", "16"))
        for start in range(0, len(molecule_order), batch_molecules):
            molecule_batch = molecule_order[start:start + batch_molecules]
            support_count = max(2, min(len(molecule_batch) - 1,
                                       round(SUPPORT_FRACTION * len(molecule_batch))))
            support_molecules = molecule_batch[:support_count]
            query_molecules = molecule_batch[support_count:]
            support_batch, query_batch = [], []
            for i in support_molecules:
                for c, y in enumerate(data["train"][i]["labels"]):
                    if (np.isfinite(y) and
                            (SPECIALIST_OBJECTIVE != "endpoint_only" or
                             c == ACTIVE_CYP_INDEX)):
                        support_batch.append((i, c, float(y)))
            for i in query_molecules:
                for c, y in enumerate(data["train"][i]["labels"]):
                    if (np.isfinite(y) and
                            (SPECIALIST_OBJECTIVE != "endpoint_only" or
                             c == ACTIVE_CYP_INDEX)):
                        query_batch.append((i, c, float(y)))
            if len(support_batch) < 2 or not query_batch:
                continue
            optimizer.zero_grad()
            combined = support_batch + query_batch
            fingerprints = model.forward_batch([
                (data["train"][i], c) for i, c, _ in combined
            ])
            support_targets = torch.tensor([y for _, _, y in support_batch],
                                           dtype=torch.float32, device=device)
            targets = torch.tensor([y for _, _, y in query_batch],
                                   dtype=torch.float32, device=device)
            ridge_state = differentiable_ridge_fit(
                fingerprints[:len(support_batch)], support_targets, RIDGE_STRENGTH
            )
            preds = differentiable_ridge_predict(
                fingerprints[len(support_batch):], ridge_state
            )
            query_endpoints = torch.tensor(
                [c for _, c, _ in query_batch], dtype=torch.long, device=device
            )
            if SPECIALIST_OBJECTIVE == "partial_pool":
                auxiliary_per_endpoint = AUXILIARY_ENDPOINT_WEIGHT / (len(CYPS) - 1)
                query_weights = torch.where(
                    query_endpoints == ACTIVE_CYP_INDEX,
                    torch.ones_like(preds),
                    torch.full_like(preds, auxiliary_per_endpoint),
                )
            else:
                query_weights = torch.ones_like(preds)
            if LOSS_MODE == "mse":
                squared_error = (preds - targets) ** 2
                prediction_loss = ((squared_error * query_weights).sum()
                                   / query_weights.sum().clamp_min(1e-8))
            elif LOSS_MODE == "hybrid_interval":
                endpoint_losses = []
                endpoint_mse = []
                endpoint_weights = []
                query_lows = torch.tensor([
                    data["train"][i]["label_conf_low"][c]
                    for i, c, _ in query_batch
                ], dtype=preds.dtype, device=device)
                query_highs = torch.tensor([
                    data["train"][i]["label_conf_high"][c]
                    for i, c, _ in query_batch
                ], dtype=preds.dtype, device=device)
                temperature = max(INTERVAL_TEMPERATURE, 1e-4)
                outside = temperature * (
                    torch.nn.functional.softplus((query_lows - preds) / temperature)
                    + torch.nn.functional.softplus((preds - query_highs) / temperature)
                )
                for cyp_index in range(len(CYPS)):
                    selected = query_endpoints == cyp_index
                    if selected.any():
                        endpoint_losses.append(
                            outside[selected].mean() / interval_normalizers[cyp_index]
                        )
                        endpoint_mse.append(((preds[selected] - targets[selected]) ** 2).mean())
                        endpoint_weights.append(
                            1.0 if SPECIALIST_OBJECTIVE != "partial_pool"
                            or cyp_index == ACTIVE_CYP_INDEX
                            else AUXILIARY_ENDPOINT_WEIGHT / (len(CYPS) - 1)
                        )
                endpoint_weight_tensor = torch.tensor(
                    endpoint_weights, dtype=preds.dtype, device=device
                )
                endpoint_weight_tensor /= endpoint_weight_tensor.sum().clamp_min(1e-8)
                balanced_interval = (torch.stack(endpoint_losses)
                                     * endpoint_weight_tensor).sum()
                balanced_mse = (torch.stack(endpoint_mse)
                                * endpoint_weight_tensor).sum()
                prediction_loss = ((1.0 - INTERVAL_LOSS_BETA) * balanced_mse
                                   + INTERVAL_LOSS_BETA * balanced_interval)
            else:
                raise ValueError(f"Unknown loss mode: {LOSS_MODE}")
            if PERTURBATION_CONSISTENCY_WEIGHT > 0:
                perturbations = [
                    PERTURBATION_CONSISTENCY_EPSILON * torch.randn(
                        (len(data["train"][i]["x"]), HIDDEN_CHANNELS),
                        dtype=preds.dtype, device=device,
                    )
                    for i, _, _ in query_batch
                ]
                perturbed_fingerprints = model.forward_batch(
                    [(data["train"][i], c) for i, c, _ in query_batch],
                    initial_perturbations=perturbations,
                )
                perturbed_preds = differentiable_ridge_predict(
                    perturbed_fingerprints, ridge_state
                )
                consistency_loss = ((perturbed_preds - preds) ** 2).mean()
                prediction_loss = (prediction_loss
                                   + PERTURBATION_CONSISTENCY_WEIGHT * consistency_loss)
            ca_penalty = CA_L2 * sum((p ** 2).sum() for p in ca_params)
            loss = prediction_loss + ca_penalty
            loss.backward()
            raw = float(torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in model.parameters()
                                       if p.grad is not None)))
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            clipped = min(raw, GRAD_CLIP)
            optimizer.step()
            total_sq += float(((preds.detach() - targets) ** 2).sum())
            total_n += len(query_batch); raw_norms.append(raw); clipped_norms.append(clipped)
        train_rmse = math.sqrt(total_sq / total_n)
        epoch_ridge_state = fitted_ridge_state()
        val_rmse, val_ma_st_rae, per_cyp_st_rae, _, _ = evaluate(
            val_pairs, epoch_ridge_state
        )
        row = {"epoch": epoch, "train_rmse": train_rmse, "validation_rmse": val_rmse,
               "validation_ma_st_rae": val_ma_st_rae,
               **{f"validation_{cyp}_st_rae": per_cyp_st_rae[cyp] for cyp in CYPS},
               "mean_raw_gradient_norm": float(np.mean(raw_norms)),
               "fraction_gradients_clipped": float(np.mean(np.asarray(raw_norms) > GRAD_CLIP))}
        history.append(row)
        scheduler.step()
        print(json.dumps(row), flush=True)
        if val_ma_st_rae < best - MIN_DELTA:
            best, best_rmse, patience = val_ma_st_rae, val_rmse, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if patience >= PATIENCE_LIMIT:
            break
    if device.type == "cuda":
        torch.cuda.synchronize()
    training_seconds = __import__("time").perf_counter() - training_started
    model.load_state_dict(best_state)
    final_ridge_state = fitted_ridge_state()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "chem_dim": chem_dim, "seed": SEED,
                "rule": RULE, "generations": GENERATIONS, "device": str(device),
                "atom_feature_profile": feature_profile,
                "atom_feature_names": list(requested_feature_names),
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "training_seconds": training_seconds,
                "prediction_mode": "residual_ca" if residual_mode else "direct",
                "residual_alpha": residual_alpha if residual_mode else None,
                "trajectory_pooling": TRAJECTORY_POOLING,
                "chemical_feature_gating": CHEMICAL_FEATURE_GATING,
                "perturbation_consistency_weight": PERTURBATION_CONSISTENCY_WEIGHT,
                "perturbation_consistency_epsilon": PERTURBATION_CONSISTENCY_EPSILON,
                "ridge_mode": RIDGE_MODE,
                "loss_mode": LOSS_MODE,
                "specialist_objective": SPECIALIST_OBJECTIVE,
                "auxiliary_endpoint_weight": AUXILIARY_ENDPOINT_WEIGHT,
                "interval_loss_beta": INTERVAL_LOSS_BETA,
                "interval_temperature": INTERVAL_TEMPERATURE,
                "ridge_state": {k: v.detach().cpu() for k, v in final_ridge_state.items()},
                "hyperparameters": {"ca_lr": CA_LR,
                    "ridge": RIDGE_STRENGTH, "ca_l2": CA_L2,
                    "gradient_clip": GRAD_CLIP, "update_scale": UPDATE_SCALE,
                    "init_scale": INIT_SCALE, "initial_noise": INITIAL_NOISE,
                    "support_fraction": SUPPORT_FRACTION,
                    "bond_temperature": BOND_TEMPERATURE,
                    "dyn_a": DYN_A, "dyn_b": DYN_B,
                    "dyn_c": DYN_C, "dyn_d": DYN_D,
                    "lr_schedule": "cosine_0.1x"}}, OUT / "model.pt")
    with (OUT / "training_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys()); writer.writeheader(); writer.writerows(history)

    train_rmse, train_ma_st_rae, train_per_cyp, train_y, train_p = evaluate(
        fit_pairs, final_ridge_state, bootstrap=True
    )
    val_rmse, val_ma_st_rae, val_per_cyp, val_y, val_p = evaluate(
        val_pairs, final_ridge_state, bootstrap=True
    )
    reserved_pairs = (observed_pairs(reserved_validation_indices)
                      if os.environ.get("SME_EVALUATE_RESERVED_HOLDOUT", "0") == "1"
                      and cv_fold_text is not None else [])
    if reserved_pairs:
        _, _, _, reserved_y, reserved_p = evaluate(
            reserved_pairs, final_ridge_state, bootstrap=False
        )
    else:
        reserved_y, reserved_p = np.asarray([]), np.asarray([])
    val_lows, val_highs, val_endpoints = credible_bounds(val_pairs)
    val_point_ma_st_rae, val_point_per_cyp = macro_soft_threshold_rae(
        val_y, val_p, val_lows, val_highs,
        metric_endpoints(val_endpoints), metric_cyps
    )
    val_point_per_cyp = complete_per_cyp(val_point_per_cyp)
    challenge_report = bootstrap_regression_report(
        val_y, val_p, val_lows, val_highs,
        metric_endpoints(val_endpoints), metric_cyps
    )
    pair_rows = []
    for split, pairs, ys, ps in (("fit", fit_pairs, train_y, train_p),
                                ("validation", val_pairs, val_y, val_p),
                                ("reserved_holdout", reserved_pairs,
                                 reserved_y, reserved_p)):
        for (i, c, _), experimental, pred in zip(pairs, ys, ps):
            r = data["train"][i]
            low, high = r["label_conf_low"][c], r["label_conf_high"][c]
            row = {"split": split, "molecule_id": r["name"], "cyp_target": CYPS[c],
                              "experimental_pic50": experimental, "predicted_pic50": pred,
                              "credible_interval_low": low,
                              "credible_interval_high": high,
                              "soft_threshold_absolute_error": max(low - pred, pred - high, 0.0),
                              "residual": pred - experimental}
            if residual_mode:
                base = r["base_predictions"][c]
                row.update({
                    "base_prediction": base,
                    "residual_target": experimental - base,
                    "predicted_residual": (pred - base) / residual_alpha
                    if residual_alpha else 0.0,
                    "residual_alpha": residual_alpha,
                })
            pair_rows.append(row)
    with (OUT / "validation_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=pair_rows[0].keys()); writer.writeheader(); writer.writerows(pair_rows)

    common_metrics = {"seed": SEED, "selection_metric": "validation_ma_st_rae",
        "best_validation_ma_st_rae": best,
        "best_validation_rmse": best_rmse,
        "restored_fit_rmse": train_rmse, "restored_validation_rmse": val_rmse,
        "restored_fit_ma_st_rae": train_ma_st_rae,
        "restored_validation_ma_st_rae": val_ma_st_rae,
        "restored_validation_point_ma_st_rae": val_point_ma_st_rae,
        "restored_validation_point_st_rae_by_cyp": val_point_per_cyp,
        "restored_fit_st_rae_by_cyp": train_per_cyp,
        "restored_validation_st_rae_by_cyp": val_per_cyp,
        "validation_bootstrap_metrics": challenge_report,
        "reserved_holdout_observations": len(reserved_pairs),
        "epochs_run": len(history), "fit_observations": len(fit_pairs),
        "validation_observations": len(val_pairs), "rule": RULE,
        "generations": GENERATIONS, "hidden_channels": hidden,
        "atom_feature_profile": feature_profile,
        "atom_feature_count": chem_dim,
        "atom_feature_names": list(requested_feature_names),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "training_seconds": training_seconds,
        "peak_gpu_memory_bytes": (torch.cuda.max_memory_allocated(device)
                                  if device.type == "cuda" else 0),
        "runtime": run_runtime,
        "active_cyp": ACTIVE_CYP or None,
        "validation_protocol": ({"kind": "scaffold_cross_validation",
                                  "fold": int(cv_fold_text), "folds": cv_folds,
                                  "split_seed": int(os.environ.get(
                                      "SME_CV_SPLIT_SEED", "260822"))}
                                 if cv_fold_text is not None else
                                 {"kind": "reserved_scaffold_holdout"}),
        "readout": "differentiable_closed_form_ridge",
        "prediction_mode": "residual_ca" if residual_mode else "direct",
        "residual_alpha": residual_alpha if residual_mode else None,
        "trajectory_pooling": TRAJECTORY_POOLING,
        "ridge_mode": RIDGE_MODE,
        "loss_mode": LOSS_MODE,
        "specialist_objective": SPECIALIST_OBJECTIVE,
        "interval_loss_beta": INTERVAL_LOSS_BETA,
        "interval_temperature": INTERVAL_TEMPERATURE,
        "hyperparameters": {"ca_lr": CA_LR,
        "ridge": RIDGE_STRENGTH, "ca_l2": CA_L2,
        "gradient_clip": GRAD_CLIP, "update_scale": UPDATE_SCALE,
        "init_scale": INIT_SCALE, "initial_noise": INITIAL_NOISE,
        "support_fraction": SUPPORT_FRACTION,
        "bond_temperature": BOND_TEMPERATURE,
        "dyn_a": DYN_A, "dyn_b": DYN_B,
        "dyn_c": DYN_C, "dyn_d": DYN_D,
        "lr_schedule": "cosine_0.1x"}}
    if TUNING_ONLY and os.environ.get("SME_ANALYSE_VALIDATION", "0") == "1":
        print(json.dumps({"dynamics_phase": "screen_start",
                          "validation_pairs": len(val_pairs)}), flush=True)
        def trajectory_scores(trajectory):
            mean_state = trajectory.mean(axis=1)
            steps = np.linalg.norm(np.diff(mean_state, axis=0), axis=1)
            late_start = min(max(2, len(mean_state) // 2), len(mean_state) - 2)
            late = mean_state[late_start:]
            late_steps = np.linalg.norm(np.diff(late, axis=0), axis=1)
            mean_step = float(np.mean(late_steps)) + 1e-12
            recurrence = []
            for lag in range(2, min(64, max(2, len(late) // 3)) + 1):
                distance = float(np.mean(np.linalg.norm(late[lag:] - late[:-lag], axis=1)))
                recurrence.append((distance / (lag * mean_step + 1e-12), lag, distance))
            if recurrence:
                recurrence_ratio, recurrence_lag, recurrence_distance = min(recurrence)
            else:
                recurrence_ratio, recurrence_lag = float("inf"), 0
                recurrence_distance = float("inf")
            centered = late - late.mean(axis=0, keepdims=True)
            power = np.abs(np.fft.rfft(centered, axis=0)) ** 2
            power = power[1:]
            normalized = power / np.maximum(power.sum(axis=0, keepdims=True), 1e-12)
            spectral_entropy = float(np.mean(
                -np.sum(normalized * np.log(np.maximum(normalized, 1e-12)), axis=0)
                / np.log(max(2, len(normalized)))
            ))
            spectral_concentration = float(np.mean(
                np.max(power, axis=0) / np.maximum(power.sum(axis=0), 1e-12)
            ))
            return {
                "late_motion": mean_step,
                "final_step": float(steps[-1]),
                "late_amplitude": float(np.mean(np.std(late, axis=0))),
                "recurrence_ratio": float(recurrence_ratio),
                "recurrence_lag": int(recurrence_lag),
                "recurrence_distance": float(recurrence_distance),
                "spectral_entropy": spectral_entropy,
                "spectral_concentration": spectral_concentration,
            }

        score_rows = []
        model.eval()
        with torch.no_grad():
            for index, (i, c, y) in enumerate(val_pairs):
                fingerprint, trajectory_tensor = model.forward_one(
                    data["train"][i], c, return_trajectory=True
                )
                prediction = differentiable_ridge_predict(
                    fingerprint.unsqueeze(0), final_ridge_state
                ).squeeze(0)
                trajectory = trajectory_tensor.cpu().numpy()
                score_rows.append({
                    "validation_pair_index": index,
                    "training_index": i,
                    "molecule_id": data["train"][i]["name"],
                    "cyp_index": c,
                    "cyp_target": CYPS[c],
                    "experimental_pic50": y,
                    "predicted_pic50": float(prediction),
                    **trajectory_scores(trajectory),
                })
        print(json.dumps({"dynamics_phase": "screen_complete",
                          "validation_pairs": len(score_rows)}), flush=True)
        late_threshold = float(np.quantile([r["late_motion"] for r in score_rows], 0.75))
        entropy_threshold = float(np.quantile([r["spectral_entropy"] for r in score_rows], 0.75))
        perturbation_case_limit = int(os.environ.get("SME_PERTURBATION_CASES", "20"))
        ranked = sorted(
            range(len(score_rows)),
            key=lambda j: (
                score_rows[j]["late_motion"] *
                (1.0 + score_rows[j]["spectral_entropy"]) /
                max(score_rows[j]["recurrence_ratio"], 1e-6)
            ),
            reverse=True,
        )[:perturbation_case_limit]
        trajectory_case_limit = int(os.environ.get("SME_SAVE_TRAJECTORY_CASES", "0"))
        trajectory_ranked = sorted(
            range(len(score_rows)),
            key=lambda j: (
                score_rows[j]["late_motion"] *
                (1.0 + score_rows[j]["spectral_entropy"]) /
                max(score_rows[j]["recurrence_ratio"], 1e-6)
            ),
            reverse=True,
        )[:trajectory_case_limit]
        trajectory_archive = OUT / "trajectory_archive"
        trajectory_manifest = []
        if trajectory_ranked:
            trajectory_archive.mkdir(parents=True, exist_ok=True)
            with torch.no_grad():
                for archive_rank, score_index in enumerate(trajectory_ranked, start=1):
                    row = score_rows[score_index]
                    rec = data["train"][row["training_index"]]
                    _, trajectory_tensor = model.forward_one(
                        rec, row["cyp_index"], return_trajectory=True,
                    )
                    archive_name = (
                        f"{archive_rank:03d}_{row['molecule_id']}_"
                        f"{row['cyp_target']}.npz"
                    )
                    np.savez_compressed(
                        trajectory_archive / archive_name,
                        trajectory=trajectory_tensor.cpu().numpy().astype(np.float32),
                        atom_features=np.asarray(rec["x"], dtype=np.float32),
                        source_indices=np.asarray(rec["src"], dtype=np.int32),
                        destination_indices=np.asarray(rec["dst"], dtype=np.int32),
                        bond_features=np.asarray(rec["edge"], dtype=np.float32),
                        molecule_id=np.asarray(rec["name"]),
                        smiles=np.asarray(rec["smiles"]),
                        cyp_target=np.asarray(row["cyp_target"]),
                        prediction_generations=np.asarray(GENERATIONS),
                        atom_feature_names=np.asarray(requested_feature_names),
                    )
                    trajectory_manifest.append({
                        "archive_rank": archive_rank,
                        "file": archive_name,
                        **{key: value for key, value in row.items()
                           if key != "training_index"},
                        "atoms": len(rec["x"]),
                        "directed_edges": len(rec["src"]),
                        "saved_generations": GENERATIONS + 1,
                        "hidden_channels": hidden,
                    })
            with (trajectory_archive / "manifest.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=trajectory_manifest[0].keys()
                )
                writer.writeheader(); writer.writerows(trajectory_manifest)
        perturbation_deferred_reason = None
        if ranked and device.type == "cuda" and os.name == "nt":
            perturbation_deferred_reason = "windows_cuda_native_stability_guard"
            ranked = []
        selected_dir = OUT / "selected_validation_trajectories"
        selected_dir.mkdir(parents=True, exist_ok=True)
        perturbation_rows = []
        epsilon = 1e-5
        if ranked:
          with torch.no_grad():
            print(json.dumps({"dynamics_phase": "perturbation_start",
                              "selected": len(ranked)}), flush=True)
            for rank, score_index in enumerate(ranked, start=1):
                row = score_rows[score_index]
                rec = data["train"][row["training_index"]]
                generator = torch.Generator(device=device).manual_seed(SEED + rank)
                shape = (len(rec["x"]), hidden)
                direction = torch.randn(shape, generator=generator,
                                        device=device)
                direction = epsilon * direction / torch.linalg.vector_norm(direction)
                _, reference_tensor = model.forward_one(
                    rec, row["cyp_index"], return_trajectory=True,
                )
                _, perturbed_tensor = model.forward_one(
                    rec, row["cyp_index"], return_trajectory=True,
                    initial_perturbation=direction,
                )
                distances = torch.linalg.vector_norm(
                    perturbed_tensor - reference_tensor, dim=(1, 2)
                ).cpu().numpy()
                fit_end = min(100, len(distances) - 1)
                times = np.arange(1, fit_end + 1)
                log_growth = np.log(np.maximum(distances[1:fit_end + 1], 1e-12) / epsilon)
                exponent = float(np.polyfit(times, log_growth, 1)[0])
                classification = "complex_transient"
                if row["late_motion"] < 1e-4 and row["final_step"] < 1e-5:
                    classification = "point_attractor_candidate"
                elif (row["recurrence_ratio"] < 0.25 and
                      row["spectral_concentration"] > 0.5):
                    classification = "periodic_attractor_candidate"
                elif (exponent > 0.01 and row["late_motion"] >= late_threshold and
                      row["spectral_entropy"] >= entropy_threshold):
                    classification = "chaos_candidate_requires_confirmation"
                row["selected_for_perturbation"] = True
                row["dynamical_classification"] = classification
                perturbation_rows.append({
                    "molecule_id": row["molecule_id"],
                    "cyp_target": row["cyp_target"],
                    "epsilon": epsilon,
                    "fit_generations": f"1-{fit_end}",
                    "finite_time_lyapunov": exponent,
                    "final_separation": float(distances[-1]),
                    "maximum_separation": float(distances.max()),
                    "classification": classification,
                })
                safe_name = f"{rank:02d}_{row['molecule_id']}_{row['cyp_target']}.npz"
                np.savez_compressed(
                    selected_dir / safe_name,
                    trajectory=reference_tensor.cpu().numpy(),
                    perturbed_trajectory=perturbed_tensor.cpu().numpy(),
                    distances=distances,
                )
          print(json.dumps({"dynamics_phase": "perturbation_complete",
                            "selected": len(perturbation_rows)}), flush=True)
        for row in score_rows:
            row.setdefault("selected_for_perturbation", False)
            row.setdefault("dynamical_classification", "not_selected")
        with (OUT / "validation_dynamics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=score_rows[0].keys())
            writer.writeheader(); writer.writerows(score_rows)
        perturbation_fields = ["molecule_id", "cyp_target", "epsilon",
                               "fit_generations", "finite_time_lyapunov",
                               "final_separation", "maximum_separation",
                               "classification"]
        with (OUT / "validation_perturbations.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=perturbation_fields)
            writer.writeheader(); writer.writerows(perturbation_rows)
        common_metrics["dynamical_analysis"] = {
            "validation_trajectories_screened": len(score_rows),
            "full_atom_trajectories_archived": len(trajectory_manifest),
            "trajectory_archive": (str(trajectory_archive.relative_to(OUT))
                                   if trajectory_manifest else None),
            "perturbation_cases": len(perturbation_rows),
            "classification_counts": {
                label: sum(r["classification"] == label for r in perturbation_rows)
                for label in sorted({r["classification"] for r in perturbation_rows})
            },
            "perturbation_status": ("complete" if perturbation_rows else
                                    perturbation_deferred_reason or
                                    "not_requested"),
        }
    if TUNING_ONLY:
        (OUT / "metrics.json").write_text(json.dumps(common_metrics, indent=2) + "\n")
        print(json.dumps(common_metrics, indent=2))
        return

    def dynamical_scores(trajectory):
        mu = trajectory.mean(axis=1)
        delta = np.diff(mu, axis=0)
        step = np.linalg.norm(delta, axis=1)
        late_start = max(10, len(mu) // 3)
        late_mu = mu[late_start:]
        late_delta = np.diff(late_mu, axis=0)
        late_step = np.linalg.norm(late_delta, axis=1)
        recurrence = []
        mean_step = float(np.mean(late_step)) + 1e-8
        for lag in range(2, min(32, len(late_mu) // 2) + 1):
            distance = float(np.mean(np.linalg.norm(late_mu[lag:] - late_mu[:-lag], axis=1)))
            recurrence.append((distance / (lag * mean_step), lag, distance))
        recurrence_ratio, best_lag, recurrence_distance = min(recurrence)
        if len(late_delta) > 1:
            denominator = (np.linalg.norm(late_delta[:-1], axis=1) *
                           np.linalg.norm(late_delta[1:], axis=1))
            cosine = np.divide((late_delta[:-1] * late_delta[1:]).sum(1), denominator,
                               out=np.zeros_like(denominator), where=denominator > 1e-12)
            curvature = float(np.mean(1.0 - cosine))
        else:
            curvature = 0.0
        centred = late_mu - late_mu.mean(axis=0, keepdims=True)
        power = np.abs(np.fft.rfft(centred, axis=0)) ** 2
        nonzero = power[1:]
        spectral_concentration = float(np.mean(np.max(nonzero, axis=0) /
                                             np.maximum(nonzero.sum(axis=0), 1e-12)))
        return {
            "late_motion": float(np.mean(late_step)),
            "final_step": float(step[-1]),
            "late_amplitude": float(np.mean(np.std(late_mu, axis=0))),
            "best_recurrence_lag": int(best_lag),
            "recurrence_ratio": float(recurrence_ratio),
            "recurrence_distance": float(recurrence_distance),
            "curvature": curvature,
            "spectral_concentration": spectral_concentration,
        }

    test_rows, all_trajectory_records, score_rows = [], [], []
    model.eval()
    with torch.no_grad():
        all_preds = np.zeros((len(data["test"]), 4), dtype=float)
        for i, rec in enumerate(data["test"]):
            for c in range(4):
                fingerprint, traj = model.forward_one(rec, c, return_trajectory=True)
                pred = differentiable_ridge_predict(
                    fingerprint.unsqueeze(0), final_ridge_state
                ).squeeze(0)
                trajectory = traj.cpu().numpy()
                all_preds[i, c] = float(pred)
                test_rows.append({"molecule_id": rec["name"], "canonical_smiles": rec["canonical_smiles"],
                                  "cyp_target": CYPS[c], "predicted_pic50": all_preds[i, c]})
                scores = dynamical_scores(trajectory)
                base = {"test_index": i, "molecule_id": rec["name"], "cyp_index": c,
                        "cyp_target": CYPS[c], "predicted_pic50": float(pred), **scores}
                score_rows.append(base.copy())
                all_trajectory_records.append({**base, "trajectory": trajectory})

    def ranked(metric, reverse=True):
        return sorted(range(len(score_rows)), key=lambda j: score_rows[j][metric], reverse=reverse)

    rankings = [
        ("recurrence", ranked("recurrence_ratio", reverse=False)),
        ("spectral", ranked("spectral_concentration")),
        ("curved", ranked("curvature")),
        ("persistent", ranked("late_motion")),
    ]
    selected, reasons = [], {}
    rank_positions = {reason: 0 for reason, _ in rankings}
    while len(selected) < 20:
        for reason, order in rankings:
            rank_position = rank_positions[reason]
            while rank_position < len(order) and order[rank_position] in selected:
                rank_position += 1
            rank_positions[reason] = rank_position + 1
            if rank_position < len(order):
                idx = order[rank_position]
                selected.append(idx); reasons[idx] = reason
                if len(selected) == 20:
                    break
    trajectory_records = []
    for idx in selected:
        record = all_trajectory_records[idx]
        record["selection_reason"] = reasons[idx]
        trajectory_records.append(record)

    perturbation_rows = []
    epsilon = 1e-5
    with torch.no_grad():
        for selection_index, record in enumerate(trajectory_records, start=1):
            rec = data["test"][record["test_index"]]
            shape = record["trajectory"][0].shape
            generator = torch.Generator(device=device).manual_seed(SEED + selection_index)
            direction = torch.randn(shape, generator=generator, device=device)
            direction = epsilon * direction / torch.linalg.vector_norm(direction)
            _, perturbed = model.forward_one(rec, record["cyp_index"],
                                             return_trajectory=True,
                                             initial_perturbation=direction)
            reference = torch.as_tensor(record["trajectory"], device=device)
            distances = torch.linalg.vector_norm(perturbed - reference,
                                                  dim=(1, 2)).cpu().numpy()
            times = np.arange(1, min(31, len(distances)))
            log_growth = np.log(np.maximum(distances[times], 1e-12) / epsilon)
            finite_time_lyapunov = float(np.polyfit(times, log_growth, 1)[0])
            perturbation_rows.append({"molecule_id": record["molecule_id"],
                "cyp_target": record["cyp_target"], "epsilon": epsilon,
                "fit_generations": "1-30", "finite_time_lyapunov": finite_time_lyapunov,
                "final_separation": float(distances[-1]),
                "maximum_separation": float(distances.max())})
    selected_set = set(selected)
    for idx, row in enumerate(score_rows):
        row["selected_for_visualisation"] = idx in selected_set
        row["selection_reason"] = reasons.get(idx, "")
    with (OUT / "blinded_test_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=test_rows[0].keys()); writer.writeheader(); writer.writerows(test_rows)
    with (OUT / "selected_trajectories.pkl").open("wb") as handle:
        pickle.dump(trajectory_records, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with (OUT / "trajectory_novelty_scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=score_rows[0].keys()); writer.writeheader(); writer.writerows(score_rows)
    with (OUT / "perturbation_analysis.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=perturbation_rows[0].keys()); writer.writeheader(); writer.writerows(perturbation_rows)
    metrics = {**common_metrics,
               "selected_trajectories": [{"molecule_id": r["molecule_id"],
                                           "cyp_target": r["cyp_target"],
                                           "selection_reason": r["selection_reason"]}
                                          for r in trajectory_records]}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


def pdb_trajectory(mol, trajectory: np.ndarray, path: Path) -> None:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    heavy = mol.GetNumAtoms()
    display = Chem.AddHs(mol)
    params = AllChem.ETKDGv3(); params.randomSeed = SEED
    if AllChem.EmbedMolecule(display, params) != 0:
        AllChem.Compute2DCoords(display)
    else:
        try: AllChem.MMFFOptimizeMolecule(display, maxIters=300)
        except Exception: pass
    conf = display.GetConformer()
    parent = {}
    for atom in display.GetAtoms():
        if atom.GetAtomicNum() == 1:
            parent[atom.GetIdx()] = atom.GetNeighbors()[0].GetIdx()
    activity = np.linalg.norm(trajectory, axis=2)
    lo, hi = float(activity.min()), float(activity.max())
    scaled = 100.0 * (activity - lo) / max(hi - lo, 1e-8)
    lines = []
    scientific_states = trajectory.shape[0]
    display_states = scientific_states + 1
    for state in range(display_states):
        lines.append(f"MODEL     {state + 1:4d}")
        scientific = min(state, scientific_states - 1)
        for atom in display.GetAtoms():
            idx = atom.GetIdx(); pos = conf.GetAtomPosition(idx)
            if idx < heavy:
                b = float(scaled[scientific, idx])
            elif state == display_states - 1:
                b = float(scaled[scientific_states - 1, parent[idx]])
            else:
                b = 0.0
            symbol = atom.GetSymbol()
            name = f"{symbol}{idx + 1}"[:4]
            lines.append(
                f"HETATM{idx+1:5d} {name:<4s} MOL A   1    "
                f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}{1.00:6.2f}{b:6.2f}          {symbol:>2s}"
            )
        for bond in display.GetBonds():
            lines.append(f"CONECT{bond.GetBeginAtomIdx()+1:5d}{bond.GetEndAtomIdx()+1:5d}")
        lines.append("ENDMDL")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return lo, hi


def render() -> None:
    import pandas as pd
    import matplotlib.pyplot as plt
    from rdkit import Chem

    set_seeds()
    with CACHE.open("rb") as handle:
        data = pickle.load(handle)
    with (OUT / "selected_trajectories.pkl").open("rb") as handle:
        records = pickle.load(handle)
    traj_dir = OUT / "trajectories"; fig_dir = OUT / "figures"
    traj_dir.mkdir(parents=True, exist_ok=True); fig_dir.mkdir(parents=True, exist_ok=True)
    # This directory contains generated prototype artefacts only. Clear prior selections
    # so a rerun always delivers exactly the declared 20 trajectories.
    for old in list(traj_dir.glob("*.pdb")) + list(traj_dir.glob("*.npz")):
        old.unlink()
    manifest = []
    display_values = {}
    colours = {"cyan": "0.00, 0.90, 1.00", "violet": "0.35, 0.15, 0.85",
               "magenta": "1.00, 0.05, 0.65", "lime": "0.65, 1.00, 0.10"}
    pml = ["reinitialize", "bg_color black", "set antialias, 2", "set ray_opaque_background, off",
           "set stick_radius, 0.16", "set sphere_scale, 0.25",
           *[f"set_color cyber_{k}, [{v}]" for k, v in colours.items()]]
    for number, rec in enumerate(records, 1):
        graph = data["test"][rec["test_index"]]
        mol, _ = standardise(graph["canonical_smiles"])
        safe = f"{number:02d}_{rec['molecule_id']}_{rec['cyp_target']}"
        obj = f"traj_{number:02d}_{rec['cyp_target']}"
        pdb_path = traj_dir / f"{safe}.pdb"
        lo, hi = pdb_trajectory(mol, rec["trajectory"], pdb_path)
        state_values, current_values = [], []
        for line in pdb_path.read_text().splitlines():
            if line.startswith("MODEL"):
                current_values = []
            elif line.startswith(("ATOM", "HETATM")):
                current_values.append(float(line[60:66]))
            elif line.startswith("ENDMDL"):
                state_values.append(current_values)
        display_values[obj] = state_values
        np.savez_compressed(traj_dir / f"{safe}.npz", trajectory=rec["trajectory"],
                            molecule_id=rec["molecule_id"], cyp_target=rec["cyp_target"],
                            predicted_pic50=rec["predicted_pic50"])
        manifest.append({"object": obj, "molecule_id": rec["molecule_id"],
                         "cyp_target": rec["cyp_target"], "predicted_pic50": rec["predicted_pic50"],
                         "pdb_file": f"trajectories/{pdb_path.name}", "activity_min": lo,
                         "activity_max": hi, "states": GENERATIONS + 2,
                         "selection_reason": rec.get("selection_reason", "")})
        pml += [f'load "{pdb_path.as_posix()}", {obj}', f"hide everything, {obj}",
                f"show sticks, {obj} and not elem H", f"show spheres, {obj} and not elem H",
                f"disable {obj}"]
    controller = OUT / "gca_trajectory_controls.py"
    controller_source = (
        "GCA_STATE_COUNT = " + str(GENERATIONS + 2) + "\n" +
        "GCA_DISPLAY_VALUES = " + repr(display_values) + "\n" +
        (ROOT / "scripts" / "pymol_gca_controller.py").read_text()
    )
    controller.write_text(controller_source)
    pml += ["enable traj_01_CYP1A2", "orient traj_01_CYP1A2",
            "python", controller_source, "python end", "gca_state 1", "refresh",
            "# Select one object in the right-hand panel, click its name to enable it,",
            "# disable the previous object, then use gca_next, gca_previous,",
            f"# gca_state 1-{GENERATIONS + 2}, gca_play, or gca_stop in the PyMOL command line."]
    (OUT / "load_20_trajectories.pml").write_text("\n".join(pml) + "\n")
    with (OUT / "trajectory_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest[0].keys()); writer.writeheader(); writer.writerows(manifest)

    plt.style.use("dark_background")
    cyan, magenta, lime, violet = "#00e5ff", "#ff1493", "#a6ff00", "#6c4cff"
    hist = pd.read_csv(OUT / "training_history.csv")
    val = pd.read_csv(OUT / "validation_predictions.csv")
    fig, ax = plt.subplots(figsize=(8, 5)); ax.plot(hist.epoch, hist.train_rmse, c=cyan, label="Fit")
    ax.plot(hist.epoch, hist.validation_rmse, c=magenta, label="Grouped validation")
    ax.set(xlabel="Epoch", ylabel="RMSE (pIC50)", title="Learning curve"); ax.legend(); ax.grid(alpha=.15)
    fig.tight_layout(); fig.savefig(fig_dir / "01_learning_curve.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 6))
    for split, colour in (("fit", cyan), ("validation", magenta)):
        q = val[val.split == split]; ax.scatter(q.experimental_pic50, q.predicted_pic50,
                                                s=10, alpha=.45, c=colour, label=split)
    low = min(val.experimental_pic50.min(), val.predicted_pic50.min())
    high = max(val.experimental_pic50.max(), val.predicted_pic50.max())
    ax.plot([low, high], [low, high], '--', c=lime); ax.set(xlabel="Experimental pIC50",
        ylabel="Predicted pIC50", title="Prediction agreement"); ax.legend(); ax.grid(alpha=.15)
    fig.tight_layout(); fig.savefig(fig_dir / "02_prediction_scatter.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    for cyp, colour in zip(CYPS, (cyan, magenta, lime, violet)):
        q = val[(val.split == "validation") & (val.cyp_target == cyp)]
        ax.hist(q.residual, bins=25, alpha=.45, label=cyp, color=colour)
    ax.axvline(0, c="white", ls="--"); ax.set(xlabel="Predicted − experimental pIC50",
        ylabel="Count", title="Grouped-validation residuals"); ax.legend(); ax.grid(alpha=.12)
    fig.tight_layout(); fig.savefig(fig_dir / "03_residual_distributions.png", dpi=180); plt.close(fig)
    first = records[0]; activity = np.linalg.norm(first["trajectory"], axis=2).T
    fig, ax = plt.subplots(figsize=(10, 5)); im = ax.imshow(activity, aspect="auto", cmap="cool",
        interpolation="nearest"); ax.set(xlabel="CA generation", ylabel="Atom index",
        title=f"Atom activity: {first['molecule_id']} / {first['cyp_target']}")
    fig.colorbar(im, ax=ax, label="Eight-channel state norm")
    fig.tight_layout(); fig.savefig(fig_dir / "04_atom_activity_heatmap.png", dpi=180); plt.close(fig)

    metrics = json.loads((OUT / "metrics.json").read_text())
    readme = f"""# Graph-CA visual prototype results

This is one fixed-design scientific prototype, trained with seed {SEED}. It is an exploratory run rather than the final nested-cross-validation estimate.

## Result snapshot

- Fit observations: {metrics['fit_observations']}
- Grouped-validation observations: {metrics['validation_observations']}
- Restored fit RMSE: {metrics['restored_fit_rmse']:.3f} pIC50
- Restored grouped-validation RMSE: {metrics['restored_validation_rmse']:.3f} pIC50
- Epochs run: {metrics['epochs_run']}
- Blinded predictions: 750 molecules × four CYPs
- Visual trajectories: 20 molecule–CYP cases selected from dynamical screening

## PyMOL

Open PyMOL, choose **File → Run Script**, and select `load_20_trajectories.pml` from this directory. All 20 objects appear in the right-hand object panel; the first is enabled. Enable one desired object and disable the previous one.

The supplied controller does not use PyMOL's movie subsystem. Enter `gca_next`, `gca_previous`, `gca_state {GENERATIONS // 2 + 1}`, or `gca_play` in the PyMOL command line. `gca_play 0.25, 2` uses a 0.25-second delay and plays two cycles; `gca_stop` stops playback.

Playback runs in the background so PyMOL can repaint between states. Only the currently enabled trajectory object is recoloured, which keeps display-memory use modest.

Before recolouring, the controller explicitly installs the selected generation's activity values into the enabled object's B-factor field. This preserves the true {GENERATIONS + 1}-state gradient even in PyMOL versions that treat a multi-model PDB's B-factor as one shared atom property.

States 1–{GENERATIONS + 1} are graph-CA generations 0–{GENERATIONS}. State {GENERATIONS + 2} is the labelled visual coda: display-only hydrogens become lime using the final heavy-atom activity. The model never received 3D coordinates or hydrogen nodes.

The PDB B-factor column stores scaled eight-channel atom-state magnitude. It is unrelated to the learned ridge coefficient beta. Lossless eight-channel values are in the matching NPZ files.

## Figures

- `01_learning_curve.png`: fit and grouped-validation error over training.
- `02_prediction_scatter.png`: predictions against measurements; the diagonal is perfect agreement.
- `03_residual_distributions.png`: signed error by CYP.
- `04_atom_activity_heatmap.png`: atom-by-generation activity for one example.
"""
    (OUT / "README.md").write_text(readme)
    print(f"Rendered {len(manifest)} trajectories in {OUT}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "train", "predict", "render", "extended-dynamics"))
    args = parser.parse_args()
    if args.phase == "predict":
        os.environ["SME_INFERENCE_ONLY"] = "1"
        train(extended_dynamics=False)
        return
    {"prepare": prepare, "train": train, "render": render,
     "extended-dynamics": lambda: train(extended_dynamics=True)}[args.phase]()


if __name__ == "__main__":
    main()
