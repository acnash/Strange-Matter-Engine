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


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "openadmet-cyp-challenge-2026"
TRAIN_CSV = DATA / "cyp-challenge-TRAIN_inhibition.csv"
TEST_CSV = DATA / "cyp-challenge-TEST-BLINDED.csv"
CACHE = Path("/private/tmp/strange_matter_graph_ca_graphs.pkl")
RULE = os.environ.get("SME_CA_RULE", "gated_residual")
GENERATIONS = int(os.environ.get("SME_GENERATIONS", "16"))
RUN_NAME = os.environ.get("SME_RUN_NAME", "graph_ca_visual_prototype")
OUT = ROOT / "results" / RUN_NAME
SEED = 1701
CYPS = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")
LABEL_COLS = tuple(f"{c}_pIC50_direct_inhibition" for c in CYPS)


def set_seeds() -> None:
    random.seed(SEED)
    np.random.seed(SEED)


def atom_features(atom, donor_ids: set[int], acceptor_ids: set[int]) -> list[float]:
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
    return element + [
        atom.GetFormalCharge() / 3.0,
        float(atom.GetIsAromatic()),
    ] + hybrid + [
        atom.GetDegree() / 4.0,
        atom.GetTotalNumHs() / 4.0,
        float(atom.IsInRing()),
        float(atom.GetIdx() in donor_ids),
        float(atom.GetIdx() in acceptor_ids),
    ] + chirality


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


def graph_record(name: str, smiles: str, labels=None) -> dict | None:
    from rdkit.Chem import Lipinski
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol, canonical = standardise(smiles)
    if mol is None:
        return None
    donor_ids = {idx for match in Lipinski._HDonors(mol) for idx in match}
    acceptor_ids = {idx for match in Lipinski._HAcceptors(mol) for idx in match}
    x = [atom_features(a, donor_ids, acceptor_ids) for a in mol.GetAtoms()]
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
    }


def prepare() -> None:
    import pandas as pd

    set_seeds()
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    train, test, rejected = [], [], []
    for row in train_df.itertuples(index=False):
        labels = [getattr(row, col) for col in LABEL_COLS]
        rec = graph_record(row.Molecule_Name, row.SMILES, labels)
        (train if rec is not None else rejected).append(rec if rec is not None else row.Molecule_Name)
    for row in test_df.itertuples(index=False):
        rec = graph_record(row.Molecule_Name, row.SMILES)
        (test if rec is not None else rejected).append(rec if rec is not None else row.Molecule_Name)

    groups = sorted({r["scaffold"] for r in train})
    rng = random.Random(SEED)
    rng.shuffle(groups)
    validation_groups = set(groups[:max(1, round(0.2 * len(groups)))])
    train_idx = [i for i, r in enumerate(train) if r["scaffold"] not in validation_groups]
    val_idx = [i for i, r in enumerate(train) if r["scaffold"] in validation_groups]
    payload = {"train": train, "test": test, "train_idx": train_idx,
               "val_idx": val_idx, "rejected": rejected, "seed": SEED}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(json.dumps({"training_molecules": len(train), "blinded_molecules": len(test),
                      "fit_molecules": len(train_idx), "validation_molecules": len(val_idx),
                      "rejected": rejected, "cache": str(CACHE)}, indent=2))


def train() -> None:
    import torch
    from torch import nn

    set_seeds()
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    with CACHE.open("rb") as handle:
        data = pickle.load(handle)
    device = torch.device("cpu")
    chem_dim = len(data["train"][0]["x"][0])
    bond_dim = 9
    hidden = 8

    class GraphCA(nn.Module):
        def __init__(self):
            super().__init__()
            self.init = nn.Linear(chem_dim, hidden)
            self.self_layer = nn.Linear(hidden, hidden, bias=False)
            self.neighbour = nn.Linear(hidden, hidden, bias=False)
            self.bond = nn.Linear(bond_dim, hidden, bias=False)
            self.chem = nn.Linear(chem_dim, hidden, bias=False)
            self.context = nn.Linear(4, hidden, bias=False)
            self.bias = nn.Parameter(torch.zeros(hidden))
            if RULE == "gated_residual":
                self.gate = nn.Linear(hidden * 2 + chem_dim + 4, hidden)
            elif RULE == "inertial_reaction_diffusion":
                self.raw_gamma = nn.Parameter(torch.zeros(hidden))
                self.raw_dt = nn.Parameter(torch.zeros(hidden))
                self.raw_diffusion = nn.Parameter(torch.full((hidden,), -1.5))
                self.raw_restoring = nn.Parameter(torch.full((hidden,), -1.5))
            else:
                raise ValueError(f"Unknown CA rule: {RULE}")
            self.readout = nn.Linear(hidden * 5, 1)

        def forward_one(self, rec, cyp: int, return_trajectory=False):
            x = torch.as_tensor(rec["x"], device=device)
            src = torch.as_tensor(rec["src"], device=device)
            dst = torch.as_tensor(rec["dst"], device=device)
            edge = torch.as_tensor(rec["edge"], device=device)
            context = torch.zeros(4, device=device); context[cyp] = 1.0
            h = torch.tanh(self.init(x))
            velocity = torch.zeros_like(h)
            states = [h]
            means = [h.mean(0)]
            step_energy = torch.zeros(hidden, device=device)
            for _ in range(GENERATIONS):
                agg = torch.zeros_like(h)
                neighbour_mean = torch.zeros_like(h)
                degree = torch.zeros((h.shape[0], 1), device=device)
                if src.numel():
                    msg = self.neighbour(h[src]) + self.bond(edge)
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
                    new_h = (1.0 - alpha) * h + alpha * reaction
                else:
                    gamma = 0.99 * torch.sigmoid(self.raw_gamma)
                    dt = 0.25 * torch.sigmoid(self.raw_dt)
                    diffusion = torch.nn.functional.softplus(self.raw_diffusion)
                    restoring = torch.nn.functional.softplus(self.raw_restoring)
                    force = reaction + diffusion * (neighbour_mean - h) - restoring * h
                    velocity = gamma * velocity + dt * force
                    new_h = torch.tanh(h + dt * velocity)
                step_energy += ((new_h - h) ** 2).mean(0) / float(GENERATIONS)
                h = new_h
                states.append(h); means.append(h.mean(0))
            mean_series = torch.stack(means)
            fingerprint = torch.cat((
                h.mean(0), h.var(0, unbiased=False),
                torch.stack(states).mean((0, 1)),
                mean_series.var(0, unbiased=False), step_energy,
            ))
            pred = self.readout(fingerprint).squeeze()
            if return_trajectory:
                return pred, torch.stack(states)
            return pred

    model = GraphCA().to(device)
    readout_params = list(model.readout.parameters())
    readout_ids = {id(p) for p in readout_params}
    ca_params = [p for p in model.parameters() if id(p) not in readout_ids]
    optimizer = torch.optim.Adam([
        {"params": ca_params, "lr": 1e-3},
        {"params": readout_params, "lr": 3e-3},
    ], betas=(0.9, 0.999), eps=1e-8)

    def observed_pairs(indices):
        pairs = []
        for i in indices:
            for c, y in enumerate(data["train"][i]["labels"]):
                if np.isfinite(y): pairs.append((i, c, float(y)))
        return pairs

    fit_pairs = observed_pairs(data["train_idx"])
    val_pairs = observed_pairs(data["val_idx"])

    def evaluate(pairs):
        model.eval(); ys, ps = [], []
        with torch.no_grad():
            for i, c, y in pairs:
                ps.append(float(model.forward_one(data["train"][i], c)))
                ys.append(y)
        ys, ps = np.asarray(ys), np.asarray(ps)
        return float(np.sqrt(np.mean((ys - ps) ** 2))), ys, ps

    history, best, best_state, patience = [], math.inf, None, 0
    rng = random.Random(SEED)
    max_epochs = int(os.environ.get("SME_MAX_EPOCHS", "200"))
    for epoch in range(1, max_epochs + 1):
        model.train()
        molecule_order = list(data["train_idx"]); rng.shuffle(molecule_order)
        total_sq, total_n, raw_norms, clipped_norms = 0.0, 0, [], []
        for start in range(0, len(molecule_order), 16):
            molecule_batch = molecule_order[start:start + 16]
            batch = []
            for i in molecule_batch:
                for c, y in enumerate(data["train"][i]["labels"]):
                    if np.isfinite(y): batch.append((i, c, float(y)))
            if not batch:
                continue
            optimizer.zero_grad()
            preds = torch.stack([model.forward_one(data["train"][i], c) for i, c, _ in batch])
            targets = torch.tensor([y for _, _, y in batch], dtype=torch.float32, device=device)
            mse = ((preds - targets) ** 2).mean()
            ridge = 1e-3 * sum((p ** 2).sum() for p in readout_params)
            ca_penalty = 1e-5 * sum((p ** 2).sum() for p in ca_params)
            loss = mse + ridge + ca_penalty
            loss.backward()
            raw = float(torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in model.parameters()
                                       if p.grad is not None)))
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            clipped = min(raw, 1.0)
            optimizer.step()
            total_sq += float(((preds.detach() - targets) ** 2).sum())
            total_n += len(batch); raw_norms.append(raw); clipped_norms.append(clipped)
        train_rmse = math.sqrt(total_sq / total_n)
        val_rmse, _, _ = evaluate(val_pairs)
        row = {"epoch": epoch, "train_rmse": train_rmse, "validation_rmse": val_rmse,
               "mean_raw_gradient_norm": float(np.mean(raw_norms)),
               "fraction_gradients_clipped": float(np.mean(np.asarray(raw_norms) > 1.0))}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_rmse < best - 0.005:
            best, patience = val_rmse, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if patience >= 20:
            break
    model.load_state_dict(best_state)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "chem_dim": chem_dim, "seed": SEED,
                "rule": RULE, "generations": GENERATIONS}, OUT / "model.pt")
    with (OUT / "training_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys()); writer.writeheader(); writer.writerows(history)

    train_rmse, train_y, train_p = evaluate(fit_pairs)
    val_rmse, val_y, val_p = evaluate(val_pairs)
    pair_rows = []
    for split, pairs, ys, ps in (("fit", fit_pairs, train_y, train_p),
                                ("validation", val_pairs, val_y, val_p)):
        for (i, c, y), pred in zip(pairs, ps):
            r = data["train"][i]
            pair_rows.append({"split": split, "molecule_id": r["name"], "cyp_target": CYPS[c],
                              "experimental_pic50": y, "predicted_pic50": pred,
                              "residual": pred - y})
    with (OUT / "validation_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=pair_rows[0].keys()); writer.writeheader(); writer.writerows(pair_rows)

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
                pred, traj = model.forward_one(rec, c, return_trajectory=True)
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
    metrics = {"seed": SEED, "best_validation_rmse": best, "restored_fit_rmse": train_rmse,
               "restored_validation_rmse": val_rmse, "epochs_run": len(history),
               "fit_observations": len(fit_pairs), "validation_observations": len(val_pairs),
               "rule": RULE, "generations": GENERATIONS,
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
    parser.add_argument("phase", choices=("prepare", "train", "render"))
    args = parser.parse_args()
    {"prepare": prepare, "train": train, "render": render}[args.phase]()


if __name__ == "__main__":
    main()
