#!/usr/bin/env python3
"""Render the four Figure 1 Graph-CA trajectories as terminal atom cascades."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import AllChem


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "long_horizon_attractor_campaign_v1" / "base_trajectories"
OUTPUT = ROOT / "results" / "long_horizon_attractor_campaign_v1" / "figures"
TRAIN = ROOT / "data" / "openadmet-cyp-challenge-2026" / "cyp-challenge-TRAIN_inhibition.csv"


@dataclass(frozen=True)
class Cascade:
    panel: str
    regime: str
    rule: str
    molecule: str
    target: str
    trajectory: Path


CASES = (
    Cascade("A", "Point Attractor", "Gated residual", "OCNT-2328519", "CYP1A2",
            BASE / "gated_residual" / "node_trajectories" / "case_001.npz"),
    Cascade("B", "Strange Attractor", "Kuramoto–Sakaguchi", "OCNT-0494110", "CYP2C9",
            BASE / "kuramoto_sakaguchi" / "node_trajectories" / "case_002.npz"),
    Cascade("C", "Persistent or Complex Candidate", "Kuramoto–Sakaguchi", "OCNT-0495493", "CYP2D6",
            BASE / "kuramoto_sakaguchi" / "node_trajectories" / "case_003.npz"),
    Cascade("D", "Period-2 Oscillator Candidate", "Coupled map", "OCNT-0495275", "CYP3A4",
            BASE / "coupled_map" / "node_trajectories" / "case_004.npz"),
)


def molecular_coordinates(smiles: str, seed: int) -> np.ndarray:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    embedded = Chem.AddHs(molecule)
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = seed
    if AllChem.EmbedMolecule(embedded, parameters) != 0:
        raise RuntimeError("ETKDG embedding failed")
    AllChem.MMFFOptimizeMolecule(embedded, maxIters=500)
    molecule = Chem.RemoveHs(embedded)
    conformer = molecule.GetConformer()
    xyz = np.asarray([list(conformer.GetAtomPosition(index))
                      for index in range(molecule.GetNumAtoms())], dtype=np.float64)
    xyz -= xyz.mean(axis=0, keepdims=True)
    return xyz


def rotation_y(angle: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.cos(angle), np.sin(angle)


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    name = "seguisb.ttf" if bold else "segoeui.ttf"
    try:
        return ImageFont.truetype(f"C:/Windows/Fonts/{name}", size)
    except OSError:
        return ImageFont.load_default()


def viridis(fraction: np.ndarray) -> np.ndarray:
    anchors = np.asarray([
        [0.267, 0.005, 0.329], [0.230, 0.322, 0.546], [0.128, 0.567, 0.551],
        [0.369, 0.789, 0.383], [0.993, 0.906, 0.144],
    ])
    position = np.clip(fraction, 0, 1) * (len(anchors) - 1)
    lower = np.floor(position).astype(int)
    upper = np.minimum(lower + 1, len(anchors) - 1)
    weight = (position - lower)[..., None]
    return anchors[lower] * (1 - weight) + anchors[upper] * weight


def main() -> None:
    table = pd.read_csv(TRAIN).set_index("Molecule_Name")
    width, height = 4800, 1950
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    title_font, panel_font = font(55, True), font(36, True)
    detail_font, tick_font = font(25), font(22)
    draw.text((width / 2, 42),
              "Molecular Information Cascades Across Four Graph-CA Dynamical Regimes",
              font=title_font, fill="#14202B", anchor="ma")
    column_width = width / 4
    plot_top, plot_bottom = 235, 1690

    for case_index, case in enumerate(CASES):
        state = np.load(case.trajectory)["trajectory"].astype(np.float64)
        xyz = molecular_coordinates(table.loc[case.molecule, "SMILES"], 260830 + case_index)
        if state.shape[1] != len(xyz):
            raise ValueError(f"{case.molecule}: {state.shape[1]} states for {len(xyz)} atoms")

        generations = np.arange(0, len(state), 10, dtype=int)
        angle = generations * (2.0 * np.pi * 14.0 / 5000.0)
        cosine, sine = rotation_y(angle)
        base_x, base_y, base_z = xyz.T
        x = cosine[:, None] * base_x[None, :] + sine[:, None] * base_z[None, :]
        z = -sine[:, None] * base_x[None, :] + cosine[:, None] * base_z[None, :]
        molecular_span = max(float(np.ptp(base_x)), float(np.ptp(base_z)), 1.0)
        y = base_y[None, :] - generations[:, None] * molecular_span / 370.0

        atom_signal = state.mean(axis=2)
        low, high = np.quantile(atom_signal, [0.02, 0.98])
        signal = np.clip((atom_signal[generations] - low) / max(high - low, 1e-12), 0, 1)
        fraction = np.sqrt(generations / 5000.0)
        time_colour = viridis(fraction)
        brightness = 0.62 + 0.38 * signal[..., None]
        rgb = np.clip(time_colour[:, None, :] * brightness, 0, 1)

        centre_x = (case_index + 0.5) * column_width
        projected_x = x + 0.28 * z
        x_scale = column_width * 0.34 / max(float(np.max(np.abs(projected_x))), 1e-6)
        screen_x = centre_x + projected_x * x_scale
        y_min, y_max = float(y.min()), float(y.max())
        screen_y = plot_top + (y_max - y) / max(y_max - y_min, 1e-9) * (plot_bottom - plot_top)
        screen_y += z / max(float(np.ptp(z)), 1e-9) * 7.0
        depth_order = np.argsort(z.ravel())
        flat_x, flat_y = screen_x.ravel(), screen_y.ravel()
        flat_rgb = (rgb.reshape(-1, 3) * 255).astype(np.uint8)
        for index in depth_order:
            red, green, blue = (int(value) for value in flat_rgb[index])
            radius = 2.25
            draw.ellipse((flat_x[index] - radius, flat_y[index] - radius,
                          flat_x[index] + radius, flat_y[index] + radius),
                         fill=(red, green, blue, 150))
        draw.text((centre_x, 135), f"{case.panel}  |  {case.regime}",
                  font=panel_font, fill="#14202B", anchor="ma")
        draw.text((centre_x, 185), f"{case.rule} · {case.molecule} · {case.target}",
                  font=detail_font, fill="#4A4A4A", anchor="ma")

    bar_left, bar_right, bar_top, bar_bottom = 1050, 3750, 1780, 1820
    for pixel in range(bar_left, bar_right):
        colour = (viridis(np.asarray([(pixel - bar_left) / (bar_right - bar_left)]))[0] * 255).astype(int)
        draw.line((pixel, bar_top, pixel, bar_bottom), fill=tuple(colour) + (255,))
    draw.rectangle((bar_left, bar_top, bar_right, bar_bottom), outline="#333333", width=2)
    for generation in range(0, 5001, 1000):
        x_tick = bar_left + generation / 5000 * (bar_right - bar_left)
        draw.line((x_tick, bar_bottom, x_tick, bar_bottom + 10), fill="#333333", width=2)
        draw.text((x_tick, bar_bottom + 15), f"{generation:,}", font=tick_font,
                  fill="#333333", anchor="ma")
    draw.text(((bar_left + bar_right) / 2, 1905), "Cellular-Automata Generation",
              font=detail_font, fill="#222222", anchor="ms")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    png = OUTPUT / "19_four_graph_ca_terminal_atom_cascades.png"
    pdf = OUTPUT / "19_four_graph_ca_terminal_atom_cascades.pdf"
    image.save(png, dpi=(300, 300), optimize=True)
    image.save(pdf, resolution=300.0)
    print(png.resolve())
    print(pdf.resolve())


if __name__ == "__main__":
    main()
