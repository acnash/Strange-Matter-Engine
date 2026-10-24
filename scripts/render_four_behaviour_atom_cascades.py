#!/usr/bin/env python3
"""Render the four Figure 1 Graph-CA trajectories as terminal atom cascades."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import AllChem, Draw


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
    Cascade("A", "Point attractor", "Gated residual", "OCNT-2328519", "CYP1A2",
            BASE / "gated_residual" / "node_trajectories" / "case_001.npz"),
    Cascade("B", "Hyperchaotic candidate", "Kuramoto–Sakaguchi", "OCNT-0494110", "CYP2C9",
            BASE / "kuramoto_sakaguchi" / "node_trajectories" / "case_002.npz"),
    Cascade("C", "Persistent complex variation", "Kuramoto–Sakaguchi", "OCNT-0495493", "CYP2D6",
            BASE / "kuramoto_sakaguchi" / "node_trajectories" / "case_003.npz"),
    Cascade("D", "Period-two oscillator candidate", "Coupled map", "OCNT-0495275", "CYP3A4",
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
    name = "arialbd.ttf" if bold else "arial.ttf"
    try:
        return ImageFont.truetype(f"C:/Windows/Fonts/{name}", size * 2)
    except OSError:
        return ImageFont.load_default()


def cyan_magenta(fraction: np.ndarray) -> np.ndarray:
    cyan = np.asarray([0.00, 0.78, 0.94])
    violet = np.asarray([0.50, 0.28, 0.88])
    magenta = np.asarray([0.96, 0.04, 0.72])
    fraction = np.clip(fraction, 0, 1)[..., None]
    first = cyan + np.minimum(fraction * 2.0, 1.0) * (violet - cyan)
    return first + np.maximum(fraction * 2.0 - 1.0, 0.0) * (magenta - violet)


class HighResolutionDraw:
    """Draw original logical coordinates directly at twice the pixel resolution."""
    def __init__(self, image):
        self.draw = ImageDraw.Draw(image, "RGBA")

    def __getattr__(self, name):
        def scaled(xy, *args, **kwargs):
            xy = tuple(value * 2 for value in xy)
            if 'width' in kwargs:
                kwargs['width'] *= 2
            return getattr(self.draw, name)(xy, *args, **kwargs)
        return scaled


def main() -> None:
    table = pd.read_csv(TRAIN).set_index("Molecule_Name")
    width, height = 4800, 2800
    image = Image.new("RGB", (width * 2, height * 2), "white")
    draw = HighResolutionDraw(image)
    panel_font = font(86, True)
    detail_font, tick_font = font(78), font(78)
    column_width = width / 4
    plot_top, plot_bottom = 370, 1680

    for case_index, case in enumerate(CASES):
        state = np.load(case.trajectory)["trajectory"].astype(np.float64)
        xyz = molecular_coordinates(table.loc[case.molecule, "SMILES"], 260830 + case_index)
        if state.shape[1] != len(xyz):
            raise ValueError(f"{case.molecule}: {state.shape[1]} states for {len(xyz)} atoms")

        generations = np.arange(len(state), dtype=int)
        angle = generations * (2.0 * np.pi * 5.25 / 5000.0)
        cosine, sine = rotation_y(angle)
        base_x, base_y, base_z = xyz.T
        x = cosine[:, None] * base_x[None, :] + sine[:, None] * base_z[None, :]
        z = -sine[:, None] * base_x[None, :] + cosine[:, None] * base_z[None, :]
        molecular_span = max(float(np.ptp(base_x)), float(np.ptp(base_z)), 1.0)
        y = base_y[None, :] - generations[:, None] * molecular_span / 370.0

        atom_signal = state.mean(axis=2)
        low, high = np.quantile(atom_signal, [0.02, 0.98])
        signal = np.clip((atom_signal[generations] - low) / max(high - low, 1e-12), 0, 1)
        rgb = cyan_magenta(signal)

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
            radius = 4.2
            draw.ellipse((flat_x[index] - radius, flat_y[index] - radius,
                          flat_x[index] + radius, flat_y[index] + radius),
                         fill=(red, green, blue, 92))
        heading = f"{case.panel}) {case.regime}"
        if case.panel in ('B', 'C', 'D'):
            heading = heading.replace('Hyperchaotic candidate', 'Hyperchaotic\ncandidate').replace('complex variation', 'complex\nvariation').replace('oscillator candidate', 'oscillator\ncandidate')
        draw.text((centre_x, 90), heading,
                  font=panel_font, fill="#000000", anchor="ma")

        molecule = Chem.MolFromSmiles(table.loc[case.molecule, "SMILES"])
        drawing_options = Draw.MolDrawOptions()
        drawing_options.padding = 0.06
        drawing_options.bondLineWidth = 6.0
        drawing_options.fixedFontSize = 68
        depiction = Draw.MolToImage(molecule, size=(2080, 1220), options=drawing_options)
        image.paste(depiction, (int(centre_x - 520) * 2, 4140))
        draw.text((centre_x, 2680), case.molecule, font=detail_font,
                  fill="#000000", anchor="ma")

    bar_left, bar_right, bar_top, bar_bottom = 850, 3950, 1780, 1835
    for pixel in range(bar_left, bar_right):
        colour = (cyan_magenta(np.asarray([(pixel - bar_left) / (bar_right - bar_left)]))[0] * 255).astype(int)
        draw.line((pixel, bar_top, pixel, bar_bottom), fill=tuple(colour) + (255,))
    draw.rectangle((bar_left, bar_top, bar_right, bar_bottom), outline="#333333", width=2)
    for step in range(6):
        x_tick = bar_left + step / 5 * (bar_right - bar_left)
        draw.line((x_tick, bar_bottom, x_tick, bar_bottom + 14), fill="#333333", width=3)
        draw.text((x_tick, bar_bottom + 20), f"{step / 5:.1f}", font=tick_font,
                  fill="#000000", anchor="ma")
    draw.text(((bar_left + bar_right) / 2, 2040), "Normalized Mean Atom State Across 16 Dynamical Channels",
              font=detail_font, fill="#000000", anchor="ms")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    png = OUTPUT / "19_four_graph_ca_terminal_atom_cascades.png"
    pdf = OUTPUT / "19_four_graph_ca_terminal_atom_cascades.pdf"
    image.save(png, dpi=(600, 600), optimize=True)
    image.save(pdf, resolution=600.0)
    image.resize((1920, 1120), Image.Resampling.LANCZOS).save(
        OUTPUT / '19_four_graph_ca_terminal_atom_cascades_preview.png')
    print(png.resolve())
    print(pdf.resolve())


if __name__ == "__main__":
    main()
