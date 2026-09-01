#!/usr/bin/env python3
"""Render a molecular Graph-CA trajectory as a classic 1D space-time cascade."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1]
FFMPEG = Path(r"C:\ffmpeg\bin\ffmpeg.exe")
CAMPAIGN = ROOT / "results" / "long_horizon_attractor_campaign_v1"
PRESETS = {
    "strange": {
        "source": CAMPAIGN / "base_trajectories" / "kuramoto_sakaguchi" /
                  "node_trajectories" / "case_002.npz",
        "output": CAMPAIGN / "videos" /
                  "trajectory_07_molecular_1d_cellular_automaton.mp4",
        "smiles": "CC/C(=C(/CC)C1=CC=C(O)C=C1)C1=CC=C(O)C=C1",
        "subtitle": "Trajectory 7 | Kuramoto-Sakaguchi confirmed strange attractor",
        "display": "circular_phase",
    },
    "point": {
        "source": CAMPAIGN / "base_trajectories" / "gated_residual" /
                  "node_trajectories" / "case_001.npz",
        "output": CAMPAIGN / "videos" /
                  "trajectory_01_point_attractor_molecular_1d_cellular_automaton.mp4",
        "smiles": "CCCCCCOC(=O)CCC(=O)CN",
        "subtitle": "Trajectory 1 | Gated-residual point attractor",
        "display": "robust_mean",
    },
}


def font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.ImageFont:
    name = "consola.ttf" if mono else ("seguisb.ttf" if bold else "segoeui.ttf")
    try:
        return ImageFont.truetype(f"C:/Windows/Fonts/{name}", size)
    except OSError:
        return ImageFont.load_default()


def phase_colours(phases: np.ndarray) -> np.ndarray:
    """Map wrapped phase to the established cyclic cyberpunk palette."""
    fraction = ((phases + np.pi) / (2.0 * np.pi)) % 1.0
    anchors = np.asarray([
        [0, 229, 255], [72, 126, 255], [151, 69, 235],
        [255, 20, 147], [255, 116, 199], [166, 255, 0], [0, 229, 255],
    ], dtype=np.float64)
    position = fraction * (len(anchors) - 1)
    lower = np.floor(position).astype(int)
    upper = np.minimum(lower + 1, len(anchors) - 1)
    weight = (position - lower)[..., None]
    return np.clip(
        anchors[lower] * (1.0 - weight) + anchors[upper] * weight, 0, 255
    ).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=sorted(PRESETS), default="strange")
    parser.add_argument("--tall-panel", action="store_true")
    args = parser.parse_args()
    preset = PRESETS[args.preset]
    source = Path(preset["source"])
    output = Path(preset["output"])
    if args.tall_panel:
        output = output.with_name(output.stem + "_tall_panel.mp4")
    poster = output.with_suffix(".png")
    smiles = str(preset["smiles"])

    archive = np.load(source, allow_pickle=True)
    trajectory = np.asarray(archive["trajectory"], dtype=np.float64)
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("RDKit could not parse the source SMILES")
    atom_labels = [f"{atom.GetSymbol()}{atom.GetIdx() + 1}" for atom in molecule.GetAtoms()]
    if trajectory.shape[1] != len(atom_labels):
        raise ValueError(
            f"Trajectory has {trajectory.shape[1]} atoms; SMILES has {len(atom_labels)}"
        )

    if preset["display"] == "circular_phase":
        # Circular averaging preserves continuity at the -pi/pi boundary.
        channel_phases = np.pi * trajectory
        atom_phase = np.angle(np.mean(np.exp(1j * channel_phases), axis=2))
        display_note = "colour is the circular mean of its 16 learned phase channels."
    else:
        # Gated-residual states are non-circular. Robust global normalization
        # converts the mean channel state to the same colour coordinate.
        atom_state = trajectory.mean(axis=2)
        low, high = np.quantile(atom_state, [0.01, 0.99])
        normalized = np.clip((atom_state - low) / max(high - low, 1e-9), 0.0, 1.0)
        atom_phase = normalized * (2.0 * np.pi) - np.pi
        display_note = "colour is its robust-normalized mean across 16 learned channels."
    colours = phase_colours(atom_phase)

    width = 1440 if args.tall_panel else 1920
    height, fps, frames = 1080, 30, 900
    plot_left = 105 if args.tall_panel else 145
    plot_top = 255
    plot_width = 1230 if args.tall_panel else 1640
    plot_height = 690
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(FFMPEG), "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ]
    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    assert process.stdin is not None

    title = font(30 if args.tall_panel else 38, bold=True)
    subtitle = font(22)
    atom_font = font(17, bold=True, mono=True)
    small = font(17)
    mono = font(18, mono=True)
    total_generations = len(trajectory) - 1

    try:
        for frame in range(frames):
            generation = round(total_generations * frame / (frames - 1))
            canvas = Image.new("RGB", (width, height), "#05070D")
            draw = ImageDraw.Draw(canvas, "RGBA")
            draw.text(
                (width / 2, 38),
                "MOLECULAR GRAPH CELLULAR AUTOMATON | ONE-DIMENSIONAL VIEW",
                font=title, fill="#E9F3FF", anchor="ma",
            )
            draw.text(
                (width / 2, 92),
                str(preset["subtitle"]),
                font=subtitle, fill="#00E5FF", anchor="ma",
            )
            draw.text(
                (width / 2, 135), f"SMILES  {smiles}",
                font=mono, fill="#B7C6D8", anchor="ma",
            )
            draw.text(
                (plot_left, 196),
                "atoms in SMILES/RDKit order  →",
                font=small, fill="#A9B9C9", anchor="la",
            )

            cell_width = plot_width / len(atom_labels)
            for index, label in enumerate(atom_labels):
                x = plot_left + (index + 0.5) * cell_width
                draw.text((x, 228), label, font=atom_font,
                          fill="#DCE6F2", anchor="ms")
                draw.line((plot_left + index * cell_width, plot_top,
                           plot_left + index * cell_width, plot_top + plot_height),
                          fill=(65, 90, 110, 70), width=1)

            active = colours[:generation + 1]
            cascade = Image.fromarray(active, mode="RGB").resize(
                (plot_width, max(2, round(plot_height * (generation + 1) /
                                          len(trajectory)))),
                Image.Resampling.NEAREST,
            )
            canvas.paste(cascade, (plot_left, plot_top))
            cascade_bottom = plot_top + cascade.height
            draw.line((plot_left, cascade_bottom, plot_left + plot_width,
                       cascade_bottom), fill=(166, 255, 0, 235), width=3)
            draw.rectangle((plot_left - 2, plot_top - 2,
                            plot_left + plot_width + 1,
                            plot_top + plot_height + 1),
                           outline=(0, 229, 255, 150), width=2)

            draw.text((55, plot_top + plot_height / 2), "generation  ↓",
                      font=small, fill="#A9B9C9", anchor="mm",
                      stroke_width=0)
            draw.text(
                (plot_left, 1005),
                f"Each coloured cell is one atom; {display_note}",
                font=small, fill="#B7C6D8", anchor="ls",
            )
            draw.text(
                (width - 135, 1005),
                f"generation {generation:04d} / {total_generations:04d}",
                font=mono, fill="#A6FF00", anchor="rs",
            )
            process.stdin.write(np.asarray(canvas, dtype=np.uint8).tobytes())
            if frame == frames - 1:
                canvas.save(poster, dpi=(180, 180))
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("FFmpeg failed while encoding the molecular CA video")
    print(output.resolve())
    print(poster.resolve())


if __name__ == "__main__":
    main()
