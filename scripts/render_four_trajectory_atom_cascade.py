"""Render graph-CA trajectories 5, 6, 7, and 8 as four atom cascades."""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class TrajectorySpec:
    rank: int
    stem: str
    rule_label: str
    molecule: str
    target: str


SPECS = (
    TrajectorySpec(5, "05_inertial-reaction-diffusion_OCNT-2312382_CYP1A2",
                   "INERTIAL REACTION-DIFFUSION", "OCNT-2312382", "CYP1A2"),
    TrajectorySpec(6, "06_inertial-reaction-diffusion_OCNT-2315030_CYP2D6",
                   "INERTIAL REACTION-DIFFUSION", "OCNT-2315030", "CYP2D6"),
    TrajectorySpec(7, "07_kuramoto-sakaguchi_OCNT-0494110_CYP2C9",
                   "KURAMOTO-SAKAGUCHI", "OCNT-0494110", "CYP2C9"),
    TrajectorySpec(8, "08_kuramoto-sakaguchi_OCNT-2328784_CYP1A2",
                   "KURAMOTO-SAKAGUCHI", "OCNT-2328784", "CYP1A2"),
)


def parse_pdb(path: Path) -> tuple[np.ndarray, list[str]]:
    coordinates, elements = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            coordinates.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            elements.append((line[76:78].strip() or line[12:14].strip())[0].upper())
    points = np.asarray(coordinates, dtype=np.float32)
    points -= points.mean(axis=0)
    return points, elements


def rotation_y(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray([[cosine, 0.0, sine], [0.0, 1.0, 0.0],
                       [-sine, 0.0, cosine]], dtype=np.float32)


def colours(values: np.ndarray) -> np.ndarray:
    fraction = np.clip(values / 100.0, 0.0, 1.0)[..., None]
    return np.asarray([0.0, 1.0, 1.0]) + fraction * np.asarray([1.0, -1.0, 0.0])


def get_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", size)
    except OSError:
        return ImageFont.load_default()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--fps", type=int, default=40)
    parser.add_argument("--ffmpeg", type=Path, default=Path("C:/ffmpeg/bin/ffmpeg.exe"))
    args = parser.parse_args()

    source = args.root / "results" / "ds_gcae_1000_generation_pymol"
    output = args.output or source / "trajectories_05_06_07_08_four_column_atom_cascade.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_count = 1001
    top_margin, bottom_margin = 138, 55
    usable_height = args.height - top_margin - bottom_margin
    column_width = args.width / 4.0
    twist_step = 2.0 * np.pi * 5.25 / (frame_count - 1)

    cascades = []
    for spec in SPECS:
        xyz, elements = parse_pdb(source / "structures" / f"{spec.stem}.pdb")
        values = np.load(source / "display_values" / f"{spec.stem}.npz")["display_values"]
        if values.shape != (frame_count, len(elements)):
            raise ValueError(f"Unexpected trajectory shape for {spec.stem}: {values.shape}")
        radius = max(np.ptp(xyz[:, 0]), np.ptp(xyz[:, 2])) / 2.0
        vertical_step = max(radius * 0.0105, 0.035)
        history = []
        for generation in range(frame_count):
            points = xyz @ rotation_y(generation * twist_step).T
            points[:, 1] -= generation * vertical_step
            history.append(points)
        history = np.asarray(history, dtype=np.float32)
        scale = min(column_width * 0.31 / max(radius, 0.1), usable_height / np.ptp(history[:, :, 1]))
        cascades.append((spec, history, colours(values), elements, scale,
                         float(history[0, :, 1].max())))

    command = [str(args.ffmpeg), "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{args.width}x{args.height}", "-r", str(args.fps), "-i", "-",
               "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    title_font, label_font, small_font = get_font(29), get_font(16), get_font(14)
    atom_radii = {"H": 1.3, "C": 2.4, "N": 2.6, "O": 2.8, "S": 3.0, "P": 3.0}

    try:
        for frame in range(frame_count):
            image = Image.new("RGB", (args.width, args.height), "black")
            draw = ImageDraw.Draw(image, "RGBA")
            draw.text((args.width / 2, 22), "GRAPH CELLULAR AUTOMATA | 1,000-GENERATION CASCADES",
                      font=title_font, fill=(235, 240, 250, 255), anchor="ma")
            draw.text((args.width - 25, args.height - 24), f"generation {frame:04d} / 1000",
                      font=small_font, fill=(155, 170, 185, 255), anchor="ra")
            camera = rotation_y(0.28 * np.sin(frame / 1000.0 * 2.0 * np.pi))

            for column, (spec, history, state_colours, elements, scale, top_y) in enumerate(cascades):
                centre_x = (column + 0.5) * column_width
                draw.text((centre_x, 67), f"TRAJECTORY {spec.rank}", font=label_font,
                          fill=(240, 240, 245, 255), anchor="ma")
                draw.text((centre_x, 91), spec.rule_label, font=small_font,
                          fill=(120, 175, 205, 255), anchor="ma")
                draw.text((centre_x, 112), f"{spec.molecule} | {spec.target}", font=small_font,
                          fill=(125, 140, 155, 255), anchor="ma")

                points = history[:frame + 1].reshape(-1, 3) @ camera.T
                rgb = state_colours[:frame + 1].reshape(-1, 3)
                atom_indices = np.tile(np.arange(len(elements)), frame + 1)
                generations = np.repeat(np.arange(frame + 1), len(elements))
                x = centre_x + points[:, 0] * scale
                y = top_margin + (top_y - points[:, 1]) * scale
                order = np.argsort(points[:, 2])
                alpha = np.clip(220.0 - (frame - generations) * 0.15, 60.0, 220.0)
                depth = np.clip(0.78 + 0.22 * (points[:, 2] - points[:, 2].min()) /
                                max(float(np.ptp(points[:, 2])), 1e-6), 0.65, 1.0)
                for index in order:
                    red, green, blue = (rgb[index] * 255.0 * depth[index]).astype(int)
                    radius = atom_radii.get(elements[atom_indices[index]], 2.5)
                    draw.ellipse((x[index] - radius, y[index] - radius,
                                  x[index] + radius, y[index] + radius),
                                 fill=(int(red), int(green), int(blue), int(alpha[index])))

            process.stdin.write(np.asarray(image, dtype=np.uint8).tobytes())
            if frame % 100 == 0:
                print(f"Rendered generation {frame}/1000", flush=True)
    finally:
        process.stdin.close()
    if process.wait():
        raise RuntimeError("FFmpeg failed")
    print(f"Wrote {output.resolve()}")


if __name__ == "__main__":
    main()
