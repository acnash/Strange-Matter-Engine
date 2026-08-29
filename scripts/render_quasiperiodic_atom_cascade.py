"""Render trajectory 7 as a rotating, bond-free atom cascade.

Each graph-CA generation is retained, rotated slightly around the vertical axis,
and displaced downward.  Frames are streamed directly to FFmpeg, so the script
does not create thousands of temporary images.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_pdb_atoms(path: Path) -> tuple[np.ndarray, list[str]]:
    coordinates: list[list[float]] = []
    elements: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            coordinates.append(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            )
            elements.append((line[76:78].strip() or line[12:14].strip())[0].upper())
    return np.asarray(coordinates, dtype=np.float32), elements


def rotation_y(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float32,
    )


def cyan_magenta(values: np.ndarray) -> np.ndarray:
    """Match PyMOL's cyan_magenta spectrum over the saved 0..100 values."""
    fraction = np.clip(values / 100.0, 0.0, 1.0)[..., None]
    cyan = np.asarray([0.0, 1.0, 1.0], dtype=np.float32)
    magenta = np.asarray([1.0, 0.0, 1.0], dtype=np.float32)
    return cyan + fraction * (magenta - cyan)


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", size)
    except OSError:
        return ImageFont.load_default()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--size", type=int, default=900)
    parser.add_argument("--fps", type=int, default=40)
    parser.add_argument("--frames", type=int, default=1001)
    parser.add_argument("--ffmpeg", type=Path, default=Path("C:/ffmpeg/bin/ffmpeg.exe"))
    args = parser.parse_args()

    source = args.root / "results" / "ds_gcae_1000_generation_pymol"
    pdb = source / "structures" / "07_kuramoto-sakaguchi_OCNT-0494110_CYP2C9.pdb"
    npz = source / "display_values" / "07_kuramoto-sakaguchi_OCNT-0494110_CYP2C9.npz"
    output = args.output or source / "trajectory_07_quasiperiodic_atom_cascade.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)

    coordinates, elements = parse_pdb_atoms(pdb)
    values = np.load(npz)["display_values"]
    if len(coordinates) != values.shape[1]:
        raise ValueError(f"PDB has {len(coordinates)} atoms but trajectory has {values.shape[1]}")

    coordinates -= coordinates.mean(axis=0)
    molecular_radius = max(np.ptp(coordinates[:, 0]), np.ptp(coordinates[:, 2])) / 2.0
    generation_count = min(args.frames, values.shape[0])
    vertical_step = max(molecular_radius * 0.0105, 0.035)
    twist_step = 2.0 * np.pi * 5.25 / max(generation_count - 1, 1)

    # Precompute the helical history. The camera adds a slower global rotation.
    all_points = []
    for generation in range(generation_count):
        points = coordinates @ rotation_y(generation * twist_step).T
        points[:, 1] -= generation * vertical_step
        all_points.append(points)
    all_points = np.asarray(all_points, dtype=np.float32)
    all_colours = cyan_magenta(values[:generation_count])

    width = height = args.size
    top_margin, bottom_margin = 105, 65
    usable_height = height - top_margin - bottom_margin
    full_height = np.ptp(all_points[:, :, 1])
    scale = min(width * 0.35 / max(molecular_radius, 0.1), usable_height / full_height)
    centre_x = width / 2.0
    top_world_y = float(all_points[0, :, 1].max())
    title_font, small_font = font(max(19, width // 36)), font(max(14, width // 55))

    command = [
        str(args.ffmpeg), "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(args.fps), "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None

    radii = {"H": 2.0, "C": 3.3, "N": 3.5, "O": 3.7, "S": 4.0, "P": 4.0}
    try:
        for frame in range(generation_count):
            image = Image.new("RGB", (width, height), "black")
            draw = ImageDraw.Draw(image, "RGBA")
            draw.text((width // 2, 24), "QUASI-PERIODIC GRAPH-CA TRAJECTORY", font=title_font,
                      fill=(235, 240, 250, 255), anchor="ma")
            draw.text((width // 2, 58), "Kuramoto-Sakaguchi | OCNT-0494110 | CYP2C9",
                      font=small_font, fill=(135, 155, 175, 255), anchor="ma")
            draw.text((width - 24, height - 28), f"generation {frame:04d} / 1000",
                      font=small_font, fill=(160, 170, 185, 255), anchor="ra")

            camera = rotation_y(0.28 * np.sin(frame / max(generation_count - 1, 1) * 2.0 * np.pi))
            points = all_points[: frame + 1].reshape(-1, 3) @ camera.T
            colours = all_colours[: frame + 1].reshape(-1, 3)
            atom_indices = np.tile(np.arange(len(elements)), frame + 1)
            generations = np.repeat(np.arange(frame + 1), len(elements))

            x = centre_x + points[:, 0] * scale
            y = top_margin + (top_world_y - points[:, 1]) * scale
            depth_order = np.argsort(points[:, 2])
            age = frame - generations
            alpha = np.clip(225.0 - age * 0.15, 65.0, 225.0)
            depth_factor = np.clip(0.78 + 0.22 * (points[:, 2] - points[:, 2].min()) /
                                   max(float(np.ptp(points[:, 2])), 1e-6), 0.65, 1.0)

            for index in depth_order:
                red, green, blue = (colours[index] * 255.0 * depth_factor[index]).astype(int)
                radius = radii.get(elements[atom_indices[index]], 3.4)
                draw.ellipse(
                    (x[index] - radius, y[index] - radius, x[index] + radius, y[index] + radius),
                    fill=(int(red), int(green), int(blue), int(alpha[index])),
                )
            process.stdin.write(np.asarray(image, dtype=np.uint8).tobytes())
            if frame % 100 == 0:
                print(f"Rendered generation {frame}/{generation_count - 1}", flush=True)
    finally:
        process.stdin.close()
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"FFmpeg exited with status {return_code}")
    print(f"Wrote {output.resolve()}")


if __name__ == "__main__":
    main()
