#!/usr/bin/env python3
"""Render a 30-second one-dimensional Gray-Scott CA explainer."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "gray_scott_1d_ca_video"
FFMPEG = Path(r"C:\ffmpeg\bin\ffmpeg.exe")


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    filename = "seguisb.ttf" if bold else "segoeui.ttf"
    try:
        return ImageFont.truetype(f"C:/Windows/Fonts/{filename}", size)
    except OSError:
        return ImageFont.load_default()


def concentration_colours(values: np.ndarray) -> np.ndarray:
    fraction = np.clip(values, 0.0, 1.0)
    anchors = np.asarray([
        [5, 7, 13], [0, 74, 105], [0, 229, 255],
        [123, 72, 235], [255, 20, 147], [255, 212, 244],
    ], dtype=np.float64)
    position = fraction * (len(anchors) - 1)
    lower = np.floor(position).astype(int)
    upper = np.minimum(lower + 1, len(anchors) - 1)
    weight = (position - lower)[..., None]
    return np.clip(anchors[lower] * (1.0 - weight) + anchors[upper] * weight,
                   0, 255).astype(np.uint8)


def simulate(cells: int = 256, generations: int = 900) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1701)
    u = np.ones(cells, dtype=np.float64)
    v = np.zeros(cells, dtype=np.float64)
    centre = cells // 2
    u[centre - 8:centre + 9] = 0.48
    v[centre - 8:centre + 9] = 0.26
    u += 0.008 * rng.standard_normal(cells)
    v += 0.004 * rng.standard_normal(cells)
    diffusion_u, diffusion_v = 0.16, 0.08
    feed, kill, time_step = 0.035, 0.061, 1.0
    u_history = np.empty((generations, cells), dtype=np.float64)
    v_history = np.empty((generations, cells), dtype=np.float64)
    for generation in range(generations):
        u_history[generation], v_history[generation] = u, v
        laplacian_u = np.roll(u, 1) + np.roll(u, -1) - 2.0 * u
        laplacian_v = np.roll(v, 1) + np.roll(v, -1) - 2.0 * v
        reaction = u * v * v
        u = u + time_step * (diffusion_u * laplacian_u - reaction + feed * (1.0 - u))
        v = v + time_step * (diffusion_v * laplacian_v + reaction - (feed + kill) * v)
        u, v = np.clip(u, 0.0, 1.0), np.clip(v, 0.0, 1.0)
    return u_history, v_history


def main() -> None:
    width, height, fps, frames = 1280, 720, 30, 900
    _, v_history = simulate(generations=frames)
    scale_low, scale_high = np.quantile(v_history, [0.01, 0.995])
    display = np.clip((v_history - scale_low) / max(scale_high - scale_low, 1e-12), 0, 1)
    colours = concentration_colours(display)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    video = OUTPUT / "gray_scott_1d_cellular_automaton.mp4"
    poster = OUTPUT / "gray_scott_1d_cellular_automaton.png"

    command = [
        str(FFMPEG), "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert process.stdin is not None
    title_font, subtitle_font = font(34, True), font(19)
    small_font, mono_font = font(15), font(15)
    plot_left, plot_top, plot_width, plot_height = 80, 155, 1120, 490

    try:
        for frame in range(frames):
            canvas = Image.new("RGB", (width, height), "#05070D")
            draw = ImageDraw.Draw(canvas, "RGBA")
            draw.text((width / 2, 30), "ONE-DIMENSIONAL GRAY-SCOTT CELLULAR AUTOMATON",
                      font=title_font, fill="#E9F3FF", anchor="ma")
            draw.text((width / 2, 78),
                      "Local diffusion and reaction generating a molecular concentration pattern",
                      font=subtitle_font, fill="#00E5FF", anchor="ma")
            draw.text((plot_left, 119), "cell position  →", font=small_font,
                      fill="#A9B9C9", anchor="la")
            vertical_label = Image.new("RGBA", (150, 28), (0, 0, 0, 0))
            label_draw = ImageDraw.Draw(vertical_label)
            label_draw.text((75, 14), "generation  ↓", font=small_font,
                            fill="#A9B9C9", anchor="mm")
            vertical_label = vertical_label.rotate(90, expand=True)
            canvas.paste(vertical_label,
                         (20, int(plot_top + (plot_height - vertical_label.height) / 2)),
                         vertical_label)

            active = colours[:frame + 1]
            diagram = Image.fromarray(active, mode="RGB")
            shown_height = max(2, int(plot_height * (frame + 1) / frames))
            diagram = diagram.resize((plot_width, shown_height), Image.Resampling.NEAREST)
            canvas.paste(diagram, (plot_left, plot_top))
            draw.rectangle((plot_left - 1, plot_top - 1,
                            plot_left + plot_width, plot_top + plot_height),
                           outline=(105, 125, 145, 180), width=1)
            draw.line((plot_left, plot_top + shown_height,
                       plot_left + plot_width, plot_top + shown_height),
                      fill=(166, 255, 0, 220), width=2)
            draw.text((plot_left, 678),
                      "du/dt = D_u Lap(u) - uv^2 + F(1-u)   |   dv/dt = D_v Lap(v) + uv^2 - (F+k)v",
                      font=mono_font, fill="#DCE6F2", anchor="ls")
            draw.text((width - 80, 678), f"generation {frame:04d} / 0899",
                      font=mono_font, fill="#A6FF00", anchor="rs")
            process.stdin.write(np.asarray(canvas, dtype=np.uint8).tobytes())
            if frame == frames - 1:
                canvas.save(poster, dpi=(180, 180))
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("FFmpeg failed while encoding the animation")
    print(video.resolve())
    print(poster.resolve())


if __name__ == "__main__":
    main()
