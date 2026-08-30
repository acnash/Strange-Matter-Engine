#!/usr/bin/env python3
"""Render a 30-second one-dimensional Kuramoto-Sakaguchi CA explainer."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "kuramoto_sakaguchi_1d_ca_video"
FFMPEG = Path(r"C:\ffmpeg\bin\ffmpeg.exe")


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    filename = "seguisb.ttf" if bold else "segoeui.ttf"
    try:
        return ImageFont.truetype(f"C:/Windows/Fonts/{filename}", size)
    except OSError:
        return ImageFont.load_default()


def phase_colours(phases: np.ndarray) -> np.ndarray:
    fraction = ((phases + np.pi) / (2.0 * np.pi)) % 1.0
    anchors = np.asarray([
        [0, 229, 255], [72, 126, 255], [151, 69, 235],
        [255, 20, 147], [255, 116, 199], [0, 229, 255],
    ], dtype=np.float64)
    position = fraction * (len(anchors) - 1)
    lower = np.floor(position).astype(int)
    upper = np.minimum(lower + 1, len(anchors) - 1)
    weight = (position - lower)[..., None]
    return np.clip(anchors[lower] * (1.0 - weight) + anchors[upper] * weight,
                   0, 255).astype(np.uint8)


def simulate(cells: int = 256, generations: int = 900) -> np.ndarray:
    rng = np.random.default_rng(1701)
    position = np.arange(cells)
    phase = 0.018 * rng.standard_normal(cells)
    distance = np.minimum(position, cells - position)
    phase += 2.45 * np.exp(-(distance / 7.0) ** 2)
    frequency = 0.055 * np.sin(2.0 * np.pi * position / cells)
    frequency += 0.018 * rng.standard_normal(cells)
    coupling, lag, step = 1.42, 0.78, 0.105
    history = np.empty((generations, cells), dtype=np.float64)
    history[0] = phase
    for generation in range(1, generations):
        left, right = np.roll(phase, 1), np.roll(phase, -1)
        neighbour_drive = 0.5 * (
            np.sin(left - phase - lag) + np.sin(right - phase - lag)
        )
        phase = phase + step * (frequency + coupling * neighbour_drive)
        phase = (phase + np.pi) % (2.0 * np.pi) - np.pi
        history[generation] = phase
    return history


def main() -> None:
    width, height, fps, frames = 1280, 720, 30, 900
    history = simulate(generations=frames)
    colours = phase_colours(history)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    video = OUTPUT / "kuramoto_sakaguchi_1d_cellular_automaton.mp4"
    poster = OUTPUT / "kuramoto_sakaguchi_1d_cellular_automaton.png"

    command = [
        str(FFMPEG), "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert process.stdin is not None
    title_font, subtitle_font = font(32, True), font(19)
    small_font, mono_font = font(15), font(16)
    plot_left, plot_top, plot_width, plot_height = 80, 155, 1120, 490

    try:
        for frame in range(frames):
            canvas = Image.new("RGB", (width, height), "#05070D")
            draw = ImageDraw.Draw(canvas, "RGBA")
            draw.text((width / 2, 30), "ONE-DIMENSIONAL KURAMOTO-SAKAGUCHI CELLULAR AUTOMATON",
                      font=title_font, fill="#E9F3FF", anchor="ma")
            draw.text((width / 2, 78),
                      "Local phase information propagating through nearest-neighbour interactions",
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
                      "theta_i(t+1) = wrap[theta_i(t) + dt(omega_i + K/2 Σ sin(theta_j - theta_i - psi))]",
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
