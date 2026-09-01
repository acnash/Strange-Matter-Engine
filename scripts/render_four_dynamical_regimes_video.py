#!/usr/bin/env python3
"""Combine the four long-horizon Graph-CA regimes into one 2 x 2 film."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VIDEO_DIR = ROOT / "results" / "long_horizon_attractor_campaign_v1" / "videos"
SOURCES = (
    VIDEO_DIR / "trajectory_01_point_attractor_convergence.mp4",
    VIDEO_DIR / "trajectory_07_hyperchaotic_strange_attractor.mp4",
    VIDEO_DIR / "trajectory_kuramoto_persistent_complex_candidate.mp4",
    VIDEO_DIR / "trajectory_coupled_map_period2_oscillator_candidate.mp4",
)
OUTPUT = VIDEO_DIR / "four_graph_ca_dynamical_regimes_2x2.mp4"


def main() -> None:
    ffmpeg = shutil.which("ffmpeg") or r"C:\ffmpeg\bin\ffmpeg.exe"
    missing = [str(path) for path in SOURCES if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing source videos: {missing}")

    # The strange-attractor source contains 45 seconds, while the other three
    # contain 30.  Its timeline is compressed to 30 seconds so every panel
    # begins and ends together while retaining the complete source trajectory.
    filter_graph = (
        "[0:v]trim=duration=30,setpts=PTS-STARTPTS[a];"
        "[1:v]setpts=(2/3)*PTS,trim=duration=30,setpts=PTS-STARTPTS[b];"
        "[2:v]trim=duration=30,setpts=PTS-STARTPTS[c];"
        "[3:v]trim=duration=30,setpts=PTS-STARTPTS[d];"
        "[a][b]hstack=inputs=2[top];"
        "[c][d]hstack=inputs=2[bottom];"
        "[top][bottom]vstack=inputs=2,"
        "drawbox=x=1918:y=0:w=4:h=2160:color=0x00d9ff@0.55:t=fill,"
        "drawbox=x=0:y=1078:w=3840:h=4:color=0x00d9ff@0.55:t=fill,"
        "format=yuv420p[out]"
    )
    command = [ffmpeg, "-y"]
    for source in SOURCES:
        command.extend(["-i", str(source)])
    command.extend([
        "-filter_complex", filter_graph,
        "-map", "[out]", "-an", "-r", "30",
        "-c:v", "libx264", "-preset", "medium", "-crf", "19",
        "-movflags", "+faststart", str(OUTPUT),
    ])
    subprocess.run(command, cwd=ROOT, check=True)
    print(OUTPUT)


if __name__ == "__main__":
    main()
