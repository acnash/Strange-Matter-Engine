#!/usr/bin/env python3
"""Combine point and strange molecular 1D Graph-CA cascades side by side."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VIDEOS = ROOT / "results" / "long_horizon_attractor_campaign_v1" / "videos"
POINT = VIDEOS / "trajectory_01_point_attractor_molecular_1d_cellular_automaton_tall_panel.mp4"
STRANGE = VIDEOS / "trajectory_07_molecular_1d_cellular_automaton_tall_panel.mp4"
OUTPUT = VIDEOS / "point_and_strange_molecular_1d_cellular_automata_side_by_side_tall.mp4"
FFMPEG = Path(r"C:\ffmpeg\bin\ffmpeg.exe")


def main() -> None:
    command = [
        str(FFMPEG), "-y", "-i", str(POINT), "-i", str(STRANGE),
        "-filter_complex",
        "[0:v]trim=duration=30,setpts=PTS-STARTPTS[left];"
        "[1:v]trim=duration=30,setpts=PTS-STARTPTS[right];"
        "[left][right]hstack=inputs=2,"
        "drawbox=x=1438:y=0:w=4:h=1080:color=0x00d9ff@0.65:t=fill,"
        "format=yuv420p[out]",
        "-map", "[out]", "-an", "-r", "30", "-c:v", "libx264",
        "-preset", "medium", "-crf", "19", "-movflags", "+faststart",
        str(OUTPUT),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()
