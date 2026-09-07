#!/usr/bin/env python3
"""Render one deterministic navigable view per existing Habitat-GS scene.

This is intentionally a tiny availability check, not a benchmark generator.
It establishes the indoor P1 split before any formal samples are created.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from execfield_p0_generate import ROOT, make_sim, render, yaw_rotation

SCENES = [
    "interior_0405_840145", "interior_0047_839892", "interior_0108_839984",
    "interior_0184_840116", "interior_0093_839966", "interior_0121_840013",
    "interior_0135_840032",
]


def main() -> None:
    out = ROOT / "outputs/formal/TrajectoryGrounding/P1/scene_audit"
    out.mkdir(parents=True, exist_ok=True)
    thumbs = []
    for index, scene in enumerate(SCENES):
        rng = np.random.default_rng(202609070 + index)
        sim = make_sim(scene)
        try:
            agent = sim.initialize_agent(0)
            pos = np.asarray(sim.pathfinder.get_random_navigable_point(), np.float32)
            path = out / f"{scene}.png"
            rgb = render(sim, agent, pos, yaw_rotation(float(rng.uniform(-math.pi, math.pi))), path)
            thumbs.append((scene, rgb))
        finally:
            sim.close()
    canvas = Image.new("RGB", (4 * 256, 2 * 286), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (scene, rgb) in enumerate(thumbs):
        x, y = (i % 4) * 256, (i // 4) * 286
        canvas.paste(Image.fromarray(rgb), (x, y))
        draw.rectangle((x, y + 256, x + 256, y + 286), fill="black")
        draw.text((x + 8, y + 264), scene, fill="white")
    asset = ROOT / "paper_assets/figures/trajectory_grounding_p1_scene_availability.png"
    asset.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(asset)
    print(asset)


if __name__ == "__main__":
    main()
