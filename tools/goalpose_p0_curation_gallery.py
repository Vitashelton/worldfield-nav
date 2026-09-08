#!/usr/bin/env python3
"""Render an auditable gallery of existing indoor Goal-Pose P0 candidates.

This does not create semantic labels.  It presents deterministic existing
Habitat-GS renders for explicit human curation of target anchors.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/formal/TrajectoryGrounding/P1/dataset_manifest.json"
OUT = ROOT / "outputs/formal/GoalPose/P0"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--count", type=int, default=24)
    args = ap.parse_args()
    records = [x for x in json.loads(SOURCE.read_text()) if x["scene_id"] == args.scene]
    if not records:
        raise ValueError(f"no existing P1 records for {args.scene}")
    picks = [records[i] for i in range(0, len(records), max(1, len(records) // args.count))][:args.count]
    tile_w, tile_h, header, cols = 256, 288, 32, 4
    rows = (len(picks) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * tile_w, rows * tile_h + header), "#121820")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 9), f"Goal-Pose P0 curation gallery — {args.scene}; existing Habitat-GS goal renders; no semantic labels", fill="white")
    for n, item in enumerate(picks):
        image = Image.open(ROOT / "outputs/formal/TrajectoryGrounding/P1" / item["goal_image"]).convert("RGB")
        x, y = (n % cols) * tile_w, header + (n // cols) * tile_h
        canvas.paste(image.resize((tile_w, tile_w)), (x, y))
        draw.text((x + 6, y + tile_w + 7), item["episode_id"], fill="white")
        draw.text((x + 6, y + tile_w + 20), f"goal: {item['goal_xyz_hidden_for_evaluation']}", fill="#afc8df")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"curation_gallery_{args.scene}.png"
    canvas.save(path)
    print(json.dumps({"scene": args.scene, "images": len(picks), "path": str(path)}, indent=2))


if __name__ == "__main__":
    main()
