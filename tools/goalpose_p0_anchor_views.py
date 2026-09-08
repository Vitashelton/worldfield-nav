#!/usr/bin/env python3
"""Create reproducible RGB-D anchor-view candidates for human P0 curation.

An anchor is not labelled automatically.  This utility records exact camera
pose, intrinsics and depth for a finite set of views, so a later manually chosen
target pixel can be metric-backprojected without inventing a target center.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from trajectory_grounding_p1_generate import OUT as P1, make_sim, render, yaw_rotation
OUT = ROOT / "outputs/formal/GoalPose/P0/anchor_views"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--episode-indices", type=int, nargs="+", default=[0, 60, 120])
    ap.add_argument("--yaw-count", type=int, default=12)
    args = ap.parse_args()
    manifest = [x for x in json.loads((P1 / "dataset_manifest.json").read_text()) if x["scene_id"] == args.scene]
    sim = make_sim(args.scene); agent = sim.initialize_agent(0); records = []
    try:
        for episode_index in args.episode_indices:
            source = manifest[episode_index]
            anchor_id = f"{args.scene}_{episode_index:04d}"
            root = OUT / anchor_id; root.mkdir(parents=True, exist_ok=True)
            tiles = []
            for yaw_index, yaw in enumerate(np.linspace(-math.pi, math.pi, args.yaw_count, endpoint=False)):
                rgb, depth, camera_xyz, c2w = render(sim, agent, np.asarray(source["goal_xyz_hidden_for_evaluation"], np.float32), yaw_rotation(float(yaw)))
                rgb_path, depth_path = root / f"view_{yaw_index:02d}.png", root / f"view_{yaw_index:02d}_depth.npy"
                Image.fromarray(rgb).save(rgb_path); np.save(depth_path, depth.astype(np.float16))
                record = {"anchor_view_id": f"{anchor_id}_view_{yaw_index:02d}", "scene_id": args.scene,
                          "source_episode_id": source["episode_id"], "camera_xyz": camera_xyz.tolist(),
                          "camera_c2w": c2w.tolist(), "yaw_rad": float(yaw), "intrinsics": {"fx": 128.0, "fy": 128.0, "cx": 127.5, "cy": 127.5},
                          "rgb": str(rgb_path.relative_to(OUT.parent.parent)), "depth": str(depth_path.relative_to(OUT.parent.parent)),
                          "curation_status": "unselected"}
                records.append(record)
                tile = Image.fromarray(rgb).resize((192, 192)); tiles.append((tile, record["anchor_view_id"], float(yaw)))
            gallery = Image.new("RGB", (4 * 192, 3 * 216 + 30), "#111820"); draw = ImageDraw.Draw(gallery)
            draw.text((8, 8), f"{anchor_id}: select only clear physical targets; record pixel/template explicitly", fill="white")
            for n, (image, name, yaw) in enumerate(tiles):
                x, y = (n % 4) * 192, 30 + (n // 4) * 216
                gallery.paste(image, (x, y)); draw.text((x + 4, y + 194), f"v{n:02d}, yaw={yaw:+.2f}", fill="white")
            gallery.save(root / "contact_sheet.png")
    finally:
        sim.close()
    (OUT / f"anchor_view_candidates_{args.scene}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in records))
    print(json.dumps({"scene": args.scene, "views": len(records), "out": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
