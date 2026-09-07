#!/usr/bin/env python3
"""Cache frozen DINOv3 descriptors for ImageNav state/goal conditioning.

This deliberately does not score or plan trajectories.  A branch-value model
receives the same current/goal visual condition for all eight planner branches
and learns branch value from its geometric and execution inputs.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import timm
from timm.data import resolve_model_data_config

ROOT = Path(__file__).resolve().parents[1]
MODEL = "hf_hub:timm/vit_small_patch16_dinov3.lvd1689m"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT / "outputs/formal/TrajectoryGrounding/P1")
    ap.add_argument("--batch-size", type=int, default=96)
    args = ap.parse_args()
    manifest = json.loads((args.root / "dataset_manifest.json").read_text())
    if not torch.cuda.is_available():
        raise RuntimeError("Frozen DINOv3 descriptor extraction requires CUDA")

    model = timm.create_model(MODEL, pretrained=True).cuda().eval()
    cfg = resolve_model_data_config(model)
    mean = torch.tensor(cfg["mean"], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device="cuda").view(1, 3, 1, 1)

    def image(relative: str) -> torch.Tensor:
        im = Image.open(args.root / relative).convert("RGB").resize((256, 256), Image.Resampling.BICUBIC)
        return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float().div_(255)

    cur, goal = [], []
    started = time.perf_counter()
    for begin in range(0, len(manifest), args.batch_size):
        chunk = manifest[begin : begin + args.batch_size]
        x = torch.stack([image(r["current_rgb"]) for r in chunk] + [image(r["goal_image"]) for r in chunk]).cuda(non_blocking=True)
        with torch.inference_mode():
            y = model.forward_features((x - mean) / std)
            if not torch.is_tensor(y):
                y = y.get("x_norm", y.get("x"))
            y = torch.nn.functional.normalize(y[:, int(model.num_prefix_tokens) :].mean(1), dim=-1)
        n = len(chunk)
        cur.append(y[:n].cpu().half().numpy())
        goal.append(y[n:].cpu().half().numpy())
    torch.cuda.synchronize()
    np.savez_compressed(
        args.root / "dino_pair_features.npz",
        episode_id=np.asarray([r["episode_id"] for r in manifest]),
        current=np.concatenate(cur),
        goal=np.concatenate(goal),
    )
    (args.root / "dino_pair_features_metrics.json").write_text(json.dumps({
        "episodes": len(manifest), "dim": 384, "model": MODEL, "frozen": True,
        "seconds": time.perf_counter() - started,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
