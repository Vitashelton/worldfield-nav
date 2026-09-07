#!/usr/bin/env python3
"""Cache frozen DINOv3 trajectory-corridor to goal-image similarities."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
import timm
from timm.data import resolve_model_data_config

ROOT = Path(__file__).resolve().parents[1]
MODEL = "hf_hub:timm/vit_small_patch16_dinov3.lvd1689m"


def normalize(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x, dim=-1)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--root", type=Path, default=ROOT / "outputs/formal/TrajectoryGrounding/P1"); ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args(); manifest = json.loads((args.root / "dataset_manifest.json").read_text())
    if not torch.cuda.is_available(): raise RuntimeError("DINOv3 grounding requires CUDA")
    model = timm.create_model(MODEL, pretrained=True).cuda().eval(); cfg = resolve_model_data_config(model)
    mean = torch.tensor(cfg["mean"], device="cuda").view(1, 3, 1, 1); std = torch.tensor(cfg["std"], device="cuda").view(1, 3, 1, 1)

    def read(relative: str) -> torch.Tensor:
        image = Image.open(args.root / relative).convert("RGB").resize((256, 256), Image.Resampling.BICUBIC)
        return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().div_(255)

    rows, started = [], time.perf_counter()
    for begin in range(0, len(manifest), args.batch_size):
        chunk = manifest[begin: begin + args.batch_size]
        images = torch.stack([read(x["current_rgb"]) for x in chunk] + [read(x["goal_image"]) for x in chunk]).cuda(non_blocking=True)
        images = (images - mean) / std
        with torch.inference_mode():
            tokens = model.forward_features(images)
            if not torch.is_tensor(tokens): tokens = tokens.get("x_norm", tokens.get("x"))
            dense = normalize(tokens[:, int(model.num_prefix_tokens):])
        current, goals = dense[:len(chunk)], dense[len(chunk):]
        goal = normalize(goals.mean(1))
        for record, visual, target in zip(chunk, current, goal):
            scores = []
            for traj in record["trajectories"]:
                uv = np.asarray(traj["image_uv"], np.float32); valid = np.asarray(traj["image_projection_valid"], bool)
                xx = np.clip((uv[:, 0] // 16).astype(int), 0, 15); yy = np.clip((uv[:, 1] // 16).astype(int), 0, 15)
                ids = torch.as_tensor(yy[valid] * 16 + xx[valid], device="cuda", dtype=torch.long)
                if len(ids): scores.append(float((normalize(visual[ids].mean(0)) * target).sum().item()))
                else: scores.append(-1.0)
            rows.append({"episode_id": record["episode_id"], "model": MODEL, "trajectory_goal_scores": scores})
    torch.cuda.synchronize()
    (args.root / "dino_cache.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    (args.root / "dino_cache_metrics.json").write_text(json.dumps({"episodes": len(rows), "seconds": time.perf_counter() - started, "model": MODEL, "frozen": True}, indent=2) + "\n")


if __name__ == "__main__": main()
