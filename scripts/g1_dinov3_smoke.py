#!/usr/bin/env python3
"""G1 prerequisite: frozen DINOv3-S/16 forward on exactly two C1 RGB frames."""
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
MODEL_NAME = "hf_hub:timm/vit_small_patch16_dinov3.lvd1689m"


def tensor_summary(value):
    if torch.is_tensor(value):
        return {"type": "tensor", "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {str(k): tensor_summary(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_summary(v) for v in value]
    return {"type": type(value).__name__}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/formal/G1/smoke")
    parser.add_argument("--trajectory", default="interior_0405_840145_traj00")
    parser.add_argument("--frames", nargs=2, type=int, default=[8, 23])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("G1 requires CUDA; CPU fallback is prohibited")
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    source = ROOT / "outputs/formal/C1/pilot/trajectories" / args.trajectory / "sequence.npz"
    with np.load(source, allow_pickle=False) as data:
        rgb_frames = [data["rgb"][frame].copy() for frame in args.frames]
    torch.cuda.reset_peak_memory_stats()
    model = timm.create_model(MODEL_NAME, pretrained=True).cuda().eval()
    cfg = resolve_model_data_config(model)
    image_h, image_w = cfg["input_size"][-2:]
    images = [Image.fromarray(rgb).resize((image_w, image_h), Image.Resampling.BICUBIC) for rgb in rgb_frames]
    mean = torch.tensor(cfg["mean"], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device="cuda").view(1, 3, 1, 1)
    batch = torch.from_numpy(np.stack([np.asarray(image) for image in images])).permute(0, 3, 1, 2).float().div(255).cuda()
    batch = (batch - mean) / std
    torch.cuda.synchronize(); started = time.perf_counter()
    with torch.no_grad():
        features = model.forward_features(batch)
    torch.cuda.synchronize(); elapsed = time.perf_counter() - started
    tokens = features if torch.is_tensor(features) else features.get("x_norm", features.get("x", None))
    if not torch.is_tensor(tokens) or tokens.ndim != 3:
        raise RuntimeError(f"Could not locate dense token tensor: {tensor_summary(features)}")
    prefix = int(getattr(model, "num_prefix_tokens", 0))
    dense = tokens[:, prefix:]
    grid = int(round(dense.shape[1] ** 0.5))
    result = {
        "experiment": "G1 frozen DINOv3 two-image smoke",
        "model": MODEL_NAME,
        "trajectory": args.trajectory,
        "frames": args.frames,
        "device": torch.cuda.get_device_name(0),
        "forward_features": tensor_summary(features),
        "prefix_token_count": prefix,
        "register_token_count": int(getattr(model, "num_reg_tokens", 0)),
        "patch_token_count": int(dense.shape[1]),
        "dense_feature_shape": list(dense.shape),
        "patch_grid": [grid, grid] if grid * grid == dense.shape[1] else None,
        "inference_seconds": elapsed,
        "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "transform": {"input_size": list(cfg["input_size"]), "mean": list(cfg["mean"]), "std": list(cfg["std"])},
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
