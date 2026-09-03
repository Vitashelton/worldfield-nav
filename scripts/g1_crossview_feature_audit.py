#!/usr/bin/env python3
"""Small frozen-DINOv3 physical-correspondence audit for GeoAnchor G1."""
from __future__ import annotations

import argparse, csv, json, math, time
from collections import defaultdict
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch, timm, yaml
from timm.data import resolve_model_data_config

ROOT = Path(__file__).resolve().parents[1]
# The repository uses a src-layout but G1 is intentionally executed directly
# from the locked simulation environment, rather than installed into it.
sys.path.insert(0, str(ROOT / "src"))
from geoanchor.correspondence import backproject_pixel, depth_at, intrinsics, mine_correspondences, patch_center

MODEL = "hf_hub:timm/vit_small_patch16_dinov3.lvd1689m"


def load_config(path): return yaml.safe_load(path.read_text())


def pairs():
    # Fixed physical-motion regimes, selected by a geometry-only preflight.  All
    # three split roles retain overlap at 0.42 m / 0 deg (small), 1.56 m / 30 deg
    # (medium), and ~1.8 m / 75 deg (large), respectively.
    return [
        ("interior_0405_840145_traj00", 0, 12, "train", "small"),
        ("interior_0405_840145_traj00", 0, 45, "train", "medium"),
        ("interior_0405_840145_traj00", 0, 60, "train", "large"),
        # The known visible -> occluded -> revisit sequence is retained
        # explicitly.  The occluded pair may legitimately yield no positives.
        ("interior_0405_840145_traj00", 8, 15, "train", "occluded"),
        ("interior_0405_840145_traj00", 8, 23, "train", "revisit"),
        ("scene05_traj00", 0, 12, "train", "small"),
        ("scene05_traj00", 0, 45, "train", "medium"),
        ("scene05_traj00", 0, 60, "train", "large"),
        ("scene04_traj00", 0, 12, "validation", "small"),
        ("scene04_traj00", 0, 45, "validation", "medium"),
        ("scene04_traj00", 0, 60, "validation", "large"),
        ("scene56_traj00", 0, 12, "unseen", "small"),
        ("scene56_traj00", 0, 45, "unseen", "medium"),
        ("scene56_traj00", 0, 60, "unseen", "large"),
    ]


def preprocess(rgb, cfg):
    h, w = cfg["input_size"][-2:]
    x = np.asarray(Image.fromarray(rgb).resize((w, h), Image.Resampling.BICUBIC))
    x = torch.from_numpy(x).permute(2, 0, 1).float().div(255)
    mean = torch.tensor(cfg["mean"]).view(3, 1, 1); std = torch.tensor(cfg["std"]).view(3, 1, 1)
    return (x - mean) / std


def dense(model, batch):
    with torch.no_grad(): out = model.forward_features(batch)
    tokens = out if torch.is_tensor(out) else out.get("x_norm", out.get("x"))
    return tokens[:, int(getattr(model, "num_prefix_tokens", 0)):]


def target_patch_world(depth, c2w, patch, grid):
    u, v = patch_center(patch, grid, grid)
    sample = depth_at(depth, u, v)
    return None if sample is None else backproject_pixel(sample[0], sample[1], sample[2], c2w, intrinsics())


def summarize(rows, key):
    grouped = defaultdict(list)
    for row in rows: grouped[row[key]].append(row)
    result = {}
    for name, values in grouped.items():
        result[name] = {metric: float(np.mean([x[metric] for x in values])) for metric in ("positive_similarity", "negative_similarity", "margin", "r_at_1", "r_at_5", "world_localization_error_m", "depth_residual_m", "world_residual_m")}
        result[name]["correspondences"] = len(values)
    return result


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/benchmark/g1_crossview_feature_audit.yaml"); ap.add_argument("--output", default="outputs/formal/G1/audit"); args = ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("G1 CUDA required; CPU fallback prohibited")
    cfg = load_config(ROOT / args.config); out = ROOT / args.output; out.mkdir(parents=True, exist_ok=True)
    figures, tables = ROOT / "paper_assets/figures", ROOT / "paper_assets/tables"; figures.mkdir(parents=True, exist_ok=True); tables.mkdir(parents=True, exist_ok=True)
    model = timm.create_model(MODEL, pretrained=True).cuda().eval(); model_cfg = resolve_model_data_config(model)
    grid = int(model_cfg["input_size"][-1] // 16)
    all_rows, exemplars, pair_counts = [], [], {}
    started = time.perf_counter()
    for traj, a, b, split, regime in pairs():
        source = ROOT / "outputs/formal/C1/pilot/trajectories" / traj / "sequence.npz"
        with np.load(source, allow_pickle=False) as d:
            rgb_a, rgb_b = d["rgb"][a].copy(), d["rgb"][b].copy()
            depth_a, depth_b = d["depth"][a], d["depth"][b]
            c2w_a, c2w_b = d["sensor_pose_c2w"][a], d["sensor_pose_c2w"][b]
            lidar_b, count_b = d["sim_lidar_xyz"][b], int(d["sim_lidar_count"][b])
        batch = torch.stack([preprocess(rgb_a, model_cfg), preprocess(rgb_b, model_cfg)]).cuda()
        feats = torch.nn.functional.normalize(dense(model, batch), dim=-1).float().cpu().numpy()
        if feats.shape[1] != grid * grid:
            raise RuntimeError(
                f"DINO patch-token grid mismatch: expected {grid}x{grid}, got {feats.shape[1]} tokens"
            )
        corr = mine_correspondences(depth_a, c2w_a, depth_b, c2w_b, grid, grid, cfg["geometry"]["depth_agreement_m"], cfg["geometry"]["world_residual_m"], lidar_b, count_b)
        corr = corr[:cfg["geometry"]["max_correspondences_per_pair"]]
        pair_counts[f"{traj}:{a}->{b}:{regime}"] = len(corr)
        for c in corr:
            similarities = feats[0, c.source_patch] @ feats[1].T
            order = np.argsort(-similarities); rank = int(np.where(order == c.target_patch)[0][0])
            negatives = []
            for patch in range(grid * grid):
                world = target_patch_world(depth_b, c2w_b, patch, grid)
                if world is not None and np.linalg.norm(world - c.world_xyz) >= cfg["geometry"]["hard_negative_min_world_distance_m"]: negatives.append(patch)
            neg = float(np.max(similarities[negatives])) if negatives else float("nan")
            best_world = target_patch_world(depth_b, c2w_b, int(order[0]), grid)
            error = float(np.linalg.norm(best_world - c.world_xyz)) if best_world is not None else 10.0
            all_rows.append({"trajectory": traj, "split": split, "viewpoint_bin": regime, "source_frame": a, "target_frame": b, "positive_similarity": float(similarities[c.target_patch]), "negative_similarity": neg, "margin": float(similarities[c.target_patch] - neg), "r_at_1": float(rank == 0), "r_at_5": float(rank < 5), "world_localization_error_m": error, "depth_residual_m": c.depth_residual_m, "world_residual_m": c.world_residual_m, "lidar_residual_m": c.lidar_residual_m if c.lidar_residual_m is not None else -1.0, "source_u": c.source_uv[0], "source_v": c.source_uv[1], "target_u": c.target_uv[0], "target_v": c.target_uv[1]})
        if corr and (len(exemplars) < 3 or regime == "revisit"):
            exemplars.append((traj, a, b, regime, rgb_a, rgb_b, corr[0]))
    if not all_rows: raise RuntimeError("No geometry-valid correspondences mined")
    fields = list(all_rows[0]);
    with (tables / "g1_dinov3_baseline.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(all_rows)
    by_regime, by_split = summarize(all_rows, "viewpoint_bin"), summarize(all_rows, "split")
    small, large = by_regime.get("small"), by_regime.get("large")
    degradation = bool(small and large and (small["r_at_1"] - large["r_at_1"] >= .10 or small["positive_similarity"] - large["positive_similarity"] >= .10 or small["margin"] - large["margin"] >= .10))
    # Cross-view correspondence figure.
    fig, axes = plt.subplots(len(exemplars), 2, figsize=(9, 3 * len(exemplars)))
    axes = np.atleast_2d(axes)
    for row, (_, a, b, regime, ra, rb, c) in zip(axes, exemplars):
        for ax, image, uv, title in ((row[0], ra, c.source_uv, f"source ({regime})"), (row[1], rb, c.target_uv, "metric reprojection")):
            ax.imshow(image); ax.scatter(*uv, s=70, facecolors="none", edgecolors="red", linewidths=2); ax.set_title(title); ax.axis("off")
    fig.tight_layout(); fig.savefig(figures / "g1_crossview_correspondence.png", dpi=180); plt.close(fig)
    labels = [x for x in ("small", "medium", "large", "revisit") if x in by_regime]
    fig, ax = plt.subplots(figsize=(8, 4)); ax.plot(labels, [by_regime[x]["positive_similarity"] for x in labels], "o-", label="same physical surface"); ax.plot(labels, [by_regime[x]["negative_similarity"] for x in labels], "o-", label="hard negative"); ax.set_ylabel("cosine similarity"); ax.set_title("Frozen DINOv3 physical consistency vs viewpoint"); ax.legend(); fig.tight_layout(); fig.savefig(figures / "g1_dinov3_viewpoint_degradation.png", dpi=180); plt.close(fig)
    revisit = [x for x in exemplars if x[3] == "revisit"] or exemplars[:1]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4)); ex = revisit[0]; axes[0].imshow(ex[4]); axes[0].scatter(*ex[6].source_uv, s=80, facecolors="none", edgecolors="red", linewidths=2); axes[0].set_title("visible: frame 8"); axes[1].imshow(ex[5]); axes[1].scatter(*ex[6].target_uv, s=80, facecolors="none", edgecolors="red", linewidths=2); axes[1].set_title(f"revisit: frame {ex[2]}"); [ax.axis("off") for ax in axes]; fig.tight_layout(); fig.savefig(figures / "g1_revisit_failure.png", dpi=180); plt.close(fig)
    result = {"experiment": "G1 frozen DINOv3 physical correspondence audit", "model": MODEL, "audit_pairs": len(pairs()), "valid_correspondences": len(all_rows), "geometry": {"depth_residual_m_mean": float(np.mean([r["depth_residual_m"] for r in all_rows])), "world_residual_m_mean": float(np.mean([r["world_residual_m"] for r in all_rows])), "lidar_residual_m_mean": float(np.mean([r["lidar_residual_m"] for r in all_rows if r["lidar_residual_m"] >= 0]))}, "pair_valid_correspondence_counts": pair_counts, "by_viewpoint": by_regime, "by_split": by_split, "elapsed_seconds": time.perf_counter() - started, "go_recommendation": "GO" if degradation else "NO-GO", "go_rationale": "preregistered large-vs-small degradation met" if degradation else "preregistered large-vs-small degradation not met", "assets": ["paper_assets/figures/g1_crossview_correspondence.png", "paper_assets/figures/g1_dinov3_viewpoint_degradation.png", "paper_assets/figures/g1_revisit_failure.png", "paper_assets/tables/g1_dinov3_baseline.csv"]}
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
