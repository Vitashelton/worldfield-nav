#!/usr/bin/env python3
"""Train a lightweight branch-value model from real Habitat branch outcomes.

The model never produces motion.  It ranks eight fixed planner proposals using
the frozen ImageNav visual condition, proposal-local geometry/pose and, at
deployment, an externally maintained failure-history mask.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


class BranchValueModel(nn.Module):
    def __init__(self, visual_dim: int = 384, branch_dim: int = 6) -> None:
        super().__init__()
        # [current, goal, current*goal, current-goal] + local proposal state.
        self.visual = nn.Sequential(nn.Linear(visual_dim * 4, 192), nn.GELU(), nn.LayerNorm(192))
        self.branch = nn.Sequential(nn.Linear(branch_dim, 64), nn.GELU())
        self.fuse = nn.Sequential(nn.Linear(256, 192), nn.GELU(), nn.Dropout(0.10), nn.Linear(192, 96), nn.GELU())
        self.reached = nn.Linear(96, 1)
        self.progress = nn.Linear(96, 1)
        self.failure = nn.Linear(96, 1)

    def forward(self, visual: torch.Tensor, branch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        v = torch.cat((visual[:, :384], visual[:, 384:768], visual[:, :384] * visual[:, 384:768], visual[:, :384] - visual[:, 384:768]), 1)
        h = self.fuse(torch.cat((self.visual(v), self.branch(branch)), 1))
        return self.reached(h).squeeze(1), self.progress(h).squeeze(1), self.failure(h).squeeze(1)


def load(root: Path):
    manifest = json.loads((root / "dataset_manifest.json").read_text())
    pairs = np.load(root / "dino_pair_features.npz")
    ids = [str(x) for x in pairs["episode_id"]]
    if ids != [r["episode_id"] for r in manifest]:
        raise RuntimeError("DINO pair-cache order does not match dataset manifest")
    outcome = {r["key"]: r for r in (json.loads(x) for x in (root / "rollout_outcomes.jsonl").read_text().splitlines())}
    visual, branch, reached, progress, failure, groups, meta = [], [], [], [], [], [], []
    for ei, record in enumerate(manifest):
        start = len(visual)
        for candidate in record["trajectories"]:
            o = outcome[f"{record['episode_id']}:{candidate['trajectory_id']}"]
            g = candidate["geometry"]
            yaw = float(candidate["relative_yaw_rad"])
            # All geometry inputs are observation/proposal derived, never
            # privileged NavMesh clearance/reachability labels.
            branch.append([g["proposal_length_m"], g["image_visible_ratio"], g["observed_depth_min_m"], g["observed_depth_mean_m"], np.sin(yaw), np.cos(yaw)])
            visual.append(np.concatenate((pairs["current"][ei], pairs["goal"][ei])).astype(np.float32))
            reached.append(float(o["reached"]))
            progress.append(float(o["progress_m"]) / max(float(record["initial_geodesic_m"]), 0.1))
            failure.append(float(bool(o["collision"] or o["stuck"])))
            meta.append({"episode_id": record["episode_id"], "scene_id": record["scene_id"], "split": record["split"], "trajectory_id": candidate["trajectory_id"], "initial_geodesic_m": record["initial_geodesic_m"], "outcome": o})
        groups.append((record, start, len(visual)))
    return (np.asarray(visual, np.float32), np.asarray(branch, np.float32), np.asarray(reached, np.float32),
            np.asarray(progress, np.float32), np.asarray(failure, np.float32), groups, meta)


def choose_metrics(groups, meta, values, method):
    per_split = defaultdict(list)
    for record, a, b in groups:
        choice = a + int(np.argmax(values[a:b]))
        chosen = meta[choice]
        all_rows = meta[a:b]
        best = max(all_rows, key=lambda x: (x["outcome"]["reached"], x["outcome"]["progress_m"]))
        out = chosen["outcome"]
        per_split[record["split"]].append({
            "success": float(out["reached"]), "collision": float(out["collision"]), "stuck": float(out["stuck"]),
            "progress": float(out["progress_m"]), "path": float(out["executed_path_length_m"]),
            "regret": max(0.0, float(best["outcome"]["progress_m"]) - float(out["progress_m"])),
            "rank_exact": float(chosen["trajectory_id"] == best["trajectory_id"]),
        })
    rows = []
    for split, values_ in per_split.items():
        rows.append({"method": method, "split": split, "episodes": len(values_),
                     "ranking_accuracy": np.mean([x["rank_exact"] for x in values_]), "regret_m": np.mean([x["regret"] for x in values_]),
                     "branch_reached_rate": np.mean([x["success"] for x in values_]), "collision_rate": np.mean([x["collision"] for x in values_]),
                     "stuck_rate": np.mean([x["stuck"] for x in values_]), "mean_progress_m": np.mean([x["progress"] for x in values_]),
                     "mean_executed_path_m": np.mean([x["path"] for x in values_])})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--root", type=Path, default=ROOT / "outputs/formal/TrajectoryGrounding/P1")
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--seed", type=int, default=20260907); args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    vis, branch, y_reach, y_progress, y_fail, groups, meta = load(args.root)
    train = np.asarray([x["split"] == "train" for x in meta])
    mean, std = branch[train].mean(0), branch[train].std(0).clip(0.05)
    branch = (branch - mean) / std
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BranchValueModel().to(device); opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    v = torch.tensor(vis, device=device); b = torch.tensor(branch, device=device); r = torch.tensor(y_reach, device=device); p = torch.tensor(y_progress, device=device); f = torch.tensor(y_fail, device=device)
    train_ids = torch.as_tensor(np.flatnonzero(train), device=device); started = time.perf_counter()
    for _ in range(args.epochs):
        for ids in train_ids[torch.randperm(len(train_ids), device=device)].split(1024):
            lr, lp, lf = model(v[ids], b[ids])
            loss = nn.functional.binary_cross_entropy_with_logits(lr, r[ids]) + nn.functional.smooth_l1_loss(lp, p[ids]) + nn.functional.binary_cross_entropy_with_logits(lf, f[ids])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    with torch.inference_mode():
        lr, lp, lf = model(v, b)
        value = (torch.sigmoid(lr) + 0.35 * lp - 0.75 * torch.sigmoid(lf)).cpu().numpy()
    # Both non-learned comparators see exactly the same K raw planner branches.
    # PlannerGeometry only uses observation/proposal-derived geometric features;
    # Oracle is evaluation-only and ranks by the collected physical outcome.
    planner = 0.65 * branch[:, 1] + 0.25 * branch[:, 3] - 0.15 * branch[:, 0]
    oracle = y_reach + 0.35 * y_progress - 0.75 * y_fail
    rows = (choose_metrics(groups, meta, planner, "Planner Geometry") +
            choose_metrics(groups, meta, value, "Learned BranchValue") +
            choose_metrics(groups, meta, oracle, "Outcome Oracle"))
    out = {"parameters": sum(x.numel() for x in model.parameters()), "seconds": time.perf_counter() - started, "metrics": rows,
           "inputs": ["frozen_dinov3_current_goal", "proposal_local_geometry", "candidate_relative_pose", "failure_history_at_rollout"],
           "labels": ["reached", "collision_or_stuck", "normalized_geodesic_progress"]}
    (args.root / "branch_value_metrics.json").write_text(json.dumps(out, indent=2) + "\n")
    (ROOT / "paper_assets/tables").mkdir(parents=True, exist_ok=True)
    with (ROOT / "paper_assets/tables/failure_aware_p1_branch_value.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    torch.save({"state_dict": model.state_dict(), "branch_mean": mean, "branch_std": std, "model": "BranchValueModel", "frozen_visual_encoder": "DINOv3-S/16"}, args.root / "branch_value_model.pt")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
