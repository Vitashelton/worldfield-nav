#!/usr/bin/env python3
"""Rolling ImageNav evaluation for fixed planner branches and Execution Bridge.

Hidden goal poses are queried solely to score terminal DTG/SR and to construct
the evaluation-only Oracle.  Planner Geometry and Learned Execution Bridge do
not read goal coordinates or NavMesh during candidate selection.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import timm
from timm.data import resolve_model_data_config
import habitat_sim

from trajectory_grounding_p1_generate import ROOT, make_sim, propose, project, geometric_features, shortest
from failure_aware_p1_branch_value import BranchValueModel

OUT = ROOT / "outputs/formal/TrajectoryGrounding/P1"


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def execute_path(agent, path: np.ndarray, max_actions: int = 32) -> dict:
    """Directly execute the raw candidate, with no follower/NavMesh routing."""
    collision, length, steps = False, 0.0, 0
    actions = []
    for waypoint in path:
        while steps < max_actions:
            state = agent.get_state(); pos = np.asarray(state.position, np.float32)
            delta = waypoint - pos; delta[1] = 0.0
            if np.linalg.norm(delta) < 0.16:
                break
            forward = habitat_sim.utils.common.quat_rotate_vector(state.rotation, np.array([0., 0., -1.], np.float32))
            forward[1] = 0.; forward /= np.linalg.norm(forward)
            err = wrap(math.atan2(float(delta[0]), float(delta[2])) - math.atan2(float(forward[0]), float(forward[2])))
            action = "turn_left" if err > .12 else "turn_right" if err < -.12 else "move_forward"
            before = pos; hit = bool(agent.act(action)); after = np.asarray(agent.get_state().position, np.float32)
            length += float(np.linalg.norm(after - before)); steps += 1; actions.append(action)
            collision |= hit
            if collision:
                break
        if collision or steps >= max_actions:
            break
    return {"collision": collision, "stuck": bool(collision or length < .10), "path_length": length, "actions": actions}


class VisualCondition:
    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError("Rolling Execution Bridge evaluation requires CUDA for frozen DINOv3")
        self.model = timm.create_model("hf_hub:timm/vit_small_patch16_dinov3.lvd1689m", pretrained=True).cuda().eval()
        cfg = resolve_model_data_config(self.model)
        self.mean = torch.tensor(cfg["mean"], device="cuda").view(1, 3, 1, 1)
        self.std = torch.tensor(cfg["std"], device="cuda").view(1, 3, 1, 1)

    def descriptor(self, images: list[np.ndarray]) -> np.ndarray:
        x = torch.stack([torch.from_numpy(np.asarray(Image.fromarray(i).resize((256, 256), Image.Resampling.BICUBIC)).copy()).permute(2, 0, 1).float().div_(255) for i in images]).cuda()
        with torch.inference_mode():
            y = self.model.forward_features((x - self.mean) / self.std)
            if not torch.is_tensor(y): y = y.get("x_norm", y.get("x"))
            y = torch.nn.functional.normalize(y[:, int(self.model.num_prefix_tokens):].mean(1), dim=-1)
        return y.cpu().numpy().astype(np.float32)


def agent_observation(sim, agent):
    obs = sim.get_sensor_observations()
    rgb = np.asarray(obs["rgb"])[..., :3].copy(); depth = np.asarray(obs["depth"], np.float32).copy()
    sensor = agent.get_state().sensor_states["rgb"]
    c2w = np.asarray(habitat_sim.utils.common.quat_to_magnum(sensor.rotation).to_matrix(), np.float32)
    return rgb, depth, np.asarray(sensor.position, np.float32), c2w


def candidate_features(depth, camera_t, c2w, paths):
    rows = []
    for i, path in enumerate(paths):
        uv, valid = project(path, camera_t, c2w)
        g = geometric_features(depth, uv, valid, path)
        angle = -0.875 * math.pi + i * (1.75 * math.pi / 7.0)
        rows.append([g["proposal_length_m"], g["image_visible_ratio"], g["observed_depth_min_m"], g["observed_depth_mean_m"], math.sin(angle), math.cos(angle)])
    return np.asarray(rows, np.float32)


def load_bridge():
    ckpt = torch.load(OUT / "branch_value_model.pt", map_location="cuda", weights_only=False)
    model = BranchValueModel().cuda().eval(); model.load_state_dict(ckpt["state_dict"])
    return model, np.asarray(ckpt["branch_mean"], np.float32), np.asarray(ckpt["branch_std"], np.float32)


def run_episode(sim, agent, record, method, visual, bridge, mean, std, max_decisions):
    # The episode manifest supplies a fixed start / goal image. Goal position is
    # never fed to Geometry or Bridge; it is evaluator-only below.
    goal = np.asarray(record["goal_xyz_hidden_for_evaluation"], np.float32)
    agent_state = agent.get_state(); agent_state.position = np.asarray(record["start_xyz"], np.float32)
    rot = np.asarray(record["camera_c2w"], np.float32) @ np.array([0., 0., -1.], np.float32)
    yaw = math.atan2(float(rot[0]), float(rot[2]))
    agent_state.rotation = habitat_sim.utils.common.quat_from_angle_axis(yaw, np.array([0., 1., 0.]))
    agent.set_state(agent_state, reset_sensors=True)
    goal_rgb = np.asarray(Image.open(OUT / record["goal_image"]).convert("RGB"))
    goal_desc = visual.descriptor([goal_rgb])[0]
    _, initial = shortest(sim, np.asarray(agent.get_state().position, np.float32), goal)
    history = np.zeros(8, np.float32); total_path = 0.; failures = 0; traces = []
    for decision in range(max_decisions):
        pos = np.asarray(agent.get_state().position, np.float32); _, dtg = shortest(sim, pos, goal)
        if dtg <= 1.0:
            break
        rgb, depth, camera_t, c2w = agent_observation(sim, agent); paths = propose(camera_t, c2w); raw = candidate_features(depth, camera_t, c2w, paths)
        if method == "Planner Geometry":
            score = .65 * raw[:, 1] + .25 * raw[:, 3] - .15 * raw[:, 0] - .7 * history
        elif method == "Learned Execution Bridge":
            b = (raw - mean) / std
            pair = np.repeat(np.concatenate((visual.descriptor([rgb])[0], goal_desc))[None], 8, axis=0)
            with torch.inference_mode():
                lr, lp, lf = bridge(torch.tensor(pair, device="cuda"), torch.tensor(b, device="cuda"))
                score = (torch.sigmoid(lr) + .35 * lp - .75 * torch.sigmoid(lf)).cpu().numpy() - .7 * history
        elif method == "Outcome Oracle":
            saved = agent.get_state(); trial = []
            for path in paths:
                agent.set_state(saved, reset_sensors=True); result = execute_path(agent, path)
                _, post = shortest(sim, np.asarray(agent.get_state().position, np.float32), goal)
                trial.append((float(not result["stuck"]) + .35 * float(dtg - post) - .75 * float(result["collision"]), result))
            agent.set_state(saved, reset_sensors=True); score = np.asarray([x[0] for x in trial]) - .7 * history
        else:
            raise ValueError(method)
        chosen = int(np.argmax(score)); result = execute_path(agent, paths[chosen]); total_path += result["path_length"]
        _, after_dtg = shortest(sim, np.asarray(agent.get_state().position, np.float32), goal)
        failed = bool(result["collision"] or result["stuck"] or after_dtg >= dtg - .05)
        if failed:
            history[chosen] += 1.; failures += 1
        traces.append({"decision": decision, "choice": chosen, "dtg_before": dtg, "dtg_after": after_dtg, "failed": failed, **result})
    _, final_dtg = shortest(sim, np.asarray(agent.get_state().position, np.float32), goal)
    success = bool(final_dtg <= 1.0)
    spl = float(success * initial / max(initial, total_path, 1e-6))
    return {"episode_id": record["episode_id"], "scene_id": record["scene_id"], "split": record["split"], "method": method,
            "success": success, "spl": spl, "initial_dtg_m": initial, "final_dtg_m": final_dtg, "path_length_m": total_path,
            "decisions": len(traces), "failed_decisions": failures, "trace": traces}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--per-split", type=int, default=100); ap.add_argument("--max-decisions", type=int, default=10)
    ap.add_argument("--methods", nargs="+", default=["Planner Geometry", "Learned Execution Bridge", "Outcome Oracle"]); args = ap.parse_args()
    manifest = json.loads((OUT / "dataset_manifest.json").read_text())
    selected = {s: [r for r in manifest if r["split"] == s][:args.per_split] for s in ("val", "unseen")}
    visual = VisualCondition(); bridge, mean, std = load_bridge(); results = []
    started = time.time()
    for method in args.methods:
        for split, records in selected.items():
            by_scene = defaultdict(list)
            for r in records: by_scene[r["scene_id"]].append(r)
            for scene, scene_records in by_scene.items():
                sim = make_sim(scene); agent = sim.initialize_agent(0)
                try:
                    for record in scene_records:
                        results.append(run_episode(sim, agent, record, method, visual, bridge, mean, std, args.max_decisions))
                finally:
                    sim.close()
    (OUT / "rolling_episode_results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))
    summary = []
    for method in args.methods:
        for split in ("val", "unseen"):
            rows = [r for r in results if r["method"] == method and r["split"] == split]
            summary.append({"method": method, "split": split, "episodes": len(rows), "SR": float(np.mean([r["success"] for r in rows])),
                            "SPL": float(np.mean([r["spl"] for r in rows])), "Final_DTG_m": float(np.mean([r["final_dtg_m"] for r in rows])),
                            "Path_Length_m": float(np.mean([r["path_length_m"] for r in rows])), "Failed_Decisions": float(np.mean([r["failed_decisions"] for r in rows]))})
    (OUT / "rolling_summary.json").write_text(json.dumps({"seconds": time.time() - started, "metrics": summary}, indent=2) + "\n")
    print(json.dumps({"seconds": time.time() - started, "metrics": summary}, indent=2))


if __name__ == "__main__":
    main()
