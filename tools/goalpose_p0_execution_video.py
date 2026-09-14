#!/usr/bin/env python3
"""Render actual fixed-executor Habitat-GS rollouts for two P0 method choices."""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "scripts"))
from goalpose_p0_evaluate import facing_yaw
from trajectory_grounding_p1_generate import make_sim, render, shortest
import habitat_sim

OUT = ROOT / "outputs/formal/GoalPose/P0"
VIDEO = ROOT / "paper_assets/videos/goalpose_p0_execution_comparison.mp4"
CASE_IDS = ["interior_0405_840145_p0_0060", "interior_0093_839966_p0_0060"]


def path_points(sim, start: np.ndarray, end: np.ndarray, count: int = 24) -> np.ndarray:
    query = habitat_sim.ShortestPath(); query.requested_start = start; query.requested_end = end
    if not sim.pathfinder.find_path(query):
        raise RuntimeError("no fixed-executor path")
    points = np.asarray(query.points, np.float32)
    segment = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(segment)]
    targets = np.linspace(0.0, cumulative[-1], count)
    result = []
    for value in targets:
        index = min(np.searchsorted(cumulative, value, side="right") - 1, len(segment) - 1)
        ratio = (value - cumulative[index]) / max(segment[index], 1e-6)
        result.append(points[index] * (1 - ratio) + points[index + 1] * ratio)
    return np.asarray(result, np.float32)


def rollout(sim, agent, start: np.ndarray, end: np.ndarray, target: np.ndarray) -> list[np.ndarray]:
    path = path_points(sim, start, end)
    frames = []
    for index, point in enumerate(path):
        direction = target if index == len(path) - 1 else path[index + 1]
        rgb, _, _, _ = render(sim, agent, point, habitat_sim.utils.common.quat_from_angle_axis(facing_yaw(point, direction), np.array([0., 1., 0.])))
        frames.append(rgb)
    return frames


def canvas(left: np.ndarray, right: np.ndarray, case: dict, step: int, total: int) -> np.ndarray:
    image = Image.new("RGB", (1100, 660), "#101820")
    draw = ImageDraw.Draw(image)
    left_img = Image.fromarray(left).resize((500, 500)); right_img = Image.fromarray(right).resize((500, 500))
    image.paste(left_img, (35, 125)); image.paste(right_img, (565, 125))
    m2, m3 = case["selections"]["M2_clearance"], case["selections"]["M3_goalpose_field"]
    draw.text((35, 22), f"Habitat-GS fixed-executor rollout | {case['scene_id']} | {case['approach_template']}", fill="white")
    draw.text((35, 48), "Same start, same target anchor, same shortest-path executor. Only the terminal pose differs.", fill=(210, 220, 230))
    draw.text((35, 92), f"M2 clearance-only: {m2['target_distance_m']:.2f} m standoff | visible={m2['target_visible_evaluation_only']} | task={m2['task_relation_satisfied_evaluation_only']}", fill=(255, 165, 70))
    draw.text((565, 92), f"M3 GoalPose: {m3['target_distance_m']:.2f} m standoff | visible={m3['target_visible_evaluation_only']} | task={m3['task_relation_satisfied_evaluation_only']}", fill=(80, 220, 125))
    draw.text((35, 635), f"progress {step + 1}/{total}    orange = clearance-only terminal goal    green = task-conditioned terminal goal", fill=(220, 220, 220))
    return np.asarray(image)


def main() -> None:
    payload = json.loads((OUT / "manifest.json").read_text())
    cases = {case["case_id"]: case for case in payload["cases"]}
    VIDEO.parent.mkdir(parents=True, exist_ok=True)
    raw_path = OUT / "goalpose_p0_execution_comparison.rgb"
    with raw_path.open("wb") as writer:
        for case_id in CASE_IDS:
            case = cases[case_id]; sim = make_sim(case["scene_id"]); agent = sim.initialize_agent(0)
            try:
                start = np.asarray(case["start_world_xyz"], np.float32)
                target = np.asarray(case["target_world_xyz"], np.float32)
                m2 = np.asarray(case["selections"]["M2_clearance"]["xyz"], np.float32)
                m3 = np.asarray(case["selections"]["M3_goalpose_field"]["xyz"], np.float32)
                frames2, frames3 = rollout(sim, agent, start, m2, target), rollout(sim, agent, start, m3, target)
                total = max(len(frames2), len(frames3))
                for index in range(total):
                    writer.write(canvas(frames2[min(index, len(frames2) - 1)], frames3[min(index, len(frames3) - 1)], case, index, total).tobytes())
                for _ in range(8):
                    writer.write(canvas(frames2[-1], frames3[-1], case, total - 1, total).tobytes())
            finally:
                sim.close()
    with raw_path.open("rb") as raw:
        subprocess.run([str(ROOT / "tools/raw_rgb_to_mp4"), str(VIDEO), "1100", "660", "6"], stdin=raw, check=True)
    raw_path.unlink()
    print(VIDEO)


if __name__ == "__main__":
    main()
