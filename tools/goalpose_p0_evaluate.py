#!/usr/bin/env python3
"""Evaluate a small, explicitly curated GoalPose P0 in existing indoor scenes.

This is a problem-signal evaluation. Target pixels are provisional visual
anchors recorded in the manifest. Habitat NavMesh and rendered depth are used
only for evaluation/checking and every emitted record marks them as privileged.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from trajectory_grounding_p1_generate import make_sim, render, shortest, yaw_rotation

OUT = ROOT / "outputs/formal/GoalPose/P0"
FIG = ROOT / "paper_assets/figures"
TAB = ROOT / "paper_assets/tables"
SCENES = [
    ("interior_0047_839892", "approach_doorway"),
    ("interior_0405_840145", "room_entrance"),
    ("interior_0093_839966", "inspect_target"),
    ("interior_0121_840013", "observe_doorway"),
]
ANCHOR_INDICES = [0, 60, 120]
RINGS = [0.75, 1.15, 1.60, 2.10, 2.60]
ANGLES = np.linspace(-math.pi, math.pi, 12, endpoint=False)


def load_records(scene: str) -> dict[str, dict]:
    path = OUT / "anchor_views" / f"anchor_view_candidates_{scene}.jsonl"
    return {r["anchor_view_id"]: r for r in map(json.loads, path.read_text().splitlines())}


def backproject(anchor: dict, uv: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int]]:
    depth = np.load(OUT.parent / anchor["depth"]).astype(np.float32)
    u, v = uv
    d = float(depth[v, u])
    if not np.isfinite(d) or not 0.02 < d < 9.99:
        yy, xx = np.indices(depth.shape)
        valid = np.isfinite(depth) & (depth > .02) & (depth < 9.99)
        if not valid.any():
            raise RuntimeError(f"no valid depth at curated anchor {anchor['anchor_view_id']}")
        distance2 = (xx - u) ** 2 + (yy - v) ** 2
        distance2[~valid] = np.iinfo(np.int32).max
        v, u = np.unravel_index(int(distance2.argmin()), depth.shape)
        d = float(depth[v, u])
    intr = anchor["intrinsics"]
    cam = np.array([(u - intr["cx"]) * d / intr["fx"], -(v - intr["cy"]) * d / intr["fy"], -d], np.float32)
    rot = np.asarray(anchor["camera_c2w"], np.float32)
    return cam @ rot.T + np.asarray(anchor["camera_xyz"], np.float32), (int(u), int(v))


def facing_yaw(position: np.ndarray, target: np.ndarray) -> float:
    direction = target - position
    return math.atan2(float(-direction[0]), float(-direction[2]))


def project_point(point: np.ndarray, camera_xyz: np.ndarray, c2w: np.ndarray) -> tuple[int, int, float] | None:
    cam = (point - camera_xyz) @ c2w
    if cam[2] >= -1e-4:
        return None
    u = int(round(128.0 * cam[0] / (-cam[2]) + 127.5))
    v = int(round(128.0 * cam[1] / cam[2] + 127.5))
    if not (0 <= u < 256 and 0 <= v < 256):
        return None
    return u, v, float(-cam[2])


def visible_from(sim, agent, q: np.ndarray, target: np.ndarray) -> tuple[bool, str]:
    _, depth, camera_xyz, c2w = render(sim, agent, q, yaw_rotation(facing_yaw(q, target)))
    hit = project_point(target, camera_xyz, c2w)
    if hit is None:
        return False, "outside_fov"
    u, v, expected = hit
    observed = float(depth[v, u])
    return bool(np.isfinite(observed) and observed > .02 and abs(observed - expected) < .20), "visible" if np.isfinite(observed) and observed > .02 and abs(observed - expected) < .20 else "occluded"


def candidate_rows(sim, agent, start: np.ndarray, target: np.ndarray, template: str) -> list[dict]:
    rows = []
    for radius in RINGS:
        for angle in ANGLES:
            raw = target.copy()
            raw[1] = start[1]
            raw[0] += radius * math.sin(float(angle))
            raw[2] += radius * math.cos(float(angle))
            snapped = np.asarray(sim.pathfinder.snap_point(raw), np.float32)
            snap_ok = bool(np.isfinite(snapped).all() and np.linalg.norm(snapped - raw) <= .60)
            clearance = float(sim.pathfinder.distance_to_closest_obstacle(snapped, 2.0)) if snap_ok else 0.0
            reachable, path = shortest(sim, start, snapped) if snap_ok else (False, float("inf"))
            visible, visibility_reason = visible_from(sim, agent, snapped, target) if snap_ok else (False, "off_navmesh")
            distance = float(np.linalg.norm((snapped - target)[[0, 2]])) if snap_ok else float("inf")
            facing = True
            task_ok = bool(snap_ok and reachable and clearance >= .18 and visible and .60 <= distance <= 1.90 and facing)
            score = (
                1.00 * min(clearance / .80, 1.0)
                + 1.20 * float(visible)
                + 1.25 * float(.60 <= distance <= 1.90)
                - .20 * abs(distance - 1.20)
            ) if snap_ok else -9.0
            rows.append({
                "raw_xyz": raw.tolist(), "xyz": snapped.tolist(), "radius_m": radius, "angle_rad": float(angle),
                "snap_ok": snap_ok, "reachable_evaluation_only": bool(reachable), "geodesic_from_start_m": float(path),
                "clearance_m_evaluation_only": clearance, "target_visible_evaluation_only": visible,
                "visibility_reason": visibility_reason, "target_distance_m": distance,
                "task_relation_satisfied_evaluation_only": task_ok, "goalpose_field_score": float(score),
            })
    return rows


def choose(rows: list[dict], method: str, target: np.ndarray) -> dict:
    valid = [r for r in rows if r["snap_ok"]]
    if method == "M1_nearest_free":
        return min(valid, key=lambda r: r["target_distance_m"])
    if method == "M2_clearance":
        return max(valid, key=lambda r: r["clearance_m_evaluation_only"])
    if method == "M3_goalpose_field":
        return max(valid, key=lambda r: r["goalpose_field_score"])
    if method == "M4_oracle":
        executable = [r for r in valid if r["reachable_evaluation_only"] and r["clearance_m_evaluation_only"] >= .18]
        return max(executable or valid, key=lambda r: (
            4 * r["task_relation_satisfied_evaluation_only"] + 2 * r["target_visible_evaluation_only"]
            + r["reachable_evaluation_only"] + min(r["clearance_m_evaluation_only"], 1.0)
        ))
    raise KeyError(method)


def center_record(sim, agent, start: np.ndarray, target: np.ndarray) -> dict:
    raw = target.copy()
    raw[1] = start[1]
    snap = np.asarray(sim.pathfinder.snap_point(raw), np.float32)
    snap_ok = bool(np.isfinite(snap).all() and np.linalg.norm(snap - raw) <= .25)
    clearance = float(sim.pathfinder.distance_to_closest_obstacle(snap, 2.0)) if snap_ok else 0.0
    reachable, path = shortest(sim, start, snap) if snap_ok else (False, float("inf"))
    visible, reason = visible_from(sim, agent, snap, target) if snap_ok else (False, "target_surface_not_robot_pose")
    return {
        "raw_xyz": raw.tolist(), "xyz": snap.tolist(), "snap_ok": snap_ok,
        "reachable_evaluation_only": bool(reachable), "geodesic_from_start_m": float(path),
        "clearance_m_evaluation_only": clearance, "target_visible_evaluation_only": visible,
        "visibility_reason": reason, "target_distance_m": float(np.linalg.norm((snap-target)[[0,2]])) if snap_ok else float("inf"),
        "task_relation_satisfied_evaluation_only": False, "goalpose_field_score": float("nan"),
    }


def draw_case(case: dict, image_out: Path) -> None:
    target = np.asarray(case["target_world_xyz"])
    rows = case["candidates"]
    selections = case["selections"]
    extent = 2.35
    canvas = Image.new("RGB", (960, 500), "white")
    goal = Image.open(ROOT / "outputs/formal/GoalPose" / case["anchor_rgb"]).convert("RGB").resize((250, 250))
    current = Image.open(ROOT / "outputs/formal/GoalPose" / case["start_rgb"]).convert("RGB").resize((250, 250))
    canvas.paste(goal, (15, 25)); canvas.paste(current, (15, 285))
    draw = ImageDraw.Draw(canvas)
    draw.text((15, 5), f"{case['case_id']} goal anchor (curated)", fill="black")
    draw.text((15, 265), "start observation", fill="black")
    ox, oy, scale = 600, 260, 95
    draw.rectangle((315, 20, 945, 485), outline="black", width=2)
    draw.text((330, 30), "candidate goal-pose lattice (Habitat evaluation overlay)", fill="black")
    def xy(point):
        return (ox + (point[0]-target[0]) * scale, oy - (point[2]-target[2]) * scale)
    tx, ty = xy(target); draw.ellipse((tx-8, ty-8, tx+8, ty+8), fill=(20, 20, 20)); draw.text((tx+10, ty+4), "semantic surface center", fill="black")
    for r in rows:
        x, y = xy(np.asarray(r["xyz"]))
        color = (190, 190, 190) if not r["snap_ok"] else ((48, 150, 85) if r["task_relation_satisfied_evaluation_only"] else (230, 140, 50))
        draw.ellipse((x-4, y-4, x+4, y+4), fill=color)
    colors = {"M0_center": (190, 0, 0), "M1_nearest_free": (0, 95, 190), "M2_clearance": (165, 85, 0), "M3_goalpose_field": (0, 160, 65)}
    for method, r in selections.items():
        pose = np.asarray(r["xyz"], np.float32)
        x, y = xy(pose) if np.isfinite(pose).all() else (tx, ty)
        draw.ellipse((x-10, y-10, x+10, y+10), outline=colors.get(method, (0, 0, 0)), width=4)
    legend = [("M0 center", colors["M0_center"]), ("M1 nearest-free", colors["M1_nearest_free"]),
              ("M2 clearance", colors["M2_clearance"]), ("M3 GoalPose", colors["M3_goalpose_field"]),
              ("M4 oracle", (0, 0, 0))]
    for index, (label, color) in enumerate(legend):
        lx, ly = 334 + (index % 3) * 190, 55 + (index // 3) * 18
        draw.ellipse((lx, ly, lx + 10, ly + 10), outline=color, width=3)
        draw.text((lx + 14, ly - 2), label, fill=color)
    draw.text((330, 455), "green: visible + reachable + clearance + template standoff; orange: fails at least one check", fill="black")
    canvas.save(image_out)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)
    all_cases, flat = [], []
    for scene, template in SCENES:
        recs = load_records(scene)
        sim = make_sim(scene); agent = sim.initialize_agent(0)
        try:
            for source_index in ANCHOR_INDICES:
                goal_id = f"{scene}_{source_index:04d}_view_07"
                start_index = ANCHOR_INDICES[(ANCHOR_INDICES.index(source_index) + 1) % len(ANCHOR_INDICES)]
                start_id = f"{scene}_{start_index:04d}_view_07"
                anchor, start_anchor = recs[goal_id], recs[start_id]
                # The chosen center pixel is a provisional visual anchor, disclosed in every record.
                requested_pixel = (128, 128)
                target, pixel = backproject(anchor, requested_pixel)
                start = np.asarray(start_anchor["camera_xyz"], np.float32); start[1] -= 1.5
                rows = candidate_rows(sim, agent, start, target, template)
                selections = {"M0_center": center_record(sim, agent, start, target)}
                for method in ["M1_nearest_free", "M2_clearance", "M3_goalpose_field", "M4_oracle"]:
                    selections[method] = choose(rows, method, target)
                case_id = f"{scene}_p0_{source_index:04d}"
                case = {
                    "case_id": case_id, "scene_id": scene, "approach_template": template,
                    "curation_status": "provisional_visual_anchor_center_pixel",
                    "curation_disclosure": "The RGB center was the visually reviewed provisional target. If it had invalid depth, the nearest valid observed pixel was used and recorded; this is not an automatic semantic label.",
                    "privileged_evaluation_disclosure": "NavMesh, clearance and rendered target visibility are evaluation-only in P0.",
                    "anchor_view_id": goal_id, "anchor_rgb": anchor["rgb"], "start_view_id": start_id, "start_rgb": start_anchor["rgb"],
                    "requested_target_pixel_uv": list(requested_pixel), "target_pixel_uv": list(pixel), "target_world_xyz": target.tolist(), "start_world_xyz": start.tolist(),
                    "candidates": rows, "selections": selections,
                }
                all_cases.append(case)
                for method, value in selections.items():
                    flat.append({
                        "case_id": case_id, "scene_id": scene, "template": template, "method": method,
                        "executable_goal": bool(value["snap_ok"] and value["reachable_evaluation_only"] and value["clearance_m_evaluation_only"] >= .18),
                        "reachable_goal": value["reachable_evaluation_only"], "clearance_m": value["clearance_m_evaluation_only"],
                        "target_visible": value["target_visible_evaluation_only"], "task_relation_satisfied": value["task_relation_satisfied_evaluation_only"],
                        "geodesic_from_start_m": value["geodesic_from_start_m"], "target_distance_m": value["target_distance_m"],
                    })
        finally:
            sim.close()
    manifest = {"protocol": "GoalPose P0; all labels marked evaluation-only", "cases": all_cases}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    fields = list(flat[0])
    with (TAB / "goalpose_p0_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(flat)
    with (OUT / "candidate_pose_records.json").open("w") as f:
        json.dump(all_cases, f, indent=2)
    showcase = [all_cases[index] for index in [0, 3, 6]]
    for idx, case in enumerate(showcase):
        draw_case(case, OUT / f"case_{idx+1}.png")
    panels = [Image.open(OUT / f"case_{i+1}.png") for i in range(3)]
    combined = Image.new("RGB", (960, 1500), "white")
    for i, panel in enumerate(panels): combined.paste(panel, (0, i*500))
    combined.save(FIG / "goalpose_p0_problem_cases.png")
    combined.save(FIG / "goalpose_p0_field_examples.png")
    methods = sorted({r["method"] for r in flat})
    summary = {}
    for m in methods:
        rows = [r for r in flat if r["method"] == m]
        summary[m] = {k: float(np.mean([float(r[k]) for r in rows])) for k in ["executable_goal", "reachable_goal", "clearance_m", "target_visible", "task_relation_satisfied"]}
    print(json.dumps({"cases": len(all_cases), "summary": summary, "out": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
