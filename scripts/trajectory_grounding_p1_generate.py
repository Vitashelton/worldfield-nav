#!/usr/bin/env python3
"""Generate goal-independent local trajectory proposals and teacher outcomes.

The sampler is deliberately independent of hidden final-goal geometry: it uses
only the current camera pose and a fixed robot-centric polar template. Habitat
NavMesh is consulted *after* proposal creation for labels, never proposals.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import habitat_sim
import habitat_sim.agent
import magnum as mn
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/formal/TrajectoryGrounding/P1"
SPLIT = {
    "train": ["interior_0405_840145", "interior_0047_839892", "interior_0108_839984", "interior_0184_840116"],
    "val": ["interior_0093_839966"],
    "unseen": ["interior_0121_840013", "interior_0135_840032"],
}
K = 8
RELATIVE_YAWS = np.linspace(-0.875 * math.pi, 0.875 * math.pi, K, dtype=np.float32)


def sensor(uuid: str, typ: habitat_sim.SensorType) -> habitat_sim.CameraSensorSpec:
    spec = habitat_sim.CameraSensorSpec(); spec.uuid = uuid; spec.sensor_type = typ
    spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE; spec.resolution = [256, 256]
    spec.position = [0.0, 1.5, 0.0]; spec.hfov = 90.0
    if typ == habitat_sim.SensorType.DEPTH: spec.channels = 1
    return spec


def make_sim(scene: str) -> habitat_sim.Simulator:
    root = ROOT / "data/scene_datasets/gs_scenes/train" / scene
    ply, nav = root / f"{scene}.gs.ply", root / f"{scene}.navmesh"
    if not ply.is_file() or not nav.is_file(): raise FileNotFoundError(root)
    cfg = habitat_sim.SimulatorConfiguration(); cfg.scene_id = "NONE"; cfg.gpu_device_id = 0; cfg.create_renderer = True
    ac = habitat_sim.agent.AgentConfiguration(); ac.height = 1.5; ac.radius = 0.1
    ac.sensor_specifications = [sensor("rgb", habitat_sim.SensorType.COLOR), sensor("depth", habitat_sim.SensorType.DEPTH)]
    sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [ac]))
    helper = habitat_sim.RenderInstanceHelper(sim, use_xyzw_orientations=False)
    helper.add_instance(asset_filepath=str(ply), semantic_id=0, scale=mn.Vector3(1.0, 1.0, 1.0))
    helper.set_world_poses(np.array([[0, 0, 0]], np.float32), np.array([[1, 0, 0, 0]], np.float32))
    if not sim.pathfinder.load_nav_mesh(str(nav)): raise RuntimeError(f"navmesh failed: {scene}")
    return sim


def yaw_rotation(yaw: float):
    return habitat_sim.utils.common.quat_from_angle_axis(yaw, np.array([0.0, 1.0, 0.0]))


def render(sim, agent, pos: np.ndarray, rot):
    state = agent.get_state(); state.position = pos.astype(np.float32); state.rotation = rot
    agent.set_state(state, reset_sensors=True)
    obs = sim.get_sensor_observations()
    rgb = np.asarray(obs["rgb"])[..., :3].copy(); depth = np.asarray(obs["depth"], np.float32).copy()
    sensor_state = agent.get_state().sensor_states["rgb"]
    rotation = np.asarray(habitat_sim.utils.common.quat_to_magnum(sensor_state.rotation).to_matrix(), np.float32)
    return rgb, depth, np.asarray(sensor_state.position, np.float32), rotation


def shortest(sim, start: np.ndarray, end: np.ndarray) -> tuple[bool, float]:
    query = habitat_sim.ShortestPath(); query.requested_start = start; query.requested_end = end
    ok = sim.pathfinder.find_path(query)
    return bool(ok and np.isfinite(query.geodesic_distance)), float(query.geodesic_distance)


def propose(camera_t: np.ndarray, c2w: np.ndarray, radius: float = 2.0, points: int = 9) -> list[np.ndarray]:
    """K robot-centric paths. No goal or NavMesh query occurs here."""
    forward = c2w @ np.array([0.0, 0.0, -1.0], np.float32)
    right = c2w @ np.array([1.0, 0.0, 0.0], np.float32)
    forward[1] = 0.0; right[1] = 0.0
    forward /= np.linalg.norm(forward); right /= np.linalg.norm(right)
    result = []
    for angle in RELATIVE_YAWS:
        direction = math.cos(float(angle)) * forward + math.sin(float(angle)) * right
        fractions = np.linspace(0.15, 1.0, points, dtype=np.float32)[:, None]
        path = camera_t[None] + fractions * radius * direction[None]
        path[:, 1] = camera_t[1] - 1.5
        result.append(path.astype(np.float32))
    return result


def project(points_world: np.ndarray, camera_t: np.ndarray, c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project with the exact sensor c2w transform; Habitat camera looks -Z."""
    camera = (points_world - camera_t[None]) @ c2w
    z = camera[:, 2]
    valid = z < -1e-3
    fx = fy = 128.0; cx = cy = 127.5
    uv = np.empty((len(points_world), 2), np.float32)
    uv[:, 0] = fx * camera[:, 0] / (-z) + cx
    uv[:, 1] = fy * camera[:, 1] / z + cy
    valid &= (uv[:, 0] >= 0) & (uv[:, 0] < 256) & (uv[:, 1] >= 0) & (uv[:, 1] < 256)
    return uv, valid


def overlay(rgb: np.ndarray, paths: list[np.ndarray], camera_t: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    image = Image.fromarray(rgb).convert("RGB"); draw = ImageDraw.Draw(image)
    colors = [(55, 126, 184), (228, 26, 28), (77, 175, 74), (152, 78, 163), (255, 127, 0), (166, 86, 40), (247, 129, 191), (153, 153, 153)]
    for i, path in enumerate(paths):
        uv, valid = project(path, camera_t, c2w)
        pts = [tuple(map(float, p)) for p, ok in zip(uv, valid) if ok]
        if len(pts) > 1:
            draw.line(pts, fill=colors[i], width=3)
            draw.text(pts[-1], f"T{i+1}", fill=colors[i], stroke_width=1, stroke_fill="black")
    return np.asarray(image)


def label_candidate(sim, path: np.ndarray, goal: np.ndarray, initial_geodesic: float) -> dict[str, object]:
    raw = path[-1]
    snap = np.asarray(sim.pathfinder.snap_point(raw), np.float32)
    snap_ok = bool(np.isfinite(snap).all() and np.linalg.norm(snap - raw) <= 0.60)
    reachable, distance = shortest(sim, snap, goal) if snap_ok else (False, float("inf"))
    clearance = float(sim.pathfinder.distance_to_closest_obstacle(snap, 2.0)) if snap_ok else 0.0
    progress = float(initial_geodesic - distance) if reachable else -float(initial_geodesic)
    collision_free = bool(snap_ok and clearance >= 0.18)
    eventual = bool(collision_free and reachable and progress > 0)
    reason = "executable" if eventual else ("off_navmesh" if not snap_ok else "unreachable" if not reachable else "low_clearance" if clearance < .18 else "no_progress")
    return {"raw_endpoint_xyz": raw.tolist(), "executor_endpoint_xyz": snap.tolist(), "reachable": reachable,
            "collision_free": collision_free, "clearance_m": clearance, "geodesic_progress_m": progress,
            "eventual_outcome": eventual, "failure_reason": reason}


def geometric_features(depth: np.ndarray, uv: np.ndarray, valid: np.ndarray, path: np.ndarray) -> dict[str, float]:
    samples = []
    for point, ok in zip(uv, valid):
        if ok:
            x, y = np.rint(point).astype(int); value = float(depth[y, x])
            if np.isfinite(value) and .02 < value < 10.0: samples.append(value)
    return {"proposal_length_m": float(np.linalg.norm(path[-1] - path[0])),
            "image_visible_ratio": float(valid.mean()),
            "observed_depth_min_m": float(min(samples)) if samples else 0.0,
            "observed_depth_mean_m": float(np.mean(samples)) if samples else 0.0}


def save_case(out: Path, split: str, scene: str, episode: int, rng: np.random.Generator) -> dict | None:
    sim = make_sim(scene); agent = sim.initialize_agent(0)
    try:
        for _ in range(2000):
            start = np.asarray(sim.pathfinder.get_random_navigable_point(), np.float32)
            goal = np.asarray(sim.pathfinder.get_random_navigable_point(), np.float32)
            ok, d0 = shortest(sim, start, goal)
            if ok and 4.0 <= d0 <= 18.0: break
        else: return None
        current_rgb, depth, camera_t, c2w = render(sim, agent, start, yaw_rotation(float(rng.uniform(-math.pi, math.pi))))
        paths = propose(camera_t, c2w)
        goal_rgb, _, _, _ = render(sim, agent, goal, yaw_rotation(float(rng.uniform(-math.pi, math.pi))))
        eid = f"{scene}_{episode:04d}"; case = out / "samples" / eid; case.mkdir(parents=True, exist_ok=True)
        Image.fromarray(current_rgb).save(case / "current_rgb.png"); Image.fromarray(goal_rgb).save(case / "goal_image.png")
        Image.fromarray(overlay(current_rgb, paths, camera_t, c2w)).save(case / "trajectory_overlay.png")
        np.save(case / "depth.npy", depth.astype(np.float16))
        rows = []
        for i, path in enumerate(paths):
            uv, visible = project(path, camera_t, c2w)
            row = {"trajectory_id": i, "relative_yaw_rad": float(RELATIVE_YAWS[i]), "world_path_xyz": path.tolist(),
                   "image_uv": uv.tolist(), "image_projection_valid": visible.tolist(),
                   "geometry": geometric_features(depth, uv, visible, path)}
            row.update(label_candidate(sim, path, goal, d0)); rows.append(row)
        return {"episode_id": eid, "scene_id": scene, "split": split, "current_rgb": str((case / "current_rgb.png").relative_to(out)),
                "goal_image": str((case / "goal_image.png").relative_to(out)), "trajectory_overlay": str((case / "trajectory_overlay.png").relative_to(out)),
                "depth": str((case / "depth.npy").relative_to(out)), "camera_world_xyz": camera_t.tolist(), "camera_c2w": c2w.tolist(),
                "start_xyz": start.tolist(), "goal_xyz_hidden_for_evaluation": goal.tolist(), "initial_geodesic_m": d0, "trajectories": rows}
    finally: sim.close()


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--episodes-per-scene", type=int, default=360); ap.add_argument("--seed", type=int, default=20260907); ap.add_argument("--only-scene")
    args = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, scenes in SPLIT.items():
        for scene in scenes:
            if args.only_scene and scene != args.only_scene: continue
            rng = np.random.default_rng(args.seed + sum(map(ord, scene)))
            for episode in range(args.episodes_per_scene):
                record = save_case(OUT, split, scene, episode, rng)
                if record is not None: rows.append(record)
    (OUT / "dataset_manifest.json").write_text(json.dumps(rows, indent=2) + "\n")
    requests = [{"request_id": r["episode_id"], "goal_image": r["goal_image"], "current_rgb": r["current_rgb"], "trajectory_overlay": r["trajectory_overlay"],
                 "trajectories": [{"trajectory_id": x["trajectory_id"]} for x in r["trajectories"]]} for r in rows]
    (OUT / "vlm_request_package.json").write_text(json.dumps(requests, indent=2) + "\n")
    print(json.dumps({"episodes": len(rows), "candidates": len(rows) * K, "split": SPLIT}, indent=2))


if __name__ == "__main__": main()
