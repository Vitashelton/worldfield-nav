#!/usr/bin/env python3
"""Create deterministic RelationNav P1 multi-stage indoor episodes.

The online candidate generator is robot-centric and never accesses the final
goal.  NavMesh, portal planes, polygons and landmark visibility are used only
offline to create labels and to evaluate fixed-executor rollouts.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from trajectory_grounding_p1_generate import make_sim, overlay, propose, render, shortest, yaw_rotation

OUT = ROOT / "outputs/formal/RelationNav/P1"
ANN = ROOT / "configs/relationnav/entities.json"
SPLIT = {
    "train": ["interior_0135_840032", "interior_0121_840013", "interior_0093_839966"],
    "heldout": ["interior_0405_840145"],
}


def side(portal: dict, xyz: np.ndarray) -> float:
    p = np.asarray(portal["portal_plane"]["point_world_xyz"], np.float32)
    n = np.asarray(portal["portal_plane"]["normal_xz"], np.float32)
    return float(n @ (xyz[[0, 2]] - p[[0, 2]]))


def snap(sim, xz: np.ndarray) -> np.ndarray | None:
    q = np.asarray(sim.pathfinder.snap_point(np.array([xz[0], 0.0, xz[1]], np.float32)), np.float32)
    return q if np.isfinite(q).all() else None


def path_ok(sim, a: np.ndarray, b: np.ndarray, lo: float = 0.0, hi: float = 99.0) -> bool:
    ok, d = shortest(sim, a, b)
    return bool(ok and lo <= d <= hi)


def pick_portal_target(sim, portal: dict, signed_side: int, offset: float, rng: np.random.Generator) -> np.ndarray | None:
    plane = portal["portal_plane"]
    p = np.asarray(plane["point_world_xyz"], np.float32)[[0, 2]]
    n = np.asarray(plane["normal_xz"], np.float32)
    tangent = np.array([-n[1], n[0]], np.float32)
    for _ in range(24):
        lateral = float(rng.uniform(-0.25, 0.25)) * float(plane["width_m"])
        target = snap(sim, p + signed_side * offset * n + lateral * tangent)
        if target is not None and signed_side * side(portal, target) > max(0.18, offset * .35):
            return target
    return None


def point_in_poly(xz: np.ndarray, poly: np.ndarray) -> bool:
    x, z = float(xz[0]), float(xz[1]); inside = False
    for i in range(len(poly)):
        x1, z1 = poly[i]; x2, z2 = poly[(i + 1) % len(poly)]
        if (z1 > z) != (z2 > z) and x < (x2 - x1) * (z - z1) / (z2 - z1 + 1e-9) + x1:
            inside = not inside
    return inside


def pick_area_target(sim, area: dict, rng: np.random.Generator) -> np.ndarray | None:
    poly = np.asarray(area["area_polygon_xz"], np.float32)
    mn, mx = poly.min(0), poly.max(0)
    for _ in range(80):
        xz = rng.uniform(mn, mx)
        if point_in_poly(xz, poly):
            q = snap(sim, xz)
            if q is not None and point_in_poly(q[[0, 2]], poly): return q
    return None


def visible_target(sim, agent, position: np.ndarray, landmark: dict) -> bool:
    target = np.asarray(landmark["visual_anchor"]["world_point_xyz"], np.float32)
    to = target - position
    yaw = math.atan2(float(-to[0]), float(-to[2]))
    _, depth, camera, c2w = render(sim, agent, position, yaw_rotation(yaw))
    local = (target - camera) @ c2w
    if local[2] >= -1e-3: return False
    u = int(round(128.0 * local[0] / (-local[2]) + 127.5)); v = int(round(128.0 * local[1] / local[2] + 127.5))
    if not (0 <= u < 256 and 0 <= v < 256): return False
    observed = float(depth[v, u]); return bool(np.isfinite(observed) and .02 < observed < 10.0 and abs(observed + local[2]) < .25)


def anchor_camera_xz(entity: dict) -> np.ndarray:
    """Reference viewing position is an offline OBSERVE label, not inference."""
    if "reference_observer_xz" in entity:
        return np.asarray(entity["reference_observer_xz"], np.float32)
    aid=entity["visual_anchor"]["anchor_view_id"]; scene=entity["scene_id"]
    src=ROOT/"outputs/formal/GoalPose/P0/anchor_views"/f"anchor_view_candidates_{scene}.jsonl"
    for line in src.read_text().splitlines():
        row=json.loads(line)
        if row["anchor_view_id"]==aid: return np.asarray(row["camera_xyz"],np.float32)[[0,2]]
    raise KeyError(aid)


def pick_observe_target(sim, agent, landmark: dict, near: np.ndarray, rng: np.random.Generator) -> np.ndarray | None:
    center = anchor_camera_xz(landmark)
    q=snap(sim,center)
    if q is not None and path_ok(sim,near,q,0.0,20.0) and visible_target(sim,agent,q,landmark): return q
    for _ in range(100):
        angle, radius = float(rng.uniform(-math.pi, math.pi)), float(rng.uniform(.9, 2.2))
        q = snap(sim, center + radius * np.array([math.sin(angle), math.cos(angle)], np.float32))
        if q is not None and path_ok(sim, near, q, 0.0, 20.0) and visible_target(sim, agent, q, landmark): return q
    return None


def phase_rows(sim, agent, portal: dict, area: dict, landmark: dict, rng: np.random.Generator):
    src = int(portal["portal_plane"]["source_side"])
    approach = pick_portal_target(sim, portal, src, 1.15, rng)
    cross = pick_portal_target(sim, portal, -src, 1.15, rng)
    enter = pick_area_target(sim, area, rng)
    if approach is None or cross is None or enter is None: return None
    observe = pick_observe_target(sim, agent, landmark, enter, rng)
    phases = [
        {"relation": "APPROACH", "entity_id": portal["entity_id"], "target_xyz_privileged": approach.tolist(), "guard": "source_portal_neighborhood"},
        {"relation": "CROSS", "entity_id": portal["entity_id"], "target_xyz_privileged": cross.tolist(), "guard": "signed_portal_side_change"},
        {"relation": "ENTER", "entity_id": area["entity_id"], "target_xyz_privileged": enter.tolist(), "guard": "inside_area_polygon"},
    ]
    # The contract permits three-to-four phases.  A scene is not discarded
    # merely because the curated target is not visible from the portal's
    # destination side; that fact is recorded by omitting OBSERVE.
    if observe is not None:
        phases.append({"relation": "OBSERVE", "entity_id": landmark["entity_id"], "target_xyz_privileged": observe.tolist(), "guard": "landmark_visible"})
    return phases


def relation_label(portal: dict, area: dict, landmark: dict, relation: str, q: np.ndarray, sim, agent) -> bool:
    if relation == "APPROACH":
        return side(portal, q) * portal["portal_plane"]["source_side"] > .25 and abs(side(portal, q)) < 1.8
    if relation == "CROSS":
        return side(portal, q) * portal["portal_plane"]["destination_side"] > .35
    if relation == "ENTER": return point_in_poly(q[[0, 2]], np.asarray(area["area_polygon_xz"], np.float32))
    if relation == "OBSERVE": return visible_target(sim, agent, q, landmark)
    raise KeyError(relation)


def sample_start(sim, portal: dict, first: np.ndarray, rng: np.random.Generator) -> np.ndarray | None:
    src = int(portal["portal_plane"]["source_side"])
    for _ in range(600):
        q = np.asarray(sim.pathfinder.get_random_navigable_point(), np.float32)
        if src * side(portal, q) <= .65: continue
        if path_ok(sim, q, first, 2.0, 12.0): return q
    return None


def save_decision(base: Path, sim, agent, episode_id: str, phase_index: int, current: np.ndarray, yaw: float, target: np.ndarray, phase: dict) -> dict:
    """Save an online observation.

    ``yaw`` comes from an initial random heading or the *previous* executed
    segment.  It must never be derived from this phase's privileged target.
    """
    rgb, depth, camera, c2w = render(sim, agent, current, yaw_rotation(yaw))
    paths = propose(camera, c2w)
    root = base / "decisions" / episode_id; root.mkdir(parents=True, exist_ok=True)
    stem = f"phase_{phase_index:02d}"
    Image.fromarray(rgb).save(root / f"{stem}_rgb.png")
    np.save(root / f"{stem}_depth.npy", depth.astype(np.float16))
    Image.fromarray(overlay(rgb, paths, camera, c2w)).save(root / f"{stem}_candidates.png")
    return {"phase_index": phase_index, "relation": phase["relation"], "entity_id": phase["entity_id"],
            "current_xyz": current.tolist(), "rgb": str((root / f"{stem}_rgb.png").relative_to(base)),
            "depth": str((root / f"{stem}_depth.npy").relative_to(base)),
            "candidate_overlay": str((root / f"{stem}_candidates.png").relative_to(base)),
            "camera_xyz": camera.tolist(), "camera_c2w": c2w.tolist(), "camera_yaw_rad": float(yaw),
            "candidate_paths_goal_independent": [x.tolist() for x in paths],
            "target_xyz_privileged_for_supervision": target.tolist(), "completion_guard": phase["guard"]}


def generate_scene(scene: str, split: str, episodes: int, seed: int, entities: list[dict], out: Path) -> list[dict]:
    portal = next(x for x in entities if x["scene_id"] == scene and x["kind"] == "portal")
    area = next(x for x in entities if x["scene_id"] == scene and x["kind"] == "area")
    landmark = next(x for x in entities if x["scene_id"] == scene and x["kind"] == "landmark")
    rng = np.random.default_rng(seed + sum(map(ord, scene))); sim = make_sim(scene); agent = sim.initialize_agent(0); rows=[]
    try:
        for e in range(episodes):
            for _ in range(80):
                phases = phase_rows(sim, agent, portal, area, landmark, rng)
                if phases is None: continue
                start = sample_start(sim, portal, np.asarray(phases[0]["target_xyz_privileged"], np.float32), rng)
                if start is None: continue
                chain=[start]+[np.asarray(p["target_xyz_privileged"],np.float32) for p in phases]
                # RelationNav makes local decisions: each phase target must be
                # within the local 6 m execution horizon. This is episode
                # construction, never online candidate filtering.
                if all(path_ok(sim,a,b,0.0,5.8) for a,b in zip(chain[:-1],chain[1:])): break
            else: continue
            eid = f"{scene}_{e:04d}"; current = start; decisions=[]; phase_success=[]
            # Curated task starts face the active portal so the requested
            # relation is visually observable. It constrains episode setup,
            # never candidate generation or the learned field input labels.
            delta=np.asarray(phases[0]["target_xyz_privileged"],np.float32)-start
            heading=math.atan2(float(-delta[0]),float(-delta[2]))
            for i, phase in enumerate(phases):
                target=np.asarray(phase["target_xyz_privileged"],np.float32)
                decisions.append(save_decision(out, sim, agent, eid, i, current, heading, target, phase))
                reachable,d=shortest(sim,current,target)
                ok=bool(reachable and np.isfinite(d))
                if ok:
                    delta = target - current
                    heading = math.atan2(float(-delta[0]), float(-delta[2]))
                    current=target
                phase_success.append(ok)
            rows.append({"episode_id":eid,"scene_id":scene,"split":split,"seed":seed,"start_xyz":start.tolist(),
                         "entities":{"portal":portal["entity_id"],"area":area["entity_id"],"landmark":landmark["entity_id"]},
                         "portal_plane":portal["portal_plane"],"phases":decisions,"oracle_fixed_executor_phase_success":phase_success,
                         "oracle_fixed_executor_task_success":bool(all(phase_success))})
    finally: sim.close()
    return rows


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--episodes-per-scene",type=int,default=100); ap.add_argument("--seed",type=int,default=20260914); ap.add_argument("--only-scene"); args=ap.parse_args()
    anns=json.loads(ANN.read_text())["entities"]; out=OUT / "dataset"; out.mkdir(parents=True,exist_ok=True); rows=[]
    for split,scenes in SPLIT.items():
        for scene in scenes:
            if args.only_scene and args.only_scene != scene: continue
            rows += generate_scene(scene,split,args.episodes_per_scene,args.seed,anns,out)
    manifest={"protocol":"RelationNav P1; relation labels are offline curated-geometry supervision only.","split":SPLIT,"episodes":rows}
    (out/"dataset_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"episodes":len(rows),"decisions":sum(len(x["phases"]) for x in rows),"out":str(out)},indent=2))

if __name__ == "__main__": main()
