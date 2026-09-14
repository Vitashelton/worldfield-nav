#!/usr/bin/env python3
"""RelationNav P2: evaluate relation completion independently of goal arrival.

The evaluator deliberately keeps the semantic compiler and NavMesh executor
fixed.  It compares task-state policies around the *same* relation contracts:
arrival-only, same-goal retry, verified execution, preserving recovery and an
evaluation upper bound.  The shallow representative is a documented
portal/boundary stress realization: an ordinary navigation arrival radius can
be satisfied before a CROSS/ENTER relation is physically true.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from relationnav_p1_generate import (
    ANN, OUT as P1_OUT, anchor_camera_xz, make_sim, pick_area_target,
    pick_portal_target, point_in_poly, relation_label, shortest, side,
    visible_target, yaw_rotation,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/formal/RelationNav/P2"
METHODS = ("Arrival-only", "Same-goal retry", "Relation-verified",
           "Relation-preserving recovery", "Oracle")


def entity_index():
    return {e["entity_id"]: e for e in json.loads(ANN.read_text())["entities"]}


def set_agent(agent, xyz: np.ndarray, yaw: float = 0.0) -> None:
    state = agent.get_state()
    state.position = np.asarray(xyz, np.float32)
    state.rotation = yaw_rotation(yaw)
    agent.set_state(state, reset_sensors=True)


def route_endpoint(sim, agent, target: np.ndarray, arrival_radius: float) -> dict:
    """Follow a NavMesh shortest path until its remaining length is radius.

    This is a deterministic abstraction of a fixed navigation executor's goal
    tolerance, not a learned controller.  The returned endpoint is on the
    path, making a false completion directly auditable.
    """
    start = np.asarray(agent.get_state().position, np.float32)
    query_ok, distance = shortest(sim, start, target)
    if not query_ok:
        return {"reachable": False, "arrival": False, "path_m": 0.0,
                "endpoint_xyz": start.tolist(), "reason": "unreachable"}
    move = max(0.0, float(distance) - float(arrival_radius))
    # Habitat shortest paths expose a polyline.  Interpolate only to the
    # executor's stopping distance; all intermediate points remain NavMesh.
    import habitat_sim
    query = habitat_sim.ShortestPath(); query.requested_start = start; query.requested_end = target
    sim.pathfinder.find_path(query)
    points = [np.asarray(p, np.float32) for p in query.points]
    pos, left = points[0], move
    for a, b in zip(points[:-1], points[1:]):
        seg = float(np.linalg.norm(b - a))
        if left <= seg + 1e-6:
            pos = a if seg < 1e-9 else a + (left / seg) * (b - a)
            break
        left -= seg
        pos = b
    set_agent(agent, pos)
    return {"reachable": True, "arrival": True, "path_m": move,
            "endpoint_xyz": pos.tolist(), "reason": "arrived"}


def shallow_and_deep(sim, agent, phase: dict, entities: dict, rng: np.random.Generator):
    """Return a shallow ordinary representative and a deeper admissible one."""
    rel, entity = phase["relation"], entities[phase["entity_id"]]
    deep = np.asarray(phase["target_xyz_privileged_for_supervision"], np.float32)
    if rel in ("APPROACH", "CROSS"):
        signed = entity["portal_plane"]["source_side"] if rel == "APPROACH" else entity["portal_plane"]["destination_side"]
        shallow = pick_portal_target(sim, entity, int(signed), .34, rng)
    elif rel == "ENTER":
        poly = np.asarray(entity["area_polygon_xz"], np.float32)
        center = poly.mean(0); edge = poly[0]
        candidate = edge + .28 * (center - edge) / max(float(np.linalg.norm(center - edge)), 1e-6)
        shallow = np.asarray(sim.pathfinder.snap_point(
            np.asarray([candidate[0], deep[1], candidate[1]], np.float32)), np.float32)
        if not np.isfinite(shallow).all() or not point_in_poly(shallow[[0, 2]], poly):
            shallow = deep.copy()
    else:  # OBSERVE has no useful boundary-only proxy; retain an observed pose.
        shallow = deep.copy()
    return np.asarray(shallow if shallow is not None else deep, np.float32), deep


def satisfied(sim, agent, phase: dict, entities: dict, before: np.ndarray, after: np.ndarray) -> bool:
    rel, entity = phase["relation"], entities[phase["entity_id"]]
    if rel in ("APPROACH", "CROSS"):
        return relation_label(entity, {}, {}, rel, after, sim, agent)
    if rel == "ENTER":
        return point_in_poly(after[[0, 2]], np.asarray(entity["area_polygon_xz"], np.float32))
    if rel == "OBSERVE":
        return visible_target(sim, agent, after, entity)
    raise KeyError(rel)


def run_episode(sim, agent, episode: dict, method: str, entities: dict, seed: int,
                arrival_radius: float) -> dict:
    rng = np.random.default_rng(seed)
    set_agent(agent, np.asarray(episode["start_xyz"], np.float32))
    stages, total_path, wrong_advance, false_completion = [], 0.0, 0, 0
    task_alive = True
    for phase in episode["phases"]:
        if not task_alive:
            break
        before = np.asarray(agent.get_state().position, np.float32)
        shallow, deep = shallow_and_deep(sim, agent, phase, entities, rng)
        target = deep if method == "Oracle" else shallow
        event = route_endpoint(sim, agent, target, arrival_radius)
        after = np.asarray(agent.get_state().position, np.float32)
        total_path += event["path_m"]
        done = bool(event["arrival"] and satisfied(sim, agent, phase, entities, before, after))
        false = bool(event["arrival"] and not done)
        attempts, recovered = 1, False
        if false:
            false_completion += 1
            if method == "Same-goal retry":
                # A goal-only retry does not change the semantic realization.
                attempts += 1
                retry = route_endpoint(sim, agent, shallow, arrival_radius)
                total_path += retry["path_m"]
                after = np.asarray(agent.get_state().position, np.float32)
                done = bool(retry["arrival"] and satisfied(sim, agent, phase, entities, before, after))
            elif method == "Relation-preserving recovery":
                # Preserve (entity, relation); only replace its failed local
                # representative with a deeper member of the same admissible region.
                attempts += 1
                retry = route_endpoint(sim, agent, deep, arrival_radius)
                total_path += retry["path_m"]
                after = np.asarray(agent.get_state().position, np.float32)
                done = bool(retry["arrival"] and satisfied(sim, agent, phase, entities, before, after))
                recovered = done
            elif method == "Relation-verified":
                task_alive = False
            elif method == "Arrival-only":
                wrong_advance += 1
        advance = (method in ("Arrival-only", "Same-goal retry")) or done
        if not advance:
            task_alive = False
        stages.append({"relation": phase["relation"], "entity_id": phase["entity_id"],
                       "start_xyz": before.tolist(), "shallow_goal_xyz": shallow.tolist(),
                       "deep_goal_xyz": deep.tolist(), "endpoint_xyz": after.tolist(),
                       "arrival": bool(event["arrival"]), "relation_done": bool(done),
                       "false_completion": false, "wrong_stage_advance": bool(false and advance and not done),
                       "recovered": recovered, "attempts": attempts})
    complete = bool(len(stages) == len(episode["phases"]) and all(s["relation_done"] or method in ("Arrival-only", "Same-goal retry") for s in stages))
    # Arrival-only "task success" is intentionally separated from semantic success.
    semantic_complete = bool(len(stages) == len(episode["phases"]) and all(s["relation_done"] for s in stages))
    return {"episode_id": episode["episode_id"], "scene_id": episode["scene_id"],
            "split": episode["split"], "method": method, "nominal_task_complete": complete,
            "semantic_task_complete": semantic_complete, "path_length_m": total_path,
            "false_completions": false_completion, "wrong_stage_advances": wrong_advance,
            "recovery_successes": sum(s["recovered"] for s in stages), "stages": stages}


def summarize(rows: list[dict]) -> list[dict]:
    out = []
    for (method, split), group in sorted(((k, v) for k, v in _group(rows).items())):
        stages = [s for r in group for s in r["stages"]]
        false = [s for s in stages if s["false_completion"]]
        out.append({"method": method, "split": split, "episodes": len(group),
                    "Relation_Completion_Rate": float(np.mean([s["relation_done"] for s in stages])) if stages else 0.0,
                    "False_Completion_Rate": float(np.mean([s["false_completion"] for s in stages])) if stages else 0.0,
                    "Wrong_Stage_Advance_Rate": float(np.mean([s["wrong_stage_advance"] for s in stages])) if stages else 0.0,
                    "Recovery_Success_Rate": float(np.mean([s["recovered"] for s in false])) if false else 0.0,
                    "Semantic_Task_Success": float(np.mean([r["semantic_task_complete"] for r in group])),
                    "Nominal_Task_Success": float(np.mean([r["nominal_task_complete"] for r in group])),
                    "Path_Length_m": float(np.mean([r["path_length_m"] for r in group]))})
    return out


def _group(rows):
    d = defaultdict(list)
    for r in rows: d[(r["method"], r["split"])].append(r)
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-episodes-per-scene", type=int, default=100)
    ap.add_argument("--arrival-radius", type=float, default=.50)
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--seed", type=int, default=20260914)
    args = ap.parse_args()
    manifest = json.loads((P1_OUT / "dataset/dataset_manifest.json").read_text())["episodes"]
    selected = []
    for scene in sorted({e["scene_id"] for e in manifest}):
        selected.extend([e for e in manifest if e["scene_id"] == scene][:args.max_episodes_per_scene])
    entities, rows = entity_index(), []
    for method in args.methods:
        by_scene = defaultdict(list)
        for ep in selected: by_scene[ep["scene_id"]].append(ep)
        for scene, episodes in by_scene.items():
            sim = make_sim(scene); agent = sim.initialize_agent(0)
            try:
                for i, ep in enumerate(episodes):
                    rows.append(run_episode(sim, agent, ep, method, entities,
                                            args.seed + i + sum(map(ord, scene)), args.arrival_radius))
            finally:
                sim.close()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "episode_results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = summarize(rows)
    (OUT / "summary.json").write_text(json.dumps({"protocol": {"arrival_radius_m": args.arrival_radius,
        "shallow_goal": "0.34m portal/boundary representative", "deep_goal": "existing admissible P1 representative"},
        "metrics": summary}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
