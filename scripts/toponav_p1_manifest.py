#!/usr/bin/env python3
"""Create the fixed TopoNav P1 static indoor task manifest.

Tasks are generated only from installed scene topology.  The manifest records
semantic node/edge identifiers and metric anchors, but the online VLM is never
given a hidden shortest path or an oracle route.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GRAPH = ROOT / "outputs/formal/RelationNav/topology/spatial_semantic_topology.json"
OUT = ROOT / "outputs/formal/TopoNav/P1"
DEV_SCENES = (
    "interior_0047_839892",
    "interior_0108_839984",
    "interior_0184_840116",
    "interior_0405_840145",
)


def center(node: dict) -> list[float]:
    p = node["geometry"]["center_xz"]
    return [float(p[0]), float(p[1])]


def name(node_id: str) -> str:
    return node_id.rsplit(":", 1)[-1].replace("_", " ")


def make_task(scene: str, kind: str, index: int, edge: dict, nodes: dict,
              alternatives: list[dict]) -> dict:
    source, target, portal = edge["source"], edge["target"], edge["via"]
    source_name, target_name, portal_name = map(name, (source, target, portal))
    base = {
        "task_id": f"{scene}:P1:{kind}:{index:03d}",
        "scene_id": scene,
        "task_type": kind,
        "start_node": source,
        "goal_node": target,
        "start_anchor_xz": center(nodes[source]),
        "goal_anchor_xz": center(nodes[target]),
        "required_relation": "CROSS",
        "relation_entity": portal,
        "topology_snapshot": str(GRAPH.relative_to(ROOT)),
        "offline_only": {
            "goal_node_for_evaluation": target,
            "required_portal_for_evaluation": portal,
        },
    }
    if kind == "S1":
        base["instruction"] = f"Navigate from the current region to {target_name}."
        base["route_constraint"] = None
    elif kind == "S2":
        base["instruction"] = (
            f"Approach and cross {portal_name}, then enter {target_name}."
        )
        base["route_constraint"] = {"must_cross": portal}
    else:
        forbidden = [e["via"] for e in alternatives if e["via"] != portal]
        base["instruction"] = (
            f"Reach {target_name} through {portal_name}; do not use the alternate portal."
        )
        base["route_constraint"] = {"must_cross": portal, "avoid_portals": forbidden}
    return base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-tasks", type=int, default=80)
    args = ap.parse_args()
    graph = json.loads(GRAPH.read_text())
    nodes = {n["node_id"]: n for n in graph["nodes"]}
    adjacency: dict[str, list[dict]] = defaultdict(list)
    for edge in graph["edges"]:
        if edge.get("edge_type") == "SPATIAL_ADJACENCY":
            adjacency[edge["source"]].append(edge)

    pools: dict[str, dict[str, list[dict]]] = {}
    for scene in DEV_SCENES:
        eligible = [
            e for source, edges in adjacency.items() for e in edges
            if e["source"].startswith(scene + ":")
            and e["target"] in nodes and e.get("via") in nodes
        ]
        if not eligible:
            # The only curated-only scene contributes relation contracts, but
            # not route-choice claims.  It is not silently relabelled as S3.
            continue
        branching = [e for e in eligible if len(adjacency[e["source"]]) > 1]
        if branching:
            pools[scene] = {"S1": eligible, "S2": eligible, "S3": branching}

    # The development protocol is a single fixed 80-episode benchmark, not a
    # per-scene smoke subset.  Its task mix is globally frozen as 20/30/30.
    # Round-robin assignment keeps the three installed multi-route scenes
    # balanced while preserving deterministic task IDs and anchors.
    if not pools:
        raise RuntimeError("no installed scene has an alternative spatial route")
    requested = {"S1": 20, "S2": 30, "S3": 30}
    if args.total_tasks != 80:
        scale = args.total_tasks / 80.0
        requested = {k: round(v * scale) for k, v in requested.items()}
        requested["S3"] += args.total_tasks - sum(requested.values())
    scenes = sorted(pools)
    tasks: list[dict] = []
    local_counts: dict[str, int] = defaultdict(int)
    for kind, count in requested.items():
        for i in range(count):
            scene = scenes[i % len(scenes)]
            pool = pools[scene][kind]
            edge = pool[(i // len(scenes)) % len(pool)]
            local_index = local_counts[scene]
            local_counts[scene] += 1
            tasks.append(make_task(scene, kind, local_index, edge, nodes,
                                   adjacency[edge["source"]]))

    counts: dict[str, int] = defaultdict(int)
    for task in tasks:
        counts[task["task_type"]] += 1
    manifest = {
        "schema_version": 1,
        "name": "TopoNav Harness P1 static indoor benchmark",
        "seed": 20260915,
        "scenes": sorted(set(t["scene_id"] for t in tasks)),
        "protocol": {
            "total_tasks": len(tasks),
            "global_task_mix": requested,
            "candidate_scene_assignment": "round_robin_over_installed_multi_route_scenes",
        },
        "excluded_scenes": {
            "interior_0405_840145": "curated relation-only topology has no spatial alternative edge"
        },
        "task_counts": dict(counts),
        "tasks": tasks,
        "online_boundary": {
            "vlm_has_goal_pose": False,
            "vlm_has_navmesh": False,
            "vlm_has_oracle_route": False,
            "executor_has_metric_goal_from_validated_edge_only": True,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "task_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"tasks": len(tasks), "counts": counts,
                      "scenes": manifest["scenes"],
                      "output": str(OUT / "task_manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
