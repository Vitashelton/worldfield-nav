"""Scene-agnostic semantic topology graph and online relation state.

The graph is built from the curated portal/area/landmark contracts.  It is a
compact navigation *interface*: Habitat/NavMesh supplies geometry and a fixed
executor supplies motion; this module only represents spatial entities,
relations and completion state.  No online candidate is selected with hidden
goal information.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np


def _xz(x: Iterable[float]) -> np.ndarray:
    a = np.asarray(list(x), dtype=np.float32)
    return a[[0, 2]] if a.size >= 3 else a[:2]


def _side(point_xz: np.ndarray, plane: dict[str, Any]) -> float:
    p = np.asarray(plane["point_world_xyz"], np.float32)[[0, 2]]
    n = np.asarray(plane["normal_xz"], np.float32)
    return float(np.dot(point_xz - p, n))


def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """2-D ray crossing test, accepting boundary points."""
    x, y = map(float, point); inside = False
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        ax, ay = map(float, a); bx, by = map(float, b)
        cross = (x - ax) * (by - ay) - (y - ay) * (bx - ax)
        if abs(cross) < 1e-6 and min(ax, bx) - 1e-6 <= x <= max(ax, bx) + 1e-6 and min(ay, by) - 1e-6 <= y <= max(ay, by) + 1e-6:
            return True
        hit = ((ay > y) != (by > y)) and (x < (bx - ax) * (y - ay) / (by - ay + 1e-12) + ax)
        if hit: inside = not inside
    return inside


def build_topology(annotation_payload: dict[str, Any]) -> dict[str, Any]:
    """Build a deterministic graph without inventing scene semantics."""
    entities = annotation_payload.get("entities", [])
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    by_scene: dict[str, list[dict[str, Any]]] = {}
    for e in entities:
        scene = e["scene_id"]; by_scene.setdefault(scene, []).append(e)
        anchor = e["visual_anchor"]["world_point_xyz"]
        if e["kind"] == "area":
            poly = e["area_polygon_xz"]
            center = np.asarray(poly, np.float32).mean(0).tolist()
            geometry = {"polygon_xz": poly, "center_xz": center}
        else:
            geometry = {"center_xz": [float(anchor[0]), float(anchor[2])]}
            if e["kind"] == "portal":
                geometry["plane"] = e["portal_plane"]
        nodes.append({"node_id": e["entity_id"], "scene_id": scene,
                      "kind": e["kind"], "geometry": geometry,
                      "annotation_status": e.get("annotation_status", "unknown")})
    for scene, items in by_scene.items():
        areas = [e for e in items if e["kind"] == "area"]
        portals = [e for e in items if e["kind"] == "portal"]
        landmarks = [e for e in items if e["kind"] == "landmark"]
        # An annotated area is the declared destination region for its scene.
        # The edge is explicit and auditable; no NavMesh query is hidden here.
        for p in portals:
            for a in areas:
                edges.extend([
                    {"source": a["entity_id"], "target": p["entity_id"], "relation": "APPROACH", "directed": True},
                    {"source": p["entity_id"], "target": a["entity_id"], "relation": "CROSS", "directed": True},
                    {"source": p["entity_id"], "target": a["entity_id"], "relation": "ENTER", "directed": True},
                ])
        for a in areas:
            for l in landmarks:
                edges.append({"source": a["entity_id"], "target": l["entity_id"], "relation": "OBSERVE", "directed": True})
    return {"schema_version": 1, "graph_type": "curated_semantic_spatial_topology",
            "scenes": annotation_payload.get("scenes", sorted(by_scene)),
            "nodes": nodes, "edges": edges,
            "provenance": {"source": "configs/relationnav/entities.json",
                           "navmesh_used_for_graph_construction": False,
                           "curated_entities_only": True}}


@dataclass
class RelationState:
    """Online state machine; relation intent is preserved across failures."""
    entity_id: str
    relation: str
    status: str = "active"
    previous_pose_xz: np.ndarray | None = None
    blacklisted_realizations: set[str] = field(default_factory=set)
    attempts: int = 0
    event_log: list[dict[str, Any]] = field(default_factory=list)

    def update(self, pose_xyz: Iterable[float], entity: dict[str, Any], *, visible: bool = False,
               realization_id: str | None = None) -> bool:
        pose = _xz(pose_xyz); before = self.previous_pose_xz
        done = False
        if self.relation == "APPROACH" and entity["kind"] == "portal":
            center = _xz(entity["portal_plane"]["point_world_xyz"])
            done = float(np.linalg.norm(pose - center)) <= max(.75, float(entity["portal_plane"]["width_m"]) * .5)
            done &= _side(pose, entity["portal_plane"]) <= 0.0
        elif self.relation == "CROSS" and entity["kind"] == "portal" and before is not None:
            plane = entity["portal_plane"]
            done = (_side(before, plane) <= 0.0 and _side(pose, plane) >= 0.0 and
                    float(np.linalg.norm(pose - _xz(plane["point_world_xyz"]))) <= 2.5)
        elif self.relation == "ENTER" and entity["kind"] == "area":
            done = point_in_polygon(pose, np.asarray(entity["area_polygon_xz"], np.float32))
        elif self.relation == "OBSERVE" and entity["kind"] == "landmark":
            done = bool(visible)
        self.previous_pose_xz = pose
        self.event_log.append({"pose_xz": pose.tolist(), "relation": self.relation, "done": bool(done)})
        if done: self.status = "completed"
        return bool(done)

    def register_failure(self, realization_id: str, reason: str) -> None:
        self.attempts += 1; self.blacklisted_realizations.add(realization_id)
        self.status = "active"  # preserve the same entity/relation contract
        self.event_log.append({"event": "failure", "realization_id": realization_id, "reason": reason,
                               "relation_preserved": True})

