#!/usr/bin/env python3
"""Auditable RelationNav portal/area/landmark annotation utility.

Annotations are deliberately human-curated. This tool only backprojects a
selected RGB-D anchor pixel and validates explicit plane/area metadata; it
never invents semantic entities from a VLM response.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ANCHORS = ROOT / "outputs/formal/GoalPose/P0/anchor_views"
DEFAULT = ROOT / "configs/relationnav/entities.json"
SCENES = ["interior_0405_840145", "interior_0135_840032", "interior_0121_840013", "interior_0093_839966"]


def load_anchor(anchor_id: str) -> dict:
    scene = anchor_id.split("_view_")[0].rsplit("_", 1)[0]
    records = ANCHORS / f"anchor_view_candidates_{scene}.jsonl"
    if not records.is_file():
        raise FileNotFoundError(records)
    for line in records.read_text().splitlines():
        value = json.loads(line)
        if value["anchor_view_id"] == anchor_id:
            return value
    raise KeyError(anchor_id)


def backproject(anchor: dict, u: int, v: int) -> list[float]:
    depth = np.load(ROOT / "outputs/formal/GoalPose" / anchor["depth"]).astype(np.float32)
    if not (0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]):
        raise ValueError("pixel outside image")
    d = float(depth[v, u])
    if not np.isfinite(d) or not .02 < d < 9.99:
        raise ValueError(f"invalid depth {d} at ({u},{v})")
    intr, c2w, camera = anchor["intrinsics"], np.asarray(anchor["camera_c2w"], np.float32), np.asarray(anchor["camera_xyz"], np.float32)
    point_cam = np.array([(u - intr["cx"]) * d / intr["fx"], -(v - intr["cy"]) * d / intr["fy"], -d], np.float32)
    return (point_cam @ c2w.T + camera).astype(float).tolist()


def empty() -> dict:
    return {
        "schema_version": 1,
        "purpose": "Human-curated RelationNav spatial entities; no automatic semantic labels.",
        "scenes": SCENES,
        "entities": [],
    }


def read(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else empty()


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def validate(payload: dict) -> list[str]:
    issues = []
    identifiers = set()
    for entity in payload.get("entities", []):
        eid = entity.get("entity_id")
        if not eid or eid in identifiers: issues.append(f"duplicate/missing entity_id: {eid}")
        identifiers.add(eid)
        if entity.get("scene_id") not in SCENES: issues.append(f"{eid}: unknown scene")
        if entity.get("kind") not in {"portal", "area", "landmark"}: issues.append(f"{eid}: invalid kind")
        anchor = entity.get("visual_anchor", {})
        if not anchor.get("anchor_view_id") or len(anchor.get("pixel_uv", [])) != 2:
            issues.append(f"{eid}: visual anchor missing")
        if entity.get("kind") == "portal":
            plane = entity.get("portal_plane", {})
            normal = plane.get("normal_xz", [])
            if len(plane.get("point_world_xyz", [])) != 3 or len(normal) != 2 or abs(float(np.linalg.norm(normal)) - 1.0) > .05:
                issues.append(f"{eid}: invalid portal plane")
            if plane.get("source_side") not in {-1, 1} or plane.get("destination_side") != -plane.get("source_side"):
                issues.append(f"{eid}: source/destination side invalid")
        if entity.get("kind") == "area" and len(entity.get("area_polygon_xz", [])) < 3:
            issues.append(f"{eid}: area polygon needs >=3 points")
    return issues


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", type=Path, default=DEFAULT)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    v = sub.add_parser("validate")
    p = sub.add_parser("add-portal")
    p.add_argument("--entity-id", required=True); p.add_argument("--scene", choices=SCENES, required=True)
    p.add_argument("--anchor-view-id", required=True); p.add_argument("--pixel", type=int, nargs=2, required=True)
    p.add_argument("--normal-yaw-rad", type=float, required=True); p.add_argument("--width-m", type=float, required=True)
    p.add_argument("--source-side", type=int, choices=[-1, 1], required=True)
    p.add_argument("--rationale", required=True)
    args = ap.parse_args()
    if args.cmd == "init":
        if args.annotations.exists(): raise FileExistsError(args.annotations)
        write(args.annotations, empty()); print(args.annotations); return
    payload = read(args.annotations)
    if args.cmd == "validate":
        issues = validate(payload)
        print(json.dumps({"entities": len(payload["entities"]), "valid": not issues, "issues": issues}, indent=2))
        if issues: raise SystemExit(1)
        return
    anchor = load_anchor(args.anchor_view_id)
    if anchor["scene_id"] != args.scene: raise ValueError("anchor scene mismatch")
    point = backproject(anchor, *args.pixel)
    normal = [math.sin(args.normal_yaw_rad), math.cos(args.normal_yaw_rad)]
    payload["entities"].append({
        "entity_id": args.entity_id, "scene_id": args.scene, "kind": "portal",
        "annotation_status": "curated",
        "visual_anchor": {"anchor_view_id": args.anchor_view_id, "pixel_uv": args.pixel, "world_point_xyz": point,
                          "rgb": anchor["rgb"], "depth": anchor["depth"]},
        "portal_plane": {"point_world_xyz": point, "normal_xz": normal, "width_m": args.width_m,
                          "source_side": args.source_side, "destination_side": -args.source_side},
        "rationale": args.rationale,
    })
    issues = validate(payload)
    if issues: raise ValueError("; ".join(issues))
    write(args.annotations, payload); print(json.dumps(payload["entities"][-1], indent=2))


if __name__ == "__main__":
    main()
