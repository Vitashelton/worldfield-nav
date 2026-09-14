#!/usr/bin/env python3
"""Audit official InteriorGS sidecars for automatically constructible tasks."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIDE = ROOT / "data/interiorgs_semantics"
OUT = ROOT / "outputs/formal/RelationNav/benchmark_asset_audit.json"


def centroid(profile):
    pts = [[float(p[0]), float(p[1])] for p in profile]
    return [sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)]


def main():
    rows = []
    for structure_path in sorted(SIDE.glob("*/structure.json")):
        scene = structure_path.parent.name
        structure = json.loads(structure_path.read_text())
        labels_path = structure_path.parent / "labels.json"
        labels = json.loads(labels_path.read_text()) if labels_path.exists() else []
        doors = []
        for idx, opening in enumerate(structure.get("holes", [])):
            if str(opening.get("type", "")).upper() != "DOOR":
                continue
            doors.append({"door_id": f"{scene}_door_{idx:03d}",
                          "centroid_xy": centroid(opening["profile"]),
                          "width_m": max(float(max(p[0] for p in opening["profile"]) - min(p[0] for p in opening["profile"])),
                                         float(max(p[1] for p in opening["profile"]) - min(p[1] for p in opening["profile"])))})
        objects = []
        for item in labels if isinstance(labels, list) else labels.get("objects", []):
            label = str(item.get("label", "")).lower()
            if label in {"door", "window", "sign", "fire extinguisher", "elevator", "stairs"}:
                objects.append({"label": label, "instance_id": str(item.get("ins_id", ""))})
        rows.append({"scene_id": scene, "rooms": len(structure.get("rooms", [])),
                     "door_openings": len(doors), "door_like_objects": len(objects),
                     "objects_total": len(labels) if isinstance(labels, list) else len(labels.get("objects", [])),
                     "rooms_with_transition_potential": max(0, len(structure.get("rooms", [])) - 1),
                     "doors": doors, "semantic_objects": objects})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {"protocol": "InteriorGS sidecars only; no per-scene manual task curation",
               "scenes": rows,
               "totals": {"scenes": len(rows), "rooms": sum(x["rooms"] for x in rows),
                          "door_openings": sum(x["door_openings"] for x in rows),
                          "objects": sum(x["objects_total"] for x in rows)}}
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["totals"], indent=2))


if __name__ == "__main__":
    main()
