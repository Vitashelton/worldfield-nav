"""Deterministic B1 P1 episode-manifest construction and validation."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


CONTEXT_ORDER = {"open_space": 0, "turn_corner": 1, "doorway_occlusion": 2}
FAMILIES = ("straight", "left", "right", "mixed_turn")
HORIZONS_S = (0.5, 1.0, 2.0, 3.0)
CONTROL_DT_S = 0.2
CONTROL_TICKS = 15
# Ten Hz is the minimum clock that represents 0.5 s exactly.  The simulator
# action sequence remains the original 5 Hz (dt=0.2 s) control sequence.
GT_HZ = 10
GT_TICKS = 30


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _integrate(controls: list[dict[str, float]]) -> list[dict[str, float]]:
    x = z = yaw = 0.0
    poses = [{"t_s": 0.0, "x_m": 0.0, "z_m": 0.0, "yaw_rad": 0.0}]
    for i, control in enumerate(controls, 1):
        dt = float(control["dt_s"])
        v = float(control["v_mps"])
        omega = float(control["omega_radps"])
        x += v * math.cos(yaw) * dt
        z += v * math.sin(yaw) * dt
        yaw += omega * dt
        poses.append({"t_s": round(i * dt, 10), "x_m": x, "z_m": z, "yaw_rad": yaw})
    return poses


def _load_rows(path: Path) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[(row["scene_id"], row["candidate_id"])].append(row)
    return grouped


def build_manifest(root: Path, output: Path, candidate_source: Path, config_path: Path) -> dict[str, Any]:
    grouped = _load_rows(candidate_source)
    by_scene: dict[str, list[tuple[tuple[int, int], list[dict[str, str]]]]] = defaultdict(list)
    for key, rows in grouped.items():
        if rows[0]["eligible_after_threshold"].lower() != "true":
            continue
        by_scene[key[0]].append(((CONTEXT_ORDER.get(rows[0]["context"], 99), int(key[1])), rows))

    episodes: list[dict[str, Any]] = []
    scene_records: list[dict[str, Any]] = []
    missing_pose: list[str] = []
    missing_gt: list[str] = []
    for scene in sorted(by_scene):
        _, rows = sorted(by_scene[scene], key=lambda item: item[0])[0]
        rows = sorted(rows, key=lambda row: FAMILIES.index(row["action_family"]))
        if {row["action_family"] for row in rows} != set(FAMILIES):
            raise ValueError(f"candidate {scene} does not contain all four action families")
        first = rows[0]
        split = first["split"]
        candidate_id = int(first["candidate_id"])
        start_key = f"p0:{scene}:{candidate_id}"
        pose_ref = f"{candidate_source.relative_to(root)}#scene_id={scene}&candidate_id={candidate_id}"
        initial_ref = f"{candidate_source.relative_to(root)}#scene_id={scene}&candidate_id={candidate_id}:initial_field"
        missing_pose.append(start_key)
        missing_gt.append(start_key)
        for row in rows:
            controls = json.loads(row["continuous_control"])
            timestamps = [round(i / GT_HZ, 10) for i in range(GT_TICKS + 1)]
            if len(controls) != CONTROL_TICKS or any(abs(float(c["dt_s"]) - CONTROL_DT_S) > 1e-8 for c in controls):
                raise ValueError(f"invalid control clock for {scene}/{candidate_id}/{row['action_family']}")
            branch_id = f"B1-P1-{scene}-{candidate_id:03d}-{row['action_family']}"
            episodes.append({
                "episode_id": branch_id,
                "scene_id": scene,
                "split": split,
                "context": first["context"],
                "candidate_id": candidate_id,
                "seed": 20260902,
                "initial_state_ref": initial_ref,
                "start_pose": {"available": False, "source_ref": pose_ref, "reason": "P0 did not persist absolute pose"},
                "action_family": row["action_family"],
                "controls": controls,
                "timestamps_s": timestamps,
                "horizon_frames": {str(h): int(round(h * GT_HZ)) for h in HORIZONS_S},
                "integrated_relative_poses": _integrate(controls),
                "requested_translation_m": float(row["requested_translation_m"]),
                "realized_translation_m": float(row["realized_translation_m"]),
                "translation_ratio": float(row["translation_ratio"]),
                "gt_snapshots": [{"horizon_s": h, "frame_index": int(round(h * GT_HZ)), "frame_time_s": h, "horizon_exact": True, "available": False, "source_ref": initial_ref, "reason": "P0 did not persist O/H/V/A snapshots"} for h in HORIZONS_S],
                "source_revelation_area_m2": {str(h): float(row[f"revelation_m2_{h:g}s"]) for h in HORIZONS_S},
                "source_revelation_clock": {"sample_hz": 5, "legacy_frame_map": {"0.5": 2, "1.0": 5, "2.0": 10, "3.0": 15}},
            })
        scene_records.append({"scene_id": scene, "split": split, "context": first["context"], "candidate_id": candidate_id, "initial_state_ref": initial_ref})

    manifest = {
        "schema_version": 1,
        "manifest_type": "B1_P1_deterministic_episode_manifest",
        "config": str(config_path.relative_to(root)),
        "source_candidate_table": str(candidate_source.relative_to(root)),
        "sampling": {"gt_hz": GT_HZ, "gt_dt_s": 1.0 / GT_HZ, "control_dt_s": CONTROL_DT_S, "control_hz": 1.0 / CONTROL_DT_S, "duration_s": CONTROL_TICKS * CONTROL_DT_S, "horizons_s": list(HORIZONS_S)},
        "action_families": list(FAMILIES),
        "scene_records": scene_records,
        "episodes": episodes,
        "p0_artifact_gaps": {"missing_absolute_start_pose": missing_pose, "missing_OHVA_snapshots": missing_gt},
    }
    manifest["manifest_sha256"] = _json_hash({k: v for k, v in manifest.items() if k != "manifest_sha256"})
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    episodes = manifest["episodes"]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        groups[episode["initial_state_ref"]].append(episode)
    branches_ok = all({e["action_family"] for e in group} == set(FAMILIES) and len(group) == 4 for group in groups.values())
    clocks_ok = all(len(e["controls"]) == CONTROL_TICKS and e["timestamps_s"] == [round(i / GT_HZ, 10) for i in range(GT_TICKS + 1)] and {str(h): int(round(h * GT_HZ)) for h in HORIZONS_S} == e["horizon_frames"] and all(abs(e["gt_snapshots"][i]["frame_time_s"] - h) < 1e-8 and e["gt_snapshots"][i]["horizon_exact"] for i, h in enumerate(HORIZONS_S)) for e in episodes)
    integration_ok = all(len(e["integrated_relative_poses"]) == CONTROL_TICKS + 1 and abs(e["integrated_relative_poses"][-1]["t_s"] - 3.0) < 1e-8 for e in episodes)
    source_refs_ok = all(len(e["gt_snapshots"]) == len(HORIZONS_S) and all(snapshot.get("source_ref") and snapshot["frame_index"] == int(round(snapshot["horizon_s"] * GT_HZ)) for snapshot in e["gt_snapshots"]) for e in episodes)
    pose_available = all(e["start_pose"].get("available") for e in episodes)
    gt_available = all(snapshot.get("available") for e in episodes for snapshot in e["gt_snapshots"])
    checks = {
        "four_matched_branches_per_start": branches_ok,
        "exact_0.5_1_2_3s_horizon_clock": clocks_ok,
        "kinematic_relative_pose_integration": integration_ok,
        "gt_horizon_references_structural": source_refs_ok,
        "absolute_start_pose_available": pose_available,
        "OHVA_snapshots_available": gt_available,
    }
    return {"episode_count": len(episodes), "unique_starts": len(groups), "checks": checks, "passed": all(checks.values())}
