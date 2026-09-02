#!/usr/bin/env python3
"""B1 P0: select a 10-scene pilot split with a measured revelation probe.

This is deliberately a data-harness probe, not model training or B1 P1.
It samples navigable starts, labels them from deterministic depth geometry,
executes four equal-duration / equal-forward-budget branches, and measures
initially-unknown -> future-observed metric area at the required horizons.
"""
from __future__ import annotations

import copy
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import habitat_sim
import habitat_sim.agent
import magnum as mn
import matplotlib.pyplot as plt
import numpy as np
import quaternion
import yaml


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/scene_datasets/gs_scenes"
OUT = ROOT / "outputs/formal/B1/p0"
FIGURES = ROOT / "paper_assets/figures"
TABLES = ROOT / "paper_assets/tables"
CONFIG = ROOT / "configs/benchmark/b1_pilot.yaml"
REGISTRY = ROOT / "experiments/registry.yaml"
RESULTS = ROOT / "docs/results/B1_P0_RESULTS.md"

FIELD_M, GRID, CELL_M = 10.0, 128, 10.0 / 128
WIDTH = HEIGHT = 192
STRIDE, HFOV = 3, 90.0
DT, HZ, TICKS = 0.2, 5, 15
HORIZONS = {0.5: 2, 1.0: 5, 2.0: 10, 3.0: 15}
V_MPS, TURN_DEG = 0.35, 3.0
FAMILIES = ("straight", "left", "right", "mixed_turn")
CONTEXTS = ("open_space", "turn_corner", "doorway_occlusion")
SPLITS = {
    "train": ["scene01", "scene02", "scene03", "scene09", "interior_0405_840145"],
    "validation": ["scene04", "scene05"],
    "unseen_test": ["scene56", "scene57", "scene58"],
}


def scene_split(scene: str) -> str:
    return "val" if scene in {"scene56", "scene57", "scene58"} else "train"


def sensor() -> habitat_sim.CameraSensorSpec:
    spec = habitat_sim.CameraSensorSpec()
    spec.uuid = "depth"
    spec.sensor_type = habitat_sim.SensorType.DEPTH
    spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    spec.resolution = [HEIGHT, WIDTH]
    spec.position = [0.0, 1.5, 0.0]
    spec.hfov = HFOV
    spec.near, spec.far = 0.01, 10.0
    return spec


def action_space() -> dict[str, habitat_sim.agent.ActionSpec]:
    spec, act = habitat_sim.agent.ActionSpec, habitat_sim.agent.ActuationSpec
    return {
        "move_forward": spec("move_forward", act(amount=V_MPS * DT)),
        "turn_left": spec("turn_left", act(amount=TURN_DEG)),
        "turn_right": spec("turn_right", act(amount=TURN_DEG)),
    }


def make_sim(scene: str) -> habitat_sim.Simulator:
    split = scene_split(scene)
    directory = DATA / split / scene
    gs, nav = directory / f"{scene}.gs.ply", directory / f"{scene}.navmesh"
    if not gs.is_file() or not nav.is_file():
        raise FileNotFoundError(f"Missing authorized P0 assets for {scene}: {gs}, {nav}")
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id, cfg.enable_physics, cfg.create_renderer, cfg.gpu_device_id = "NONE", False, True, 0
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.height, agent_cfg.radius = 1.5, 0.1
    agent_cfg.sensor_specifications, agent_cfg.action_space = [sensor()], action_space()
    sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent_cfg]))
    helper = habitat_sim.RenderInstanceHelper(sim, use_xyzw_orientations=False)
    helper.add_instance(str(gs), semantic_id=0, scale=mn.Vector3(1.0, 1.0, 1.0))
    helper.set_world_poses(np.array([[0, 0, 0]], np.float32), np.array([[1, 0, 0, 0]], np.float32))
    if not sim.pathfinder.load_nav_mesh(str(nav)):
        sim.close()
        raise RuntimeError(f"Could not load navmesh: {nav}")
    return sim


def vector(value: Any) -> np.ndarray:
    return np.asarray([value[0], value[1], value[2]], np.float32)


def camera_to_world(sim: habitat_sim.Simulator) -> np.ndarray:
    transform = sim._sensors["depth"].node.absolute_transformation()
    axes = np.column_stack([
        vector(transform.transform_vector(mn.Vector3(1, 0, 0))),
        vector(transform.transform_vector(mn.Vector3(0, 1, 0))),
        vector(transform.transform_vector(mn.Vector3(0, 0, 1))),
    ])
    output = np.eye(4, dtype=np.float32)
    output[:3, :3] = axes
    output[:3, 3] = vector(transform.transform_point(mn.Vector3(0, 0, 0)))
    return output


def depth_cells(depth: np.ndarray, c2w: np.ndarray, intrinsics: np.ndarray) -> set[tuple[int, int]]:
    fx, fy, cx, cy = intrinsics
    vv, uu = np.mgrid[0:HEIGHT:STRIDE, 0:WIDTH:STRIDE]
    sampled = depth[::STRIDE, ::STRIDE]
    valid = np.isfinite(sampled) & (sampled > 0.02) & (sampled < 9.99)
    d, u, v = sampled[valid], uu[valid], vv[valid]
    camera = np.column_stack(((u - cx) * d / fx, -(v - cy) * d / fy, -d))
    world = camera @ c2w[:3, :3].T + c2w[:3, 3]
    cells = np.floor(world[:, [0, 2]] / CELL_M).astype(np.int32)
    return {(int(x), int(z)) for x, z in cells}


def observe(sim: habitat_sim.Simulator, intrinsics: np.ndarray) -> tuple[np.ndarray, set[tuple[int, int]]]:
    depth = np.asarray(sim.get_sensor_observations()["depth"], np.float32)
    return depth, depth_cells(depth, camera_to_world(sim), intrinsics)


def pose_xz(agent: habitat_sim.Agent) -> np.ndarray:
    p = agent.get_state().position
    return np.asarray([p[0], p[2]], np.float32)


def context_from_depth(depth: np.ndarray) -> tuple[str, dict[str, float]]:
    """Deterministic proxy labels; semantic room labels are intentionally not used."""
    valid = np.where(np.isfinite(depth) & (depth > 0.05), depth, 10.0)
    left, center, right = np.array_split(valid, 3, axis=1)
    lm, cm, rm = (float(np.median(x)) for x in (left, center, right))
    far = float(np.mean(valid > 4.0))
    asym = abs(lm - rm) / max(0.1, lm + rm)
    # Strong view-dependent discontinuity with an open center is a doorway/
    # occlusion proxy; left-right asymmetric close geometry is a corner proxy.
    discontinuity = float(np.mean(np.abs(np.diff(valid, axis=1)) > 0.75))
    scores = {
        "open_space": far + 0.15 * min(lm, cm, rm) / 10.0,
        "turn_corner": asym + 0.25 * (1.0 - far),
        "doorway_occlusion": discontinuity * 12.0 + max(0.0, cm - min(lm, rm)) / 4.0,
    }
    return max(scores, key=scores.get), {
        "median_left_m": lm, "median_center_m": cm, "median_right_m": rm,
        "far_fraction": far, "left_right_asymmetry": asym,
        "depth_discontinuity_fraction": discontinuity, **{f"score_{k}": v for k, v in scores.items()},
    }


def branch_commands(family: str) -> list[tuple[str | None, str]]:
    commands = []
    for tick in range(TICKS):
        turn: str | None = None
        if family == "left":
            turn = "turn_left"
        elif family == "right":
            turn = "turn_right"
        elif family == "mixed_turn":
            turn = "turn_left" if tick < TICKS // 2 else "turn_right"
        commands.append((turn, "move_forward"))
    return commands


def continuous_controls(family: str) -> list[dict[str, float]]:
    omega = math.radians(TURN_DEG) / DT
    controls = []
    for tick in range(TICKS):
        sign = 0.0
        if family == "left": sign = 1.0
        elif family == "right": sign = -1.0
        elif family == "mixed_turn": sign = 1.0 if tick < TICKS // 2 else -1.0
        controls.append({"v_mps": V_MPS, "omega_radps": sign * omega, "dt_s": DT})
    return controls


def rollout(sim: habitat_sim.Simulator, agent: habitat_sim.Agent, start: Any, family: str,
            initial_cells: set[tuple[int, int]], intrinsics: np.ndarray) -> dict[str, float]:
    agent.set_state(copy.deepcopy(start))
    observed = set(initial_cells)
    previous, length = pose_xz(agent), 0.0
    measured: dict[float, float] = {}
    for tick, (turn, forward) in enumerate(branch_commands(family), start=1):
        if turn is not None:
            sim.step(turn)
        sim.step(forward)
        current = pose_xz(agent)
        length += float(np.linalg.norm(current - previous))
        previous = current
        _, cells = observe(sim, intrinsics)
        observed.update(cells)
        for horizon, end_tick in HORIZONS.items():
            if tick == end_tick:
                measured[horizon] = len(observed - initial_cells) * CELL_M * CELL_M
    return {f"revelation_m2_{horizon:g}s": measured[horizon] for horizon in HORIZONS} | {
        "requested_translation_m": V_MPS * DT * TICKS,
        "realized_translation_m": length,
        "translation_ratio": length / (V_MPS * DT * TICKS),
    }


def row_writer(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows: return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, np.float64), q)) if values else 0.0


def make_figures(rows: list[dict[str, Any]], threshold: float) -> list[str]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    values = np.asarray([float(row["revelation_m2_3s"]) for row in rows])
    paths: list[str] = []
    def save(name: str) -> None:
        path = FIGURES / name; plt.tight_layout(); plt.savefig(path, dpi=180); plt.close(); paths.append(str(path.relative_to(ROOT)))
    # 1 distribution: small multiples by scene, colored context and action.
    scenes = sorted({str(row["scene_id"]) for row in rows})
    fig, axes = plt.subplots(2, 5, figsize=(16, 6), sharex=True, sharey=True)
    for ax, scene in zip(axes.flat, scenes):
        for context, color in zip(CONTEXTS, ["#377eb8", "#ff7f00", "#4daf4a"]):
            for family, marker in zip(FAMILIES, ["o", "^", "s", "D"]):
                x = [float(r["revelation_m2_3s"]) for r in rows if r["scene_id"] == scene and r["context"] == context and r["action_family"] == family]
                if x: ax.scatter([context] * len(x), x, c=color, marker=marker, alpha=.58, s=15)
        ax.axhline(threshold, color="crimson", ls="--", lw=1); ax.set_title(scene); ax.tick_params(axis="x", labelrotation=25, labelsize=7)
    fig.supylabel("newly observed area at 3 s (m²)"); fig.suptitle("B1 P0 revelation area by scene / context / action")
    save("b1_p0_revelation_distribution.png")
    # 2 CDF.
    plt.figure(figsize=(8, 5))
    for context, color in zip(CONTEXTS, ["#377eb8", "#ff7f00", "#4daf4a"]):
        x = np.sort([float(r["revelation_m2_3s"]) for r in rows if r["context"] == context])
        if len(x): plt.step(x, np.arange(1, len(x) + 1) / len(x), where="post", label=context, color=color)
    plt.axvline(threshold, color="crimson", ls="--", label=f"frozen threshold={threshold:.4f} m²")
    plt.xlabel("newly observed area at 3 s (m²)"); plt.ylabel("empirical CDF"); plt.legend(); plt.title("B1 P0 revelation CDF")
    save("b1_p0_revelation_cdf.png")
    # 3 action-family boxplot.
    plt.figure(figsize=(8, 5))
    data = [[float(r["revelation_m2_3s"]) for r in rows if r["action_family"] == family] for family in FAMILIES]
    plt.boxplot(data, tick_labels=FAMILIES, showfliers=True)
    plt.axhline(threshold, color="crimson", ls="--", label="frozen threshold"); plt.ylabel("newly observed area at 3 s (m²)"); plt.legend(); plt.title("Matched action-family revelation distribution")
    save("b1_p0_action_family_boxplot.png")
    # 4 threshold selection.
    positive = np.sort(values[values > 0]); q25 = percentile(list(positive), 25)
    plt.figure(figsize=(8, 5)); plt.hist(values, bins=35, color="#6a9fb5", alpha=.8)
    plt.axvline(CELL_M * CELL_M * 4, color="#555555", ls=":", label="4-cell floor")
    plt.axvline(q25, color="#ff7f00", ls="--", label="positive-area 25th percentile")
    plt.axvline(threshold, color="crimson", lw=2, label="frozen threshold")
    plt.xlabel("newly observed area at 3 s (m²)"); plt.ylabel("probe branch count"); plt.legend(); plt.title("Quantitative revelation-threshold selection")
    save("b1_p0_threshold_selection.png")
    # 5 split overview.
    plt.figure(figsize=(11, 3.8)); ax = plt.gca(); colors = {"train":"#377eb8", "validation":"#ff7f00", "unseen_test":"#4daf4a"}
    x = 0
    for split, scene_ids in SPLITS.items():
        for scene in scene_ids:
            ax.barh(0, 1, left=x, color=colors[split], edgecolor="white"); ax.text(x+.5, 0, scene, ha="center", va="center", fontsize=8, rotation=35); x += 1
    ax.set_xlim(0, 10); ax.set_yticks([]); ax.set_xticks([]); ax.set_title("B1 pilot 10-scene split (no scene overlap)")
    ax.legend(handles=[plt.Rectangle((0,0),1,1,color=colors[s],label=s) for s in SPLITS], loc="upper center", ncol=3)
    save("b1_p0_split_overview.png")
    return paths


def update_contract_outputs(summary: dict[str, Any], threshold: float, figure_paths: list[str]) -> None:
    with CONFIG.open() as handle: config = yaml.safe_load(handle)
    config.update({
        "status": "p0_frozen_p1_not_authorized", "splits": {**SPLITS, "selection_rule": "P0 fixed split; official Habitat-GS assets only."},
        "episodes": {"contexts": list(CONTEXTS), "branches": list(FAMILIES), "matched_counterfactual": True, "candidate_starts_per_scene_target": 30,
                     "action_controls": {"representation": "continuous (v, omega, dt)", "dt_s": DT, "sample_hz": HZ, "duration_s": TICKS*DT, "requested_translation_m": V_MPS*DT*TICKS,
                                         "families_are_labels_only": True, "execution_note": "Habitat discrete turn+forward proxy; controls and realized motion are saved."}},
        "horizons_s": list(HORIZONS),
        "revelation_probe": {"metric": "initially unknown -> observed union area at 3 s", "threshold_m2": threshold, "rule": "max(4 field cells, 25th percentile of positive P0 branch areas)", "eligibility": "open: valid; corner/doorway: every matched family >= threshold", "summary": summary, "figures": figure_paths},
        "outputs": {"root": "outputs/formal/B1", "paper_assets_root": "paper_assets", "resumable": True, "avoid_duplicate_rgb_depth": True},
    })
    CONFIG.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    with REGISTRY.open() as handle: registry = yaml.safe_load(handle)
    b1 = registry["formal"]["B1"]
    b1.update({"status": "p0_completed_p1_not_authorized", "p0_outputs": {"metrics": "outputs/formal/B1/p0/metrics.json", "tables": ["paper_assets/tables/b1_p0_revelation_candidates.csv", "paper_assets/tables/b1_p0_summary.csv"], "figures": figure_paths}, "p0_acceptance": summary["acceptance"]})
    REGISTRY.write_text(yaml.safe_dump(registry, sort_keys=False, allow_unicode=True))
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text("# B1 P0 — Revelation Probe Results\n\n"
        f"- Candidate starts attempted: {summary['candidate_starts']} across 10 fixed scenes.\n"
        f"- Branch probes: {summary['branch_probes']}; each accepted candidate ran all four fixed-duration matched families.\n"
        f"- Frozen 3 s threshold: {threshold:.6f} m², max of four cells ({4*CELL_M*CELL_M:.6f} m²) and the positive-area 25th percentile.\n"
        f"- Eligible starts after threshold: {summary['eligible_starts']}.\n"
        f"- P0 acceptance: **{summary['acceptance']['passed']}**. P1 remains unauthorized.\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); TABLES.mkdir(parents=True, exist_ok=True)
    fx = (WIDTH / 2) / math.tan(math.radians(HFOV) / 2)
    intrinsics = np.asarray([fx, fx, (WIDTH-1)/2, (HEIGHT-1)/2], np.float32)
    rng = np.random.default_rng(20260902)
    rows: list[dict[str, Any]] = []; starts: list[dict[str, Any]] = []; attempts_by_scene: dict[str, int] = {}
    for scene in [scene for split in SPLITS.values() for scene in split]:
        sim = make_sim(scene); sim.seed(int(rng.integers(0, 2**31-1))); agent = sim.initialize_agent(0)
        chosen: dict[str, list[tuple[Any, dict[str, float]]]] = {context: [] for context in CONTEXTS}
        attempts, maximum_attempts = 0, 600
        while attempts < maximum_attempts and sum(len(x) for x in chosen.values()) < 30:
            attempts += 1
            state = agent.get_state(); state.position = sim.pathfinder.get_random_navigable_point()
            state.rotation = quaternion.from_rotation_vector([0.0, float(rng.uniform(-math.pi, math.pi)), 0.0])
            agent.set_state(state); depth, initial = observe(sim, intrinsics); context, features = context_from_depth(depth)
            if len(chosen[context]) < 10: chosen[context].append((copy.deepcopy(state), features))
        attempts_by_scene[scene] = attempts
        candidate_id = 0
        for context in CONTEXTS:
            for state, features in chosen[context]:
                candidate_id += 1; agent.set_state(copy.deepcopy(state)); _, initial = observe(sim, intrinsics)
                start_row = {"scene_id": scene, "split": next(k for k,v in SPLITS.items() if scene in v), "candidate_id": candidate_id, "context": context, "initial_observed_cells": len(initial), "sampling_attempts_until_selection": attempts, **features}
                starts.append(start_row)
                for family in FAMILIES:
                    measured = rollout(sim, agent, state, family, initial, intrinsics)
                    rows.append({**start_row, "action_family": family, **measured, "continuous_control": json.dumps(continuous_controls(family), separators=(",", ":")), "control_representation": "(v,omega,dt)", "duration_s": TICKS*DT})
        sim.close()
        print(f"{scene}: selected {candidate_id}/30 candidates after {attempts} sampling attempts", flush=True)
    positive = [float(row["revelation_m2_3s"]) for row in rows if float(row["revelation_m2_3s"]) > 0]
    threshold = max(4 * CELL_M * CELL_M, percentile(positive, 25))
    eligible: dict[tuple[str, int], bool] = {}
    for start in starts:
        key = (start["scene_id"], int(start["candidate_id"]))
        branch = [r for r in rows if r["scene_id"] == key[0] and r["candidate_id"] == key[1]]
        eligible[key] = start["context"] == "open_space" or (len(branch) == 4 and min(float(r["revelation_m2_3s"]) for r in branch) >= threshold)
    for row in rows: row["eligible_after_threshold"] = eligible[(row["scene_id"], int(row["candidate_id"]))]
    for start in starts: start["eligible_after_threshold"] = eligible[(start["scene_id"], int(start["candidate_id"]))]
    by_context = {c: {"count": sum(r["context"] == c for r in rows), "median_m2": percentile([float(r["revelation_m2_3s"]) for r in rows if r["context"] == c], 50), "p25_m2": percentile([float(r["revelation_m2_3s"]) for r in rows if r["context"] == c], 25), "p75_m2": percentile([float(r["revelation_m2_3s"]) for r in rows if r["context"] == c], 75)} for c in CONTEXTS}
    by_family = {f: {"count": sum(r["action_family"] == f for r in rows), "median_m2": percentile([float(r["revelation_m2_3s"]) for r in rows if r["action_family"] == f], 50), "p25_m2": percentile([float(r["revelation_m2_3s"]) for r in rows if r["action_family"] == f], 25), "p75_m2": percentile([float(r["revelation_m2_3s"]) for r in rows if r["action_family"] == f], 75)} for f in FAMILIES}
    selected_per_scene = Counter(s["scene_id"] for s in starts)
    context_per_scene = {scene: dict(Counter(s["context"] for s in starts if s["scene_id"] == scene)) for scene in selected_per_scene}
    summary: dict[str, Any] = {"benchmark": "B1 P0 revelation probe", "sample_hz": HZ, "horizons_s": list(HORIZONS), "candidate_starts": len(starts), "branch_probes": len(rows), "candidates_per_scene": dict(selected_per_scene), "sampling_attempts_by_scene": attempts_by_scene, "contexts_per_scene": context_per_scene, "by_context": by_context, "by_action_family": by_family, "eligible_starts": sum(eligible.values()), "eligible_by_context": dict(Counter(s["context"] for s in starts if s["eligible_after_threshold"])), "threshold_m2": threshold, "control": {"duration_s": TICKS*DT, "dt_s": DT, "requested_translation_m": V_MPS*DT*TICKS, "families": list(FAMILIES)}, "acceptance": {"at_least_30_candidates_attempted_each_scene": all(attempts_by_scene.get(scene, 0) >= 30 for scene_ids in SPLITS.values() for scene in scene_ids), "all_candidates_have_four_branches": len(rows) == 4 * len(starts), "all_contexts_covered": set().union(*(set(v) for v in context_per_scene.values())) == set(CONTEXTS), "passed": False}}
    # Threshold eligibility must retain real turning/occlusion candidates, not merely open starts.
    summary["acceptance"]["eligible_corner_and_doorway"] = all(summary["eligible_by_context"].get(context, 0) > 0 for context in ("turn_corner", "doorway_occlusion"))
    summary["acceptance"]["passed"] = all(
        value for key, value in summary["acceptance"].items() if key != "passed"
    )
    figure_paths = make_figures(rows, threshold)
    row_writer(OUT / "candidates.csv", rows); row_writer(TABLES / "b1_p0_revelation_candidates.csv", rows)
    summary_rows = [{"group_type":"context", "group": k, **v} for k,v in by_context.items()] + [{"group_type":"action_family", "group": k, **v} for k,v in by_family.items()]
    row_writer(TABLES / "b1_p0_summary.csv", summary_rows)
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    update_contract_outputs(summary, threshold, figure_paths)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
