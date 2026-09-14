#!/usr/bin/env python3
"""Render a small deterministic visual/spatial audit of installed GS scenes."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import habitat_sim
import habitat_sim.agent
import magnum as mn
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/formal/GoalPose/scene_spatial_audit"


def sensor() -> habitat_sim.CameraSensorSpec:
    s = habitat_sim.CameraSensorSpec()
    s.uuid = "rgb"; s.sensor_type = habitat_sim.SensorType.COLOR
    s.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    s.resolution = [256, 256]; s.position = [0.0, 1.5, 0.0]; s.hfov = 90.0
    return s


def make_sim(asset: Path) -> habitat_sim.Simulator:
    cfg = habitat_sim.SimulatorConfiguration(); cfg.scene_id = "NONE"; cfg.gpu_device_id = 0; cfg.create_renderer = True
    ac = habitat_sim.agent.AgentConfiguration(); ac.height = 1.5; ac.radius = .1; ac.sensor_specifications = [sensor()]
    sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [ac]))
    helper = habitat_sim.RenderInstanceHelper(sim, use_xyzw_orientations=False)
    helper.add_instance(asset_filepath=str(asset), semantic_id=0, scale=mn.Vector3(1.0, 1.0, 1.0))
    helper.set_world_poses(np.array([[0, 0, 0]], np.float32), np.array([[1, 0, 0, 0]], np.float32))
    nav = asset.with_suffix("").with_suffix(".navmesh")
    if not sim.pathfinder.load_nav_mesh(str(nav)): raise RuntimeError(f"missing navmesh: {nav}")
    return sim


def render(sim: habitat_sim.Simulator, agent, point: np.ndarray, yaw: float) -> np.ndarray:
    st = agent.get_state(); st.position = point
    st.rotation = habitat_sim.utils.common.quat_from_angle_axis(yaw, np.array([0., 1., 0.]))
    agent.set_state(st, reset_sensors=True)
    return np.asarray(sim.get_sensor_observations()["rgb"])[..., :3].copy()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    assets = sorted((ROOT / "data/scene_datasets/gs_scenes").glob("*/*/*.gs.ply"))
    rows = []
    for asset in assets:
        scene = asset.stem.replace(".gs", "")
        split = asset.parent.parent.name
        sim = make_sim(asset); agent = sim.initialize_agent(0)
        try:
            rng = np.random.default_rng(sum(map(ord, scene)))
            points = [np.asarray(sim.pathfinder.get_random_navigable_point(), np.float32) for _ in range(3)]
            tiles = []
            for point_id, point in enumerate(points):
                for yaw_id, yaw in enumerate(np.linspace(-math.pi, math.pi, 4, endpoint=False)):
                    tile = Image.fromarray(render(sim, agent, point, float(yaw))).resize((192, 192))
                    tiles.append((tile, point_id, yaw_id))
            sheet = Image.new("RGB", (4 * 192, 3 * 220 + 34), "#101820")
            draw = ImageDraw.Draw(sheet)
            bounds = sim.pathfinder.get_bounds()
            draw.text((8, 8), f"{scene} | {split} | 3 NavMesh samples × 4 yaw views", fill="white")
            for index, (tile, point_id, yaw_id) in enumerate(tiles):
                x, y = (index % 4) * 192, 34 + (index // 4) * 220
                sheet.paste(tile, (x, y))
                draw.text((x + 4, y + 194), f"p{point_id} yaw{yaw_id}", fill="white")
            path = OUT / f"{scene}.png"; sheet.save(path)
            rows.append({
                "scene_id": scene, "split": split, "asset": str(asset.relative_to(ROOT)),
                "navmesh_bounds_min": list(map(float, bounds[0])), "navmesh_bounds_max": list(map(float, bounds[1])),
                "dynamic_avatar_available": scene == "scene09",
                "contact_sheet": str(path.relative_to(ROOT)),
            })
        finally:
            sim.close()
    (OUT / "scene_inventory.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps({"scenes": len(rows), "out": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
