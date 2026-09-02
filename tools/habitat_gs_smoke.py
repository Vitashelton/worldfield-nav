#!/usr/bin/env python3
"""Standalone, single-scene Habitat-GS RGB/depth smoke test."""

import json
import math
import subprocess
import time
from pathlib import Path

import habitat_sim
import habitat_sim.agent
import magnum as mn
import numpy as np


ROOT = Path("/root/autodl-tmp/worldfield_nav")
SCENE_DIR = ROOT / "data/scene_datasets/gs_scenes/train/interior_0405_840145"
GS_PLY = SCENE_DIR / "interior_0405_840145.gs.ply"
NAVMESH = SCENE_DIR / "interior_0405_840145.navmesh"
OUT = ROOT / "logs/habitat_gs_smoke_metrics.json"
FRAMES = 500


def gpu_memory_mib() -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    return int(result.stdout.splitlines()[0].strip())


def camera(uuid: str, sensor_type: habitat_sim.SensorType) -> habitat_sim.CameraSensorSpec:
    spec = habitat_sim.CameraSensorSpec()
    spec.uuid = uuid
    spec.sensor_type = sensor_type
    spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    spec.resolution = [256, 256]
    spec.position = [0.0, 1.5, 0.0]
    spec.hfov = 90.0
    if sensor_type == habitat_sim.SensorType.DEPTH:
        spec.channels = 1
    return spec


def make_action_space():
    return {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.07)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=1.5)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=1.5)
        ),
    }


def main() -> None:
    if not GS_PLY.is_file() or not NAVMESH.is_file():
        raise FileNotFoundError(f"Expected scene assets under {SCENE_DIR}")

    initial_memory = gpu_memory_mib()
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = "NONE"
    sim_cfg.enable_physics = False
    sim_cfg.create_renderer = True
    sim_cfg.gpu_device_id = 0

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.height = 1.5
    agent_cfg.radius = 0.1
    agent_cfg.sensor_specifications = [
        camera("rgb", habitat_sim.SensorType.COLOR),
        camera("depth", habitat_sim.SensorType.DEPTH),
    ]
    agent_cfg.action_space = make_action_space()

    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    try:
        helper = habitat_sim.RenderInstanceHelper(sim, use_xyzw_orientations=False)
        instance_id = helper.add_instance(
            asset_filepath=str(GS_PLY), semantic_id=0, scale=mn.Vector3(1.0, 1.0, 1.0)
        )
        helper.set_world_poses(
            np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        )
        navmesh_loaded = sim.pathfinder.load_nav_mesh(str(NAVMESH))
        if not navmesh_loaded:
            raise RuntimeError("Failed to load the supplied navmesh")

        agent = sim.initialize_agent(0)
        state = agent.get_state()
        state.position = sim.pathfinder.get_random_navigable_point()
        agent.set_state(state)

        # Warm up renderer and establish the observed output schema.
        obs = sim.get_sensor_observations()
        if set(obs) != {"rgb", "depth"}:
            raise RuntimeError(f"Unexpected sensor keys: {list(obs)}")
        peak_memory = gpu_memory_mib()
        frame_times = []
        nan_inf_frames = 0
        invalid_pose_frames = 0
        actions = ("move_forward", "turn_left", "move_forward", "turn_right")

        for frame in range(FRAMES):
            begin = time.perf_counter()
            obs = sim.step(actions[frame % len(actions)])
            frame_times.append(time.perf_counter() - begin)
            rgb, depth = obs["rgb"], obs["depth"]
            if not np.isfinite(rgb).all() or not np.isfinite(depth).all():
                nan_inf_frames += 1
            pose = agent.get_state()
            position = np.asarray(pose.position)
            rotation = np.asarray([pose.rotation.w, pose.rotation.x, pose.rotation.y, pose.rotation.z])
            if not np.isfinite(position).all() or not np.isfinite(rotation).all():
                invalid_pose_frames += 1
            if frame % 10 == 0:
                peak_memory = max(peak_memory, gpu_memory_mib())

        peak_memory = max(peak_memory, gpu_memory_mib())
        metrics = {
            "scene": str(GS_PLY),
            "navmesh": str(NAVMESH),
            "render_instance_id": int(instance_id),
            "resolution": [256, 256],
            "agents": 1,
            "frames": FRAMES,
            "rgb_shape": list(obs["rgb"].shape),
            "rgb_dtype": str(obs["rgb"].dtype),
            "rgb_min": int(np.min(obs["rgb"])),
            "rgb_max": int(np.max(obs["rgb"])),
            "depth_shape": list(obs["depth"].shape),
            "depth_dtype": str(obs["depth"].dtype),
            "depth_min": float(np.min(obs["depth"])),
            "depth_max": float(np.max(obs["depth"])),
            "navmesh_loaded": bool(navmesh_loaded),
            "nan_inf_frames": nan_inf_frames,
            "invalid_pose_frames": invalid_pose_frames,
            "mean_fps": FRAMES / sum(frame_times),
            "mean_frame_ms": 1000.0 * float(np.mean(frame_times)),
            "p95_frame_ms": 1000.0 * float(np.percentile(frame_times, 95)),
            "gpu_memory_initial_mib": initial_memory,
            "gpu_memory_peak_mib": peak_memory,
            "stable": nan_inf_frames == 0 and invalid_pose_frames == 0,
        }
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(metrics, indent=2) + "\n")
        print(json.dumps(metrics, indent=2))
    finally:
        sim.close()


if __name__ == "__main__":
    main()
