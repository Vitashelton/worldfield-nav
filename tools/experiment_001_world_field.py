#!/usr/bin/env python3
"""Experiment 001: persistent, world-aligned local field from Habitat-GS GT.

The simulator is Habitat's Y-up frame.  Thus the field's horizontal "XY"
coordinates are Habitat world X and Z; the height channel is Habitat world Y.
No estimated pose, visual matching, or learned component is used here.
"""

import json
import math
import subprocess
from pathlib import Path

import habitat_sim
import habitat_sim.agent
import magnum as mn
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path("/root/autodl-tmp/worldfield_nav")
SCENE_DIR = ROOT / "data/scene_datasets/gs_scenes/train/interior_0405_840145"
GS_PLY = SCENE_DIR / "interior_0405_840145.gs.ply"
NAVMESH = SCENE_DIR / "interior_0405_840145.navmesh"
OUT = ROOT / "outputs/exp001"
ENCODER = ROOT / "tools/raw_rgb_to_mp4"

T = 320
IMG_H = IMG_W = 256
HFOV_DEG = 90.0
FIELD_METERS = 10.0
FIELD_SIZE = 128
CELL = FIELD_METERS / FIELD_SIZE
DEPTH_STRIDE = 2
ANCHOR_WARMUP = 30
MAX_AGE = T
FPS = 12


def v3(value):
    return np.asarray([value[0], value[1], value[2]], dtype=np.float32)


def camera_spec(uuid, sensor_type):
    spec = habitat_sim.CameraSensorSpec()
    spec.uuid = uuid
    spec.sensor_type = sensor_type
    spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    spec.resolution = [IMG_H, IMG_W]
    spec.position = [0.0, 1.5, 0.0]
    spec.hfov = HFOV_DEG
    spec.near = 0.01
    spec.far = 10.0
    if sensor_type == habitat_sim.SensorType.DEPTH:
        spec.channels = 1
    return spec


def action_space():
    make = habitat_sim.agent.ActionSpec
    act = habitat_sim.agent.ActuationSpec
    return {
        "move_forward": make("move_forward", act(amount=0.07)),
        "turn_left": make("turn_left", act(amount=1.5)),
        "turn_right": make("turn_right", act(amount=1.5)),
    }


def deterministic_actions(count):
    # Every loop has translation, left and right turns, then more translation.
    loop = (["move_forward"] * 28 + ["turn_left"] * 20 +
            ["move_forward"] * 32 + ["turn_right"] * 36 +
            ["move_forward"] * 28 + ["turn_left"] * 16 +
            ["move_forward"] * 24)
    return (loop * math.ceil(count / len(loop)))[:count]


def sensor_c2w(sim):
    """Exact sensor-node transform, rather than assuming camera=agent pose."""
    node = sim._sensors["depth"].node
    transform = node.absolute_transformation()
    origin = v3(transform.transform_point(mn.Vector3(0.0, 0.0, 0.0)))
    axes = np.column_stack([
        v3(transform.transform_vector(mn.Vector3(1.0, 0.0, 0.0))),
        v3(transform.transform_vector(mn.Vector3(0.0, 1.0, 0.0))),
        v3(transform.transform_vector(mn.Vector3(0.0, 0.0, 1.0))),
    ])
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = axes
    c2w[:3, 3] = origin
    return c2w


def depth_to_world(depth, c2w, intrinsics):
    """Pinhole backprojection for Habitat depth (-camera-Z, in metres)."""
    fx, fy, cx, cy = intrinsics
    vv, uu = np.mgrid[0:IMG_H:DEPTH_STRIDE, 0:IMG_W:DEPTH_STRIDE]
    d = depth[::DEPTH_STRIDE, ::DEPTH_STRIDE]
    valid = np.isfinite(d) & (d > 0.02) & (d < 9.99)
    d, uu, vv = d[valid], uu[valid], vv[valid]
    # Habitat camera frame is +X right, +Y up, and looks along -Z.
    points_cam = np.column_stack(((uu - cx) * d / fx, -(vv - cy) * d / fy, -d))
    points_world = points_cam @ c2w[:3, :3].T + c2w[:3, 3]
    return points_world.astype(np.float32), points_cam.astype(np.float32)


def project_world(points_world, c2w, intrinsics):
    fx, fy, cx, cy = intrinsics
    points_cam = (points_world - c2w[:3, 3]) @ c2w[:3, :3]
    forward_depth = -points_cam[:, 2]
    safe = forward_depth > 0.02
    u = fx * points_cam[:, 0] / np.maximum(forward_depth, 1e-8) + cx
    v = cy - fy * points_cam[:, 1] / np.maximum(forward_depth, 1e-8)
    return u, v, forward_depth, safe, points_cam


class PersistentField:
    """Incremental sparse global grid; snapshots are world-aligned local crops."""

    def __init__(self):
        # key=(world X cell, world Z cell); value=[occupancy, height, last_seen, visits]
        self.cells = {}

    def update(self, points_world, frame):
        keys = np.floor(points_world[:, [0, 2]] / CELL).astype(np.int32)
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        counts = np.bincount(inverse)
        mean_height = np.bincount(inverse, weights=points_world[:, 1]) / counts
        updated = []
        for index, (ix, iz) in enumerate(unique):
            key = (int(ix), int(iz))
            previous = self.cells.get(key)
            if previous is None:
                # Repeated observations saturate rather than accumulating an ever-growing cloud.
                self.cells[key] = np.array([0.35, mean_height[index], frame, 1.0], dtype=np.float32)
            else:
                previous[0] = min(1.0, previous[0] + 0.18)
                previous[1] = 0.80 * previous[1] + 0.20 * mean_height[index]
                previous[2] = frame
                previous[3] += 1.0
            updated.append(key)
        return updated

    def snapshot(self, center_xz, frame):
        origin = center_xz - FIELD_METERS / 2.0
        phi = np.zeros((4, FIELD_SIZE, FIELD_SIZE), dtype=np.float32)
        phi[3].fill(MAX_AGE)  # explicit unobserved/unknown age sentinel
        ix0, iz0 = np.floor(origin / CELL).astype(np.int32)
        for (ix, iz), value in self.cells.items():
            x, z = ix - ix0, iz - iz0
            if 0 <= x < FIELD_SIZE and 0 <= z < FIELD_SIZE:
                row = FIELD_SIZE - 1 - z  # visualization convention: +world Z points up
                phi[0, row, x] = value[0]
                phi[1, row, x] = value[1]
                phi[2, row, x] = 1.0
                phi[3, row, x] = frame - value[2]
        return phi, origin.astype(np.float32)

    def occupied_keys(self):
        return {key for key, value in self.cells.items() if value[0] >= 0.5}


class RawVideoWriter:
    def __init__(self, path, width, height, fps=FPS):
        self.process = subprocess.Popen(
            [str(ENCODER), str(path), str(width), str(height), str(fps)], stdin=subprocess.PIPE
        )

    def append(self, frame):
        frame = np.ascontiguousarray(frame[..., :3].astype(np.uint8))
        self.process.stdin.write(frame.tobytes())

    def close(self):
        self.process.stdin.close()
        if self.process.wait() != 0:
            raise RuntimeError("MP4 encoder failed")


def draw_robot(image, center=(FIELD_SIZE // 2, FIELD_SIZE // 2)):
    draw = ImageDraw.Draw(image)
    x, y = center
    draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(255, 45, 45))


def field_rgb(phi, mode):
    observed = phi[2] > 0
    image = np.zeros((FIELD_SIZE, FIELD_SIZE, 3), dtype=np.uint8)
    if mode == "occupancy":
        image[:] = (20, 20, 24)
        image[observed] = (45, 70, 75)
        image[phi[0] >= 0.5] = (35, 220, 195)
    elif mode == "height":
        image[:] = (18, 18, 22)
        if np.any(observed):
            values = phi[1][observed]
            lo, hi = np.percentile(values, [5, 95])
            scaled = np.clip((phi[1] - lo) / max(hi - lo, 1e-5), 0, 1)
            image[..., 0] = (40 + 215 * scaled).astype(np.uint8)
            image[..., 1] = (30 + 140 * (1.0 - np.abs(scaled - 0.5) * 2)).astype(np.uint8)
            image[..., 2] = (230 - 190 * scaled).astype(np.uint8)
            image[~observed] = (18, 18, 22)
    elif mode == "visibility":
        image[:] = (15, 15, 18)
        image[observed] = (40, 210, 95)
    elif mode == "age":
        image[:] = (15, 15, 18)
        age = np.clip(phi[3] / 60.0, 0.0, 1.0)
        image[observed, 0] = (255 * age[observed]).astype(np.uint8)
        image[observed, 1] = (220 * (1.0 - age[observed])).astype(np.uint8)
        image[observed, 2] = 80
    pil = Image.fromarray(image).resize((256, 256), Image.Resampling.NEAREST)
    draw_robot(pil, (128, 128))
    return np.asarray(pil)


def depth_rgb(depth):
    scaled = np.clip(depth / 4.0, 0.0, 1.0)
    image = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    image[..., 0] = (255 * scaled).astype(np.uint8)
    image[..., 1] = (255 * (1.0 - np.abs(2 * scaled - 1))).astype(np.uint8)
    image[..., 2] = (255 * (1.0 - scaled)).astype(np.uint8)
    return image


def choose_anchors(accumulator, max_anchors=100):
    candidates = []
    for key, value in accumulator.items():
        if value[3] >= 3:
            candidates.append((int(value[3]), key, value[:3] / value[3]))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    chosen = []
    for _, _, xyz in candidates:
        if all(np.linalg.norm(xyz[[0, 2]] - old[[0, 2]]) >= 0.18 for old in chosen):
            chosen.append(xyz.astype(np.float32))
        if len(chosen) == max_anchors:
            break
    return np.asarray(chosen, dtype=np.float32)


def update_anchor_accumulator(accumulator, points_world):
    # 10 cm quantization gives repeatable static surface support across warmup frames.
    keys = np.floor(points_world / 0.10).astype(np.int32)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    sums = np.column_stack([np.bincount(inverse, weights=points_world[:, d]) for d in range(3)])
    for index, key in enumerate(unique):
        k = tuple(int(value) for value in key)
        record = accumulator.setdefault(k, np.zeros(4, dtype=np.float64))
        # Exactly one contribution per voxel per frame: this is a temporal
        # stability count, not a count of adjacent depth pixels.
        record[:3] += sums[index] / counts[index]
        record[3] += 1.0


def annotate_anchors(rgb, depth, anchors, c2w, intrinsics, colors):
    image = Image.fromarray(rgb[..., :3])
    draw = ImageDraw.Draw(image)
    u, v, projected_depth, forward, _ = project_world(anchors, c2w, intrinsics)
    visible = []
    for index in range(len(anchors)):
        if not forward[index]:
            continue
        px, py = int(round(u[index])), int(round(v[index]))
        if not (1 <= px < IMG_W - 1 and 1 <= py < IMG_H - 1):
            continue
        # Ground-truth visibility: projected anchor must agree with rendered depth.
        if abs(float(depth[py, px]) - float(projected_depth[index])) > 0.12:
            continue
        color = colors[index % len(colors)]
        draw.ellipse((px - 3, py - 3, px + 3, py + 3), outline=color, width=2)
        visible.append((index, px, py, projected_depth[index]))
    return np.asarray(image), visible


def save_topdown(path, poses, anchors, field):
    xz = poses[:, [0, 2]]
    points = [xz]
    if len(anchors):
        points.append(anchors[:, [0, 2]])
    for ix, iz in field.cells:
        points.append(np.array([[ix * CELL, iz * CELL]], dtype=np.float32))
    all_points = np.concatenate(points, axis=0)
    low, high = all_points.min(axis=0) - 0.5, all_points.max(axis=0) + 0.5
    span = max(float(np.max(high - low)), 1.0)
    canvas = Image.new("RGB", (900, 900), (18, 20, 24))
    draw = ImageDraw.Draw(canvas)
    def xy(point):
        return (70 + (point[0] - low[0]) / span * 760,
                830 - (point[1] - low[1]) / span * 760)
    for ix, iz in field.occupied_keys():
        x, y = xy(np.array([ix * CELL, iz * CELL]))
        draw.point((x, y), fill=(35, 130, 125))
    path_points = [xy(point) for point in xz]
    if len(path_points) > 1:
        draw.line(path_points, fill=(255, 205, 55), width=3)
    for point in anchors[::max(1, len(anchors) // 25)]:
        x, y = xy(point[[0, 2]])
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(255, 80, 160))
    draw.text((20, 20), "Experiment 001 — world X-Z top-down (Habitat Y-up)", fill=(235, 235, 235))
    canvas.save(path)


def main():
    if not ENCODER.is_file() or not GS_PLY.is_file() or not NAVMESH.is_file():
        raise RuntimeError("Experiment assets or local MP4 encoder are missing")
    OUT.mkdir(parents=True, exist_ok=True)

    fx = (IMG_W / 2.0) / math.tan(math.radians(HFOV_DEG) / 2.0)
    intrinsics = np.array([fx, fx, (IMG_W - 1) / 2.0, (IMG_H - 1) / 2.0], dtype=np.float32)
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = "NONE"
    sim_cfg.enable_physics = False
    sim_cfg.create_renderer = True
    sim_cfg.gpu_device_id = 0
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.height, agent_cfg.radius = 1.5, 0.1
    agent_cfg.sensor_specifications = [
        camera_spec("rgb", habitat_sim.SensorType.COLOR),
        camera_spec("depth", habitat_sim.SensorType.DEPTH),
    ]
    agent_cfg.action_space = action_space()
    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    try:
        sim.seed(17)
        helper = habitat_sim.RenderInstanceHelper(sim, use_xyzw_orientations=False)
        helper.add_instance(str(GS_PLY), semantic_id=0, scale=mn.Vector3(1.0, 1.0, 1.0))
        helper.set_world_poses(np.array([[0, 0, 0]], np.float32), np.array([[1, 0, 0, 0]], np.float32))
        if not sim.pathfinder.load_nav_mesh(str(NAVMESH)):
            raise RuntimeError("Navmesh did not load")
        agent = sim.initialize_agent(0)
        state = agent.get_state()
        state.position = sim.pathfinder.get_random_navigable_point()
        agent.set_state(state)

        field = PersistentField()
        actions = deterministic_actions(T - 1)
        rgbs, depths, agent_poses, sensor_poses, origins, fields = [], [], [], [], [], []
        raw_depth_change, coverage, ious, height_drifts = [], [], [], []
        anchor_accumulator, anchors = {}, None
        previous_depth, previous_occ = None, set()
        visible_anchor_errors, visible_anchor_count = [], []
        camera_roundtrip_max = 0.0

        for frame in range(T):
            obs = sim.get_sensor_observations() if frame == 0 else sim.step(actions[frame - 1])
            rgb = np.asarray(obs["rgb"])[..., :3].copy()
            depth = np.asarray(obs["depth"], dtype=np.float32).copy()
            c2w = sensor_c2w(sim)
            agent_state = agent.get_state()
            position = v3(agent_state.position)
            rotation = agent_state.rotation
            pose = np.array([position[0], position[1], position[2], rotation.w, rotation.x, rotation.y, rotation.z], dtype=np.float32)
            points_world, points_cam = depth_to_world(depth, c2w, intrinsics)
            updated = field.update(points_world, frame)
            if frame < ANCHOR_WARMUP:
                update_anchor_accumulator(anchor_accumulator, points_world)
            if frame == ANCHOR_WARMUP - 1:
                anchors = choose_anchors(anchor_accumulator)
                if len(anchors) < 50:
                    raise RuntimeError(f"Only {len(anchors)} stable anchors found in warmup")

            if len(points_cam):
                # Explicit camera->world->camera validation using the exact sensor transform.
                recovered = (points_world[:128] - c2w[:3, 3]) @ c2w[:3, :3]
                camera_roundtrip_max = max(camera_roundtrip_max, float(np.max(np.abs(recovered - points_cam[:128]))))
            phi, origin = field.snapshot(position[[0, 2]], frame)
            current_occ = field.occupied_keys()
            if frame:
                valid = (depth > 0.02) & (previous_depth > 0.02) & np.isfinite(depth) & np.isfinite(previous_depth)
                raw_depth_change.append(float(np.mean(np.abs(depth[valid] - previous_depth[valid]))))
                union = len(current_occ | previous_occ)
                ious.append(float(len(current_occ & previous_occ) / union) if union else 1.0)
            else:
                raw_depth_change.append(0.0)
                ious.append(1.0)
            # Per-update surface-height change on already-observed cells only.
            changes = []
            for key in updated:
                value = field.cells[key]
                if value[3] > 1:
                    changes.append(float(value[1]))
            height_drifts.append(float(np.std(changes)) if len(changes) > 1 else 0.0)
            coverage.append(float(len(field.cells) * CELL * CELL))
            rgbs.append(rgb); depths.append(depth); agent_poses.append(pose); sensor_poses.append(c2w)
            origins.append(origin); fields.append(phi)
            previous_depth, previous_occ = depth, current_occ

        anchors = np.asarray(anchors, dtype=np.float32)
        colors = [(255, 75, 75), (75, 220, 255), (255, 225, 60), (170, 90, 255), (70, 255, 145)]
        anchor_writer = RawVideoWriter(OUT / "anchor_correspondence.mp4", IMG_W, IMG_H)
        for rgb, depth, c2w in zip(rgbs, depths, sensor_poses):
            annotated, visible = annotate_anchors(rgb, depth, anchors, c2w, intrinsics, colors)
            anchor_writer.append(annotated)
            visible_anchor_count.append(len(visible))
            for index, px, py, expected_depth in visible:
                d = float(depth[py, px])
                point_cam = np.array([[(px - intrinsics[2]) * d / intrinsics[0],
                                       -(py - intrinsics[3]) * d / intrinsics[1], -d]], dtype=np.float32)
                reconstructed = point_cam @ c2w[:3, :3].T + c2w[:3, 3]
                visible_anchor_errors.append(float(np.linalg.norm(reconstructed[0] - anchors[index])))
        anchor_writer.close()

        fields = np.stack(fields).astype(np.float32)
        agent_poses = np.stack(agent_poses).astype(np.float32)
        sensor_poses = np.stack(sensor_poses).astype(np.float32)
        origins = np.stack(origins).astype(np.float32)
        rgb_array = np.stack(rgbs).astype(np.uint8)
        depth_array = np.stack(depths).astype(np.float32)
        timestamps = np.arange(T, dtype=np.float32) / FPS

        video_specs = [
            ("rgb.mp4", IMG_W, IMG_H, (frame for frame in rgb_array)),
            ("depth.mp4", IMG_W, IMG_H, (depth_rgb(frame) for frame in depth_array)),
            ("occupancy_field.mp4", 256, 256, (field_rgb(phi, "occupancy") for phi in fields)),
            ("height_field.mp4", 256, 256, (field_rgb(phi, "height") for phi in fields)),
            ("visibility_field.mp4", 256, 256, (field_rgb(phi, "visibility") for phi in fields)),
            ("age_field.mp4", 256, 256, (field_rgb(phi, "age") for phi in fields)),
        ]
        for filename, width, height, frames in video_specs:
            writer = RawVideoWriter(OUT / filename, width, height)
            for image in frames:
                writer.append(image)
            writer.close()

        final_tiles = [field_rgb(fields[-1], mode) for mode in ("occupancy", "height", "visibility", "age")]
        montage = Image.new("RGB", (512, 512))
        for index, tile in enumerate(final_tiles):
            montage.paste(Image.fromarray(tile), ((index % 2) * 256, (index // 2) * 256))
        montage.save(OUT / "field_final.png")
        save_topdown(OUT / "trajectory_topdown.png", agent_poses, anchors, field)

        metadata = {
            "field_plane": "Habitat world X-Z (called field XY); Habitat world Y is height",
            "field_meters": FIELD_METERS,
            "resolution": FIELD_SIZE,
            "cell_size_m": CELL,
            "intrinsics": {"fx": float(intrinsics[0]), "fy": float(intrinsics[1]), "cx": float(intrinsics[2]), "cy": float(intrinsics[3]), "hfov_deg": HFOV_DEG},
            "depth_semantics": "Habitat depth equals -camera-Z in metres",
            "trajectory_actions": actions,
        }
        np.savez_compressed(OUT / "field_sequence.npz", phi=fields, pose=agent_poses,
                            sensor_pose=sensor_poses, field_origin=origins, timestamps=timestamps,
                            anchors_world=anchors, metadata=json.dumps(metadata))
        np.savez_compressed(OUT / "trajectory_observations.npz", rgb=rgb_array, depth=depth_array,
                            pose=agent_poses, sensor_pose=sensor_poses, timestamps=timestamps,
                            actions=np.asarray(["initialize"] + actions))

        trajectory_length = float(np.linalg.norm(np.diff(agent_poses[:, :3], axis=0), axis=1).sum())
        metrics = {
            "experiment": "001_persistent_local_world_field_reconstruction",
            "frames": T,
            "field_tensor_shape": list(fields.shape),
            "field_range_m": [FIELD_METERS, FIELD_METERS],
            "field_resolution": [FIELD_SIZE, FIELD_SIZE],
            "cell_size_m": CELL,
            "trajectory_length_m": trajectory_length,
            "actions": {"forward": actions.count("move_forward"), "turn_left": actions.count("turn_left"), "turn_right": actions.count("turn_right")},
            "anchors": {"count": int(len(anchors)), "visible_observations": int(len(visible_anchor_errors)),
                        "mean_visible_per_frame": float(np.mean(visible_anchor_count)),
                        "world_error_mean_m": float(np.mean(visible_anchor_errors)),
                        "world_error_median_m": float(np.median(visible_anchor_errors)),
                        "world_error_p95_m": float(np.percentile(visible_anchor_errors, 95))},
            "raw_depth_change": {"mean_abs_m": float(np.mean(raw_depth_change[1:])), "series_m": raw_depth_change},
            "field_static_drift": {"height_update_std_mean_m": float(np.mean(height_drifts)), "height_update_std_series_m": height_drifts},
            "consecutive_field_iou": {"mean": float(np.mean(ious[1:])), "series": ious},
            "coverage_over_time": {"area_m2": coverage, "initial_m2": coverage[0], "final_m2": coverage[-1], "growth_m2": coverage[-1] - coverage[0]},
            "coordinate_validation": {"camera_world_camera_max_abs_error_m": camera_roundtrip_max,
                                      "anchor_visibility_depth_gated": True,
                                      "coordinate_error_detected": bool(camera_roundtrip_max > 1e-4)},
            "files": ["rgb.mp4", "depth.mp4", "occupancy_field.mp4", "height_field.mp4", "visibility_field.mp4", "age_field.mp4", "anchor_correspondence.mp4", "trajectory_topdown.png", "field_final.png", "metrics.json", "field_sequence.npz", "trajectory_observations.npz"],
        }
        (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(json.dumps(metrics, indent=2))
    finally:
        sim.close()


if __name__ == "__main__":
    main()
