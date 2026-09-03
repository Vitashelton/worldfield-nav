"""Physical cross-view correspondence construction without appearance labels."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Correspondence:
    source_patch: int
    target_patch: int
    source_uv: tuple[float, float]
    target_uv: tuple[float, float]
    world_xyz: np.ndarray
    projected_depth_m: float
    target_depth_m: float
    depth_residual_m: float
    world_residual_m: float
    lidar_residual_m: float | None


def intrinsics(width: int = 192, height: int = 192, hfov_deg: float = 90.0) -> np.ndarray:
    fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return np.asarray([fx, fx, (width - 1) / 2.0, (height - 1) / 2.0], dtype=np.float32)


def backproject_pixel(depth: float, u: float, v: float, c2w: np.ndarray, k: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = k
    cam = np.asarray([(u - cx) * depth / fx, -(v - cy) * depth / fy, -depth], dtype=np.float32)
    return cam @ c2w[:3, :3].T + c2w[:3, 3]


def project_world(world_xyz: np.ndarray, c2w: np.ndarray, k: np.ndarray) -> tuple[float, float, float]:
    cam = (world_xyz - c2w[:3, 3]) @ c2w[:3, :3]
    depth = float(-cam[2])
    fx, fy, cx, cy = k
    return float(fx * cam[0] / max(depth, 1e-8) + cx), float(cy - fy * cam[1] / max(depth, 1e-8)), depth


def patch_center(patch_index: int, grid_h: int, grid_w: int, image_h: int = 192, image_w: int = 192) -> tuple[float, float]:
    row, col = divmod(patch_index, grid_w)
    return (col + 0.5) * image_w / grid_w, (row + 0.5) * image_h / grid_h


def patch_index_from_uv(u: float, v: float, grid_h: int, grid_w: int, image_h: int = 192, image_w: int = 192) -> int | None:
    col, row = int(u * grid_w / image_w), int(v * grid_h / image_h)
    if not (0 <= row < grid_h and 0 <= col < grid_w):
        return None
    return row * grid_w + col


def depth_at(depth: np.ndarray, u: float, v: float) -> tuple[float, int, int] | None:
    x, y = int(round(u)), int(round(v))
    if not (0 <= y < depth.shape[0] and 0 <= x < depth.shape[1]):
        return None
    value = float(depth[y, x])
    if not np.isfinite(value) or not (0.02 < value < 9.99):
        return None
    return value, x, y


def mine_correspondences(
    source_depth: np.ndarray,
    source_c2w: np.ndarray,
    target_depth: np.ndarray,
    target_c2w: np.ndarray,
    grid_h: int,
    grid_w: int,
    depth_threshold_m: float,
    world_threshold_m: float,
    target_lidar_xyz: np.ndarray | None = None,
    target_lidar_count: int | None = None,
) -> list[Correspondence]:
    """Return only metric/visibility-valid positive correspondences."""
    k = intrinsics(source_depth.shape[1], source_depth.shape[0])
    output: list[Correspondence] = []
    lidar = None
    if target_lidar_xyz is not None and target_lidar_count:
        lidar = target_lidar_xyz[:target_lidar_count]
    for source_patch in range(grid_h * grid_w):
        u, v = patch_center(source_patch, grid_h, grid_w, source_depth.shape[0], source_depth.shape[1])
        sample = depth_at(source_depth, u, v)
        if sample is None:
            continue
        source_d, sx, sy = sample
        world = backproject_pixel(source_d, sx, sy, source_c2w, k)
        tu, tv, projected_d = project_world(world, target_c2w, k)
        if projected_d <= 0.02:
            continue
        target_sample = depth_at(target_depth, tu, tv)
        target_patch = patch_index_from_uv(tu, tv, grid_h, grid_w, source_depth.shape[0], source_depth.shape[1])
        if target_sample is None or target_patch is None:
            continue
        target_d, tx, ty = target_sample
        depth_residual = abs(target_d - projected_d)
        # A closer rendered surface is an occluder; both it and depth mismatch reject the pair.
        if depth_residual > depth_threshold_m or target_d + depth_threshold_m < projected_d:
            continue
        reconstructed = backproject_pixel(target_d, tx, ty, target_c2w, k)
        world_residual = float(np.linalg.norm(reconstructed - world))
        if world_residual > world_threshold_m:
            continue
        lidar_residual = None
        if lidar is not None and len(lidar):
            lidar_residual = float(np.min(np.linalg.norm(lidar - world[None], axis=1)))
        output.append(Correspondence(source_patch, target_patch, (float(sx), float(sy)), (float(tx), float(ty)), world, projected_d, target_d, depth_residual, world_residual, lidar_residual))
    return output
