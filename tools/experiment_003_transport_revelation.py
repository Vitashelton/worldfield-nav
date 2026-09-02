#!/usr/bin/env python3
"""Exp003: deterministic transport vs. action-dependent information revelation.

This reads only the inherited Exp002 snapshots.  T is built from Phi_0 and
the recorded robot displacement; future Phi is opened only after T exists to
measure R = Phi_GT - T.  The X-Z world field is globally aligned, so its SE(2)
transport is the action-induced translation of the local crop (no artificial
rotation is applied to a world-aligned tensor).
"""
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path("/root/autodl-tmp/worldfield_nav")
SRC = ROOT / "outputs/exp002_counterfactual_ego_flow"
OUT = ROOT / "outputs/exp003_transport_revelation"
ENC = ROOT / "tools/raw_rgb_to_mp4"
N, CELL, FPS = 128, 10.0 / 128.0, 12
BRANCHES = ("A_straight", "B_left", "C_right")
TARGET_HORIZONS = (0.5, 1.0, 2.0, 3.0)


def l1_energy(residual):
    """Per-cell O/H/V residual energy; age is bookkeeping, not world content."""
    return np.abs(residual[:3]).sum(axis=0)


def iou(a, b):
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else 1.0


def transport(phi0, origin0, pose0, pose_h, h):
    """T(Phi_0, action_0:h), with crop shift derived from recorded robot pose.

    The displacement is equivalent to integrating the branch's actions in this
    world-aligned field.  No future RGB, depth, visibility, or future Phi value
    is read by this function.
    """
    future_origin = origin0 + (pose_h[[0, 2]] - pose0[[0, 2]])
    out = np.zeros_like(phi0)
    out[3].fill(float(N + h))
    rows, cols = np.indices((N, N))
    wx = future_origin[0] + (cols + 0.5) * CELL
    wz = future_origin[1] + (N - rows - 0.5) * CELL
    src_col = np.floor((wx - origin0[0]) / CELL).astype(np.int32)
    src_row = N - 1 - np.floor((wz - origin0[1]) / CELL).astype(np.int32)
    valid = (src_col >= 0) & (src_col < N) & (src_row >= 0) & (src_row < N)
    for channel in range(3):
        out[channel, valid] = phi0[channel, src_row[valid], src_col[valid]]
    old_age = phi0[3, src_row[valid], src_col[valid]]
    old_seen = phi0[2, src_row[valid], src_col[valid]] > 0.5
    out[3, valid] = np.where(old_seen, old_age + h, float(N + h))
    return out, future_origin


def field_image(phi, title):
    """Occupancy+height+visibility: dark unknown, blue free, teal occupied."""
    seen = phi[2] > 0.5
    occ = (phi[0] > 0.5) & seen
    height = phi[1]
    image = np.full((N, N, 3), (17, 20, 26), dtype=np.uint8)
    if np.any(seen):
        h = height[seen]
        lo, hi = np.percentile(h, [2, 98])
        normalized = np.clip((height - lo) / max(hi - lo, 1e-6), 0, 1)
        image[seen] = np.stack((35 + 25 * normalized, 75 + 75 * normalized,
                                105 + 95 * normalized), axis=-1)[seen].astype(np.uint8)
    image[occ] = (35, 225, 190)
    im = Image.fromarray(image).resize((256, 256), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(im)
    draw.rectangle((0, 0, 255, 21), fill=(12, 14, 18))
    draw.text((6, 5), title, fill=(240, 240, 240))
    draw.ellipse((124, 124, 132, 132), fill=(255, 70, 70))
    return np.asarray(im)


def residual_image(residual, title="Revelation residual"):
    energy = l1_energy(residual)
    vmax = max(float(np.percentile(energy, 99)), 1e-6)
    x = np.clip(energy / vmax, 0, 1)
    image = np.stack((255 * x, 130 * np.sqrt(x), 30 * (1 - x)), axis=-1).astype(np.uint8)
    im = Image.fromarray(image).resize((256, 256), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(im)
    draw.rectangle((0, 0, 255, 21), fill=(12, 14, 18)); draw.text((6, 5), title, fill="white")
    return np.asarray(im)


def mask_image(mask, title="Newly observed"):
    image = np.full((N, N, 3), (17, 20, 26), dtype=np.uint8)
    image[mask] = (255, 165, 45)
    im = Image.fromarray(image).resize((256, 256), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(im)
    draw.rectangle((0, 0, 255, 21), fill=(12, 14, 18)); draw.text((6, 5), title, fill="white")
    return np.asarray(im)


def overlay_image(residual, new_mask, title="Residual vs revelation"):
    image = residual_image(residual, title)
    draw = ImageDraw.Draw(Image.fromarray(image))
    # Re-create drawing surface as PIL and mark mask boundary at 2x scale.
    im = Image.fromarray(image)
    draw = ImageDraw.Draw(im)
    m = np.kron(new_mask.astype(np.uint8), np.ones((2, 2), dtype=np.uint8))
    # Boundary pixels make the relation legible without hiding residual magnitude.
    edge = m.astype(bool) & ~(
        np.roll(m, 1, 0).astype(bool) & np.roll(m, -1, 0).astype(bool) &
        np.roll(m, 1, 1).astype(bool) & np.roll(m, -1, 1).astype(bool))
    arr = np.asarray(im).copy(); arr[edge] = (255, 255, 255)
    return arr


def four_panel(current, transported, gt, residual, branch, seconds):
    panels = [field_image(current, "Current World"), field_image(transported, "Transported World"),
              field_image(gt, "GT Future World"), residual_image(residual, "Revelation Residual")]
    im = Image.new("RGB", (1024, 256), (10, 12, 16))
    for i, panel in enumerate(panels):
        im.paste(Image.fromarray(panel), (256 * i, 0))
    draw = ImageDraw.Draw(im); draw.text((6, 235), f"{branch} | t+{seconds:.1f}s", fill="white")
    return np.asarray(im)


class Video:
    def __init__(self, path, width=1024, height=256):
        self.proc = subprocess.Popen([str(ENC), str(path), str(width), str(height), str(FPS)], stdin=subprocess.PIPE)
    def add(self, frame):
        self.proc.stdin.write(np.ascontiguousarray(frame[:, :, :3], dtype=np.uint8).tobytes())
    def close(self):
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("MP4 encoding failed")


def region_stats(energy, new, previous, still):
    def one(mask):
        values = energy[mask]
        return {"cells": int(values.size), "area_m2": float(values.size * CELL * CELL),
                "energy_sum": float(values.sum()), "energy_mean_per_cell": float(values.mean()) if values.size else None}
    inside, prev, unobserved = one(new), one(previous), one(still)
    outside_mask = ~new
    outside = one(outside_mask)
    ratio = (inside["energy_mean_per_cell"] / outside["energy_mean_per_cell"]
             if inside["energy_mean_per_cell"] is not None and outside["energy_mean_per_cell"] not in (None, 0)
             else None)
    total = inside["energy_sum"] + outside["energy_sum"]
    return {"newly_observed": inside, "previously_observed": prev, "still_unobserved": unobserved,
            "outside_newly_observed": outside, "inside_outside_mean_energy_ratio": ratio,
            "inside_energy_fraction": float(inside["energy_sum"] / total) if total else 0.0,
            "newly_observed_area_fraction": float(new.mean())}


def point_biserial(new, energy):
    x, y = new.ravel().astype(np.float64), energy.ravel().astype(np.float64)
    return float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else None


def selected_indices(timestamps):
    answer = []
    positive = timestamps[timestamps > 0]
    for requested in TARGET_HORIZONS:
        exact = np.where(np.isclose(timestamps, requested, atol=1e-5))[0]
        if exact.size:
            answer.append((requested, int(exact[0]), float(timestamps[exact[0]]), True))
        elif positive.size and requested < positive.min():
            answer.append((requested, None, None, False))
        else:
            eligible = np.where(timestamps <= requested + 1e-5)[0]
            index = int(eligible[-1]) if eligible.size else None
            answer.append((requested, index, float(timestamps[index]) if index is not None else None, False))
    return answer


def save_key_images(reference):
    current, transport_phi, gt, residual, new = reference
    Image.fromarray(field_image(current, "Current World")).save(OUT / "01_current_field.png")
    Image.fromarray(field_image(transport_phi, "Transport only")).save(OUT / "02_transport_only.png")
    Image.fromarray(field_image(gt, "GT future")).save(OUT / "03_gt_future.png")
    Image.fromarray(residual_image(residual)).save(OUT / "04_revelation_residual.png")
    Image.fromarray(mask_image(new)).save(OUT / "05_newly_observed_mask.png")
    Image.fromarray(overlay_image(residual, new)).save(OUT / "06_residual_vs_revelation.png")


def main():
    if not ENC.is_file():
        raise RuntimeError(f"Missing local encoder: {ENC}")
    OUT.mkdir(parents=True, exist_ok=True)
    all_metrics, reference = {}, None
    transport_video, revelation_video = Video(OUT / "transport_vs_gt.mp4"), Video(OUT / "revelation_field.mp4")
    try:
        for branch in BRANCHES:
            x = np.load(SRC / f"exp002_{branch}.npz", allow_pickle=False)
            gt, poses, origins, ts = x["phi"].astype(np.float32), x["pose"].astype(np.float32), x["field_origin"].astype(np.float32), x["timestamps"].astype(np.float32)
            if not (np.isfinite(gt).all() and np.isfinite(poses).all() and np.isfinite(origins).all()):
                raise ValueError(f"Non-finite inherited data in {branch}")
            phi0, origin0, pose0 = gt[0], origins[0], poses[0]
            predicted, residuals, new_masks, horizon_metrics = [], [], [], {}
            max_origin_error = 0.0
            for h in range(len(gt)):
                pred, derived_origin = transport(phi0, origin0, pose0, poses[h], h)
                max_origin_error = max(max_origin_error, float(np.abs(derived_origin - origins[h]).max()))
                future = gt[h]  # GT is consulted only here, after deterministic transport.
                residual = future - pred
                new = (future[2] > 0.5) & ~(pred[2] > 0.5)
                predicted.append(pred); residuals.append(residual); new_masks.append(new)
                transport_video.add(four_panel(phi0, pred, future, residual, branch, float(ts[h])))
                revelation_video.add(np.concatenate([residual_image(residual), mask_image(new), overlay_image(residual, new), field_image(future, "GT Future")], axis=1))
            for requested, h, actual, exact in selected_indices(ts):
                key = f"{requested:.1f}s"
                if h is None:
                    horizon_metrics[key] = {"requested_horizon_s": requested, "supported": False,
                                            "reason": f"trajectory sampling starts at {float(ts[1]):.1f}s"}
                    continue
                pred, future, residual, new = predicted[h], gt[h], residuals[h], new_masks[h]
                visible_future, visible_transport = future[2] > 0.5, pred[2] > 0.5
                shared_visible = visible_future & visible_transport
                height_mae = float(np.abs(future[1][shared_visible] - pred[1][shared_visible]).mean()) if shared_visible.any() else None
                energy = l1_energy(residual)
                previous = visible_future & visible_transport
                still = ~visible_future
                stats = region_stats(energy, new, previous, still)
                horizon_metrics[key] = {
                    "requested_horizon_s": requested, "actual_horizon_s": actual, "supported": True,
                    "exact_timestamp": exact, "transport_occupancy_iou_vs_gt": iou((pred[0] > 0.5) & visible_transport, (future[0] > 0.5) & visible_future),
                    "transport_height_mae_shared_visible": height_mae,
                    "transport_visibility_iou_vs_gt": iou(visible_transport, visible_future),
                    "newly_observed_prediction_target_area_m2": float(new.sum() * CELL * CELL),
                    "residual_vs_newly_observed_point_biserial_r": point_biserial(new, energy),
                    "residual_regions": stats,
                }
                Image.fromarray(four_panel(phi0, pred, future, residual, branch, actual)).save(OUT / f"{branch}_t{actual:.1f}s_4panel.png")
                if actual == 3.0 and reference is None:
                    reference = (phi0, pred, future, residual, new)
            np.savez_compressed(OUT / f"exp003_{branch}.npz", transport_only=np.stack(predicted), residual=np.stack(residuals), newly_observed=np.stack(new_masks), timestamps=ts, actions=x["actions"], metadata=json.dumps({"transport": "Phi_0 plus recorded robot-pose/action displacement only; no future RGB/depth/Phi used during prediction", "residual": "GT minus transport", "channels": ["occupancy", "height", "visibility", "age"]}))
            all_metrics[branch] = {"frames": int(len(ts)), "duration_s": float(ts[-1]), "max_pose_derived_vs_recorded_crop_origin_error_m": max_origin_error, "horizons": horizon_metrics}
    finally:
        transport_video.close(); revelation_video.close()
    if reference is None:
        raise RuntimeError("No supported 3.0 s frame for required figures")
    save_key_images(reference)
    metrics = {"experiment": "003_transport_revelation_decomposition", "input": str(SRC),
               "construction": "Persistent field transport from shared initial Phi and actual robot SE(2) trajectory; future GT is comparison-only.",
               "world_alignment": "X-Z global field; SE(2) rotation is represented in robot trajectory but field crop axes remain world aligned, therefore regridding uses the derived translation.",
               "sampling_note": "Inherited Exp002 timestamps are 1 Hz; 0.5 s has no exact sample and is explicitly reported unavailable.",
               "residual_energy": "per-cell L1 over occupancy, height, visibility; age excluded.", "branches": all_metrics}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
