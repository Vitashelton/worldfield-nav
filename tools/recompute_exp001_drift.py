#!/usr/bin/env python3
"""Recompute Experiment 001 drift in common world-cell coordinates."""
import json
from pathlib import Path

import numpy as np

OUT = Path("/root/autodl-tmp/worldfield_nav/outputs/exp001")
CELL = 10.0 / 128.0
SIZE = 128


def world_cells(phi, origin):
    observed = phi[2] > 0
    rows, cols = np.nonzero(observed)
    ix0, iz0 = np.floor(origin / CELL).astype(np.int32)
    ix = ix0 + cols
    iz = iz0 + (SIZE - 1 - rows)
    # A reversible 64-bit key preserves the signed global grid coordinates.
    keys = (ix.astype(np.int64) << 32) ^ (iz.astype(np.int64) & 0xFFFFFFFF)
    order = np.argsort(keys)
    return keys[order], phi[0, rows, cols][order], phi[1, rows, cols][order]


def main():
    sequence = np.load(OUT / "field_sequence.npz")
    phi, origins = sequence["phi"], sequence["field_origin"]
    previous = None
    occupancy_changes, height_changes, overlap_counts = [], [], []
    for frame in range(phi.shape[0]):
        current = world_cells(phi[frame], origins[frame])
        if previous is None:
            occupancy_changes.append(0.0); height_changes.append(0.0); overlap_counts.append(0)
        else:
            common, old_indices, new_indices = np.intersect1d(previous[0], current[0], return_indices=True)
            overlap_counts.append(int(len(common)))
            if len(common):
                occupancy_changes.append(float(np.mean(np.abs(previous[1][old_indices] - current[1][new_indices]))))
                height_changes.append(float(np.mean(np.abs(previous[2][old_indices] - current[2][new_indices]))))
            else:
                occupancy_changes.append(0.0); height_changes.append(0.0)
        previous = current
    metrics = json.loads((OUT / "metrics.json").read_text())
    metrics["field_static_drift"] = {
        "world_aligned_overlap_mean_abs_occupancy": float(np.mean(occupancy_changes[1:])),
        "world_aligned_overlap_mean_abs_height_m": float(np.mean(height_changes[1:])),
        "occupancy_series": occupancy_changes,
        "height_m_series": height_changes,
        "mean_overlap_cells": float(np.mean(overlap_counts[1:])),
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics["field_static_drift"], indent=2))


if __name__ == "__main__":
    main()
