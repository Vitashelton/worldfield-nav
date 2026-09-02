"""PyTorch reader for C1 sequential multimodal WorldFlow trajectories."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class SequentialWorldFieldDataset(Dataset[dict[str, Any]]):
    """Flatten completed C1 trajectories into causally valid frame samples.

    `oracle_field` is returned under an explicit reference-only key. Callers
    must never use it as an online model input.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.items: list[tuple[Path, int, dict[str, Any]]] = []
        for marker in sorted(self.root.glob("trajectories/*/complete.json")):
            metadata = json.loads(marker.read_text())
            sequence = marker.parent / "sequence.npz"
            if not sequence.is_file():
                raise FileNotFoundError(f"Completion marker without trajectory data: {marker}")
            frame_count = int(metadata["frame_count"])
            self.items.extend((sequence, frame, metadata) for frame in range(frame_count))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sequence, frame, metadata = self.items[index]
        with np.load(sequence, allow_pickle=False) as data:
            return {
                "rgb": torch.from_numpy(data["rgb"][frame]),
                "depth": torch.from_numpy(data["depth"][frame].astype(np.float32)),
                "sim_lidar_xyz": torch.from_numpy(data["sim_lidar_xyz"][frame]),
                "sim_lidar_count": torch.tensor(int(data["sim_lidar_count"][frame])),
                "pose_wxyz": torch.from_numpy(data["agent_pose_wxyz"][frame]),
                "timestamp_s": torch.tensor(float(data["timestamps_s"][frame])),
                "field_origin_xz": torch.from_numpy(data["field_origin_xz"][frame]),
                "G_lidar": torch.from_numpy(data["G_lidar"][frame]),
                "G_rgbd": torch.from_numpy(data["G_rgbd"][frame]),
                "V": torch.from_numpy(data["V"][frame]),
                "A": torch.from_numpy(data["A"][frame]),
                "causal_field": torch.from_numpy(data["causal_field"][frame]),
                "oracle_field_reference_only": torch.from_numpy(data["oracle_field"][frame]),
                "scene_id": metadata["scene_id"],
                "trajectory_id": metadata["trajectory_id"],
                "frame_index": torch.tensor(frame),
            }
