#!/usr/bin/env python3
"""Fast, error-driven preflight for the existing WorldFlow runtime."""
from __future__ import annotations

import sys


def main() -> int:
    import torch
    import habitat_sim

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; investigate the concrete runtime error before changing the environment.")
    props = torch.cuda.get_device_properties(0)
    summary = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_gib": round(props.total_memory / 1024**3, 2),
        "habitat_sim": habitat_sim.__file__,
    }
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
