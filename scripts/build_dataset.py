#!/usr/bin/env python3
"""Reserved B1 entry point; implement only under B1 P1 authorization."""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = ROOT / "configs/benchmark/b1_pilot.yaml"
    if args.dry_run:
        print(f"B1 dataset entry point reserved; config is {config}")
        print("No data generated. Complete B1 P0 scene inventory before B1 P1 implementation.")
        return 0
    raise SystemExit("Dataset generation is not yet authorized: complete B1 P0 then implement B1 P1.")


if __name__ == "__main__":
    raise SystemExit(main())
