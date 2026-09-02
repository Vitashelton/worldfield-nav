#!/usr/bin/env python3
"""Run a formal command and atomically register its immutable run record.

Usage:
  python scripts/run_experiment.py B1 --config configs/benchmark/b1_pilot.yaml \
      --output outputs/formal/B1/smoke --metrics outputs/formal/B1/smoke/metrics.json \
      -- python scripts/build_dataset.py --dry-run
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "experiments/registry.yaml"
RUNS = ROOT / "experiments/runs"


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def atomic_yaml(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(data, sort_keys=False))
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def append_record(experiment_id: str, record_relative: str) -> None:
    registry = load_yaml(REGISTRY)
    formal = registry.get("formal", {})
    if experiment_id not in formal:
        raise ValueError(f"Unknown formal experiment {experiment_id!r}")
    experiment = formal[experiment_id]
    runs = experiment.setdefault("runs", [])
    runs.append(record_relative)
    experiment["last_run"] = record_relative
    atomic_yaml(REGISTRY, registry)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_id", help="Formal registry key, for example B1")
    parser.add_argument("--config", required=True, help="Versioned config path relative to project root")
    parser.add_argument("--output", required=True, help="Run output directory relative to project root")
    parser.add_argument("--metrics", help="Expected metrics file relative to project root")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command after '--'")
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after '--'")
    return args


def main() -> int:
    args = parse_args()
    config = (ROOT / args.config).resolve()
    output = (ROOT / args.output).resolve()
    metrics = (ROOT / args.metrics).resolve() if args.metrics else None
    if not config.is_file():
        raise FileNotFoundError(f"Config does not exist: {config}")
    # Fail before execution if the ID is not formally authorized.
    registry = load_yaml(REGISTRY)
    if args.experiment_id not in registry.get("formal", {}):
        raise ValueError(f"Unknown formal experiment {args.experiment_id!r}")

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUNS / args.experiment_id / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    record = {
        "schema_version": 1,
        "experiment_id": args.experiment_id,
        "status": "running",
        "started_at_utc": started,
        "config": rel(config),
        "config_sha256": sha256(config),
        "output": rel(output),
        "metrics": rel(metrics) if metrics else None,
        "command": args.command,
        "cwd": rel(ROOT),
    }
    record_path = run_dir / "run.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    append_record(args.experiment_id, rel(record_path))

    stdout = (run_dir / "stdout.log").open("w")
    stderr = (run_dir / "stderr.log").open("w")
    try:
        result = subprocess.run(args.command, cwd=ROOT, stdout=stdout, stderr=stderr, check=False)
    finally:
        stdout.close()
        stderr.close()
    record["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    record["returncode"] = result.returncode
    record["status"] = "completed" if result.returncode == 0 else "failed"
    record["metrics_exists"] = metrics.is_file() if metrics else None
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
