#!/usr/bin/env python3
"""Build and validate the deterministic B1 P1 smoke manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "src").resolve()))
from datasets.episode_manifest import build_manifest, validate_manifest

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-source", default="paper_assets/tables/b1_p0_revelation_candidates.csv")
    parser.add_argument("--config", default="configs/experiments/b1_p1_manifest.yaml")
    parser.add_argument("--output", default="outputs/formal/B1/p1_manifest")
    parser.add_argument("--compare-output", default=None)
    args = parser.parse_args()
    config = ROOT / args.config
    source = ROOT / args.candidate_source
    output = ROOT / args.output
    manifest = build_manifest(ROOT, output, source, config)
    validation = validate_manifest(manifest)
    if args.compare_output:
        second = build_manifest(ROOT, ROOT / args.compare_output, source, config)
        canonical_a = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
        canonical_b = {k: v for k, v in second.items() if k != "manifest_sha256"}
        validation["byte_equivalent_second_build"] = canonical_a == canonical_b
    else:
        validation["byte_equivalent_second_build"] = None
    metrics = {"benchmark": "B1 P1 deterministic manifest", "manifest": str((output / "manifest.json").relative_to(ROOT)), "validation": validation, "limitations": ["P0 did not persist absolute start poses or O/H/V/A arrays; manifest stores explicit source references and marks them unavailable."]}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
