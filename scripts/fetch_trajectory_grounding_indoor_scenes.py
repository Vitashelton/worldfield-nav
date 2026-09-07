#!/usr/bin/env python3
"""Fetch only the official indoor assets required by Trajectory Grounding P1."""
from __future__ import annotations

import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/scene_datasets/gs_scenes/train"
OUT = ROOT / "outputs/formal/TrajectoryGrounding/P1"
REPOSITORY = "RukawaY/gs_scenes"
MIRROR = "https://hf-mirror.com"
SCENES = (
    "interior_0047_839892", "interior_0108_839984", "interior_0184_840116",
    "interior_0093_839966", "interior_0121_840013", "interior_0135_840032",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def remote_size(url: str) -> int:
    header = subprocess.check_output(
        ["curl", "--silent", "--show-error", "--location", "--head", url], text=True
    )
    for line in header.splitlines():
        if line.lower().startswith("x-linked-size:"):
            return int(line.split(":", 1)[1].strip())
    raise RuntimeError(f"Missing X-Linked-Size for {url}")


def fetch(scene: str, suffix: str) -> dict[str, object]:
    destination = DATA / scene / f"{scene}{suffix}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"{MIRROR}/datasets/{REPOSITORY}/resolve/main/train/{scene}/{scene}{suffix}?download=true"
    expected = remote_size(url)
    preexisting = destination.is_file() and destination.stat().st_size == expected
    if not preexisting:
        subprocess.run([
            "curl", "--location", "--fail", "--retry", "3", "--continue-at", "-",
            "--output", str(destination), url,
        ], check=True)
    if destination.stat().st_size != expected:
        raise RuntimeError(f"Incomplete asset: {destination}")
    return {
        "scene_id": scene, "suffix": suffix, "source": f"{MIRROR}/datasets/{REPOSITORY}",
        "path": str(destination.relative_to(ROOT)), "bytes": destination.stat().st_size,
        "sha256": sha256(destination), "preexisting": preexisting,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        files = list(pool.map(lambda x: fetch(*x), [(s, ext) for s in SCENES for ext in (".gs.ply", ".navmesh")]))
    split = {
        "train": ["interior_0405_840145", "interior_0047_839892", "interior_0108_839984", "interior_0184_840116"],
        "val": ["interior_0093_839966"],
        "unseen": ["interior_0121_840013", "interior_0135_840032"],
    }
    manifest = {
        "purpose": "Trajectory Grounding P1 scene-disjoint indoor benchmark",
        "download_policy": "six explicitly listed official interior scene assets only; no full collection",
        "split": split, "assets": files,
        "downloaded_bytes": sum(x["bytes"] for x in files if not x["preexisting"]),
    }
    (OUT / "indoor_scene_download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUT / "split_manifest.json").write_text(json.dumps(split, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
