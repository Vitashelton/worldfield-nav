#!/usr/bin/env python3
"""Fetch exactly the eight missing official Habitat-GS assets for B1 P0."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/scene_datasets/gs_scenes"
OUT = ROOT / "outputs/formal/B1/p0"
REPOSITORY = "RukawaY/gs_scenes"
REVISION = "main"
MISSING = (
    ("train", "scene01"), ("train", "scene02"), ("train", "scene03"),
    ("train", "scene04"), ("train", "scene05"),
    ("val", "scene56"), ("val", "scene57"), ("val", "scene58"),
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def remote_size(url: str) -> int:
    headers = subprocess.check_output(["curl", "--silent", "--show-error", "--location", "--head", url], text=True)
    for line in headers.splitlines():
        if line.lower().startswith("x-linked-size:"):
            return int(line.split(":", 1)[1].strip())
    raise RuntimeError(f"official asset did not expose X-Linked-Size: {url}")


def fetch(split: str, scene: str, suffix: str) -> dict[str, object]:
    target = DATA / split / scene / f"{scene}{suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{split}/{scene}/{scene}{suffix}?download=true"
    expected_size = remote_size(url)
    existed = target.is_file() and target.stat().st_size == expected_size
    if not existed:
        subprocess.run([
            "curl", "--location", "--fail", "--retry", "3", "--continue-at", "-",
            "--output", str(target), url,
        ], check=True)
    if target.stat().st_size != expected_size:
        raise RuntimeError(f"incomplete asset {target}: {target.stat().st_size} != {expected_size}")
    return {
        "scene_id": scene,
        "official_source": f"https://huggingface.co/datasets/{REPOSITORY}",
        "repository": REPOSITORY,
        "revision": REVISION,
        "url": url,
        "path": str(target.relative_to(ROOT)),
        "size_bytes": target.stat().st_size,
        "official_size_bytes": expected_size,
        "sha256": digest(target),
        "preexisting": existed,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Independent scene files are safe to fetch concurrently.  This remains
    # the exact P0 allow-list, rather than a repository snapshot download.
    jobs = [(split, scene, suffix) for split, scene in MISSING for suffix in (".gs.ply", ".navmesh")]
    with ThreadPoolExecutor(max_workers=4) as pool:
        assets = list(pool.map(lambda args: fetch(*args), jobs))
    manifest = {
        "purpose": "B1 P0 targeted official Habitat-GS pilot-scene acquisition",
        "excluded": ["full scene collection", "mesh assets", "episodes", "avatars", "other datasets"],
        "assets": assets,
        "downloaded_scene_count": len(MISSING),
        "downloaded_bytes": sum(x["size_bytes"] for x in assets if not x["preexisting"]),
        "total_selected_asset_bytes": sum(x["size_bytes"] for x in assets),
    }
    (OUT / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
