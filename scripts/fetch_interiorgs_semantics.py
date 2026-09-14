#!/usr/bin/env python3
"""Fetch only lightweight official InteriorGS semantic sidecars for local scenes."""
from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENE_ROOT = ROOT / "data/scene_datasets/gs_scenes"
OUT = ROOT / "data/interiorgs_semantics"
BASE = "https://hf-mirror.com/datasets/spatialverse/InteriorGS/resolve/main"
FILES = ("labels.json", "occupancy.json", "structure.json")


def main() -> None:
    scenes = sorted(p.name for split in ("train", "val") for p in (SCENE_ROOT / split).glob("interior_*") if p.is_dir())
    manifest = {"source": "spatialverse/InteriorGS", "files": list(FILES), "scenes": []}
    for scene in scenes:
        upstream = scene.removeprefix("interior_")
        target = OUT / upstream; target.mkdir(parents=True, exist_ok=True)
        entry = {"scene_id": scene, "upstream_id": upstream, "downloads": []}
        for filename in FILES:
            path = target / filename
            if not path.exists():
                url = f"{BASE}/{upstream}/{filename}?download=true"
                headers = {"User-Agent": "worldfield-nav-relationbench/1.0"}
                if token := os.environ.get("HF_TOKEN"):
                    headers["Authorization"] = f"Bearer {token}"
                request = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(request, timeout=60) as response:
                    path.write_bytes(response.read())
            payload = path.read_bytes()
            entry["downloads"].append({"file": str(path.relative_to(ROOT)), "bytes": len(payload),
                                       "sha256": hashlib.sha256(payload).hexdigest()})
        structure = json.loads((target / "structure.json").read_text())
        labels = json.loads((target / "labels.json").read_text())
        # The official schema has rooms/walls/holes; tolerate future wrappers.
        entry["official_stats"] = {
            "rooms": len(structure.get("rooms", [])),
            "doors": sum(1 for hole in structure.get("holes", []) if str(hole.get("type", "")).upper() == "DOOR"),
            "objects": len(labels if isinstance(labels, list) else labels.get("objects", labels.get("labels", []))),
        }
        manifest["scenes"].append(entry)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
