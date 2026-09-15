#!/usr/bin/env python3
"""Build and visualize the RelationNav semantic topology graph."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from semantic_topology import build_topology

ROOT = Path(__file__).resolve().parents[1]


def draw(graph: dict, out: Path) -> None:
    scenes = graph["scenes"]; n = len(scenes)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5.2), squeeze=False)
    axes = axes[0]
    colors = {"area": "#4daf4a", "portal": "#e41a1c", "landmark": "#377eb8"}
    for ax, scene in zip(axes, scenes):
        ns = [x for x in graph["nodes"] if x["scene_id"] == scene]
        by_id = {x["node_id"]: x for x in ns}
        for e in graph["edges"]:
            if e["source"] not in by_id or e["target"] not in by_id: continue
            a, b = by_id[e["source"]], by_id[e["target"]]
            pa = np.asarray(a["geometry"]["center_xz"]); pb = np.asarray(b["geometry"]["center_xz"])
            ax.annotate("", xy=pb, xytext=pa, arrowprops={"arrowstyle": "->", "color": "#888", "lw": 1.0, "alpha": .75})
            ax.text(*(0.55 * pa + .45 * pb), e["relation"], fontsize=7, color="#555")
        for node in ns:
            p = np.asarray(node["geometry"]["center_xz"])
            if node["kind"] == "area":
                poly = np.asarray(node["geometry"]["polygon_xz"])
                ax.fill(poly[:, 0], poly[:, 1], color=colors["area"], alpha=.18)
            ax.scatter(*p, s=90, color=colors[node["kind"]], edgecolor="white", linewidth=.8, zorder=3)
            ax.text(p[0], p[1] + .18, node["node_id"], fontsize=8, ha="center")
        ax.set_title(scene, fontsize=10); ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("world X (m)"); ax.set_ylabel("world Z (m)"); ax.grid(alpha=.18)
    fig.suptitle("RelationNav semantic topology: entity nodes and relation edges", fontsize=15)
    fig.tight_layout(); out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=220, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--annotations", type=Path, default=ROOT / "configs/relationnav/entities.json")
    ap.add_argument("--output", type=Path, default=ROOT / "outputs/formal/RelationNav/topology/semantic_topology.json")
    ap.add_argument("--figure", type=Path, default=ROOT / "paper_assets/figures/relationnav_semantic_topology.png")
    args = ap.parse_args(); payload = json.loads(args.annotations.read_text()); graph = build_topology(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(graph, indent=2) + "\n"); draw(graph, args.figure)
    counts = {s: {k: sum(n["scene_id"] == s and n["kind"] == k for n in graph["nodes"]) for k in ("portal", "area", "landmark")} for s in graph["scenes"]}
    print(json.dumps({"nodes": len(graph["nodes"]), "edges": len(graph["edges"]), "scene_counts": counts, "output": str(args.output), "figure": str(args.figure)}, indent=2))


if __name__ == "__main__": main()
