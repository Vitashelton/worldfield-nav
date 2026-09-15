#!/usr/bin/env python3
"""Build a heterogeneous spatial/semantic topology from InteriorGS structure.

Room polygons and door/hole profiles are supplied by the downloaded InteriorGS
semantic structure files.  The resulting graph separates spatial connectivity
(`CONNECTS`, `CONTAINS`) from executable task relations (`APPROACH`, `CROSS`,
`ENTER`, `OBSERVE`).  No hidden goal, trajectory outcome or VLM response is
used during construction.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ANN = ROOT / "configs/relationnav/entities.json"
SEM = ROOT / "data/interiorgs_semantics"


def poly_xy(profile: list) -> np.ndarray:
    a = np.asarray(profile, np.float32)
    # Structure room profiles are [x,z] pairs; tolerate [x,y,z].
    if a.ndim != 2: return np.zeros((0, 2), np.float32)
    if a.shape[1] >= 3: return a[:, [0, 2]]
    return a[:, :2]


def center_profile(profile: list) -> np.ndarray:
    a = np.asarray(profile, np.float32)
    if a.ndim != 2 or not len(a): return np.zeros(2, np.float32)
    # InteriorGS room profiles are [x,z], whereas hole profiles are [x,z,y].
    # A previous version treated the third hole coordinate (height) as z,
    # collapsing doors onto false room adjacencies.  A doorway center must stay
    # in the horizontal x-z plane, therefore it is always the first two values.
    return a[:, :2].mean(0)


def point_poly_distance(p: np.ndarray, poly: np.ndarray) -> float:
    if len(poly) < 2: return float("inf")
    a, b = poly, np.roll(poly, -1, axis=0); ab = b - a
    t = np.clip(((p - a) * ab).sum(1) / ((ab * ab).sum(1) + 1e-8), 0, 1)
    return float(np.min(np.linalg.norm(p - (a + t[:, None] * ab), axis=1)))


def build_scene(scene: str, structure: dict[str, Any], entities: list[dict[str, Any]]) -> tuple[list, list]:
    nodes, edges = [], []
    rooms = []
    for i, room in enumerate(structure.get("rooms", [])):
        poly = poly_xy(room.get("profile", [])); nid = f"{scene}:room_{i:02d}"
        if len(poly) < 3: continue
        rooms.append((nid, poly)); nodes.append({"node_id": nid, "scene_id": scene, "kind": "area",
                      "geometry": {"polygon_xz": poly.tolist(), "center_xz": poly.mean(0).tolist()}, "source": "structure.json"})
    for i, hole in enumerate(structure.get("holes", [])):
        if hole.get("type", "").upper() not in {"DOOR", "HOLE", "ARCH"}: continue
        p = center_profile(hole.get("profile", [])); nid = f"{scene}:portal_{i:02d}"
        nodes.append({"node_id": nid, "scene_id": scene, "kind": "portal",
                      "geometry": {"center_xz": p.tolist(), "portal_type": hole.get("type", "HOLE"), "width_m": float(np.ptp(np.asarray(hole.get("profile", []), np.float32)[:, 0])) if hole.get("profile") else 0.0},
                      "source": "structure.json"})
        ranked = sorted(((point_poly_distance(p, poly), rid) for rid, poly in rooms), key=lambda x: x[0])[:2]
        if not ranked: continue
        if len(ranked) == 1:
            outside = f"{scene}:corridor_unknown"; 
            if not any(n["node_id"] == outside for n in nodes):
                nodes.append({"node_id": outside, "scene_id": scene, "kind": "corridor", "geometry": {"center_xz": p.tolist()}, "source": "derived_boundary"})
            ranked.append((float("inf"), outside))
        for _, rid in ranked:
            edges.extend([
                {"source": rid, "target": nid, "edge_type": "CONNECTS", "relation": "APPROACH", "cost": None},
                {"source": nid, "target": rid, "edge_type": "CONNECTS", "relation": "CROSS", "cost": None},
                {"source": nid, "target": rid, "edge_type": "CONNECTS", "relation": "ENTER", "cost": None},
            ])
        if len(ranked) == 2:
            a, b = ranked[0][1], ranked[1][1]
            edges.append({"source": a, "target": b, "edge_type": "SPATIAL_ADJACENCY", "relation": "CONNECTS", "via": nid})
            edges.append({"source": b, "target": a, "edge_type": "SPATIAL_ADJACENCY", "relation": "CONNECTS", "via": nid})
    # Curated landmarks become observation relations to their nearest area.
    for e in entities:
        if e.get("scene_id") != scene or e.get("kind") != "landmark": continue
        p = np.asarray(e["visual_anchor"]["world_point_xyz"], np.float32)[[0, 2]]; nid = e["entity_id"]
        nodes.append({"node_id": nid, "scene_id": scene, "kind": "landmark", "geometry": {"center_xz": p.tolist()}, "source": "entities.json"})
        if rooms:
            rid = min(rooms, key=lambda rp: point_poly_distance(p, rp[1]))[0]
            edges.append({"source": rid, "target": nid, "edge_type": "RELATION", "relation": "OBSERVE"})
    return nodes, edges


def draw(graph: dict, out: Path) -> None:
    scenes = graph["scenes"]
    cols = 3; rows = int(np.ceil(len(scenes) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.7 * cols, 5.1 * rows), squeeze=False); axes = axes.ravel()
    colors = {"area": "#4daf4a", "portal": "#e41a1c", "landmark": "#377eb8", "corridor": "#984ea3"}
    for ax, scene in zip(axes, scenes):
        ns = {n["node_id"]: n for n in graph["nodes"] if n["scene_id"] == scene}
        for n in ns.values():
            if n["kind"] == "area":
                p = np.asarray(n["geometry"]["polygon_xz"]); ax.fill(p[:,0],p[:,1],color=colors["area"],alpha=.12)
        for e in graph["edges"]:
            if e["source"] not in ns or e["target"] not in ns: continue
            a=np.asarray(ns[e["source"]]["geometry"]["center_xz"]); b=np.asarray(ns[e["target"]]["geometry"]["center_xz"])
            if e["edge_type"] == "SPATIAL_ADJACENCY": ax.plot([a[0],b[0]],[a[1],b[1]],"-",color="#555",lw=2,alpha=.8)
        for n in ns.values():
            p=np.asarray(n["geometry"]["center_xz"]); ax.scatter(*p,s=76,color=colors.get(n["kind"],"#222"),edgecolor="white",zorder=3)
            short=n["node_id"].split(":")[-1].replace("room_","R").replace("portal_","P").replace("landmark_","L")
            ax.annotate(short,xy=p,xytext=(4,5),textcoords="offset points",fontsize=6.6,ha="left",va="bottom",bbox={"boxstyle":"round,pad=.12","fc":"white","ec":"none","alpha":.82})
        ax.set_title(scene,fontsize=11,pad=8); ax.set_aspect("equal",adjustable="datalim"); ax.set_xlabel("world X (m)"); ax.set_ylabel("world Z (m)"); ax.grid(alpha=.15)
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([0],[0],marker="o",color="w",label="area",markerfacecolor="#4daf4a",markersize=7),Line2D([0],[0],marker="o",color="w",label="portal",markerfacecolor="#e41a1c",markersize=7),Line2D([0],[0],marker="o",color="w",label="landmark",markerfacecolor="#377eb8",markersize=7)],loc="best",fontsize=7,framealpha=.9)
    for ax in axes[len(scenes):]: ax.axis("off")
    fig.suptitle("Habitat-GS spatial semantic topology\nsolid links = spatial alternatives; labels are abbreviated",fontsize=16); fig.tight_layout(rect=(0,0,1,.93)); out.parent.mkdir(parents=True,exist_ok=True); fig.savefig(out,dpi=220,bbox_inches="tight"); plt.close(fig)


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--output",type=Path,default=ROOT/"outputs/formal/RelationNav/topology/spatial_semantic_topology.json"); ap.add_argument("--figure",type=Path,default=ROOT/"paper_assets/figures/relationnav_spatial_topology.png"); args=ap.parse_args()
    ann=json.loads(ANN.read_text())
    asset_root=ROOT/"data/scene_datasets/gs_scenes/train"
    discovered={
        "interior_"+p.parent.name
        for p in SEM.glob("*/structure.json")
        if (asset_root/("interior_"+p.parent.name)/(("interior_"+p.parent.name)+".gs.ply")).is_file()
        and (asset_root/("interior_"+p.parent.name)/(("interior_"+p.parent.name)+".navmesh")).is_file()
    }
    scenes=sorted(set(ann["scenes"]) | discovered); nodes=[]; edges=[]
    for scene_id in scenes:
        key=scene_id.replace("interior_",""); path=SEM/key/"structure.json"
        if path.exists():
            n,e=build_scene(scene_id,json.loads(path.read_text()),ann["entities"]); nodes.extend(n); edges.extend(e)
        else:
            # 0405 currently has labels/occupancy but no structure.json. Keep
            # its audited contracts in the graph rather than silently dropping
            # the scene; this fallback is explicitly marked curated-only.
            local=[x for x in ann["entities"] if x.get("scene_id")==scene_id]
            for x in local:
                a=x["visual_anchor"]["world_point_xyz"]
                if x["kind"]=="area":
                    poly=x["area_polygon_xz"]; center=np.asarray(poly,np.float32).mean(0).tolist(); geom={"polygon_xz":poly,"center_xz":center}
                else:
                    geom={"center_xz":[float(a[0]),float(a[2])]}
                    if x["kind"]=="portal": geom["plane"]=x["portal_plane"]
                nodes.append({"node_id":x["entity_id"],"scene_id":scene_id,"kind":x["kind"],"geometry":geom,"source":"entities.json"})
            area=next((x for x in local if x["kind"]=="area"),None)
            for x in local:
                if x["kind"]!="portal" or area is None: continue
                edges.extend([{ "source":area["entity_id"],"target":x["entity_id"],"edge_type":"CONNECTS","relation":"APPROACH","cost":None}, {"source":x["entity_id"],"target":area["entity_id"],"edge_type":"CONNECTS","relation":"CROSS","cost":None}, {"source":x["entity_id"],"target":area["entity_id"],"edge_type":"CONNECTS","relation":"ENTER","cost":None}])
            landmark=next((x for x in local if x["kind"]=="landmark"),None)
            if area and landmark: edges.append({"source":area["entity_id"],"target":landmark["entity_id"],"edge_type":"RELATION","relation":"OBSERVE"})
    graph={"schema_version":1,"graph_type":"heterogeneous_spatial_semantic_topology","scenes":scenes,"nodes":nodes,"edges":edges,"provenance":{"structure_source":"data/interiorgs_semantics/*/structure.json","curated_entity_source":"configs/relationnav/entities.json","hidden_goal_oracle_used":False,"navmesh_outcomes_used":False}}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(graph,indent=2)+"\n"); draw(graph,args.figure)
    print(json.dumps({"scenes":len(scenes),"nodes":len(nodes),"edges":len(edges),"output":str(args.output),"figure":str(args.figure)},indent=2))

if __name__ == "__main__": main()
