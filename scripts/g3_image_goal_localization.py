"""Build a geometry-anchored landmark map and evaluate image-goal queries.

The query path intentionally uses RGB descriptors only. Depth/pose are used
to create the offline map and to score hidden ground truth correspondences.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from geoanchor.metricanchor import ResidualMetricAdapter
from geoanchor.correspondence import intrinsics, patch_center

SPLITS = {"val": ["scene04", "scene05"], "unseen": ["scene56", "scene57", "scene58"]}
TRAIN_SCENES = ["scene01", "scene02", "scene03", "scene09", "interior_0405_840145"]

def files():
    fs = sorted((ROOT / "outputs/formal/G2/feature_cache").glob("*_traj*.npz"))
    return {p.stem: p for p in fs}

def load_feature(p):
    x = np.load(p)["features"].astype(np.float32)
    return x.reshape(x.shape[0], -1, x.shape[-1])

def encode_feature(x, adapter=None):
    if adapter is not None:
        with torch.no_grad(): x = adapter(torch.from_numpy(x).cuda()).cpu().numpy()
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)

def global_offsets(fs):
    out, n = {}, 0
    for k, p in sorted(fs.items()):
        x = np.load(p)["features"]
        out[k] = n
        n += x.shape[0]
    return out

def map_for_scene(scene, fs, off, xyz, valid, adapter=None, voxel=0.15):
    # Reference traversal is offline and may use metric pose/depth-derived xyz.
    sums, counts = {}, {}
    for key, p in fs.items():
        if not key.startswith(scene + "_") or not key.endswith(("traj00", "traj01")):
            continue
        f = encode_feature(load_feature(p), adapter)
        g = off[key]; q = xyz[g:g+len(f)]; m = valid[g:g+len(f)]
        for dframe, pframe, mframe in zip(f, q, m):
          for d, pt, ok in zip(dframe, pframe, mframe):
              if not ok or not np.isfinite(pt).all(): continue
              k = tuple(np.floor(pt / voxel).astype(np.int32))
              sums[k] = sums.get(k, np.zeros_like(d)) + d
              counts[k] = counts.get(k, 0) + 1
    keys = list(sums)
    desc = np.stack([sums[k] / counts[k] for k in keys]).astype(np.float32)
    desc /= np.maximum(np.linalg.norm(desc, axis=1, keepdims=True), 1e-8)
    pos = np.stack([np.asarray(k, np.float32) * voxel for k in keys])
    return pos, desc

def query_scene(scene, fs, off, xyz, valid, lm_pos, lm_desc, adapter=None):
    rows = []
    map_tree = cKDTree(lm_pos)
    for key, p in fs.items():
        if not key.startswith(scene + "_") or not key.endswith("traj02"): continue
        z = np.load(ROOT / "outputs/formal/C1/pilot/trajectories" / key / "sequence.npz")
        f = encode_feature(load_feature(p), adapter)
        g = off[key]; q = xyz[g:g+len(f)]; m = valid[g:g+len(f)]
        for fi, (dframe, qframe, mframe) in enumerate(zip(f, q, m)):
            # A query is scored only when its physical place is represented by
            # the offline reference map; uncovered space is reported separately.
            coverage = map_tree.query(qframe, k=1)[0] <= 2.0
            sims = dframe @ lm_desc.T; nn = sims.argmax(1)
            for patch, j in enumerate(nn):
                if not mframe[patch] or not coverage[patch] or not np.isfinite(qframe[patch]).all(): continue
                err = float(np.linalg.norm(lm_pos[j] - qframe[patch]))
                rows.append({"scene": scene, "trajectory": key, "frame": fi,
                             "patch": patch, "similarity": float(sims[patch,j]),
                             "world_error_m": err, "gt_xyz": qframe[patch].tolist(),
                             "retrieved_xyz": lm_pos[j].tolist(), "method": "GeoAnchor" if adapter else "Frozen"})
    return rows

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--checkpoint"); ap.add_argument("--max-queries", type=int, default=3000); args = ap.parse_args()
    fs = files(); off = global_offsets(fs)
    geom = ROOT / "outputs/formal/G2/geometry_cache"
    xyz = torch.load(geom / "candidate_world_xyz.pt", map_location="cpu").numpy()
    valid = torch.load(geom / "candidate_world_valid.pt", map_location="cpu").numpy().astype(bool)
    adapter = None
    if args.checkpoint:
        adapter = ResidualMetricAdapter().cuda().eval(); adapter.load_state_dict(torch.load(args.checkpoint, map_location="cuda"))
    out = ROOT / "outputs/formal/G3/image_goal"; out.mkdir(parents=True, exist_ok=True)
    allrows = []
    for split, scenes in SPLITS.items():
        for scene in scenes:
            pos, desc = map_for_scene(scene, fs, off, xyz, valid, adapter=None)
            rows = query_scene(scene, fs, off, xyz, valid, pos, desc, adapter=adapter)
            allrows.extend(rows[:args.max_queries])
    if not allrows: raise RuntimeError("no valid image-goal queries")
    json.dump(allrows, open(out / "image_goal_results.json", "w"), indent=2)
    arr = np.asarray([r["world_error_m"] for r in allrows]); sim = np.asarray([r["similarity"] for r in allrows])
    summary = {"queries": len(allrows), "mean_world_error_m": float(arr.mean()), "median_world_error_m": float(np.median(arr)),
               "success_0.5m": float((arr <= .5).mean()), "success_1m": float((arr <= 1.).mean()), "mean_similarity": float(sim.mean())}
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    pa = ROOT / "paper_assets"; (pa / "figures").mkdir(parents=True, exist_ok=True); (pa / "tables").mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8,5)); plt.hist(arr, bins=40, color="#3568a8"); plt.axvline(.5,color="r",ls="--",label="0.5 m"); plt.xlabel("Goal pose proxy error (m)"); plt.ylabel("Query patches"); plt.title("Image-goal localization with geometry-anchored landmarks"); plt.legend(); plt.tight_layout(); plt.savefig(pa / "figures/metricanchor_image_goal_localization.png", dpi=180); plt.close()
    import csv
    with open(pa / "tables/metricanchor_image_goal_localization.csv", "w", newline="") as h:
        w=csv.DictWriter(h, fieldnames=["queries","mean_world_error_m","median_world_error_m","success_0.5m","success_1m","mean_similarity"]); w.writeheader(); w.writerow(summary)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__": main()
