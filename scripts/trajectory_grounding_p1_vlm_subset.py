#!/usr/bin/env python3
"""Create the bounded, stratified local-only VLM request subset for P1."""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
LIMITS={"train":160,"val":120,"unseen":100}

def main():
    root=ROOT/"outputs/formal/TrajectoryGrounding/P1"; rows=json.loads((root/"dataset_manifest.json").read_text()); grouped=defaultdict(list)
    for row in rows: grouped[(row["split"],row["scene_id"])].append(row)
    selected=[]
    for key,items in sorted(grouped.items()): selected.extend(items[:LIMITS[key[0]]])
    package=root/"vlm_budgeted"; package.mkdir(exist_ok=True); requests=[]
    for row in selected:
        d=package/"images"/row["episode_id"]; d.mkdir(parents=True,exist_ok=True)
        for name in ("goal_image","trajectory_overlay"):
            link=d/f"{name}.png"; target=Path("../../../samples")/row["episode_id"]/f"{name}.png"
            if not link.exists(): link.symlink_to(target)
        requests.append({"request_id":row["episode_id"],"goal_image":f"images/{row['episode_id']}/goal_image.png","trajectory_overlay":f"images/{row['episode_id']}/trajectory_overlay.png","trajectories":[{"trajectory_id":t["trajectory_id"]} for t in row["trajectories"]]})
    (package/"vlm_request_package.json").write_text(json.dumps(requests,indent=2)+"\n")
    summary={"requests":len(requests),"per_scene_limits":LIMITS,"by_split":{s:sum(1 for x in selected if x["split"]==s) for s in LIMITS},"note":"VLM cache only; all P1 outcomes/DINO remain full dataset."}
    (package/"selection_summary.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
