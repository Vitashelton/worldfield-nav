#!/usr/bin/env python3
"""Merge independently generated deterministic RelationNav scene manifests."""
from __future__ import annotations
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"outputs/formal/RelationNav/P1/dataset"
SPLIT={"train":["interior_0135_840032","interior_0121_840013","interior_0093_839966"],"heldout":["interior_0405_840145"]}

def main():
    rows=[]; missing=[]
    for scene in SPLIT["train"]+SPLIT["heldout"]:
        p=DATA/"scene_manifests"/f"{scene}.json"
        if not p.is_file(): missing.append(scene); continue
        r=json.loads(p.read_text())
        if not r: missing.append(scene)
        rows += r
    if missing: raise RuntimeError(f"missing/empty scene manifests: {missing}")
    payload={"protocol":"RelationNav P1; relation labels are offline curated-geometry supervision only.","split":SPLIT,"episodes":rows}
    (DATA/"dataset_manifest.json").write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps({"episodes":len(rows),"decisions":sum(len(x['phases']) for x in rows),"by_scene":{s:sum(x['scene_id']==s for x in rows) for s in SPLIT['train']+SPLIT['heldout']}},indent=2))

if __name__=="__main__":main()
