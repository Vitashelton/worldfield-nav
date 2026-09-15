#!/usr/bin/env python3
"""Attach cached VLM visual evidence to a geometry-derived topology.

This is an offline map-construction step.  It does not create task labels from
hidden goal poses: each node gets a real Habitat RGB keyframe and a conservative
VLM description that is retained with its evidence path and confidence.
"""
from __future__ import annotations
import base64, io, json, math, time, urllib.request
from pathlib import Path
import numpy as np
from PIL import Image
from relationnav_p1_generate import make_sim, render, yaw_rotation

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"outputs/formal/RelationNav/topology/spatial_semantic_topology.json"
OUT=ROOT/"outputs/formal/TopoNav/P1"
SCENES={"interior_0047_839892","interior_0108_839984","interior_0135_840032"}

def b64(rgb):
 b=io.BytesIO();Image.fromarray(rgb).save(b,format="JPEG",quality=85);return base64.b64encode(b.getvalue()).decode()
def ask(rgb,kind):
 prompt=("Describe this indoor robot-navigation keyframe conservatively. Return JSON only: "
 "{\"place_type\": one of [room,corridor,doorway,entrance,unknown], "
 "\"short_description\": <=12 words, \"visible_landmarks\": [<=3 strings], "
 "\"confidence\": number 0..1}. Do not invent room names, coordinates, or objects not visible. "
 f"The topology node type is {kind}.")
 body={"model":"qwen3-vl:8b-instruct","stream":False,"format":"json","options":{"temperature":0,"num_predict":96},"messages":[{"role":"user","content":prompt,"images":[b64(rgb)]}]}
 req=urllib.request.Request("http://127.0.0.1:11434/api/chat",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
 t=time.perf_counter()
 try:
  raw=json.loads(urllib.request.urlopen(req,timeout=120).read());txt=raw["message"]["content"]; out=json.loads(txt[txt.find("{"):txt.rfind("}")+1])
 except Exception as e: out={"place_type":"unknown","short_description":"visual description unavailable","visible_landmarks":[],"confidence":0.0,"error":type(e).__name__}
 out["latency_s"]=time.perf_counter()-t;return out
def main():
 g=json.loads(SRC.read_text());nodes=[n for n in g["nodes"] if n["scene_id"] in SCENES]; by={}
 for n in nodes:by.setdefault(n["scene_id"],[]).append(n)
 evidence=OUT/"node_evidence";evidence.mkdir(parents=True,exist_ok=True);cache=OUT/"node_semantic_cache.jsonl";old={}
 if cache.exists():old={r["node_id"]:r for r in map(json.loads,cache.read_text().splitlines()) if r}
 rows=[]
 for scene,ns in by.items():
  sim=make_sim(scene);agent=sim.initialize_agent(0)
  try:
   for n in ns:
    nid=n["node_id"];stem=nid.replace(":","_");path=evidence/f"{stem}.jpg"
    p=n["geometry"]["center_xz"];pos=np.asarray(sim.pathfinder.snap_point(np.asarray([p[0],0,p[1]],np.float32)),np.float32)
    if not np.isfinite(pos).all(): continue
    rgb,_,_,_=render(sim,agent,pos,yaw_rotation(0.0));Image.fromarray(rgb).save(path)
    semantic=old[nid]["semantic"] if nid in old else ask(rgb,n["kind"])
    n["semantic"]={**semantic,"evidence_rgb":str(path.relative_to(ROOT)),"source":"frozen_qwen3_vl_cached"}
    rows.append({"node_id":nid,"semantic":n["semantic"]})
    if nid not in old:
     with cache.open("a") as f:f.write(json.dumps(rows[-1],ensure_ascii=False)+"\n")
  finally:sim.close()
 g["semantic_annotation"]={"backend":"qwen3-vl:8b-instruct","policy":"conservative visual descriptions only; no hidden poses/routes","node_count":len(rows)}
 (OUT/"semantic_topology.json").write_text(json.dumps(g,indent=2,ensure_ascii=False)+"\n")
 print(json.dumps({"nodes":len(rows),"output":str(OUT/"semantic_topology.json"),"cache":str(cache)},indent=2))
if __name__=="__main__":main()
