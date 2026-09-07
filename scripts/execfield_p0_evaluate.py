"""Evaluate the fixed-candidate ExecField P0 oracle gap."""
from __future__ import annotations
import csv, json
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
P0=ROOT/"outputs/formal/ExecField/P0"

def choose_oracle(cands, rank):
    for i in rank:
        if cands[i]["eventual_outcome"]: return i
    return rank[0]

def record(ep, method, idx):
    c=ep["candidates"][idx]; d0=float(ep["initial_geodesic_m"]); progress=float(c["geodesic_progress_m"])
    dtg=max(0., d0-progress) if c["reachable"] else 2*d0
    path=float(np.linalg.norm(np.asarray(c["raw_xyz"])-np.asarray(ep["start_xyz"])))+dtg
    succ=bool(c["eventual_outcome"])
    return {"episode_id":ep["episode_id"],"scene_id":ep["scene_id"],"method":method,"selected_candidate":idx,"success":int(succ),"spl":float((d0/max(d0,path)) if succ else 0.),"final_dtg_m":dtg,"invalid_subgoal":int(not c["collision_free"]),"wrong_goal":int(c["collision_free"] and not succ),"failure_reason":c["failure_reason"],"clearance_m":c["clearance_m"],"progress_m":progress}

def main():
    eps=json.loads((P0/"episodes.json").read_text()); cache={json.loads(x)["request_id"]:json.loads(x) for x in (P0/"vlm_cache.jsonl").read_text().splitlines() if x.strip()}
    if set(e["episode_id"] for e in eps)!=set(cache): raise RuntimeError("cache/episode IDs mismatch")
    rows=[]
    for ep in eps:
        rank=cache[ep["episode_id"]]["ranking"]; cands=ep["candidates"]
        rows += [record(ep,"M0_geometry_only",0),record(ep,"M1_vlm_only",rank[0]),record(ep,"M1_oracle",choose_oracle(cands,rank))]
    out=P0/"metrics"; out.mkdir(exist_ok=True)
    fields=list(rows[0]);
    with (out/"episode_results.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    summary=[]
    for method in ["M0_geometry_only","M1_vlm_only","M1_oracle"]:
        a=[r for r in rows if r["method"]==method]
        summary.append({"method":method,"episodes":len(a),"SR":np.mean([r["success"] for r in a]),"SPL":np.mean([r["spl"] for r in a]),"Final_DTG_m":np.mean([r["final_dtg_m"] for r in a]),"Invalid_Subgoal":np.mean([r["invalid_subgoal"] for r in a]),"Wrong_Goal":np.mean([r["wrong_goal"] for r in a])})
    pa=ROOT/"paper_assets"; (pa/"tables").mkdir(parents=True,exist_ok=True);(pa/"figures").mkdir(parents=True,exist_ok=True)
    with (pa/"tables/execfield_p0_oracle_gap.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    json.dump({"summary":summary,"failure_reasons":{m:dict(Counter(r["failure_reason"] for r in rows if r["method"]==m)) for m in ["M0_geometry_only","M1_vlm_only","M1_oracle"]}},open(out/"metrics.json","w"),indent=2)
    metrics=["SR","SPL","Invalid_Subgoal","Wrong_Goal"]; fig,axs=plt.subplots(1,4,figsize=(16,4)); labels=[x["method"].replace("_","\n") for x in summary]
    for ax,key in zip(axs,metrics):
        vals=[x[key] for x in summary]; ax.bar(labels,vals,color=["#777777","#e89b3c","#2878b5"]); ax.set_title(key); ax.set_ylim(0,1); ax.grid(axis="y",alpha=.25)
        for i,v in enumerate(vals): ax.text(i,v+.025,f"{v:.2f}",ha="center")
    fig.suptitle("ExecField P0: Oracle Executability Gap (24 fixed ImageNav episodes)",fontsize=15);fig.tight_layout();fig.savefig(pa/"figures/execfield_p0_oracle_gap.png",dpi=200);print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
