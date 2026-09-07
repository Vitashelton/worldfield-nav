#!/usr/bin/env python3
"""Train/evaluate the only learned P1 component: a lightweight trajectory ranker."""
from __future__ import annotations

import argparse, csv, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]

class Ranker(nn.Module):
    def __init__(self, dim=6):
        super().__init__(); self.net=nn.Sequential(nn.Linear(dim,128),nn.GELU(),nn.Linear(128,128),nn.GELU())
        self.success=nn.Linear(128,1); self.progress=nn.Linear(128,1)
    def forward(self,x):
        h=self.net(x); return self.success(h).squeeze(-1),self.progress(h).squeeze(-1)

def cache(path):
    return {x["request_id"] if "request_id" in x else x["episode_id"]:x for x in (json.loads(z) for z in path.read_text().splitlines() if z.strip())}

def build(root: Path):
    manifest=json.loads((root/"dataset_manifest.json").read_text()); dino=cache(root/"dino_cache.jsonl"); vlm=cache(root/"vlm_cache.jsonl")
    samples=[]; groups=[]
    for ei,e in enumerate(manifest):
        if e["episode_id"] not in dino or e["episode_id"] not in vlm: continue
        ds=np.asarray(dino[e["episode_id"]]["trajectory_goal_scores"],np.float32); vs=np.asarray(vlm[e["episode_id"]]["semantic_scores"],np.float32)
        start=len(samples)
        for i,t in enumerate(e["trajectories"]):
            g=t["geometry"]; samples.append([ei,i,g["proposal_length_m"],g["image_visible_ratio"],g["observed_depth_min_m"],g["observed_depth_mean_m"],vs[i],ds[i],float(t["eventual_outcome"]),t["geodesic_progress_m"],e["initial_geodesic_m"],e["split"]])
        groups.append((e,start,start+len(e["trajectories"])))
    return samples,groups

def metrics(groups, rows, score_fn):
    decisions=[]
    for e,a,b in groups:
        chunk=np.asarray(rows[a:b], object); idx=int(np.argmax(score_fn(chunk, a))); chosen=chunk[idx]
        best=max(chunk,key=lambda r:r[9])
        decisions.append({"episode_id":e["episode_id"],"scene_id":e["scene_id"],"split":e["split"],"selected_trajectory":int(chosen[1]),"success":float(chosen[8]),"progress_m":float(chosen[9]),"initial_geodesic_m":float(chosen[10]),"final_dtg_m":max(0.,float(chosen[10]-chosen[9])),"regret_m":max(0.,float(best[9]-chosen[9]))})
    n=len(decisions); return {"episodes":n,"ranking_accuracy":float(np.mean([d["regret_m"]<1e-6 for d in decisions])) if n else 0.,"regret_m":float(np.mean([d["regret_m"] for d in decisions])) if n else 0.,"SR":float(np.mean([d["success"] for d in decisions])) if n else 0.,"SPL_proxy":float(np.mean([d["success"]*max(0,d["progress_m"])/max(d["initial_geodesic_m"],.1) for d in decisions])) if n else 0.,"Final_DTG_m":float(np.mean([d["final_dtg_m"] for d in decisions])) if n else 0.,"Invalid_collision_rate":float(np.mean([1-d["success"] for d in decisions])) if n else 0.},decisions

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=ROOT/"outputs/formal/TrajectoryGrounding/P1");ap.add_argument("--epochs",type=int,default=25);ap.add_argument("--seed",type=int,default=20260907);args=ap.parse_args()
    torch.manual_seed(args.seed); rows,groups=build(args.root)
    if not rows: raise RuntimeError("DINO and VLM caches must be complete before ranker training")
    arr=np.asarray([[r[i] for i in range(2,8)] for r in rows],np.float32); y=np.asarray([r[8] for r in rows],np.float32); p=np.asarray([r[9]/max(r[10],.1) for r in rows],np.float32)
    train=np.asarray([r[11]=="train" for r in rows]); mean=arr[train].mean(0); std=arr[train].std(0).clip(.1); x=(arr-mean)/std
    device="cuda" if torch.cuda.is_available() else "cpu"; model=Ranker(x.shape[1]).to(device); opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    xx=torch.tensor(x[train],device=device); yy=torch.tensor(y[train],device=device); pp=torch.tensor(p[train],device=device); started=time.time()
    for _ in range(args.epochs):
        order=torch.randperm(len(xx),device=device)
        for ids in order.split(1024):
            logits,progress=model(xx[ids]); loss=nn.functional.binary_cross_entropy_with_logits(logits,yy[ids])+nn.functional.smooth_l1_loss(progress,pp[ids]);opt.zero_grad();loss.backward();opt.step()
    with torch.inference_mode(): logits,progress=model(torch.tensor(x,device=device)); learned=torch.sigmoid(logits).cpu().numpy()+.25*progress.cpu().numpy()
    variants={"Planner-only":lambda q,a:q[:,3]+.25*q[:,5],"VLM-only":lambda q,a:q[:,6],"DINO-only":lambda q,a:q[:,7],"Planner+VLM":lambda q,a:q[:,3]+.25*q[:,5]+q[:,6],"Planner+DINO":lambda q,a:q[:,3]+.25*q[:,5]+q[:,7],"Fixed Fusion":lambda q,a:q[:,3]+.25*q[:,5]+q[:,6]+q[:,7],"Learned Ranker":lambda q,a:learned[a:a+len(q)],"Oracle upper bound":lambda q,a:q[:,8]+q[:,9]}
    all_groups=[(e,a,b) for e,a,b in groups]
    output=[]; result_rows=[]
    for name,fn in variants.items():
        m,detail=metrics(all_groups,rows,lambda q,a, fn=fn:fn(np.asarray(q,np.float32),a))
        for split in ("val","unseen"):
            indices=[i for i,(e,_,_) in enumerate(all_groups) if e["split"]==split]; subset=[detail[i] for i in indices];
            if subset: output.append({"method":name,"split":split,**{k:float(np.mean([d[k] for d in subset])) for k in ["ranking_accuracy","regret_m","SR","SPL_proxy","Final_DTG_m","Invalid_collision_rate"]}})
    (args.root/"ranker_metrics.json").write_text(json.dumps({"rows":len(rows),"parameters":sum(x.numel() for x in model.parameters()),"seconds":time.time()-started,"metrics":output},indent=2)+"\n")
    with (ROOT/"paper_assets/tables/trajectory_grounding_p1_main.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(output[0]));w.writeheader();w.writerows(output)
    torch.save({"state_dict":model.state_dict(),"mean":mean,"std":std,"feature_order":["length","visible","depth_min","depth_mean","vlm","dino"]},args.root/"trajectory_ranker.pt")

if __name__=="__main__": main()
