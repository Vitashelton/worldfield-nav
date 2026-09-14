#!/usr/bin/env python3
"""Train/evaluate the lightweight RelationNav spatial goal field.

No gradient reaches DINOv3.  The cache contains frozen 16x16 descriptors;
this script projects them with aligned depth into a robot-centric 64x64 map,
combines relation/entity/history channels, and learns a dense executable
region mask.  Curated world geometry is used exclusively for supervision.
"""
from __future__ import annotations

import argparse, json, math, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"outputs/formal/RelationNav/P1/dataset"
ANN=ROOT/"configs/relationnav/entities.json"
REL={"APPROACH":0,"CROSS":1,"ENTER":2,"OBSERVE":3}


class Field(nn.Module):
    def __init__(self, cin=54, width=96):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(cin,width,3,padding=1),nn.GELU(),nn.Conv2d(width,width,3,padding=1),nn.GELU(),
            nn.MaxPool2d(2),nn.Conv2d(width,width*2,3,padding=1),nn.GELU(),nn.Conv2d(width*2,width*2,3,padding=1),nn.GELU(),
            nn.Upsample(scale_factor=2,mode="bilinear",align_corners=False),nn.Conv2d(width*2,width,3,padding=1),nn.GELU(),nn.Conv2d(width,1,1))
    def forward(self,x): return self.net(x)


def load():
    manifest=json.loads((DATA/"dataset_manifest.json").read_text())["episodes"]
    rows=[]
    for ep in manifest:
        for p in ep["phases"]:
            p=dict(p); p["scene_id"]=ep["scene_id"];p["split"]=ep["split"];p["portal_plane"]=ep["portal_plane"];p["episode_id"]=ep["episode_id"]; rows.append(p)
    anns={x["entity_id"]:x for x in json.loads(ANN.read_text())["entities"]}
    cache=torch.load(DATA/"dino_cache.pt",map_location="cpu",weights_only=False)
    key={k:i for i,k in enumerate(cache["decision_keys"])}; ent={k:i for i,k in enumerate(cache["entity_ids"])}
    for r in rows: r["cache_i"]=key[f"{r['episode_id']}:{r['phase_index']}"];r["entity_i"]=ent[r["entity_id"]]
    return rows,anns,cache


def point_poly(x,z,poly):
    inside=torch.zeros_like(x,dtype=torch.bool)
    for i in range(len(poly)):
        x1,z1=poly[i];x2,z2=poly[(i+1)%len(poly)]
        inside ^= ((z1>z)!=(z2>z)) & (x < (x2-x1)*(z-z1)/(z2-z1+1e-6)+x1)
    return inside


def observer_xz(entity):
    aid=entity["visual_anchor"]["anchor_view_id"]; scene=entity["scene_id"]
    src=ROOT/"outputs/formal/GoalPose/P0/anchor_views"/f"anchor_view_candidates_{scene}.jsonl"
    for line in src.read_text().splitlines():
        v=json.loads(line)
        if v["anchor_view_id"]==aid: return v["camera_xyz"][0],v["camera_xyz"][2]
    raise KeyError(aid)


def batch_maps(rows, anns, cache, ids, device, grid=64):
    # Frozen descriptors are projected from camera coordinates. Inputs have no
    # privileged target position; labels below are separate offline supervision.
    dense=cache["current_dense"][torch.tensor([rows[i]["cache_i"] for i in ids])].float().to(device) # B,256,384
    ent=cache["entity_global"][torch.tensor([rows[i]["entity_i"] for i in ids])].float().to(device)
    B=dense.shape[0]; H=W=grid; yy,xx=torch.meshgrid(torch.arange(H,device=device),torch.arange(W,device=device),indexing="ij")
    local_x=(xx.float()+.5-W/2)*.125; local_f=(yy.float()+.5)*.125-2.0
    # Geometry-aware lifting: each frozen patch descriptor goes to its metric
    # robot-centric depth location, then collisions use normalized mean pooling.
    visual=torch.zeros((B,48,H,W),device=device)
    geom=torch.zeros((B,3,H,W),device=device); sim=torch.zeros((B,1,H,W),device=device); rel=torch.zeros((B,2,H,W),device=device); labels=torch.zeros((B,1,H,W),device=device)
    for j,idx in enumerate(ids):
        r=rows[idx]; c2w=np.asarray(r["camera_c2w"],np.float32); cam=np.asarray(r["camera_xyz"],np.float32); yaw=float(r["camera_yaw_rad"])
        # The 16x16 patch centers are backprojected using aligned depth. Camera
        # x is robot-right and -camera-z is robot-forward for this rig.
        depth=np.load(DATA/r["depth"]).astype(np.float32); ds=torch.tensor(depth[7::16,7::16],device=device)
        valid=torch.isfinite(ds)&(ds>.02)&(ds<10.0)
        u=torch.arange(16,device=device).repeat(16)*16+7.5
        px=(u-127.5)*ds.reshape(-1)/128.0; pf=ds.reshape(-1); keep=valid.reshape(-1)&(px>-4)&(px<4)&(pf>-2)&(pf<6)
        gx=((px+4)/.125).long().clamp(0,W-1); gy=((pf+2)/.125).long().clamp(0,H-1); cell=gy*W+gx
        count=torch.zeros(H*W,device=device); count.index_add_(0,cell[keep],torch.ones(int(keep.sum()),device=device))
        acc=torch.zeros((48,H*W),device=device); acc.index_add_(1,cell[keep],dense[j,keep,:48].T)
        visual[j]= (acc/count.clamp_min(1)).reshape(48,H,W)
        geom[j,0]=(count.reshape(H,W)>0).float(); geom[j,1]=(count.reshape(H,W)>0).float()
        depth_acc=torch.zeros(H*W,device=device); depth_acc.index_add_(0,cell[keep],pf[keep]/10.0);geom[j,2]=(depth_acc/count.clamp_min(1)).reshape(H,W)
        e=ent[j]; token_sim=F.cosine_similarity(dense[j],e[None],dim=-1).reshape(1,1,16,16)
        sim[j:j+1]=F.interpolate(token_sim,size=(H,W),mode="bilinear",align_corners=False)
        rel[j,0].fill_(REL[r["relation"]]/3.0); rel[j,1].fill_(1.0)
        # local robot-centric cell -> world xz; yaw follows executor history.
        forward=np.array([-math.sin(yaw),-math.cos(yaw)],np.float32); right=np.array([math.cos(yaw),-math.sin(yaw)],np.float32)
        wx=torch.tensor(cam[0],device=device)+local_x*float(right[0])+local_f*float(forward[0]); wz=torch.tensor(cam[2],device=device)+local_x*float(right[1])+local_f*float(forward[1])
        entity=anns[r["entity_id"]]; relation=r["relation"]
        if relation in {"APPROACH","CROSS"}:
            pl=r["portal_plane"]; p=np.asarray(pl["point_world_xyz"],np.float32); n=np.asarray(pl["normal_xz"],np.float32)
            signed=n[0]*(wx-float(p[0]))+n[1]*(wz-float(p[2])); tang=-n[1]*(wx-float(p[0]))+n[0]*(wz-float(p[2]))
            desired=float(pl["source_side"] if relation=="APPROACH" else pl["destination_side"])
            mask=(signed*desired>.35)&(signed*desired<1.8)&(tang.abs()<float(pl["width_m"])*.7)
        elif relation=="ENTER": mask=point_poly(wx,wz,entity["area_polygon_xz"])
        else:
            ox,oz=observer_xz(entity);dist=((wx-float(ox))**2+(wz-float(oz))**2).sqrt();mask=(dist<.75)
        labels[j,0]=mask.float()
    # 48 visual + 3 geometry + 1 visual entity similarity + 2 relation + 0 history = 54
    x=torch.cat([visual,geom,sim,rel],1)
    return x,labels


def evaluate(model, rows, anns, cache, indices, device):
    model.eval(); vals=[]
    with torch.no_grad():
        for at in range(0,len(indices),32):
            x,y=batch_maps(rows,anns,cache,indices[at:at+32],device); p=torch.sigmoid(model(x)); pred=p>.5
            inter=(pred&y.bool()).sum((1,2,3)).float(); union=(pred|y.bool()).sum((1,2,3)).float().clamp_min(1); recall=inter/y.sum((1,2,3)).clamp_min(1)
            per_sample_bce=F.binary_cross_entropy(p,y,reduction="none").mean((1,2,3))
            vals += list(zip((inter/union).cpu().tolist(),recall.cpu().tolist(),per_sample_bce.cpu().tolist()))
    a=np.asarray(vals);return {"field_iou":float(a[:,0].mean()),"field_recall":float(a[:,1].mean()),"bce":float(a[:,2].mean()),"samples":len(indices)}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--epochs",type=int,default=20);ap.add_argument("--batch-size",type=int,default=32);ap.add_argument("--lr",type=float,default=3e-4);ap.add_argument("--seed",type=int,default=0);args=ap.parse_args()
    torch.manual_seed(args.seed); device="cuda" if torch.cuda.is_available() else "cpu"; rows,anns,cache=load(); train=[i for i,r in enumerate(rows) if r["split"]=="train"]; held=[i for i,r in enumerate(rows) if r["split"]=="heldout"]
    model=Field().to(device); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4); history=[]; start=time.time()
    rng=np.random.default_rng(args.seed)
    for ep in range(args.epochs):
        model.train();rng.shuffle(train); losses=[]
        for at in range(0,len(train),args.batch_size):
            x,y=batch_maps(rows,anns,cache,train[at:at+args.batch_size],device); logits=model(x); pos=((1-y).sum()/y.sum().clamp_min(1)).clamp(1,15)
            loss=F.binary_cross_entropy_with_logits(logits,y,pos_weight=pos);opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.item()))
        report={"epoch":ep,"train_loss":float(np.mean(losses)),"heldout":evaluate(model,rows,anns,cache,held,device)};history.append(report);print(json.dumps(report),flush=True)
    out=ROOT/"artifacts/relationnav";out.mkdir(parents=True,exist_ok=True);torch.save({"state_dict":model.state_dict(),"params":sum(p.numel() for p in model.parameters()),"relations":REL},out/"relationnav_field.pt")
    result={"model":"RelationNav lightweight spatial field","frozen_dino":True,"params":sum(p.numel() for p in model.parameters()),"seconds":time.time()-start,"history":history,"final_heldout":history[-1]["heldout"]}
    (ROOT/"outputs/formal/RelationNav/P1/training_metrics.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2))

if __name__=="__main__":main()
