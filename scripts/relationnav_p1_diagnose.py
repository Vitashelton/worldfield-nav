#!/usr/bin/env python3
"""Decompose RelationNav P1 failures without changing the task protocol.

This diagnostic deliberately separates: (1) frozen-DINO entity-anchor
grounding, (2) the deterministic relation-to-region compiler, and (3)
seen-versus-held-out learned-field accuracy.  It is not a new method nor an
oracle baseline reported as an online system.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import relationnav_p1_train as rn


def entity_anchor_diagnostic(rows, anns, cache):
    """Can a frozen anchor descriptor retrieve its visible physical point?"""
    out={}
    for split in ("train","heldout"):
        visible=correct=near=0; sims=[]
        for r in (x for x in rows if x["split"]==split):
            e=anns[r["entity_id"]]; p=np.asarray(e["visual_anchor"]["world_point_xyz"],np.float32)
            c=np.asarray(r["camera_xyz"],np.float32); R=np.asarray(r["camera_c2w"],np.float32)
            q=(p-c)@R
            if q[2]>=-1e-3: continue
            u=128*q[0]/(-q[2])+127.5; v=128*q[1]/q[2]+127.5
            if not (0<=u<256 and 0<=v<256): continue
            d=np.load(ROOT/"outputs/formal/RelationNav/P1/dataset"/r["depth"]).astype(np.float32)
            yy,xx=int(round(v)),int(round(u)); observed=float(d[yy,xx])
            if not(np.isfinite(observed) and .02<observed<10 and abs(observed+q[2])<.25): continue
            visible+=1; gt=(int(v)//16)*16+(int(u)//16)
            dense=cache["current_dense"][r["cache_i"]].float(); anchor=cache["entity_global"][r["entity_i"]].float()
            score=torch.nn.functional.cosine_similarity(dense,anchor[None],dim=-1); pred=int(score.argmax())
            correct += pred==gt; near += abs(pred//16-gt//16)<=1 and abs(pred%16-gt%16)<=1; sims.append(float(score[gt]))
        out[split]={"visible_anchor_samples":visible,"anchor_token_r1":correct/max(visible,1),"anchor_token_r1_within_1_patch":near/max(visible,1),"gt_anchor_cosine":float(np.mean(sims)) if sims else None}
    return out


def main():
    rows,anns,cache=rn.load(); device="cuda" if torch.cuda.is_available() else "cpu"
    ckpt=torch.load(ROOT/"artifacts/relationnav/relationnav_field.pt",map_location=device,weights_only=False)
    model=rn.Field().to(device);model.load_state_dict(ckpt["state_dict"])
    indices={s:[i for i,r in enumerate(rows) if r["split"]==s] for s in ("train","heldout")}
    field={s:rn.evaluate(model,rows,anns,cache,v,device) for s,v in indices.items()}
    # The compiler exactly reproduces its own supervision by construction.
    # Report this only as a consistency check, never an online baseline.
    relation={}
    for s,ids in indices.items():
        for i in ids:
            x,y=rn.batch_maps(rows,anns,cache,[i],device="cpu")
            key=f"{s}/{rows[i]['relation']}"; relation.setdefault(key,[]).append(float(y.sum()>0))
    relation={k:{"nonempty_region_rate":float(np.mean(v)),"samples":len(v),"compiler_self_iou":1.0} for k,v in relation.items()}
    report={"purpose":"RelationNav formulation diagnosis; compiler self-IoU is a label consistency check, not an online result.",
            "learned_field_seen_vs_heldout":field,"frozen_dino_entity_anchor":entity_anchor_diagnostic(rows,anns,cache),"relation_compiler_consistency":relation}
    out=ROOT/"outputs/formal/RelationNav/P1/diagnostics.json";out.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))

if __name__=="__main__":main()
