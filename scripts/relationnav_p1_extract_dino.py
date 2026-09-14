#!/usr/bin/env python3
"""Cache frozen DINOv3 dense evidence for RelationNav P1 exactly once."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import timm
from timm.data import resolve_model_data_config

ROOT = Path(__file__).resolve().parents[1]
MODEL = "hf_hub:timm/vit_small_patch16_dinov3.lvd1689m"


def read(path: Path) -> torch.Tensor:
    im = Image.open(path).convert("RGB").resize((256, 256), Image.Resampling.BICUBIC)
    return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float().div_(255)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", type=Path, default=ROOT / "outputs/formal/RelationNav/P1/dataset"); ap.add_argument("--batch-size", type=int, default=96)
    args = ap.parse_args(); manifest = json.loads((args.root / "dataset_manifest.json").read_text()); episodes = manifest["episodes"]
    decisions = [(episode["episode_id"], phase) for episode in episodes for phase in episode["phases"]]
    entities = {x["entity_id"]: x for x in json.loads((ROOT / "configs/relationnav/entities.json").read_text())["entities"]}
    ids = sorted({p["entity_id"] for _, p in decisions})
    if not torch.cuda.is_available(): raise RuntimeError("RelationNav frozen DINO cache requires CUDA")
    model = timm.create_model(MODEL, pretrained=True).cuda().eval(); cfg = resolve_model_data_config(model)
    mean = torch.tensor(cfg["mean"], device="cuda").view(1,3,1,1); std = torch.tensor(cfg["std"], device="cuda").view(1,3,1,1)
    def forward(images):
        x=(torch.stack(images).cuda(non_blocking=True)-mean)/std
        with torch.inference_mode():
            z=model.forward_features(x); z=z if torch.is_tensor(z) else z.get("x_norm",z.get("x"))
            z=torch.nn.functional.normalize(z[:,int(model.num_prefix_tokens):],dim=-1)
        return z.cpu().half()
    start=time.perf_counter(); chunks=[]
    for at in range(0,len(decisions),args.batch_size):
        chunks.append(forward([read(args.root/p["rgb"]) for _,p in decisions[at:at+args.batch_size]]))
    current=torch.cat(chunks); anchors=forward([read(ROOT/"outputs/formal/GoalPose"/entities[e]["visual_anchor"]["rgb"]) for e in ids])
    payload={"model":MODEL,"frozen":True,"decision_keys":[f"{eid}:{p['phase_index']}" for eid,p in decisions],"current_dense":current,
             "entity_ids":ids,"entity_global":torch.nn.functional.normalize(anchors.float().mean(1),dim=-1).half(),"input_size":[256,256],"grid":[16,16]}
    out=args.root/"dino_cache.pt"; torch.save(payload,out)
    (args.root/"dino_cache_metrics.json").write_text(json.dumps({"frames":len(decisions),"entities":len(ids),"shape":list(current.shape),"storage_mb":out.stat().st_size/1024**2,"seconds":time.perf_counter()-start,"model":MODEL},indent=2)+"\n")
    print((args.root/"dino_cache_metrics.json").read_text())

if __name__=="__main__": main()
