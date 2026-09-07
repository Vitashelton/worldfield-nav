"""Render representative VLM-only versus oracle P0 cases."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]; P0=ROOT/"outputs/formal/ExecField/P0"

def load(path): return Image.open(path).convert("RGB")
def main():
 eps={x["episode_id"]:x for x in json.loads((P0/"episodes.json").read_text())}; cache={json.loads(x)["request_id"]:json.loads(x) for x in (P0/"vlm_cache.jsonl").read_text().splitlines() if x.strip()}
 cases=[]
 for eid,ep in eps.items():
  rank=cache[eid]["ranking"]; vi=rank[0]; oi=next((i for i in rank if ep["candidates"][i]["eventual_outcome"]),vi)
  if vi!=oi and not ep["candidates"][vi]["eventual_outcome"] and ep["candidates"][oi]["eventual_outcome"]: cases.append((eid,vi,oi))
 cases=cases[:3]
 if len(cases)<3: raise RuntimeError(f"only {len(cases)} correction cases")
 fig,ax=plt.subplots(3,4,figsize=(14,10))
 for r,(eid,vi,oi) in enumerate(cases):
  ep=eps[eid]; vc,oc=ep["candidates"][vi],ep["candidates"][oi]
  cells=[(ep["goal_image"],"Goal image"),(ep["current_rgb"],"Current RGB"),(vc["view_path"],f"VLM: C{vi} — {vc['failure_reason']}"),(oc["view_path"],f"Oracle: C{oi} — executable")]
  for c,(path,title) in enumerate(cells):
   ax[r,c].imshow(load(P0/path));ax[r,c].set_title(title,fontsize=10);ax[r,c].axis("off")
  ax[r,0].set_ylabel(eid,fontsize=9)
 fig.suptitle("ExecField P0: VLM semantic choice corrected by oracle executability",fontsize=16);fig.tight_layout()
 out=ROOT/"paper_assets/figures";out.mkdir(parents=True,exist_ok=True);fig.savefig(out/"execfield_p0_vlm_oracle_cases.png",dpi=200)
if __name__=="__main__":main()
