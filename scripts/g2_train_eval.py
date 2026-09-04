#!/usr/bin/env python3
"""GPU-resident MetricAnchor training; no per-step file or Python data work."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np,torch,torch.nn.functional as F,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from geoanchor.metricanchor import ResidualMetricAdapter
def rows(name):return [json.loads(x) for x in (ROOT/'outputs/formal/G2/correspondence_dataset'/name).read_text().splitlines()]
def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=0);p.add_argument('--smoke',action='store_true');p.add_argument('--output',default='outputs/formal/G2/train_eval');a=p.parse_args();c=yaml.safe_load((ROOT/'configs/benchmark/g2_g3_metricanchor.yaml').read_text());dev='cuda';torch.manual_seed(a.seed)
 files=sorted((ROOT/'outputs/formal/G2/feature_cache').glob('*_traj*.npz')); ids={f.stem:i for i,f in enumerate(files)}
 X=torch.from_numpy(np.concatenate([np.load(f)['features'].astype('float32') for f in files])).to(dev); X=F.normalize(X,dim=-1); # [4500,16,16,384]
 def pack(rs):
  return torch.tensor([[ids[r['trajectory_id']]*150+r['source_frame'],ids[r['trajectory_id']]*150+r['target_frame'],r['source_patch'],r['target_patch'],r['hard_negative_patch']] for r in rs],device=dev,dtype=torch.long)
 tr=pack(rows('train.jsonl')); va=pack(rows('val.jsonl')); un=pack(rows('unseen_test.jsonl')); bs=int(c['adapter']['batch_size']); steps=1000 if a.smoke else (len(tr)+bs-1)//bs*int(c['adapter']['epochs'])
 m=ResidualMetricAdapter().to(dev);opt=torch.optim.AdamW(m.parameters(),lr=c['adapter']['lr']);torch.cuda.reset_peak_memory_stats();start=time.time()
 for step in range(steps):
  ii=torch.randint(len(tr),(bs,),device=dev);b=tr[ii];fr=torch.unique(b[:,:2]);Y=m(X[fr]).reshape(len(fr),256,384);loc=torch.searchsorted(fr,b[:,:2].contiguous());q=Y[loc[:,0],b[:,2]];pos=(q*Y[loc[:,1],b[:,3]]).sum(-1);neg=(q*Y[loc[:,1],b[:,4]]).sum(-1);loss=F.softplus((neg-pos)/c['adapter']['temperature']).mean();opt.zero_grad();loss.backward();opt.step()
 torch.cuda.synchronize();sec=time.time()-start;result={'batch_size':bs,'steps':steps,'samples_per_second':steps*bs/sec,'peak_vram_mib':torch.cuda.max_memory_allocated()/2**20,'projected_m1_m2_m3_seconds':sec/steps*((len(tr)+bs-1)//bs*c['adapter']['epochs']*3)}
 out=ROOT/a.output;out.mkdir(parents=True,exist_ok=True);(out/'throughput.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
