#!/usr/bin/env python3
"""Train fixed MetricAnchor ablations exclusively from cached DINO tokens."""
from __future__ import annotations
import argparse,csv,json,sys,time
from collections import defaultdict
from pathlib import Path
import numpy as np,torch,torch.nn.functional as F,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from geoanchor.metricanchor import ResidualMetricAdapter
def load_rows(p):return [json.loads(x) for x in (ROOT/p).read_text().splitlines()]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/benchmark/g2_g3_metricanchor.yaml');ap.add_argument('--output',default='outputs/formal/G2/train_eval');ap.add_argument('--seed',type=int,default=0);a=ap.parse_args();c=yaml.safe_load((ROOT/a.config).read_text());torch.manual_seed(a.seed);np.random.seed(a.seed);out=ROOT/a.output;out.mkdir(parents=True,exist_ok=True); cache={}
 def feat(t,f):
  if t not in cache: cache[t]=np.load(ROOT/'outputs/formal/G2/feature_cache'/f'{t}.npz')['features'].astype('float32')
  return cache[t][f]
 train=load_rows('outputs/formal/G2/correspondence_dataset/train.jsonl'); val=load_rows('outputs/formal/G2/correspondence_dataset/val.jsonl'); test=load_rows('outputs/formal/G2/correspondence_dataset/unseen_test.jsonl')
 dev='cuda'; methods=['M1','M2','M3']; states={}; timing={}
 for method in methods:
  m=ResidualMetricAdapter().to(dev);opt=torch.optim.AdamW(m.parameters(),lr=c['adapter']['lr']);start=time.time()
  for ep in range(c['adapter']['epochs']):
   np.random.shuffle(train)
   for j in range(0,len(train),32):
    r=train[j:j+32]; x=torch.tensor(np.stack([feat(z['trajectory_id'],z['source_frame']) for z in r]),device=dev);y=torch.tensor(np.stack([feat(z['trajectory_id'],z['target_frame']) for z in r]),device=dev)
    ax=m(x);ay=m(y); ix=torch.arange(len(r),device=dev); pos=(ax[ix, [z['source_patch'] for z in r]]*ay[ix,[z['target_patch'] for z in r]]).sum(-1)
    negidx=[np.random.randint(256) if method=='M1' else z['hard_negative_patch'] for z in r];neg=(ax[ix,[z['source_patch'] for z in r]]*ay[ix,negidx]).sum(-1)
    loss=F.softplus((neg-pos)/.07).mean()
    if method=='M3':loss=loss+.1*((ax-x)**2).mean()
    opt.zero_grad();loss.backward();opt.step()
  states[method]={k:v.cpu() for k,v in m.state_dict().items()};timing[method]=time.time()-start
 def ev(method,rows):
  m=None if method=='M0' else ResidualMetricAdapter().to(dev).eval();
  if m:m.load_state_dict(states[method]);g=defaultdict(list)
  for r in rows:
   x=torch.tensor(feat(r['trajectory_id'],r['source_frame'])[None],device=dev);y=torch.tensor(feat(r['trajectory_id'],r['target_frame'])[None],device=dev)
   with torch.no_grad(): ax=F.normalize(x,dim=-1) if m is None else m(x);ay=F.normalize(y,dim=-1) if m is None else m(y)
   q=ax[0,r['source_patch']];s=(q@ay[0].T);order=torch.argsort(s,descending=True);rank=int((order==r['target_patch']).nonzero()[0]);pos=float(s[r['target_patch']]);neg=float(s[r['hard_negative_patch']]);g[r['regime']].append((rank,pos,neg))
  return {k:{'r1':np.mean([z[0]==0 for z in v]),'r5':np.mean([z[0]<5 for z in v]),'margin':np.mean([z[1]-z[2] for z in v]),'world_localization_error_m':float('nan'),'n':len(v)} for k,v in g.items()}
 result={};allrows=[]
 for method in ['M0']+methods:
  d=ev(method,val+test); result[method]=d
  for k,v in d.items():allrows.append({'method':method,'regime':k,**v})
 torch.save({'state_dict':states['M3'],'architecture':'384-128-dw3x3-128-384 residual L2','parameter_count':ResidualMetricAdapter().parameter_count},ROOT/'artifacts/metricanchor/adapter_best.pt') if (ROOT/'artifacts/metricanchor').mkdir(parents=True,exist_ok=True) is None else None
 with (ROOT/'paper_assets/tables/metricanchor_main_results.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=allrows[0]);w.writeheader();w.writerows(allrows)
 (out/'metrics.json').write_text(json.dumps({'methods':result,'training_seconds':timing,'params':ResidualMetricAdapter().parameter_count},indent=2)+'\n')
if __name__=='__main__':main()
