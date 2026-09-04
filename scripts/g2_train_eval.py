#!/usr/bin/env python3
"""GPU-resident MetricAnchor train/eval with real three-view track loss."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np,torch,torch.nn.functional as F,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from geoanchor.metricanchor import ResidualMetricAdapter
from geoanchor.correspondence import patch_center, depth_at, backproject_pixel, intrinsics
def read(n):return [json.loads(x) for x in (ROOT/'outputs/formal/G2/correspondence_dataset'/n).read_text().splitlines()]
def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=0);p.add_argument('--smoke',action='store_true');p.add_argument('--output',default='outputs/formal/G2/train_eval');a=p.parse_args();cfg=yaml.safe_load((ROOT/'configs/benchmark/g2_g3_metricanchor.yaml').read_text());torch.manual_seed(a.seed);np.random.seed(a.seed);dev='cuda'
 files=sorted((ROOT/'outputs/formal/G2/feature_cache').glob('*_traj*.npz')); fmap={}; chunks=[]; offset=0
 for f in files:
  arr=np.load(f)['features'].astype('float32'); fmap[f.stem]=offset; chunks.append(arr); offset+=len(arr)
 X=F.normalize(torch.from_numpy(np.concatenate(chunks)).to(dev),dim=-1)
 geometry_cache=ROOT/'outputs/formal/G2/geometry_cache'; candidate_world=torch.load(geometry_cache/'candidate_world_xyz.pt',map_location=dev); candidate_valid=torch.load(geometry_cache/'candidate_world_valid.pt',map_location=dev)
 def pack(rs):return torch.tensor([[fmap[r['trajectory_id']]+r['source_frame'],fmap[r['trajectory_id']]+r['target_frame'],r['source_patch'],r['target_patch'],r['hard_negative_patch']] for r in rs],device=dev,dtype=torch.long)
 train_rows,val_rows,unseen_rows=read('train.jsonl'),read('val.jsonl'),read('unseen_test.jsonl');tr,va,un=pack(train_rows),pack(val_rows),pack(unseen_rows);assert tr[:,:2].min()>=0 and tr[:,:2].max()<len(X);assert tr[:,2:5].min()>=0 and tr[:,2:5].max()<256;gt_va=torch.tensor([r['world_xyz'] for r in val_rows],device=dev);gt_un=torch.tensor([r['world_xyz'] for r in unseen_rows],device=dev);tracks=read('tracks.jsonl');track_t=torch.tensor([[fmap[r['trajectory_id']]+r['frames'][k] for k in range(3)]+r['patches'] for r in tracks],device=dev,dtype=torch.long);assert track_t[:,:3].min()>=0 and track_t[:,:3].max()<len(X);assert track_t[:,3:].min()>=0 and track_t[:,3:].max()<256;bs=int(cfg['adapter']['batch_size']);epochs=1 if a.smoke else int(cfg['adapter']['epochs']);out=ROOT/a.output;out.mkdir(parents=True,exist_ok=True);result={'batch_size':bs,'track_loss_nonzero':False,'methods':{}}
 for method in ['M1','M2','M3']:
  m=ResidualMetricAdapter().to(dev);opt=torch.optim.AdamW(m.parameters(),lr=cfg['adapter']['lr']);t0=time.time();last=0
  for ep in range(epochs):
   for j in range(0,len(tr),bs):
    b=tr[torch.randperm(len(tr),device=dev)[:bs]];fr=torch.unique(b[:,:2]);Y=m(X[fr]).reshape(len(fr),256,384);loc=torch.searchsorted(fr,b[:,:2].contiguous());q=Y[loc[:,0],b[:,2]];pos=(q*Y[loc[:,1],b[:,3]]).sum(-1);neg=(q*Y[loc[:,1],b[:,4]]).sum(-1);loss=F.softplus((neg-pos)/cfg['adapter']['temperature']).mean()
    if method=='M3':
     z=track_t[torch.randint(len(track_t),(min(bs//3,len(track_t)),),device=dev)];tf=torch.unique(z[:,:3]);TY=m(X[tf]).reshape(len(tf),256,384);tl=torch.searchsorted(tf,z[:,:3].contiguous());td=TY[tl,z[:,3:]];track=((td[:,0]*td[:,1]).sum(-1)-(td[:,0]*td[:,2]).sum(-1)).abs().mean();loss=loss+float(cfg['adapter']['track_weight'])*track;last=float(track.detach());result['track_loss_nonzero']=True
    opt.zero_grad();loss.backward();opt.step()
  result['methods'][method]={'train_seconds':time.time()-t0,'last_track_loss':last};torch.save(m.state_dict(),out/f'{method}.pt')
 def ev(method,rs,gt):
  m=None if method=='M0' else ResidualMetricAdapter().to(dev).eval()
  if m:m.load_state_dict(torch.load(out/f'{method}.pt',map_location=dev))
  fr=torch.unique(rs[:,:2]);Y=F.normalize(X[fr],dim=-1) if m is None else m(X[fr]).reshape(len(fr),256,384);loc=torch.searchsorted(fr,rs[:,:2].contiguous());q=Y[loc[:,0],rs[:,2]];s=torch.bmm(q[:,None,:],Y[loc[:,1]].transpose(1,2)).squeeze(1);order=s.argsort(1,descending=True);rank=(order==rs[:,3,None]).nonzero()[:,1];top=order[:,0];xyz=candidate_world[rs[:,1],top];valid=candidate_valid[rs[:,1],top];err=torch.where(valid,(xyz-gt).norm(dim=1),torch.full_like(gt[:,0],10.0));return {'r_at_1':float((rank==0).float().mean()),'r_at_5':float((rank<5).float().mean()),'margin':float((s.gather(1,rs[:,3,None])-s.gather(1,rs[:,4,None])).mean()),'world_localization_error_m':float(err.mean()),'n':len(rs)}
 for method in ['M0','M1','M2','M3']:result['methods'].setdefault(method,{})['validation']=ev(method,va,gt_va);result['methods'][method]['unseen']=ev(method,un,gt_un)
 result['params']=ResidualMetricAdapter().parameter_count;result['track_count']=len(track_t);result['seed']=a.seed;(out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
