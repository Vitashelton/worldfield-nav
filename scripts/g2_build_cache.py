#!/usr/bin/env python3
"""One-time frozen DINOv3 FP16 cache over all completed C1 RGB frames."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np, torch, timm
from PIL import Image
from timm.data import resolve_model_data_config
ROOT=Path(__file__).resolve().parents[1]
MODEL='hf_hub:timm/vit_small_patch16_dinov3.lvd1689m'
SPLITS={'scene01':'train','scene02':'train','scene03':'train','scene09':'train','interior_0405_840145':'train','scene04':'validation','scene05':'validation','scene56':'unseen','scene57':'unseen','scene58':'unseen'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',default='outputs/formal/G2/feature_cache');a=p.parse_args(); out=ROOT/a.output;out.mkdir(parents=True,exist_ok=True)
 if not torch.cuda.is_available():raise RuntimeError('CUDA required')
 m=timm.create_model(MODEL,pretrained=True).cuda().eval();cfg=resolve_model_data_config(m);h,w=cfg['input_size'][-2:];mean=torch.tensor(cfg['mean'],device='cuda').view(1,3,1,1);std=torch.tensor(cfg['std'],device='cuda').view(1,3,1,1)
 started=time.perf_counter(); n=0
 for seq in sorted((ROOT/'outputs/formal/C1/pilot/trajectories').glob('*/sequence.npz')):
  tid=seq.parent.name; scene=tid.rsplit('_traj',1)[0]; target=out/f'{tid}.npz'
  if target.exists(): n+=len(np.load(target)['features']);continue
  with np.load(seq,allow_pickle=False) as d:
   rgb=d['rgb']; feats=[]
   for i in range(0,len(rgb),32):
    x=np.stack([np.asarray(Image.fromarray(z).resize((w,h),Image.Resampling.BICUBIC)) for z in rgb[i:i+32]])
    x=torch.from_numpy(x).permute(0,3,1,2).float().div(255).cuda();x=(x-mean)/std
    with torch.no_grad(): y=m.forward_features(x)
    y=y[:,int(m.num_prefix_tokens):].reshape(-1,16,16,384).half().cpu().numpy();feats.append(y)
   np.savez_compressed(target,features=np.concatenate(feats),timestamps_s=d['timestamps_s'],sensor_pose_c2w=d['sensor_pose_c2w'],scene_id=np.array(scene),trajectory_id=np.array(tid),sequence_path=np.array(str(seq.relative_to(ROOT))))
   n+=len(rgb)
 manifest=[]
 for f in sorted(out.glob('*.npz')):
  with np.load(f,allow_pickle=False) as d: manifest.append({'scene_id':str(d['scene_id']),'trajectory_id':str(d['trajectory_id']),'frames':len(d['features']),'timestamp_ref':'timestamps_s','pose_ref':'sensor_pose_c2w','cache_file':f.name})
 elapsed=time.perf_counter()-started; result={'model':MODEL,'frames':n,'cache_size_bytes':sum(x.stat().st_size for x in out.glob('*.npz')),'extraction_seconds':elapsed,'extraction_fps':n/max(elapsed,1e-6),'peak_gpu_memory_mib':torch.cuda.max_memory_allocated()/2**20,'token_layout':[16,16,384],'dtype':'float16','records':manifest}
 (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
