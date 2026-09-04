#!/usr/bin/env python3
"""Build reusable metric correspondence/track JSONL manifests from C1."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from geoanchor.correspondence import mine_correspondences,patch_center,depth_at,backproject_pixel,intrinsics
def emit(path,rows):
 with path.open('w') as f:
  for r in rows:f.write(json.dumps(r)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',default='configs/benchmark/g2_g3_metricanchor.yaml');p.add_argument('--output',default='outputs/formal/G2/correspondence_dataset');a=p.parse_args();c=yaml.safe_load((ROOT/a.config).read_text());out=ROOT/a.output;out.mkdir(parents=True,exist_ok=True)
 split={s:k for k,v in c['splits'].items() for s in v}; rows={k:[] for k in split.values()};tracks=[];stats={k:{'positive_pairs':0,'hard_negatives':0,'tracks':0} for k in rows}
 for seq in sorted((ROOT/'outputs/formal/C1/pilot/trajectories').glob('*/sequence.npz')):
  tid=seq.parent.name;scene=tid.rsplit('_traj',1)[0];role=split[scene]
  with np.load(seq,allow_pickle=False) as d:
   dep,poses=d['depth'],d['sensor_pose_c2w'];lid=d['sim_lidar_xyz'];counts=d['sim_lidar_count']; n=len(dep)
   for off in c['geometry']['pair_offsets']:
    regime='small' if off<=5 else 'medium' if off<=15 else 'large'
    for i in range(0,n-off,5):
     cs=mine_correspondences(dep[i],poses[i],dep[i+off],poses[i+off],16,16,c['geometry']['depth_agreement_m'],c['geometry']['world_residual_m'],lid[i+off],int(counts[i+off]))[:c['geometry']['max_pairs_per_frame_pair']]
     for x in cs:
      neg=(x.target_patch+127)%256
      # use a deterministic physically distinct target surface when available
      for q in range(256):
       u,v=patch_center(q,16,16);z=depth_at(dep[i+off],u,v)
       if z is not None and np.linalg.norm(backproject_pixel(z[0],z[1],z[2],poses[i+off],intrinsics())-x.world_xyz)>=c['geometry']['hard_negative_min_world_distance_m']:
        neg=q;break
      rows[role].append({'scene_id':scene,'trajectory_id':tid,'source_frame':i,'target_frame':i+off,'source_patch':x.source_patch,'target_patch':x.target_patch,'hard_negative_patch':neg,'regime':regime,'world_xyz':x.world_xyz.tolist(),'depth_residual_m':x.depth_residual_m,'world_residual_m':x.world_residual_m,'lidar_residual_m':x.lidar_residual_m})
      stats[role]['positive_pairs']+=1;stats[role]['hard_negatives']+=1
   # explicit 3-view tracks at 0,5,15 only when the same source patch persists
   for i in range(0,n-15,10):
    a5=mine_correspondences(dep[i],poses[i],dep[i+5],poses[i+5],16,16,.12,.08);a15=mine_correspondences(dep[i],poses[i],dep[i+15],poses[i+15],16,16,.12,.08);b={x.source_patch:x for x in a15}
    for x in a5:
     if x.source_patch in b:
      y=b[x.source_patch];tracks.append({'scene_id':scene,'trajectory_id':tid,'frames':[i,i+5,i+15],'patches':[x.source_patch,x.target_patch,y.target_patch],'split':role});stats[role]['tracks']+=1
 for role in rows:emit(out/({'train':'train.jsonl','validation':'val.jsonl','unseen':'unseen_test.jsonl'}[role]),rows[role])
 emit(out/'tracks.jsonl',tracks);(out/'statistics.json').write_text(json.dumps({'splits':stats,'tracks_total':len(tracks),'geometry':c['geometry']},indent=2)+'\n')
if __name__=='__main__':main()
