#!/usr/bin/env python3
"""Vectorized, method-independent C1 candidate world-point cache."""
from pathlib import Path
import numpy as np, torch, sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from geoanchor.correspondence import patch_center, intrinsics, backproject_pixel, depth_at
def main():
 out=ROOT/'outputs/formal/G2/geometry_cache';out.mkdir(parents=True,exist_ok=True); files=sorted((ROOT/'outputs/formal/G2/feature_cache').glob('*_traj*.npz')); uv=np.array([patch_center(i,16,16) for i in range(256)],np.float32);k=intrinsics();xs=[];vs=[]
 for f in files:
  with np.load(ROOT/'outputs/formal/C1/pilot/trajectories'/f.stem/'sequence.npz',allow_pickle=False) as d:
   dep=d['depth'][:,np.rint(uv[:,1]).astype(int),np.rint(uv[:,0]).astype(int)];valid=np.isfinite(dep)&(dep>.02)&(dep<9.99);cam=np.stack([(uv[:,0][None]-k[2])*dep/k[0],-(uv[:,1][None]-k[3])*dep/k[1],-dep],-1);R=d['sensor_pose_c2w'][:,:3,:3];t=d['sensor_pose_c2w'][:,:3,3];xs.append(np.einsum('fpi,fji->fpj',cam,R)+t[:,None,:]);vs.append(valid)
 xyz=np.concatenate(xs).astype('float32');valid=np.concatenate(vs);torch.save(torch.from_numpy(xyz),out/'candidate_world_xyz.pt');torch.save(torch.from_numpy(valid),out/'candidate_world_valid.pt');
 rng=np.random.default_rng(0); valid_indices=np.argwhere(valid); chosen=valid_indices[rng.choice(len(valid_indices),20,replace=False)]
 for fi,pi in chosen:
  f=files[fi//150]; with_data=np.load(ROOT/'outputs/formal/C1/pilot/trajectories'/f.stem/'sequence.npz',allow_pickle=False);u,v=uv[pi];z=depth_at(with_data['depth'][fi%150],u,v);ref=backproject_pixel(z[0],z[1],z[2],with_data['sensor_pose_c2w'][fi%150],k);assert np.max(np.abs(ref-xyz[fi,pi]))<1e-5
 print({'shape':list(xyz.shape),'valid':int(valid.sum()),'equivalence_max_error':float(np.max(np.abs(ref-xyz[fi,pi])))})
if __name__=='__main__':main()
