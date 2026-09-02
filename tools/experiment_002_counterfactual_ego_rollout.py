#!/usr/bin/env python3
"""Exp002 — Counterfactual Ego-Flow Rollout (GT data generation only).

One common observation history builds a persistent static world field. From
the identical final robot state and exact field state, three future action
sequences are rendered in Habitat-GS.  The grid is world-XZ aligned (Y height)
and only its 10 m local crop follows the robot; it never rotates with yaw.
"""
import copy
import json
import math
import subprocess
from pathlib import Path

import habitat_sim
import habitat_sim.agent
import magnum as mn
import numpy as np
import quaternion
from PIL import Image, ImageDraw

ROOT = Path('/root/autodl-tmp/worldfield_nav')
SCENE = ROOT/'data/scene_datasets/gs_scenes/train/interior_0405_840145'
GS, NAV = SCENE/'interior_0405_840145.gs.ply', SCENE/'interior_0405_840145.navmesh'
OUT, ENCODER = ROOT/'outputs/exp002_counterfactual_controlled', ROOT/'tools/raw_rgb_to_mp4'
H=W=256; FIELD=10.; N=128; CELL=FIELD/N; STRIDE=2; HFOV=90.; WARMUP=240; HORIZON=120; VIDEO_FPS=12

def sensor(uuid, typ):
    s=habitat_sim.CameraSensorSpec(); s.uuid=uuid; s.sensor_type=typ; s.sensor_subtype=habitat_sim.SensorSubType.PINHOLE
    s.resolution=[H,W]; s.position=[0.,1.5,0.]; s.hfov=HFOV; s.near=.01; s.far=10.; return s

def actions():
    mk, act=habitat_sim.agent.ActionSpec, habitat_sim.agent.ActuationSpec
    return {'move_forward':mk('move_forward',act(amount=.07)), 'turn_left':mk('turn_left',act(amount=1.5)), 'turn_right':mk('turn_right',act(amount=1.5)), 'wait':mk('move_forward',act(amount=0.0))}

def make_sim():
    c=habitat_sim.SimulatorConfiguration(); c.scene_id='NONE'; c.enable_physics=False; c.create_renderer=True; c.gpu_device_id=0
    a=habitat_sim.agent.AgentConfiguration(); a.height=1.5; a.radius=.1; a.sensor_specifications=[sensor('rgb',habitat_sim.SensorType.COLOR),sensor('depth',habitat_sim.SensorType.DEPTH)]; a.action_space=actions()
    sim=habitat_sim.Simulator(habitat_sim.Configuration(c,[a])); helper=habitat_sim.RenderInstanceHelper(sim,use_xyzw_orientations=False)
    helper.add_instance(str(GS),semantic_id=0,scale=mn.Vector3(1.,1.,1.)); helper.set_world_poses(np.array([[0,0,0]],np.float32),np.array([[1,0,0,0]],np.float32))
    if not sim.pathfinder.load_nav_mesh(str(NAV)): raise RuntimeError('navmesh failed to load')
    return sim

def arr(x): return np.asarray([x[0],x[1],x[2]],np.float32)

def camera_pose(sim):
    m=sim._sensors['depth'].node.absolute_transformation(); axes=np.column_stack([arr(m.transform_vector(mn.Vector3(1,0,0))),arr(m.transform_vector(mn.Vector3(0,1,0))),arr(m.transform_vector(mn.Vector3(0,0,1)))])
    p=np.eye(4,dtype=np.float32); p[:3,:3]=axes; p[:3,3]=arr(m.transform_point(mn.Vector3(0,0,0))); return p

def depth_world(depth,c2w,K):
    fx,fy,cx,cy=K; vv,uu=np.mgrid[0:H:STRIDE,0:W:STRIDE]; d=depth[::STRIDE,::STRIDE]; ok=np.isfinite(d)&(d>.02)&(d<9.99); d,uu,vv=d[ok],uu[ok],vv[ok]
    pc=np.column_stack(((uu-cx)*d/fx,-(vv-cy)*d/fy,-d)); return (pc@c2w[:3,:3].T+c2w[:3,3]).astype(np.float32)

def pose(agent):
    s=agent.get_state(); p=arr(s.position); q=s.rotation; return np.array([p[0],p[1],p[2],q.w,q.x,q.y,q.z],np.float32)

class Field:
    # key=(world-X cell, world-Z cell), value=[occupancy,height,last_observed]
    def __init__(self,cells=None): self.cells={} if cells is None else copy.deepcopy(cells)
    def update(self,xyz,t):
        keys=np.floor(xyz[:,[0,2]]/CELL).astype(np.int32); uni,inv=np.unique(keys,axis=0,return_inverse=True); counts=np.bincount(inv); heights=np.bincount(inv,weights=xyz[:,1])/counts
        for i,(ix,iz) in enumerate(uni):
            k=(int(ix),int(iz)); old=self.cells.get(k)
            if old is None:self.cells[k]=[1.,float(heights[i]),t]
            else: old[1]=.85*old[1]+.15*float(heights[i]); old[2]=t
    def snapshot(self,center_xz,t):
        origin=np.asarray(center_xz,np.float32)-FIELD/2; ix0,iz0=np.floor(origin/CELL).astype(int); out=np.zeros((4,N,N),np.float32); out[3].fill(WARMUP+HORIZON)
        for (ix,iz),(occ,h,last) in self.cells.items():
            x,z=ix-ix0,iz-iz0
            if 0<=x<N and 0<=z<N:
                r=N-1-z; out[0,r,x]=occ; out[1,r,x]=h; out[2,r,x]=1.; out[3,r,x]=t-last
        return out,origin.astype(np.float32)

class Video:
    def __init__(self,path): self.p=subprocess.Popen([str(ENCODER),str(path),'256','256',str(VIDEO_FPS)],stdin=subprocess.PIPE)
    def add(self,x): self.p.stdin.write(np.ascontiguousarray(x[...,:3].astype(np.uint8)).tobytes())
    def close(self): self.p.stdin.close(); assert self.p.wait()==0

def depth_image(d):
    x=np.clip(d/4,0,1); out=np.empty((H,W,3),np.uint8); out[...,0]=(255*x).astype(np.uint8); out[...,1]=(255*(1-np.abs(2*x-1))).astype(np.uint8); out[...,2]=(255*(1-x)).astype(np.uint8); return out

def field_image(phi,visibility=False):
    out=np.full((N,N,3),(18,20,25),np.uint8); seen=phi[2]>0
    if visibility: out[seen]=(55,215,110)
    else: out[seen]=(50,80,88); out[phi[0]>.5]=(40,220,195)
    im=Image.fromarray(out).resize((256,256),Image.Resampling.NEAREST); d=ImageDraw.Draw(im); d.ellipse((124,124,132,132),fill=(255,55,55)); return np.asarray(im)

def branch_actions():
    # Equal horizon and equal translational action budget: exactly 75 forward
    # controls in every branch.  A waits rather than getting extra travel.
    return {'A_straight':['move_forward']*75+['wait']*45,
            'B_left':['turn_left']*45+['move_forward']*75,
            'C_right':['turn_right']*45+['move_forward']*75}

def rollout_length(sim, agent, state, sequence):
    agent.set_state(state); previous=pose(agent)[:3]; length=0.0
    for action in sequence:
        sim.step(action); current=pose(agent)[:3]
        length += float(np.linalg.norm(current-previous)); previous=current
    return length

def choose_controlled_start(sim, agent, candidates=120):
    """Find a common start where all three fixed-control branches are viable."""
    rng=np.random.default_rng(31); sequences=branch_actions(); best=None
    for _ in range(candidates):
        candidate=agent.get_state(); candidate.position=sim.pathfinder.get_random_navigable_point()
        candidate.rotation=quaternion.from_rotation_vector([0.0,float(rng.uniform(-math.pi,math.pi)),0.0])
        lengths={name:rollout_length(sim,agent,candidate,seq) for name,seq in sequences.items()}
        low,high=min(lengths.values()),max(lengths.values())
        score=(high-low, -low)
        if low>=4.2 and high-low<=0.35:
            return candidate,lengths
        if best is None or score<best[0]: best=(score,candidate,lengths)
    raise RuntimeError(f'No controlled common start found; best realized lengths={best[2]}')

def run_branch(name, start, cells, sequence, K):
    sim=make_sim(); agent=sim.initialize_agent(0); agent.set_state(start); field=Field(cells)
    rgb=[]; depth=[]; poses=[]; cposes=[]; origins=[]; phis=[]; changes=[]; coverage=[]; prev=None
    for t in range(HORIZON+1):
        obs=sim.get_sensor_observations() if t==0 else sim.step(sequence[t-1]); r=np.asarray(obs['rgb'])[...,:3].copy(); d=np.asarray(obs['depth'],np.float32).copy(); cp=camera_pose(sim); ap=pose(agent)
        field.update(depth_world(d,cp,K),WARMUP+t); phi,origin=field.snapshot(ap[[0,2]],WARMUP+t)
        if prev is None: changes.append(0.)
        else:
            valid=np.isfinite(d)&np.isfinite(prev)&(d>.02)&(prev>.02); changes.append(float(np.mean(np.abs(d[valid]-prev[valid]))) if np.any(valid) else 0.)
        prev=d; rgb.append(r);depth.append(d);poses.append(ap);cposes.append(cp);origins.append(origin);phis.append(phi);coverage.append(len(field.cells)*CELL*CELL)
    sim.close(); data={k:np.stack(v) if isinstance(v,list) and k not in ('depth_change','coverage') else v for k,v in {'rgb':rgb,'depth':depth,'pose':poses,'sensor_pose':cposes,'field_origin':origins,'phi':phis,'depth_change':changes,'coverage':coverage}.items()}
    data['actions']=np.asarray(['initial']+sequence); data['timestamps']=np.arange(HORIZON+1,dtype=np.float32)
    np.savez_compressed(OUT/f'exp002_{name}.npz',**data,metadata=json.dumps({'field_channels':['occupancy','height','visibility','age'],'field_plane':'Habitat world X-Z; world Y height','same_initial_state':True,'action_sequence':sequence}))
    sub=OUT/name; sub.mkdir(exist_ok=True)
    for fn,frames in [('rgb.mp4',data['rgb']),('depth.mp4',(depth_image(x) for x in data['depth'])),('occupancy_field.mp4',(field_image(x) for x in data['phi'])),('visibility_field.mp4',(field_image(x,True) for x in data['phi']))]:
        v=Video(sub/fn)
        for x in frames:v.add(x)
        v.close()
    return data

def main():
    if not all(x.is_file() for x in (GS,NAV,ENCODER)): raise RuntimeError('existing static Exp001 assets missing')
    OUT.mkdir(parents=True,exist_ok=True); fx=(W/2)/math.tan(math.radians(HFOV)/2); K=np.array([fx,fx,(W-1)/2,(H-1)/2],np.float32)
    # Common 360-degree history at the selected physical position.  It builds
    # Phi_t without moving away from the carefully controlled branch start.
    sim=make_sim(); sim.seed(17); ag=sim.initialize_agent(0); start,planned_lengths=choose_controlled_start(sim,ag); ag.set_state(start); field=Field(); history=['turn_left']*WARMUP
    for t in range(WARMUP+1):
        obs=sim.get_sensor_observations(); cp=camera_pose(sim); field.update(depth_world(np.asarray(obs['depth'],np.float32),cp,K),t)
        if t<WARMUP: sim.step('turn_left')
    start=ag.get_state(); phi0,origin0=field.snapshot(pose(ag)[[0,2]],WARMUP); sim.close()
    np.savez_compressed(OUT/'initial_field.npz',phi=phi0,field_origin=origin0,robot_pose=pose_from_state(start),history_actions=np.asarray(history[:WARMUP]),metadata=json.dumps({'definition':'shared persistent field Phi_t before all three branches'}))
    results={name:run_branch(name,start,field.cells,seq,K) for name,seq in branch_actions().items()}
    metrics={'experiment':'002_counterfactual_ego_flow_rollout_controlled','scene':'interior_0405_840145','initial_field_shape':list(phi0.shape),'rollout_field_shape':[HORIZON+1,4,N,N],'field_meters':[FIELD,FIELD],'cell_m':CELL,'world_alignment':'X-Z fixed; crop origin follows robot and is recorded','warmup_frames':WARMUP+1,'horizon_steps':HORIZON,'control':{'same_horizon_steps':True,'forward_actions_each':75,'turn_rate_deg_per_action':1.5,'common_start_selected_for_clearance':True,'planned_realized_length_m':planned_lengths},'branches':{}}
    for name,x in results.items():
        length=float(np.linalg.norm(np.diff(x['pose'][:,:3],axis=0),axis=1).sum()); metrics['branches'][name]={'frames':int(len(x['phi'])),'trajectory_length_m':length,'raw_depth_change_mean_m':float(np.mean(x['depth_change'][1:])),'coverage_initial_final_m2':[float(x['coverage'][0]),float(x['coverage'][-1])],'actions':{'forward':int(np.sum(x['actions']=='move_forward')),'left':int(np.sum(x['actions']=='turn_left')),'right':int(np.sum(x['actions']=='turn_right')),'wait':int(np.sum(x['actions']=='wait'))}}
    (OUT/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n'); (OUT/'manifest.json').write_text(json.dumps({'input':'initial_field.npz','supervision':'(Phi_t, action_sequence) -> Phi_t+1:t+H','branches':list(results),'no_learning_performed':True},indent=2)+'\n'); print(json.dumps(metrics,indent=2))

def pose_from_state(s):
    p=arr(s.position); q=s.rotation; return np.array([p[0],p[1],p[2],q.w,q.x,q.y,q.z],np.float32)

if __name__=='__main__': main()
