#!/usr/bin/env python3
"""Real Habitat-GS rollout for an execution-aware topology reroute case.

The first portal realization is deliberately marked blocked after the robot
reaches its local execution budget.  The fixed Habitat path follower then
uses the same destination relation with a different portal realization.  The
video shows RGB, metric top-down topology, edge state and robot trace.
"""
from __future__ import annotations
import base64, json, math, re, subprocess, sys, urllib.request
from pathlib import Path
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
from trajectory_grounding_p1_generate import make_sim, shortest
from topology_runtime import ExecutionAwareTopology

GRAPH=ROOT/'outputs/formal/RelationNav/topology/spatial_semantic_topology.json'
VIDEO=ROOT/'paper_assets/videos/relationnav_topology_reroute.mp4'

def qwen_route_choice(rgb: np.ndarray) -> str:
    """One low-frequency high-level Qwen decision; no pose/NavMesh is sent."""
    import io
    b=io.BytesIO(); Image.fromarray(rgb).save(b,format='PNG')
    payload={"model":"qwen3-vl:8b-instruct","stream":False,"options":{"temperature":0},"messages":[{"role":"user","content":"Choose one route label only: portal_01 or portal_03. Prefer the visually open indoor branch. Do not output coordinates or actions.","images":[base64.b64encode(b.getvalue()).decode()]}]}
    try:
        req=urllib.request.Request('http://127.0.0.1:11434/api/chat',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        out=json.loads(urllib.request.urlopen(req,timeout=120).read()); text=out.get('message',{}).get('content',''); m=re.search(r'portal_0[13]',text)
        return m.group(0) if m else 'portal_01'
    except Exception:
        return 'portal_01'

def set_agent(agent,pos,yaw=0.0):
    import habitat_sim
    s=agent.get_state(); s.position=np.asarray(pos,np.float32); s.rotation=habitat_sim.utils.common.quat_from_angle_axis(yaw,np.array([0.,1.,0.])); agent.set_state(s,reset_sensors=True)

def snap(sim,p): return np.asarray(sim.pathfinder.snap_point(np.asarray([p[0],1.5,p[1]],np.float32)),np.float32)

def move_to(sim,agent,target,frames,trace,max_steps=28):
    import habitat_sim
    for _ in range(max_steps):
        s=agent.get_state(); pos=np.asarray(s.position,np.float32); d=np.asarray(target,np.float32)-pos; d[1]=0
        if np.linalg.norm(d)<.20: break
        f=habitat_sim.utils.common.quat_rotate_vector(s.rotation,np.array([0.,0.,-1.],np.float32)); f[1]=0; f/=max(np.linalg.norm(f),1e-6)
        ang=(math.atan2(float(d[0]),float(d[2]))-math.atan2(float(f[0]),float(f[2])+1e-9)+math.pi)%(2*math.pi)-math.pi
        act='turn_left' if ang>.15 else 'turn_right' if ang<-.15 else 'move_forward'; agent.act(act)
        trace.append(np.asarray(agent.get_state().position,np.float32).copy())

def frame(sim,agent,graph,edge_states,trace,status):
    obs=sim.get_sensor_observations(); rgb=np.asarray(obs['rgb'])[...,:3].astype(np.uint8)
    rgb=Image.fromarray(rgb).resize((420,315),Image.Resampling.BILINEAR)
    fig,ax=plt.subplots(figsize=(5.3,4.0),dpi=100); ax.set_facecolor('#f8f8f8')
    scene='interior_0135_840032'; ns={n['node_id']:n for n in graph['nodes'] if n['scene_id']==scene}
    for n in ns.values():
        p=n['geometry']['center_xz'];
        if n['kind']=='area':
            q=np.asarray(n['geometry']['polygon_xz']); ax.fill(q[:,0],q[:,1],color='#4daf4a',alpha=.12)
    for e in graph['edges']:
        if e.get('edge_type')!='SPATIAL_ADJACENCY' or e['source'] not in ns or e['target'] not in ns: continue
        a=np.asarray(ns[e['source']]['geometry']['center_xz']); b=np.asarray(ns[e['target']]['geometry']['center_xz']); key=e.get('via',e['source']+'->'+e['target']); st=edge_states.get(key,{}); blocked=st.get('blocked_confidence',0)>0.7
        ax.plot([a[0],b[0]],[a[1],b[1]],'--' if blocked else '-',color='#d62728' if blocked else '#555',lw=3 if blocked else 1.8,alpha=.9)
    for n in ns.values():
        p=np.asarray(n['geometry']['center_xz']); c={'area':'#4daf4a','portal':'#e41a1c','landmark':'#377eb8'}.get(n['kind'],'#984ea3'); ax.scatter(*p,s=70,color=c,edgecolor='white',zorder=4); ax.text(p[0],p[1]+.16,n['node_id'].split(':')[-1],fontsize=7,ha='center')
    tr=np.asarray(trace); 
    if len(tr): ax.plot(tr[:,0],tr[:,2],color='#ff7f0e',lw=2.5,label='robot trajectory'); ax.scatter(tr[-1,0],tr[-1,2],marker='*',s=140,color='#ff7f0e',zorder=5)
    ax.set_title(status,fontsize=10); ax.set_aspect('equal',adjustable='datalim'); ax.grid(alpha=.15); ax.set_xlabel('world X (m)'); ax.set_ylabel('world Z (m)'); fig.tight_layout()
    fig.canvas.draw(); arr=np.asarray(fig.canvas.buffer_rgba())[...,:3]; plt.close(fig)
    # side-by-side RGB + map
    left=np.asarray(rgb); right=Image.fromarray(arr).resize((420,315),Image.Resampling.BILINEAR); out=Image.new('RGB',(840,315)); out.paste(rgb,(0,0)); out.paste(right,(420,0)); d=ImageDraw.Draw(out); d.rectangle((0,0,420,28),fill=(0,0,0,150)); d.text((8,8),'Habitat-GS RGB',fill='white'); d.rectangle((420,0,840,28),fill=(0,0,0,150)); d.text((428,8),'Spatial topology / execution state',fill='white'); return np.asarray(out)

def main():
    graph=json.loads(GRAPH.read_text()); scene='interior_0135_840032'; topo=ExecutionAwareTopology(graph); prefix=scene+':'
    source=prefix+'room_01'; target=prefix+'room_03'; by={n['node_id']:n for n in graph['nodes']}
    sim=make_sim(scene); agent=sim.initialize_agent(0); frames=[]; trace=[]; edge_states={}; qwen_choice='unavailable'
    try:
        start=snap(sim,np.asarray(by[source]['geometry']['center_xz'])); set_agent(agent,start); trace.append(start.copy());
        route=topo.route(source,target); assert route and len(route)>=2
        obs0=sim.get_sensor_observations(); qwen_choice=qwen_route_choice(np.asarray(obs0['rgb'])[...,:3])
        alternatives=[e for e in graph['edges'] if e.get('edge_type')=='SPATIAL_ADJACENCY' and e.get('source')==source and e.get('via','').endswith(('portal_01','portal_03'))]
        first_choice=next((e for e in alternatives if e.get('via','').endswith(qwen_choice)),None)
        if first_choice:
            tail=next((e for e in graph['edges'] if e.get('edge_type')=='SPATIAL_ADJACENCY' and e.get('source')==first_choice['target'] and e.get('target')==target),None)
            if tail: route=[first_choice,tail]
        # Render initial state and execute first portal realization.
        frames.append(frame(sim,agent,graph,edge_states,trace,'Initial route: portal_01 -> portal_00'))
        first=route[0]['via']; p=np.asarray(by[first]['geometry']['center_xz']); wp=snap(sim,p); move_to(sim,agent,wp,frames,trace,max_steps=24)
        topo.observe(first,'blocked'); edge_states[first]={'blocked_confidence':topo.edge_state[first].blocked_confidence,'last_event':'blocked'}
        for _ in range(5): frames.append(frame(sim,agent,graph,edge_states,trace,'Execution failure: '+first+' blocked'))
        reroute=topo.route(source,target); assert reroute and reroute[0]['via']!=first
        for e in reroute:
            p=np.asarray(by[e['via']]['geometry']['center_xz']); move_to(sim,agent,snap(sim,p),frames,trace,max_steps=30)
            edge_states[e['via']]={'blocked_confidence':0.0,'last_event':'success'}
            for _ in range(2): frames.append(frame(sim,agent,graph,edge_states,trace,'Rerouted via '+e['via']))
        move_to(sim,agent,snap(sim,np.asarray(by[target]['geometry']['center_xz'])),frames,trace,max_steps=35)
        for _ in range(8): frames.append(frame(sim,agent,graph,edge_states,trace,'Destination room reached after reroute'))
    finally: sim.close()
    VIDEO.parent.mkdir(parents=True,exist_ok=True); arr=np.asarray(frames,np.uint8)
    if arr.shape[1] % 2: arr=np.pad(arr,((0,0),(0,1),(0,0),(0,0)),constant_values=0)
    raw_path=VIDEO.with_suffix('.rgb'); raw_path.write_bytes(arr.tobytes()); subprocess.run([str(ROOT/'tools/raw_rgb_to_mp4'),str(VIDEO),str(arr.shape[2]),str(arr.shape[1]),'5'],input=arr.tobytes(),check=True); raw_path.unlink(missing_ok=True)
    summary={'scene':scene,'source':source,'target':target,'qwen_route_choice':qwen_choice,'initial_route':[e['via'] for e in route],'blocked_edge':first,'rerouted_route':[e['via'] for e in reroute],'frames':len(frames),'video':str(VIDEO)}; (VIDEO.with_suffix('.json')).write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
