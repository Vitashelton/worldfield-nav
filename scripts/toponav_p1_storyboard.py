#!/usr/bin/env python3
"""Render a paper-style storyboard exclusively from recorded TopoNav traces."""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from relationnav_p1_generate import make_sim, render, yaw_rotation

ROOT=Path(__file__).resolve().parents[1]; P1=ROOT/'outputs/formal/TopoNav/P1'; OUT=ROOT/'paper_assets/figures'
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'; BOLD='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
def font(n,b=False):return ImageFont.truetype(BOLD if b else FONT,n)
def xy(point,bounds,w,h):
 lo,hi=np.asarray(bounds[0]),np.asarray(bounds[1]); x,z=float(point[0]),float(point[2]);return ((x-lo[0])/(hi[0]-lo[0])*w,(z-lo[2])/(hi[2]-lo[2])*h)
def flat_trace(row):
 out=[]
 for h in row['history']:
  out.extend(h.get('trace_xyz',[]))
 return out
def main():
 rows=[json.loads(x) for x in (P1/'benchmark_episode_results.jsonl').read_text().splitlines()]
 good=[r for r in rows if r.get('method')=='TopoNav-Harness' and r.get('success') and len(flat_trace(r))>2]
 if not good: raise RuntimeError('no successful recorded TopoNav-Harness trace')
 row=max(good,key=lambda r:(r['tool_calls'],r['path_length_m'])); graph=json.loads((P1/'semantic_topology.json').read_text()); nodes={n['node_id']:n for n in graph['nodes']}; task=next(t for t in json.loads((P1/'task_manifest.json').read_text())['tasks'] if t['task_id']==row['task_id'])
 sim=make_sim(row['scene_id']);agent=sim.initialize_agent(0)
 try:
  bounds=sim.pathfinder.get_bounds();nav=sim.pathfinder.get_topdown_view(.05,0.0);base=Image.fromarray(np.where(nav,238,75).astype(np.uint8)).convert('RGB').resize((340,300))
  trace=flat_trace(row); idx=np.linspace(0,len(trace)-1,6).round().astype(int); W,H=6*350+30,760; canvas=Image.new('RGB',(W,H),'white');d=ImageDraw.Draw(canvas)
  d.text((25,15),'TopoNav Harness: semantic-topology-grounded closed-loop execution',font=font(26,True),fill=(20,25,35));d.text((25,48),f"Task: {task['instruction']}  |  Frozen Qwen tool calls: {row['tool_calls']}  |  Actual Habitat-GS trace",font=font(14),fill=(70,80,100))
  for col,k in enumerate(idx):
   x0=15+col*350;p=trace[k];m=base.copy();md=ImageDraw.Draw(m)
   # topology nodes and route executed so far
   for n in nodes.values():
    if n['scene_id']!=row['scene_id']:continue
    q=n['geometry']['center_xz']; px=(q[0]-bounds[0][0])/(bounds[1][0]-bounds[0][0])*340; py=(q[1]-bounds[0][2])/(bounds[1][2]-bounds[0][2])*300
    color=(50,85,220) if n['kind']=='portal' else (110,110,110);md.ellipse((px-3,py-3,px+3,py+3),fill=color)
   upto=trace[:k+1];pts=[xy(a,bounds,340,300) for a in upto]
   if len(pts)>1:md.line(pts,fill=(20,165,95),width=4)
   rx,ry=xy(p,bounds,340,300);md.ellipse((rx-7,ry-7,rx+7,ry+7),fill=(255,190,0),outline='white',width=2)
   goal=nodes[task['goal_node']]['geometry']['center_xz'];gx=(goal[0]-bounds[0][0])/(bounds[1][0]-bounds[0][0])*340;gy=(goal[1]-bounds[0][2])/(bounds[1][2]-bounds[0][2])*300;md.ellipse((gx-6,gy-6,gx+6,gy+6),fill=(230,35,35),outline='white',width=2)
   canvas.paste(m,(x0,82));d=ImageDraw.Draw(canvas);d.rectangle((x0,82,x0+340,107),fill=(33,45,62));d.text((x0+10,88),f'DECISION / TRACE STEP {k+1}',font=font(12,True),fill='white')
   nxt=trace[min(k+1,len(trace)-1)];delta=np.asarray(nxt)-np.asarray(p);yaw=math.atan2(float(-delta[0]),float(-delta[2])) if np.linalg.norm(delta)>1e-5 else 0.
   rgb,_,_,_=render(sim,agent,np.asarray(p,np.float32),yaw_rotation(yaw));im=Image.fromarray(rgb).resize((340,300));canvas.paste(im,(x0,400));d=ImageDraw.Draw(canvas);d.rectangle((x0,400,x0+340,425),fill=(33,45,62));d.text((x0+10,406),'HABITAT-GS EGO RGB (same instant)',font=font(12,True),fill='white')
   prior=[h for h in row['history'] if h.get('trace_xyz')]
   step=min(len(prior)-1,max(0,int((k/(len(idx)-1))*len(prior))))
   if prior:
    h=prior[step];call=h.get('call',{});fb=h.get('feedback',{});d.text((x0+8,715-42),f"{call.get('tool','')}({str(call.get('target',''))[-10:]})",font=font(11,True),fill=(45,55,70));d.text((x0+8,715-22),f"feedback: {fb.get('status','')} | relation: {fb.get('relation_complete','')}",font=font(10),fill=(45,120,80))
  d=ImageDraw.Draw(canvas);d.rectangle((15,730,W-15,750),fill=(238,238,238));d.text((24,734),'Legend: yellow=robot   red=semantic target area   blue=portal node   green=actual traversed path   (all panels replay the same logged episode)',font=font(10),fill=(35,40,50))
  OUT.mkdir(parents=True,exist_ok=True);out=OUT/'toponav_p1_execution_storyboard.png';canvas.save(out);print(json.dumps({'episode':row['task_id'],'output':str(out)}))
 finally:sim.close()
if __name__=='__main__':main()
