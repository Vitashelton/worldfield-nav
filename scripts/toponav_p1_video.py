#!/usr/bin/env python3
"""Render actual logged Habitat-GS trajectories as a paper/meeting video."""
import json,math,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from relationnav_p1_generate import make_sim,render,yaw_rotation

ROOT=Path(__file__).resolve().parents[1];P1=ROOT/'outputs/formal/TopoNav/P1';OUT=ROOT/'paper_assets/videos/toponav_p1_habitat_demo.mp4'
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';BOLD='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
def ft(n,b=False):return ImageFont.truetype(BOLD if b else FONT,n)
def trace(r):
 out=[]
 for h in r['history']:
  x=h.get('trace_xyz',[]);out.extend(x[1:] if out and x and out[-1]==x[0] else x)
 return out
def xy(p,b,w,h):return ((p[0]-b[0][0])/(b[1][0]-b[0][0])*w,(p[2]-b[0][2])/(b[1][2]-b[0][2])*h)
def main():
 rows=[json.loads(x) for x in (P1/'benchmark_episode_results.jsonl').read_text().splitlines()];tasks={x['task_id']:x for x in json.loads((P1/'task_manifest.json').read_text())['tasks']};graph=json.loads((P1/'semantic_topology.json').read_text());by={(r['task_id'],r['method']):r for r in rows};ids=sorted(tasks)
 cases=[]
 for typ in ('S1','S2','S3'):
  cases.append(next(i for i in ids if tasks[i]['task_type']==typ and not by[i,'Direct-VLM']['success'] and by[i,'TopoNav-Harness']['success']))
 OUT.parent.mkdir(parents=True,exist_ok=True);enc=subprocess.Popen([str(ROOT/'tools/raw_rgb_to_mp4'),str(OUT),'1280','720','6'],stdin=subprocess.PIPE)
 def emit(im,n=1):
  a=np.asarray(im.convert('RGB'),dtype=np.uint8)
  for _ in range(n):enc.stdin.write(a.tobytes())
 for ci,tid in enumerate(cases,1):
  row=by[tid,'TopoNav-Harness'];task=tasks[tid];tr=trace(row);sim=make_sim(row['scene_id']);agent=sim.initialize_agent(0)
  try:
   bounds=sim.pathfinder.get_bounds();nav=sim.pathfinder.get_topdown_view(.05,0.0);base=Image.fromarray(np.where(nav,239,70).astype('uint8')).convert('RGB').resize((390,440))
   title=Image.new('RGB',(1280,720),'#111827');d=ImageDraw.Draw(title);d.text((65,105),f'CASE {ci} · {task["task_type"]}',font=ft(36,1),fill='#5ee0ae');d.text((65,175),task['instruction'],font=ft(25,1),fill='white');d.text((65,260),'OUR CONTRIBUTION',font=ft(16,1),fill='#5ee0ae');d.text((65,302),'Task-relevant topology  →  named tool contract  →  validator',font=ft(23,1),fill='white');d.text((65,345),'→ metric executor  →  typed feedback  →  next context',font=ft(23,1),fill='white');d.text((65,430),'Frozen Qwen reasons semantically; the harness owns robot-world grounding.',font=ft(18),fill='#b8c3d5');emit(title,24)
   # Densify actual path only for visualization.
   dense=[]
   for a,b in zip(tr[:-1],tr[1:]):
    a=np.asarray(a,float);b=np.asarray(b,float);n=max(2,int(np.linalg.norm(b-a)/.35)+1);dense.extend([(a*(1-u)+b*u).tolist() for u in np.linspace(0,1,n,endpoint=False)])
   dense.append(tr[-1]);sel=np.linspace(0,len(dense)-1,min(72,len(dense))).round().astype(int);dense=[dense[i] for i in sel]
   for k,p0 in enumerate(dense):
    p=np.asarray(p0,np.float32);q=np.asarray(dense[min(k+1,len(dense)-1)],np.float32);dv=q-p;yaw=math.atan2(float(-dv[0]),float(-dv[2])) if np.linalg.norm(dv)>1e-5 else 0
    rgb,_,_,_=render(sim,agent,p,yaw_rotation(yaw));left=Image.fromarray(rgb).resize((590,440));m=base.copy();md=ImageDraw.Draw(m)
    pts=[xy(x,bounds,390,440) for x in dense[:k+1]]
    if len(pts)>1:md.line(pts,fill='#20b77a',width=7)
    x,y=pts[-1];md.ellipse((x-9,y-9,x+9,y+9),fill='#ffc400',outline='white',width=2)
    frame=Image.new('RGB',(1280,720),'white');frame.paste(left,(0,90));frame.paste(m,(600,90));d=ImageDraw.Draw(frame);d.rectangle((0,0,1280,90),fill='#172033');d.text((22,14),f'{task["task_type"]}: {task["instruction"]}',font=ft(19,1),fill='white');d.text((22,50),f'Recorded Habitat-GS execution · progress {100*k/max(1,len(dense)-1):.0f}%',font=ft(14),fill='#b8c3d5')
    # Active recorded tool segment.
    cum=0;active=row['history'][-1]
    for h in row['history']:
     cum+=max(1,len(h.get('trace_xyz',[])))
     if k/len(dense)<=cum/max(1,sum(max(1,len(z.get('trace_xyz',[]))) for z in row['history'])):active=h;break
    call=active.get('call') or {};fb=active.get('feedback') or {};target=str(call.get('target','-')).rsplit(':',1)[-1];ai=min(len(row['history'])-1,int(k/max(1,len(dense))*len(row['history'])));current=task['start_node']
    for old in row['history'][:ai]:
     oc=old.get('call') or {};of=old.get('feedback') or {}
     if oc.get('tool')=='NAVIGATE' and of.get('status')=='SUCCESS':
      edge=next((e for e in graph['edges'] if e.get('via')==oc.get('target') and e.get('source')==current),None)
      if edge:current=edge['target']
    cand=[e.get('via','').rsplit(':',1)[-1] for e in graph['edges'] if e.get('edge_type')=='SPATIAL_ADJACENCY' and e.get('source')==current]
    # Right method trace: green blocks are ours, blue/gray are reused.
    d.rectangle((1000,90,1280,530),fill='#0f172a');d.text((1016,106),'TOPONAV HARNESS',font=ft(17,1),fill='#5ee0ae')
    blocks=[('1  CONTEXT COMPILER',f"{current.rsplit(':',1)[-1]} · {len(cand)} local edges",'#5ee0ae'),('2  FROZEN QWEN',f"{call.get('tool','INVALID')}({target})",'#7fb2ff'),('3  TOOL VALIDATOR',str(active.get('validation','')), '#5ee0ae'),('4  METRIC EXECUTOR',f"travel {fb.get('travelled_m',0):.1f} m",'#b8c3d5'),('5  TYPED FEEDBACK',f"{fb.get('status','')} / {fb.get('relation','')}",'#5ee0ae')]
    yy=145
    for name,val,color in blocks:
     d.rounded_rectangle((1015,yy,1265,yy+62),7,fill='#172033',outline=color,width=2);d.text((1027,yy+9),name,font=ft(12,1),fill=color);d.text((1027,yy+34),val,font=ft(12),fill='white');yy+=72
    d.rectangle((0,545,1280,720),fill='#f1f4f8');d.text((24,566),'What is observable here?',font=ft(17,1),fill='#172033');d.text((24,600),'The VLM never outputs coordinates or controls. It names a topology edge.',font=ft(15),fill='#42516a');d.text((24,632),'Our compiler exposes only task-relevant transitions; the validator checks adjacency;',font=ft(15),fill='#42516a');d.text((24,660),'the fixed executor moves the robot and returns typed, relation-aware feedback.',font=ft(15),fill='#42516a');d.text((795,575),f"SUCCESS · path {row['path_length_m']:.1f} m",font=ft(19,1),fill='#14734f');d.text((795,616),f"final DTG {row['final_dtg_m']:.1f} m",font=ft(16),fill='#22324a');d.text((795,650),f"tool calls {row['tool_calls']}",font=ft(16),fill='#22324a');emit(frame)
   emit(frame,12)
  finally:sim.close()
 summary=Image.new('RGB',(1280,720),'#111827');d=ImageDraw.Draw(summary);d.text((65,90),'STATIC HABITAT-GS RESULT',font=ft(30,1),fill='#5ee0ae');d.text((65,165),'Direct VLM',font=ft(24),fill='white');d.text((390,165),'10.0%',font=ft(42,1),fill='#ef6a6a');d.text((65,250),'Full-history agent',font=ft(24),fill='white');d.text((390,250),'57.5%',font=ft(42,1),fill='#6fa8dc');d.text((65,335),'TopoNav Harness',font=ft(24,1),fill='white');d.text((390,335),'73.8%',font=ft(52,1),fill='#5ee0ae');d.text((65,440),'S1 / S2: 90.0%     S3 route-constrained: 46.7%',font=ft(22),fill='#d4dbea');d.text((65,510),'Honest limitation: static benchmark only; no dynamic-avatar claim.',font=ft(19),fill='#f5c76d');emit(summary,30)
 enc.stdin.close();code=enc.wait();print(json.dumps({'video':str(OUT),'cases':cases,'encoder_exit':code},indent=2))
if __name__=='__main__':main()
