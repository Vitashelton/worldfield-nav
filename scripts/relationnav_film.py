#!/usr/bin/env python3
from pathlib import Path
import json, math, subprocess, tempfile, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
ROOT=Path('/root/autodl-tmp/.autodl/worldfield_nav'); P1=ROOT/'outputs/formal/RelationNav/P1/dataset'; OUT=ROOT/'paper_assets'
sys.path.insert(0,str(ROOT/'scripts'))
from relationnav_p1_generate import make_sim, shortest, yaw_rotation, render
from relationnav_p2_evaluate import set_agent, shallow_and_deep, satisfied, orient_terminal_observation, entity_index
import habitat_sim
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'; BOLD='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
def ft(n,b=False): return ImageFont.truetype(BOLD if b else FONT,n)
def trace(sim,agent,target,radius):
 start=np.asarray(agent.get_state().position,np.float32); ok,d=shortest(sim,start,target)
 if not ok:return {'ok':False,'path':np.asarray([start]),'end':start,'length':0.}
 q=habitat_sim.ShortestPath();q.requested_start=start;q.requested_end=np.asarray(target,np.float32);sim.pathfinder.find_path(q);pts=[np.asarray(p,np.float32) for p in q.points];left=max(0.,float(d)-radius);out=[pts[0]];pos=pts[0]
 for a,b in zip(pts[:-1],pts[1:]):
  seg=float(np.linalg.norm(b-a))
  if left<=seg+1e-6:pos=a if seg<1e-9 else a+(left/max(seg,1e-9))*(b-a);out.append(pos);break
  left-=seg;pos=b;out.append(pos)
 set_agent(agent,pos);return {'ok':True,'path':np.asarray(out),'end':np.asarray(pos),'length':max(0.,float(d)-radius)}
def views(sim,agent,path,target):
 out=[]
 for i,p in enumerate(path):
  p=np.asarray(p,np.float32); nxt=np.asarray(path[min(i+1,len(path)-1)],np.float32); delta=np.asarray(target)-p; yaw=math.atan2(float(-delta[0]),float(-delta[2]))
  set_agent(agent,p); rgb,_,_,_=render(sim,agent,p,yaw_rotation(yaw)); out.append(rgb)
 return out
def run(ep,method,entities):
 sim=make_sim(ep['scene_id']);agent=sim.initialize_agent(0);set_agent(agent,np.asarray(ep['start_xyz'],np.float32));rng=np.random.default_rng(20260914+sum(map(ord,ep['episode_id']))); rows=[]
 try:
  for ph in ep['phases']:
   before=np.asarray(agent.get_state().position,np.float32);shallow,deep=shallow_and_deep(sim,agent,ph,entities,rng);target=deep if method=='Oracle' else shallow
   t=trace(sim,agent,target,.10 if method=='Oracle' else .50); first_views=views(sim,agent,t['path'],target); set_agent(agent,t['end']); after=np.asarray(agent.get_state().position,np.float32);orient_terminal_observation(agent,ph,entities);after=np.asarray(agent.get_state().position,np.float32);done=bool(t['ok'] and satisfied(sim,agent,ph,entities,before,after)); retry=None; retry_views=[]; recovered=False
   if not done and method=='Relation-preserving recovery':
    retry=trace(sim,agent,deep,.10);retry_views=views(sim,agent,retry['path'],deep);set_agent(agent,retry['end']);after=np.asarray(agent.get_state().position,np.float32);orient_terminal_observation(agent,ph,entities);after=np.asarray(agent.get_state().position,np.float32);done=bool(retry['ok'] and satisfied(sim,agent,ph,entities,before,after));recovered=done
   rows.append({'phase':ph,'before':before,'shallow':shallow,'deep':deep,'first':t,'retry':retry,'done':done,'recovered':recovered,'first_views':first_views,'retry_views':retry_views})
 finally:sim.close()
 return rows
def replace_rgb(base,rgb,label,step,status):
 im=base.copy();pic=Image.fromarray(rgb).convert('RGB').resize((260,260));im.paste(pic,(28,100));d=ImageDraw.Draw(im);d.rectangle((28,72,288,98),fill=(38,48,64));d.text((38,78),'LIVE ROBOT RGB',font=ft(15,True),fill='white');d.text((990,300),f'{label}  |  frame {step}',font=ft(18,True),fill=(255,210,110));d.text((990,335),status,font=ft(17,True),fill=(220,230,240));return im
def main():
 E=json.loads((P1/'dataset_manifest.json').read_text())['episodes'];entities=entity_index();ids=['interior_0405_840145_0000','interior_0135_840032_0000','interior_0093_839966_0000'];eps=[next(e for e in E if e['episode_id']==i) for i in ids];frames=[]
 from relationnav_story import draw_frame
 for ep in eps:
  a=run(ep,'Arrival-only',entities);b=run(ep,'Relation-preserving recovery',entities)
  for j in range(len(a)):
   # nominal path: moving camera observations
   for k,rgb in enumerate(a[j]['first_views']):
    frames.append(replace_rgb(draw_frame(ep,a,b,j),rgb,'ARRIVAL-ONLY',k,'moving toward nominal local goal'))
   for _ in range(3): frames.append(frames[-1])
   # recovery path, if relation failed
   if b[j]['retry_views']:
    for k,rgb in enumerate(b[j]['retry_views']): frames.append(replace_rgb(draw_frame(ep,a,b,j),rgb,'RELATION-PRESERVING RECOVERY',k,'nominal arrival rejected; reselecting same relation'))
    for _ in range(5): frames.append(frames[-1])
  # end card
  end=Image.new('RGB',(1400,820),(15,19,28));d=ImageDraw.Draw(end);d.text((70,100),'RELATIONNAV EXECUTION RESULT',font=ft(38,True),fill='white');d.text((70,190),f"{ep['scene_id']}  /  {ep['episode_id']}",font=ft(22),fill=(180,195,215));d.text((70,300),'Arrival-only:  false relation completion',font=ft(25,True),fill=(240,100,100));d.text((70,370),'Relation-preserving recovery:  relation complete',font=ft(25,True),fill=(90,225,140));d.text((70,500),'The task advances only after the spatial predicate is satisfied.',font=ft(22),fill=(255,210,110));frames.extend([end]*10)
 OUT.joinpath('videos').mkdir(parents=True,exist_ok=True);out=OUT/'videos/relationnav_execution_film.mp4';p=subprocess.Popen([str(ROOT/'tools/raw_rgb_to_mp4'),str(out),'1400','820','5'],stdin=subprocess.PIPE)
 for im in frames:p.stdin.write(np.asarray(im.convert('RGB'),np.uint8).tobytes())
 p.stdin.close();rc=p.wait();print({'frames':len(frames),'seconds':len(frames)/5,'returncode':rc,'output':str(out),'bytes':out.stat().st_size})
if __name__=='__main__':main()
