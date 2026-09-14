#!/usr/bin/env python3
from pathlib import Path
import json, math, subprocess, tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path('/root/autodl-tmp/.autodl/worldfield_nav')
P1=ROOT/'outputs/formal/RelationNav/P1/dataset'
OUT=ROOT/'paper_assets'
ANN=ROOT/'configs/relationnav/entities.json'
import sys
sys.path.insert(0,str(ROOT/'scripts'))
from relationnav_p1_generate import make_sim, shortest, yaw_rotation
from relationnav_p2_evaluate import set_agent, shallow_and_deep, satisfied, orient_terminal_observation, entity_index
import habitat_sim

FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
def f(size,bold=False):
    p='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else FONT
    return ImageFont.truetype(p,size)

def route_trace(sim, agent, target, radius):
    start=np.asarray(agent.get_state().position,np.float32)
    ok,d=shortest(sim,start,target)
    if not ok: return {'ok':False,'path':np.asarray([start]),'end':start,'length':0.0}
    q=habitat_sim.ShortestPath(); q.requested_start=start; q.requested_end=np.asarray(target,np.float32); sim.pathfinder.find_path(q)
    pts=[np.asarray(p,np.float32) for p in q.points]; move=max(0.0,float(d)-radius); out=[pts[0]]; left=move; pos=pts[0]
    for a,b in zip(pts[:-1],pts[1:]):
        seg=float(np.linalg.norm(b-a))
        if left<=seg+1e-6:
            pos=a if seg<1e-9 else a+(left/max(seg,1e-9))*(b-a); out.append(pos); break
        left-=seg; pos=b; out.append(pos)
    set_agent(agent,pos)
    return {'ok':True,'path':np.asarray(out),'end':np.asarray(pos),'length':move}

def run_case(ep, method, entities):
    sim=make_sim(ep['scene_id']); agent=sim.initialize_agent(0); set_agent(agent,np.asarray(ep['start_xyz'],np.float32)); rng=np.random.default_rng(20260914+sum(map(ord,ep['episode_id'])))
    rows=[]; total=0
    try:
      for ph in ep['phases']:
        before=np.asarray(agent.get_state().position,np.float32); shallow,deep=shallow_and_deep(sim,agent,ph,entities,rng)
        target=deep if method=='Oracle' else shallow
        tr=route_trace(sim,agent,target,.10 if method=='Oracle' else .50); after=np.asarray(agent.get_state().position,np.float32)
        orient_terminal_observation(agent,ph,entities); after=np.asarray(agent.get_state().position,np.float32)
        done=bool(tr['ok'] and satisfied(sim,agent,ph,entities,before,after)); recovered=False; retry=None
        if not done and method=='Relation-preserving recovery':
          retry=route_trace(sim,agent,deep,.10); after=np.asarray(agent.get_state().position,np.float32); orient_terminal_observation(agent,ph,entities); after=np.asarray(agent.get_state().position,np.float32)
          done=bool(retry['ok'] and satisfied(sim,agent,ph,entities,before,after)); recovered=done
        rows.append({'phase':ph,'before':before,'shallow':shallow,'deep':deep,'first':tr,'retry':retry,'after':after,'done':done,'recovered':recovered})
        total+=tr['length']+(retry['length'] if retry else 0)
    finally: sim.close()
    return rows

def draw_frame(ep, rows_a, rows_b, idx, method_labels=('Arrival-only','Relation-preserving recovery')):
    W,H=1400,820; im=Image.new('RGB',(W,H),(15,19,28)); d=ImageDraw.Draw(im)
    title=f"RelationNav | {ep['scene_id']} | {ep['episode_id']} | metric task execution"
    d.text((28,20),title,font=f(25,True),fill=(245,247,250)); d.text((28,55),"Same episode · same relation · only completion verification/recovery changes",font=f(16),fill=(166,177,194))
    r=rows_a[idx]; ph=r['phase']; rel=ph['relation']; ent=ph['entity_id'];
    # RGB and candidate overlay
    for x,label,path in [(28,'CURRENT RGB',P1/ph['rgb']),(328,'CANDIDATE OVERLAY',P1/ph['candidate_overlay'])]:
      try: pic=Image.open(path).convert('RGB').resize((260,260))
      except: pic=Image.new('RGB',(260,260),(40,40,50))
      im.paste(pic,(x,100)); d.rectangle((x,72,x+260,98),fill=(38,48,64)); d.text((x+10,78),label,font=f(15,True),fill='white')
    # task state rail
    x0=628; d.text((x0,80),"TASK STATE",font=f(16,True),fill=(255,210,110));
    phases=[p['relation'] for p in ep['phases']]
    for j,pname in enumerate(phases):
      xx=x0+j*155; col=(65,190,130) if j<idx else ((255,190,70) if j==idx else (90,100,120)); d.rounded_rectangle((xx,112,xx+138,164),8,fill=col); d.text((xx+12,130),pname,font=f(16,True),fill=(10,15,20) if j<=idx else 'white')
    d.text((x0,184),f"relation: {rel}",font=f(18,True),fill=(239,244,252)); d.text((x0,214),f"entity: {ent}",font=f(15),fill=(170,180,195));
    # map area
    mx,my,mw,mh=28,410,930,370; d.rectangle((mx,my,mx+mw,my+mh),fill=(24,31,43),outline=(74,91,112),width=2); d.text((mx+14,my+12),'TOP-DOWN EXECUTION TRACE (x,z)',font=f(16,True),fill='white')
    paths=[]
    for p in ph['candidate_paths_goal_independent']:
      a=np.asarray(p); paths.extend(a[:,[0,2]].tolist())
    rr=[np.asarray(x)[:,[0,2]] for x in [r['first']['path'],r['retry']['path'] if r['retry'] else np.empty((0,3))]]; bb=[np.asarray(x)[:,[0,2]] for x in [rows_b[idx]['first']['path'],rows_b[idx]['retry']['path'] if rows_b[idx]['retry'] else np.empty((0,3))]]
    allxy=np.vstack([np.asarray(paths),*rr,*bb,np.asarray([r['before'][[0,2]],r['shallow'][[0,2]],r['deep'][[0,2]]])]); mn=allxy.min(0)-.6; mxv=allxy.max(0)+.6
    def xy(q): return (mx+40+(float(q[0])-mn[0])/(mxv[0]-mn[0])*(mw-80), my+mh-40-(float(q[1])-mn[1])/(mxv[1]-mn[1])*(mh-80))
    # candidate fan
    for k,p in enumerate(ph['candidate_paths_goal_independent']):
      a=np.asarray(p)[:,[0,2]]; pts=[xy(q) for q in a]; d.line(pts,fill=(90,105,125),width=2); q=pts[-1]; d.ellipse((q[0]-5,q[1]-5,q[0]+5,q[1]+5),fill=(105,116,135)); d.text((q[0]+7,q[1]-8),f'C{k+1}',font=f(11),fill=(173,184,200))
    def trace(path,col,width=6):
      if len(path): d.line([xy(q) for q in path],fill=col,width=width,joint='curve')
    trace(rr[0],(235,90,90),6); trace(rr[1],(90,220,140),5); trace(bb[0],(235,90,90),2)
    s=xy(r['before'][[0,2]]); sh=xy(r['shallow'][[0,2]]); dp=xy(r['deep'][[0,2]]); d.ellipse((s[0]-9,s[1]-9,s[0]+9,s[1]+9),fill=(80,180,255),outline='white',width=2); d.text((s[0]+10,s[1]-12),'ROBOT',font=f(13,True),fill=(150,210,255)); d.ellipse((sh[0]-9,sh[1]-9,sh[0]+9,sh[1]+9),fill=(235,75,75),outline='white',width=2); d.text((sh[0]+10,sh[1]-12),'arrival-only',font=f(12),fill=(255,150,150)); d.ellipse((dp[0]-9,dp[1]-9,dp[0]+9,dp[1]+9),fill=(80,220,140),outline='white',width=2); d.text((dp[0]+10,dp[1]-12),'recovery target',font=f(12),fill=(145,255,170));
    d.text((mx+18,my+mh-30),'blue=start  red=nominal endpoint  green=relation-preserving retry  gray=candidate branches',font=f(13),fill=(180,190,205))
    # outcome panels
    bx=990; d.text((bx,410),'OUTCOME COMPARISON',font=f(16,True),fill=(255,210,110));
    for j,(lab,rows,col) in enumerate(zip(method_labels,[rows_a,rows_b],[(235,90,90),(90,220,140)])):
      q=rows[idx]; y=455+j*122; d.rounded_rectangle((bx,y,bx+375,y+96),10,fill=(28,37,50),outline=col,width=3); d.text((bx+16,y+12),lab,font=f(18,True),fill=col); status='RELATION COMPLETE' if q['done'] else 'FALSE COMPLETION'; d.text((bx+16,y+43),status,font=f(17,True),fill=(235,240,245)); d.text((bx+16,y+70),f"arrival={q['first']['ok']}  retry={q['recovered']}",font=f(14),fill=(170,180,195))
    d.text((bx,720),'Completion predicate:',font=f(14,True),fill=(255,210,110)); d.text((bx,746),f"{ph.get('guard','relation predicate')}",font=f(14),fill=(225,230,240))
    return im

def main():
  E=json.loads((P1/'dataset_manifest.json').read_text())['episodes']; entities=entity_index(); ids=['interior_0405_840145_0000','interior_0135_840032_0000','interior_0093_839966_0000']; eps=[next(e for e in E if e['episode_id']==x) for x in ids]
  frames=[]; boards=[]
  for ep in eps:
    a=run_case(ep,'Arrival-only',entities); b=run_case(ep,'Relation-preserving recovery',entities)
    # first key frame for storyboard, with phase where baseline fails if possible
    k=next((i for i,x in enumerate(a) if not x['done']),min(1,len(a)-1)); boards.append(draw_frame(ep,a,b,k));
    for i in range(len(a)):
      for _ in range(3): frames.append(draw_frame(ep,a,b,i))
      if i<len(a)-1:
       for _ in range(2): frames.append(draw_frame(ep,a,b,i))
  fig=Image.new('RGB',(1400,820*len(boards)),(15,19,28))
  for i,b in enumerate(boards): fig.paste(b,(0,820*i))
  OUT.joinpath('figures').mkdir(parents=True,exist_ok=True); fig.save(OUT/'figures/relationnav_qualitative_storyboard.png')
  tmp=Path(tempfile.mkdtemp());
  for i,fr in enumerate(frames): fr.save(tmp/f'{i:05d}.rgb.png')
  OUT.joinpath('videos').mkdir(parents=True,exist_ok=True); out=OUT/'videos/relationnav_execution_story.mp4'; proc=subprocess.Popen([str(ROOT/'tools/raw_rgb_to_mp4'),str(out),'1400','820','3'],stdin=subprocess.PIPE)
  for p in sorted(tmp.glob('*.png')): proc.stdin.write(np.asarray(Image.open(p).convert('RGB'),dtype=np.uint8).tobytes())
  proc.stdin.close(); rc=proc.wait(); print({'episodes':ids,'frames':len(frames),'video':str(out),'returncode':rc,'bytes':out.stat().st_size,'storyboard':str(OUT/'figures/relationnav_qualitative_storyboard.png')})
if __name__=='__main__': main()
