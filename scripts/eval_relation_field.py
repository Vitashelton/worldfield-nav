#!/usr/bin/env python3
from pathlib import Path
import json, math, sys
import numpy as np, torch
ROOT=Path('/root/autodl-tmp/.autodl/worldfield_nav'); sys.path.insert(0,str(ROOT/'scripts'))
from relationnav_p1_train import load,Field,batch_maps
from relationnav_p2_evaluate import make_sim,set_agent,route_endpoint,satisfied,entity_index
DATA=ROOT/'outputs/formal/RelationNav/P1/dataset'
def endpoint_score(p, row, ep):
 a=np.asarray(p[-1],np.float32); cam=np.asarray(row['camera_xyz'],np.float32); yaw=float(row['camera_yaw_rad']); delta=a-cam; forward=np.array([-math.sin(yaw),-math.cos(yaw)],np.float32); right=np.array([math.cos(yaw),-math.sin(yaw)],np.float32); lx=float(delta[[0,2]]@right); lf=float(delta[[0,2]]@forward); gx=int(np.clip((lx+4)/.125,0,63)); gy=int(np.clip((lf+2)/.125,0,63)); return gx,gy
def main():
 rows,anns,cache=load(); device='cuda' if torch.cuda.is_available() else 'cpu'; model=Field(); state=torch.load(ROOT/'artifacts/relationnav/relationnav_field.pt',map_location=device,weights_only=False);model.load_state_dict(state['state_dict']);model.to(device).eval();byep={}
 for r in rows: byep.setdefault(r['episode_id'],[]).append(r)
 man=json.loads((DATA/'dataset_manifest.json').read_text())['episodes']; held=[e for e in man if e['split']=='heldout']; entities=entity_index(); out=[]
 for ep in held:
  sim=make_sim(ep['scene_id']);agent=sim.initialize_agent(0);set_agent(agent,np.asarray(ep['start_xyz'],np.float32)); task=True;stage=[]
  try:
   for r in byep[ep['episode_id']]:
    with torch.no_grad(): x,_=batch_maps(rows,anns,cache,[rows.index(r)],device); score=torch.sigmoid(model(x))[0,0].detach().cpu().numpy()
    vals=[]
    for k,p in enumerate(r['candidate_paths_goal_independent']):
      gx,gy=endpoint_score(p,r,ep); vals.append(float(score[gy,gx]))
    k=int(np.argmax(vals)); target=np.asarray(r['candidate_paths_goal_independent'][k][-1],np.float32); before=np.asarray(agent.get_state().position,np.float32); ev=route_endpoint(sim,agent,target,.10); after=np.asarray(agent.get_state().position,np.float32); done=bool(ev['arrival'] and satisfied(sim,agent,r,entities,before,after)); stage.append({'relation':r['relation'],'candidate':k+1,'score':vals[k],'arrival':ev['arrival'],'relation_done':done})
    if not done: task=False
  finally:sim.close()
  out.append({'episode_id':ep['episode_id'],'task_success':bool(task and len(stage)==len(ep['phases'])),'stages':stage})
 print(json.dumps({'episodes':len(out),'task_success':float(np.mean([r['task_success'] for r in out])),'relation_stage_success':float(np.mean([s['relation_done'] for r in out for s in r['stages']])),'examples':out[:3]},indent=2))
 (ROOT/'outputs/formal/RelationNav/P1/field_eval_heldout.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
