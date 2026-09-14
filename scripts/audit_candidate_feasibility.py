from pathlib import Path
import json,sys,numpy as np
ROOT=Path('/root/autodl-tmp/.autodl/worldfield_nav');sys.path.insert(0,str(ROOT/'scripts'))
from relationnav_p2_evaluate import make_sim,set_agent,route_endpoint,satisfied,entity_index
P1=ROOT/'outputs/formal/RelationNav/P1/dataset'
def main():
 E=json.loads((P1/'dataset_manifest.json').read_text())['episodes']; ents=entity_index(); out=[]
 for scene in sorted({e['scene_id'] for e in E}):
  sim=make_sim(scene);agent=sim.initialize_agent(0)
  try:
   for ep in [e for e in E if e['scene_id']==scene]:
    for ph in ep['phases']:
     set_agent(agent,np.asarray(ph['current_xyz'],np.float32)); before=np.asarray(agent.get_state().position,np.float32);good=[]
     for k,path in enumerate(ph['candidate_paths_goal_independent']):
      target=np.asarray(path[-1],np.float32);ev=route_endpoint(sim,agent,target,.10);after=np.asarray(agent.get_state().position,np.float32);good.append(bool(ev['arrival'] and satisfied(sim,agent,ph,ents,before,after)));set_agent(agent,before)
     out.append({'episode_id':ep['episode_id'],'scene_id':scene,'relation':ph['relation'],'valid_candidates':int(sum(good)),'any_valid':bool(any(good))})
  finally:sim.close()
 print(json.dumps({'states':len(out),'any_valid_rate':float(np.mean([r['any_valid'] for r in out])),'valid_count_mean':float(np.mean([r['valid_candidates'] for r in out])),'by_relation':{rel:{'states':sum(r['relation']==rel for r in out),'any':float(np.mean([r['any_valid'] for r in out if r['relation']==rel])),'mean_valid':float(np.mean([r['valid_candidates'] for r in out if r['relation']==rel]))} for rel in sorted({r['relation'] for r in out})}},indent=2))
 (ROOT/'outputs/formal/RelationNav/P1/candidate_feasibility_audit.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
