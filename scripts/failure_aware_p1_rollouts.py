#!/usr/bin/env python3
"""Collect actual short-horizon Habitat execution outcomes for fixed P1 candidates."""
from __future__ import annotations
import json, math
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
import habitat_sim

from trajectory_grounding_p1_generate import ROOT, make_sim, yaw_rotation, shortest

OUT=ROOT/"outputs/formal/TrajectoryGrounding/P1"

def reset(agent, record):
    rot=np.asarray(record["camera_c2w"],np.float32)
    forward=rot @ np.array([0.,0.,-1.],np.float32)
    yaw=float(math.atan2(float(forward[0]),float(forward[2])))
    state=agent.get_state(); state.position=np.asarray(record["start_xyz"],np.float32); state.rotation=yaw_rotation(yaw)
    agent.set_state(state,reset_sensors=True)

def rollout(sim,agent,record,candidate,max_actions=32):
    reset(agent,record); start=np.asarray(agent.get_state().position,np.float32); endpoint=np.asarray(candidate["executor_endpoint_xyz"],np.float32)
    follower=habitat_sim.GreedyGeodesicFollower(sim.pathfinder,agent,goal_radius=.25)
    try: actions=[action for action in (follower.find_path(endpoint) or []) if action is not None]
    except Exception: actions=[]
    collision=False; length=0.; steps=0
    for action in actions[:max_actions]:
        before=np.asarray(agent.get_state().position,np.float32); hit=bool(agent.act(action)); after=np.asarray(agent.get_state().position,np.float32)
        length+=float(np.linalg.norm(after-before)); collision|=hit; steps+=1
        if collision: break
    final=np.asarray(agent.get_state().position,np.float32); reached=bool(np.linalg.norm(final-endpoint)<=.30)
    goal=np.asarray(record["goal_xyz_hidden_for_evaluation"],np.float32)
    _,final_geo=shortest(sim,final,goal); initial=float(record["initial_geodesic_m"])
    progress=float(initial-final_geo) if np.isfinite(final_geo) else -initial
    time_limited=bool(not reached and not collision and len(actions)>max_actions)
    stuck=bool(not reached and (collision or length<.10))
    return {"reached":reached,"collision":collision,"stuck":stuck,"time_limited":time_limited,"progress_m":progress,"executed_path_length_m":length,"action_count":steps,"final_goal_geodesic_m":float(final_geo),"actions":list(actions[:max_actions])}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--limit",type=int);args=ap.parse_args()
    manifest=json.loads((OUT/"dataset_manifest.json").read_text()); path=OUT/"rollout_outcomes.jsonl"; done=set()
    if path.exists(): done={json.loads(line)["key"] for line in path.read_text().splitlines() if line.strip()}
    by_scene=defaultdict(list)
    for record in manifest: by_scene[record["scene_id"]].append(record)
    count=0
    with path.open("a") as out:
        for scene,records in by_scene.items():
            sim=make_sim(scene);agent=sim.initialize_agent(0)
            try:
                for record in records:
                    for candidate in record["trajectories"]:
                        key=f"{record['episode_id']}:{candidate['trajectory_id']}"
                        if key in done: continue
                        result=rollout(sim,agent,record,candidate)
                        out.write(json.dumps({"key":key,"episode_id":record["episode_id"],"scene_id":scene,"split":record["split"],"trajectory_id":candidate["trajectory_id"],**result})+"\n");out.flush();count+=1
                        if args.limit and count>=args.limit: break
                    if args.limit and count>=args.limit: break
                if args.limit and count>=args.limit: break
            finally: sim.close()
    rows=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    summary={"outcomes":len(rows),"new":count,"reached_rate":float(np.mean([x["reached"] for x in rows])),"collision_rate":float(np.mean([x["collision"] for x in rows])),"stuck_rate":float(np.mean([x["stuck"] for x in rows])),"time_limited_rate":float(np.mean([x["time_limited"] for x in rows]))}
    (OUT/"rollout_outcome_summary.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
