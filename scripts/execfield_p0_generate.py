"""Deterministic Habitat-GS P0 episode/candidate generator.

Candidate sampling is polar geometry only. NavMesh is queried strictly after
sampling to create training-only oracle labels.
"""
from __future__ import annotations
import json, math, shutil
from pathlib import Path
import numpy as np
from PIL import Image
import habitat_sim
import habitat_sim.agent
import magnum as mn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/formal/ExecField/P0"
SCENES = ["scene01", "scene02", "scene03", "interior_0405_840145"]
K, PER_SCENE, SEED = 8, 6, 20260907

def sensor(uuid, typ):
    s = habitat_sim.CameraSensorSpec(); s.uuid = uuid; s.sensor_type = typ
    s.sensor_subtype = habitat_sim.SensorSubType.PINHOLE; s.resolution = [256, 256]
    s.position = [0.0, 1.5, 0.0]; s.hfov = 90.0
    if typ == habitat_sim.SensorType.DEPTH: s.channels = 1
    return s

def make_sim(scene):
    d = ROOT / "data/scene_datasets/gs_scenes/train" / scene
    if not d.exists(): d = ROOT / "data/scene_datasets/gs_scenes/val" / scene
    ply, nav = d / f"{scene}.gs.ply", d / f"{scene}.navmesh"
    if not ply.is_file() or not nav.is_file(): raise FileNotFoundError(d)
    cfg = habitat_sim.SimulatorConfiguration(); cfg.scene_id = "NONE"; cfg.gpu_device_id = 0; cfg.create_renderer = True
    ac = habitat_sim.agent.AgentConfiguration(); ac.height = 1.5; ac.radius = .1
    ac.sensor_specifications = [sensor("rgb", habitat_sim.SensorType.COLOR), sensor("depth", habitat_sim.SensorType.DEPTH)]
    sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [ac]))
    h = habitat_sim.RenderInstanceHelper(sim, use_xyzw_orientations=False)
    h.add_instance(asset_filepath=str(ply), semantic_id=0, scale=mn.Vector3(1.0,1.0,1.0))
    h.set_world_poses(np.array([[0,0,0]],np.float32),np.array([[1,0,0,0]],np.float32))
    if not sim.pathfinder.load_nav_mesh(str(nav)): raise RuntimeError(f"navmesh {scene}")
    return sim

def path(sim, a, b):
    p = habitat_sim.ShortestPath(); p.requested_start = a; p.requested_end = b
    ok = sim.pathfinder.find_path(p)
    return bool(ok and np.isfinite(p.geodesic_distance)), float(p.geodesic_distance)

def look_at(pos, target):
    d = np.asarray(target) - np.asarray(pos); yaw = math.atan2(float(d[0]), float(-d[2]))
    return habitat_sim.utils.common.quat_from_angle_axis(yaw, np.array([0.,1.,0.]))

def render(agent, pos, rot, image_path):
    st = agent.get_state(); st.position = np.asarray(pos, np.float32); st.rotation = rot; agent.set_state(st, reset_sensors=True)
    obs = agent._sim.get_sensor_observations(); rgb = np.asarray(obs["rgb"])[...,:3]
    Image.fromarray(rgb).save(image_path)
    return rgb

def main():
    if OUT.exists():
        for p in (OUT / "images", OUT / "requests"): shutil.rmtree(p, ignore_errors=True)
    (OUT / "images").mkdir(parents=True, exist_ok=True); (OUT / "requests").mkdir(parents=True, exist_ok=True)
    rows, requests = [], []
    for si, scene in enumerate(SCENES):
        rng = np.random.default_rng(SEED + si); sim = make_sim(scene); agent = sim.initialize_agent(0)
        try:
          accepted = 0; attempts = 0
          while accepted < PER_SCENE and attempts < 5000:
            attempts += 1; start = np.asarray(sim.pathfinder.get_random_navigable_point(), np.float32); goal = np.asarray(sim.pathfinder.get_random_navigable_point(), np.float32)
            ok, d0 = path(sim, start, goal)
            if not ok or not 3.0 <= d0 <= 15.0: continue
            eid = f"{scene}_p0_{accepted:02d}"; epdir = OUT / "images" / eid; epdir.mkdir()
            start_rot = look_at(start, goal); render(agent, start, start_rot, epdir / "current_rgb.png")
            render(agent, goal, look_at(goal, start), epdir / "goal_image.png")
            cs = []
            # No NavMesh in this generator: only polar coordinates around start.
            base = math.atan2(float(goal[0]-start[0]), float(goal[2]-start[2]))
            for ci in range(K):
                angle = base + 2*math.pi*ci/K; radius = float(rng.uniform(.8, 2.0))
                raw = start + np.array([radius*math.sin(angle),0.,radius*math.cos(angle)],np.float32)
                snap = np.asarray(sim.pathfinder.snap_point(raw),np.float32)
                snap_ok = bool(np.isfinite(snap).all() and np.linalg.norm(snap-raw) <= .6)
                reachable, d_candidate_goal = path(sim, snap, goal) if snap_ok else (False, float("inf"))
                clear = float(sim.pathfinder.distance_to_closest_obstacle(snap, 2.0)) if snap_ok else 0.0
                progress = float(d0-d_candidate_goal) if reachable else -float(d0)
                collision_free = bool(snap_ok and clear >= .18 and reachable)
                eventual = bool(collision_free and progress > 0.0)
                if not snap_ok: reason = "off_navmesh"
                elif not reachable: reason = "unreachable"
                elif clear < .18: reason = "low_clearance"
                elif progress <= 0: reason = "no_progress"
                else: reason = "executable"
                cpath = epdir / f"candidate_{ci:02d}.png"; render(agent, snap if snap_ok else start, look_at(snap if snap_ok else start, goal), cpath)
                cs.append({"candidate_id":ci,"raw_xyz":raw.tolist(),"metric_xyz":snap.tolist(),"view_path":str(cpath.relative_to(OUT)),"reachable":reachable,"collision_free":collision_free,"clearance_m":clear,"geodesic_progress_m":progress,"eventual_outcome":eventual,"failure_reason":reason})
            ep = {"episode_id":eid,"scene_id":scene,"seed":SEED+si,"start_xyz":start.tolist(),"goal_xyz_hidden_for_eval":goal.tolist(),"initial_geodesic_m":d0,"current_rgb":str((epdir/"current_rgb.png").relative_to(OUT)),"goal_image":str((epdir/"goal_image.png").relative_to(OUT)),"candidate_count":K,"candidates":cs}
            rows.append(ep)
            requests.append({"request_id":eid,"goal_image":ep["goal_image"],"current_rgb":ep["current_rgb"],"candidates":[{"candidate_id":c["candidate_id"],"image":c["view_path"]} for c in cs],"required_response":{"ranked_candidate_ids":"list[int] length 8","semantic_scores":"list[float] length 8","rationale":"short string"}})
            accepted += 1
        finally: sim.close()
    (OUT/"episodes.json").write_text(json.dumps(rows,indent=2)+"\n")
    (OUT/"requests"/"vlm_request_package.json").write_text(json.dumps(requests,indent=2)+"\n")
    summary={"episodes":len(rows),"scenes":SCENES,"candidates":len(rows)*K,"generator":"polar_no_navmesh","vlm_cache_status":"pending_external_worker"}
    (OUT/"generation_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))
if __name__ == "__main__": main()
