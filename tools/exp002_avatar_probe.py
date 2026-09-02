#!/usr/bin/env python3
"""Minimal standalone Habitat-GS dynamic-avatar probe (no installation)."""
import sys
import types
from pathlib import Path

# The official driver contains precomputed joint matrices.  Habitat-GS creates
# an SMPL object during setup even though this fast path never calls it.  Keep
# this compatibility object process-local: no package or model is installed.
class _UnusedSMPL:
    def __init__(self, *args, **kwargs): pass
    def to(self, *args, **kwargs): return self
    def eval(self): return self
sys.modules.setdefault("smplx", types.SimpleNamespace(SMPLX=_UnusedSMPL, SMPL=_UnusedSMPL))

import habitat_sim
import habitat_sim.agent
import numpy as np

ROOT = Path("/root/autodl-tmp/worldfield_nav")
DATA = ROOT / "data/scene_datasets/gs_scenes"

def spec(name, kind):
    s = habitat_sim.CameraSensorSpec(); s.uuid=name; s.sensor_type=kind
    s.sensor_subtype=habitat_sim.SensorSubType.PINHOLE; s.resolution=[256,256]
    s.position=[0.,1.5,0.]; s.hfov=90.; s.near=.01; s.far=10.
    return s

cfg = habitat_sim.SimulatorConfiguration()
cfg.scene_dataset_config_file = str(DATA / "dynamic_nav/dynamic_nav.scene_dataset_config.json")
cfg.scene_id = "scene09"
cfg.enable_physics = False; cfg.create_renderer = True; cfg.gpu_device_id = 0
agent = habitat_sim.agent.AgentConfiguration()
agent.sensor_specifications=[spec("rgb",habitat_sim.SensorType.COLOR),spec("depth",habitat_sim.SensorType.DEPTH)]
sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg,[agent]))
try:
    print("AVATARS", len(sim._gaussian_avatar_manager.avatars))
    avatar = sim._gaussian_avatar_manager.avatars[0]
    for t in (0.0, 0.5, 1.0):
        sim.gaussian_time=t
        sim._update_gaussian_avatars()
        obs=sim.get_sensor_observations()
        caps=np.asarray(avatar.get_navmesh_capsules(),dtype=np.float32)
        print("TIME",t,"RGB",np.asarray(obs['rgb']).shape,"DEPTH",np.asarray(obs['depth']).shape,"CAPS",caps.shape,"CENTER",caps[:,:6].reshape(-1,2,3).mean((0,1)))
finally:
    sim.close()
