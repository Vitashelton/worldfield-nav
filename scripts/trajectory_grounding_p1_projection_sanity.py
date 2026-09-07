#!/usr/bin/env python3
"""Create paired RGB-overlay and robot-centric top-down projection sanity asset."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
def main():
 root=ROOT/"outputs/formal/TrajectoryGrounding/P1"; record=json.loads((root/"dataset_manifest.json").read_text())[0]
 rgb=Image.open(root/record["trajectory_overlay"]).convert("RGB")
 top=Image.new("RGB",(512,512),"white"); d=ImageDraw.Draw(top); center=np.asarray(record["camera_world_xyz"],np.float32); scale=75.
 colors=[(55,126,184),(228,26,28),(77,175,74),(152,78,163),(255,127,0),(166,86,40),(247,129,191),(120,120,120)]
 def px(p): return (256+scale*(p[0]-center[0]),256-scale*(p[2]-center[2]))
 d.ellipse((250,250,262,262),fill="black");d.text((266,250),"robot",fill="black")
 for i,t in enumerate(record["trajectories"]):
  pts=[px(np.asarray(p)) for p in t["world_path_xyz"]];d.line(pts,fill=colors[i],width=3);d.text(pts[-1],f"T{i+1}",fill=colors[i])
 g=px(np.asarray(record["goal_xyz_hidden_for_evaluation"]));d.regular_polygon((g,9),5,fill="gold",outline="black");d.text((g[0]+10,g[1]),"goal (eval only)",fill="black")
 canvas=Image.new("RGB",(1024,512),"white");canvas.paste(rgb.resize((512,512)),(0,0));canvas.paste(top,(512,0));
 ImageDraw.Draw(canvas).text((12,12),"RGB: projected T1...T8",fill="white",stroke_width=2,stroke_fill="black");ImageDraw.Draw(canvas).text((524,12),"Top-down: same proposed paths",fill="black")
 out=ROOT/"paper_assets/figures/trajectory_grounding_p1_projection_sanity.png";out.parent.mkdir(parents=True,exist_ok=True);canvas.save(out);print(out)
if __name__=="__main__":main()
