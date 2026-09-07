"""Paper-readable paired P0 overview and top-down decision maps."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
import habitat_sim

ROOT=Path(__file__).resolve().parents[1]; P0=ROOT/"outputs/formal/ExecField/P0"
sys.path.insert(0,str(ROOT/"scripts")); from execfield_p0_generate import make_sim

def shortest(sim,a,b):
 p=habitat_sim.ShortestPath();p.requested_start=np.asarray(a,np.float32);p.requested_end=np.asarray(b,np.float32);ok=sim.pathfinder.find_path(p)
 return np.asarray(p.points) if ok and len(p.points)>1 else None

def load_rel(rel): return Image.open(P0/rel).convert("RGB")

def main():
 eps={e["episode_id"]:e for e in json.loads((P0/"episodes.json").read_text())}; cache={json.loads(x)["request_id"]:json.loads(x) for x in (P0/"vlm_cache.jsonl").read_text().splitlines() if x.strip()}
 pa=ROOT/"paper_assets/figures";pa.mkdir(parents=True,exist_ok=True)
 # Figure A: paired matrix plus central effect sizes.
 ordered=list(eps); vlm=[]; oracle=[]
 for eid in ordered:
  e=eps[eid]; rank=cache[eid]["ranking"];vi=rank[0];oi=next((i for i in rank if e["candidates"][i]["eventual_outcome"]),vi)
  vlm.append(int(e["candidates"][vi]["eventual_outcome"]));oracle.append(int(e["candidates"][oi]["eventual_outcome"]))
 fig=plt.figure(figsize=(15,6));gs=GridSpec(2,3,figure=fig,width_ratios=[1.25,1.25,2.1])
 ax=fig.add_subplot(gs[:,0]); ax.plot([.25,.875],[1,1],color="#477db3",lw=8,solid_capstyle="round");ax.scatter([.25,.875],[1,1],s=220,c=["#d9534f","#2e8b57"],zorder=3);ax.text(.25,1.13,"VLM-only\n25.0%",ha="center",fontweight="bold");ax.text(.875,1.13,"VLM + Oracle\n87.5%",ha="center",fontweight="bold");ax.text(.5625,.82,"+62.5 pp SR",ha="center",fontsize=16,fontweight="bold",color="#174c7e");ax.text(.5625,.67,"same episodes · same candidates · same VLM ranking",ha="center",fontsize=9);ax.set(xlim=(0,1.12),ylim=(.5,1.5));ax.axis("off");ax.set_title("Oracle Executability Gap",fontsize=17,fontweight="bold")
 ax=fig.add_subplot(gs[0,1]);ax.axis("off");ax.text(.05,.75,"SPL",fontsize=12,fontweight="bold");ax.text(.05,.48,"23.8%  →  82.1%",fontsize=20,color="#174c7e",fontweight="bold");ax.text(.05,.12,"+58.3 pp",fontsize=12)
 ax=fig.add_subplot(gs[1,1]);ax.axis("off");ax.text(.05,.70,"Invalid subgoal",fontsize=11);ax.text(.05,.51,"50.0% → 12.5%",fontsize=16,color="#b94a48",fontweight="bold");ax.text(.05,.22,"Wrong goal",fontsize=11);ax.text(.05,.03,"25.0% → 0%",fontsize=16,color="#b94a48",fontweight="bold")
 ax=fig.add_subplot(gs[:,2]);mat=np.array([vlm,oracle]);ax.imshow(mat,cmap=plt.matplotlib.colors.ListedColormap(["#d9534f","#2e8b57"]),vmin=0,vmax=1,aspect="auto");ax.set_yticks([0,1],["VLM-only","VLM + Oracle"]);ax.set_xticks(range(24),[f"{i+1}" for i in range(24)],fontsize=7);ax.set_xlabel("Same fixed ImageNav episode");ax.set_title("Paired episode outcomes: 6/24 → 21/24",fontsize=14,fontweight="bold");
 for y,row in enumerate(mat):
  for x,val in enumerate(row): ax.text(x,y,"✓" if val else "×",ha="center",va="center",color="white",fontweight="bold")
 fig.tight_layout();fig.savefig(pa/"execfield_p0_oracle_gap_overview.png",dpi=220);plt.close(fig)
 # Figure B: three representative spatial decisions.
 picks=[("scene01_p0_02","no progress"),("scene02_p0_00","low clearance"),("interior_0405_840145_p0_03","off-navmesh")]
 fig=plt.figure(figsize=(17,13));gs=GridSpec(3,4,figure=fig,width_ratios=[1,1,1.35,1.35])
 for r,(eid,label) in enumerate(picks):
  e=eps[eid];rank=cache[eid]["ranking"];vi=rank[0];oi=next((i for i in rank if e["candidates"][i]["eventual_outcome"]),vi); vc,oc=e["candidates"][vi],e["candidates"][oi]
  for c,(image,title) in enumerate([(e["goal_image"],"Goal image"),(e["current_rgb"],"Current RGB")]):
   ax=fig.add_subplot(gs[r,c]);ax.imshow(load_rel(image));ax.set_title(title,fontsize=10);ax.axis("off")
  ax=fig.add_subplot(gs[r,2:]); start=np.asarray(e["start_xyz"]);goal=np.asarray(e["goal_xyz_hidden_for_eval"]);cs=e["candidates"];xy=np.array([[x["metric_xyz"][0],x["metric_xyz"][2]] for x in cs]); s=np.array([start[0],start[2]])
  ax.scatter(xy[:,0],xy[:,1],s=58,c="#aeb7c2",edgecolors="white",zorder=3)
  for i,cand in enumerate(cs): ax.text(xy[i,0]+.08,xy[i,1]+.08,f"C{i}\n{cache[eid]['semantic_scores'][i]:.2f}",fontsize=7)
  ax.scatter(*s,s=220,c="#202020",marker="o",zorder=5);ax.text(s[0],s[1]-.28,"robot",ha="center",fontsize=8)
  # show goal direction locally rather than leaking a distant absolute map extent
  d=np.array([goal[0]-start[0],goal[2]-start[2]]);d=d/(np.linalg.norm(d)+1e-8);star=s+d*3.1;ax.scatter(*star,s=220,c="#e6b422",marker="*",zorder=5);ax.annotate("goal direction",xy=star,xytext=s+d*1.7,arrowprops={"arrowstyle":"->","color":"#e6b422"},fontsize=9,color="#8b6b00")
  ax.scatter(*xy[vi],s=150,c="#d9534f",marker="X",zorder=6,label="VLM choice");ax.scatter(*xy[oi],s=150,c="#2e8b57",marker="P",zorder=6,label="Oracle choice")
  sim=make_sim(e["scene_id"])
  try:
   for candidate,color,style in [(vc,"#d9534f","--"),(oc,"#2e8b57","-")]:
    q=shortest(sim,start,candidate["metric_xyz"])
    if q is not None: ax.plot(q[:,0],q[:,2],color=color,ls=style,lw=2.7)
  finally: sim.close()
  ax.set_aspect("equal");ax.grid(alpha=.2);ax.legend(loc="upper right",fontsize=8);ax.set_title(f"{eid}: VLM C{vi} ({vc['failure_reason']}) → Oracle C{oi} (executable)",fontsize=10,fontweight="bold");ax.set_xlabel("world x (m)");ax.set_ylabel("world z (m)")
 fig.suptitle("Why semantic ranking fails: local physical executability corrects the selected subgoal",fontsize=17,fontweight="bold");fig.tight_layout();fig.savefig(pa/"execfield_p0_why_semantic_ranking_fails.png",dpi=220);plt.close(fig)
if __name__=="__main__":main()
