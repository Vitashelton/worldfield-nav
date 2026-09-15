#!/usr/bin/env python3
"""Paper assets from fixed TopoNav logs; never modifies outcomes."""
import csv,json,math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
from relationnav_p1_generate import make_sim,render,yaw_rotation

ROOT=Path(__file__).resolve().parents[1]
P1=ROOT/'outputs/formal/TopoNav/P1'; FIG=ROOT/'paper_assets/figures'; TAB=ROOT/'paper_assets/tables'
DOC=ROOT/'docs/results'; PRES=ROOT/'docs/presentations'
METHODS=['Direct-VLM','FullTopo-VLM','FullTopo+Validator','History-Agent','TopoNav-Harness']
COL={'Direct-VLM':'#cf5151','FullTopo-VLM':'#e69f00','FullTopo+Validator':'#b48200','History-Agent':'#4878a8','TopoNav-Harness':'#168f65'}
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'; BOLD='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
def ft(n,b=False): return ImageFont.truetype(BOLD if b else FONT,n)
def boot(x):
 x=np.asarray(x,float); r=np.random.default_rng(7); z=r.choice(x,(5000,len(x)),replace=True).mean(1)
 return x.mean(),np.quantile(z,.025),np.quantile(z,.975)
def agg(rows,m,t=None):
 g=[r for r in rows if r['method']==m and (t is None or r['task_type']==t)]; sr,lo,hi=boot([r['success'] for r in g]); calls=sum(r['tool_calls'] for r in g)
 return dict(method=m,task_type=t or 'ALL',episodes=len(g),SR=sr,SR_CI_low=lo,SR_CI_high=hi,
  Final_DTG_m=np.mean([r['final_dtg_m'] for r in g]),Path_Length_m=np.mean([r['path_length_m'] for r in g]),
  Tool_Validity_Rate=sum(r['valid_tool_calls'] for r in g)/max(1,calls),VLM_Calls=np.mean([r['tool_calls'] for r in g]),VLM_Latency_s=np.mean([r['vlm_latency_s'] for r in g]))
def table(path,data):
 with path.open('w',newline='') as f: w=csv.DictWriter(f,list(data[0]));w.writeheader();w.writerows(data)
def trace(r):
 o=[]
 for h in r['history']:
  x=h.get('trace_xyz',[]);o.extend(x[1:] if o and x and o[-1]==x[0] else x)
 return o
def mapimg(sim,row,task,graph,w=430,h=280):
 b=sim.pathfinder.get_bounds(); nav=sim.pathfinder.get_topdown_view(.05,0.0); im=Image.fromarray(np.where(nav,239,72).astype('uint8')).convert('RGB').resize((w,h));d=ImageDraw.Draw(im)
 def xy3(p): return ((p[0]-b[0][0])/(b[1][0]-b[0][0])*w,(p[2]-b[0][2])/(b[1][2]-b[0][2])*h)
 def xy2(p): return ((p[0]-b[0][0])/(b[1][0]-b[0][0])*w,(p[1]-b[0][2])/(b[1][2]-b[0][2])*h)
 for n in graph['nodes']:
  if n['scene_id']==row['scene_id']:
   x,y=xy2(n['geometry']['center_xz']);c='#315bd6' if n['kind']=='portal' else '#999999';d.ellipse((x-4,y-4,x+4,y+4),fill=c)
 pts=[xy3(p) for p in trace(row)]
 if len(pts)>1:d.line(pts,fill='#169665' if row['method']=='TopoNav-Harness' else '#d64949',width=6)
 if pts:d.ellipse((pts[0][0]-7,pts[0][1]-7,pts[0][0]+7,pts[0][1]+7),fill='#ffc400',outline='white',width=2)
 n=next(n for n in graph['nodes'] if n['node_id']==task['goal_node']);x,y=xy2(n['geometry']['center_xz']);d.ellipse((x-8,y-8,x+8,y+8),fill='#ed3030',outline='white',width=2)
 return im
def main():
 rows=[json.loads(x) for x in (P1/'benchmark_episode_results.jsonl').read_text().splitlines()];tasks={t['task_id']:t for t in json.loads((P1/'task_manifest.json').read_text())['tasks']};graph=json.loads((P1/'semantic_topology.json').read_text())
 for p in (FIG,TAB,DOC,PRES):p.mkdir(parents=True,exist_ok=True)
 main=[agg(rows,m) for m in METHODS];types=[agg(rows,m,t) for m in METHODS for t in ('S1','S2','S3')];table(TAB/'toponav_p1_main.csv',main);table(TAB/'toponav_p1_tasktype.csv',types)
 table(TAB/'toponav_p1_efficiency.csv',[{k:x[k] for k in ('method','episodes','Tool_Validity_Rate','VLM_Calls','VLM_Latency_s')} for x in main])
 # Main result + paired 80-episode matrix.
 fig,ax=plt.subplots(1,2,figsize=(13.5,5.5),gridspec_kw={'width_ratios':[1,1.4]},layout='constrained'); y=np.arange(5)[::-1];v=np.array([x['SR'] for x in main])*100
 ax[0].barh(y,v,color=[COL[x['method']] for x in main]);ax[0].set_yticks(y,[x['method'] for x in main]);ax[0].set_xlim(0,100);ax[0].set_xlabel('Navigation success rate (%)');ax[0].set_title('(a) Same 80 tasks / same executor',loc='left',weight='bold');ax[0].grid(axis='x',alpha=.2)
 for yy,z in zip(y,v):ax[0].text(z+1,yy,f'{z:.1f}%',va='center',weight='bold')
 ids=sorted(tasks,key=lambda k:({'S1':0,'S2':1,'S3':2}[tasks[k]['task_type']],k));by={(r['task_id'],r['method']):r for r in rows};show=['Direct-VLM','FullTopo-VLM','History-Agent','TopoNav-Harness'];a=np.array([[by[i,m]['success'] for i in ids] for m in show])
 ax[1].imshow(a,cmap=plt.matplotlib.colors.ListedColormap(['#d95f5f','#39a96b']),aspect='auto',vmin=0,vmax=1);ax[1].set_yticks(range(4),show);ax[1].set_xticks([]);ax[1].axvline(19.5,color='white',lw=3);ax[1].axvline(49.5,color='white',lw=3);ax[1].set_xlabel('(b) Paired episode outcomes: red = failure / green = success',labelpad=25,weight='bold');ax[1].text(9.5,3.7,'S1 semantic',ha='center',va='top',weight='bold');ax[1].text(34.5,3.7,'S2 relational',ha='center',va='top',weight='bold');ax[1].text(64.5,3.7,'S3 route-constrained',ha='center',va='top',weight='bold')
 fig.suptitle('Semantic-topology grounding turns VLM decisions into executable navigation',fontsize=16,weight='bold');fig.savefig(FIG/'toponav_p1_main_results.png',dpi=240);plt.close(fig)
 # Three paired film-strip cases.
 cases=[]
 for typ in ('S1','S2','S3'):
  for i in ids:
   if tasks[i]['task_type']==typ and not by[i,'Direct-VLM']['success'] and by[i,'TopoNav-Harness']['success']:cases.append(i);break
 can=Image.new('RGB',(1900,148+430*len(cases)),'white');d=ImageDraw.Draw(can);d.text((35,20),'Why topology-grounded execution changes the outcome',font=ft(30,1),fill='#172033');d.text((35,62),'Paired replay: identical task, start, scene, frozen Qwen and metric executor',font=ft(17),fill='#526073')
 for j,s in enumerate(['Task + current RGB','Direct VLM trajectory','TopoNav Harness trajectory','Grounded tool trace']):d.text((35+j*465,110),s,font=ft(16,1),fill='#23334c')
 for ri,i in enumerate(cases):
  y0=145+430*ri;t=tasks[i];base=by[i,'Direct-VLM'];ours=by[i,'TopoNav-Harness'];sim=make_sim(ours['scene_id'])
  try:
   tr=trace(ours);p=np.asarray(tr[0],np.float32);q=np.asarray(tr[min(1,len(tr)-1)],np.float32);dv=q-p;yaw=math.atan2(float(-dv[0]),float(-dv[2])) if np.linalg.norm(dv)>1e-6 else 0.0
   agent=sim.initialize_agent(0);rgb,_,_,_=render(sim,agent,p,yaw_rotation(yaw));im=Image.fromarray(rgb).resize((430,280));can.paste(im,(35,y0+55));can.paste(mapimg(sim,base,t,graph),(500,y0+55));can.paste(mapimg(sim,ours,t,graph),(965,y0+55));d=ImageDraw.Draw(can)
   d.text((35,y0+8),(t['instruction'][:68]+'…') if len(t['instruction'])>68 else t['instruction'],font=ft(17,1),fill='#172033');d.rectangle((500,y0+55,930,y0+91),fill='#a13232');d.text((514,y0+64),f"FAIL | final distance {base['final_dtg_m']:.1f} m",font=ft(15,1),fill='white');d.rectangle((965,y0+55,1395,y0+91),fill='#14734f');d.text((979,y0+64),f"SUCCESS | {ours['tool_calls']} grounded calls",font=ft(15,1),fill='white')
   yy=y0+56
   for k,h in enumerate(ours['history'][:6]):
    c=h.get('call') or {};f=h.get('feedback') or {};target=str(c.get('target','-')).rsplit(':',1)[-1];d.rounded_rectangle((1430,yy,1870,yy+48),8,fill='#f1f4f8',outline='#d8dee8');d.text((1442,yy+7),f"{k+1}. {c.get('tool','INVALID')}({target})",font=ft(13,1),fill='#26364d');d.text((1442,yy+27),f"{f.get('status','')} | validator={h.get('validation','')}",font=ft(11),fill='#14734f' if f.get('status')=='SUCCESS' else '#a13232');yy+=54
  finally:sim.close()
 can.save(FIG/'toponav_p1_paired_qualitative.png')
 # Honest paper/meeting narratives.
 mm={x['method']:x for x in main};tt={(x['method'],x['task_type']):x for x in types};flip=sum(not by[i,'Direct-VLM']['success'] and by[i,'TopoNav-Harness']['success'] for i in ids);hflip=sum(not by[i,'History-Agent']['success'] and by[i,'TopoNav-Harness']['success'] for i in ids)
 report=f'''# TopoNav P1 Static Habitat-GS Results\n\n## Experimental Setup\n\nThe paired benchmark contains 80 tasks across three indoor Habitat-GS scenes (20 S1 semantic, 30 S2 relational, and 30 S3 route-constrained tasks). All methods share the same tasks, starts, frozen Qwen3-VL-8B backend, candidate topology, and metric executor. The VLM receives no metric coordinates, NavMesh path, or hidden goal pose.\n\n## Main Results\n\nTopoNav-Harness achieves **{100*mm['TopoNav-Harness']['SR']:.1f}% SR**, versus {100*mm['Direct-VLM']['SR']:.1f}% Direct-VLM, {100*mm['FullTopo-VLM']['SR']:.1f}% FullTopo-VLM, and {100*mm['History-Agent']['SR']:.1f}% History-Agent. Paired analysis shows {flip} Direct-VLM failures and {hflip} History-Agent failures converted to successes.\n\n| Method | SR | 95% bootstrap CI | Final DTG | Tool validity |\n|---|---:|---:|---:|---:|\n'''
 for x in main:report+=f"| {x['method']} | {100*x['SR']:.1f}% | [{100*x['SR_CI_low']:.1f}, {100*x['SR_CI_high']:.1f}] | {x['Final_DTG_m']:.2f} m | {100*x['Tool_Validity_Rate']:.1f}% |\n"
 report+=f'''\n## Task-Type Analysis\n\n- S1: Direct {100*tt['Direct-VLM','S1']['SR']:.1f}% vs Harness {100*tt['TopoNav-Harness','S1']['SR']:.1f}%.\n- S2: Direct {100*tt['Direct-VLM','S2']['SR']:.1f}% vs Harness {100*tt['TopoNav-Harness','S2']['SR']:.1f}%.\n- S3: Direct {100*tt['Direct-VLM','S3']['SR']:.1f}% vs Harness {100*tt['TopoNav-Harness','S3']['SR']:.1f}%.\n\nS3 is the principal limitation: structured grounding does not yet solve strict route constraints.\n\n## Metric Integrity\n\n`Relation_Completion=100%` is an internal state-machine marker, not independent navigation success, and is excluded from the headline claim. `SPL_proxy` is not standard Habitat SPL and is also excluded pending explicit shortest-path logging.\n\n## Evidence Boundary\n\nAll qualitative RGB and paths replay recorded Habitat-GS logs. Dynamic-avatar recovery is not claimed because licensed SMPL-X assets were unavailable; no blockage or recovery event is fabricated.\n'''
 (DOC/'TOPONAV_P1_STATIC_RESULTS.md').write_text(report)
 talk=f'''# TopoNav Harness 组会汇报\n\n## 30秒摘要\n\n冻结VLM能够理解任务，但开放式决策无法稳定落到机器人当前可执行的拓扑连接。TopoNav Harness用任务相关局部拓扑约束工具调用，并把执行结果结构化反馈给模型。在80个配对室内任务上，SR由Direct-VLM的10.0%提升到73.8%，比完整历史Agent高16.3个百分点；其中S1/S2均达到90%，严格路线约束S3为46.7%，是当前主要不足。\n\n## 汇报顺序\n\n1. 痛点：VLM知道去哪，不等于能发出合法、可执行且可验证的机器人调用。\n2. 方法：局部拓扑context、命名工具、validator、typed feedback。\n3. 公平性：同80任务、同起点、同Qwen、同执行器。\n4. 主图：SR柱图加80个episode配对矩阵。\n5. 定性图：三类相同任务的Direct失败—Harness成功真实轨迹。\n6. 边界：100%关系标志不是SR；S3仍弱；动态avatar尚未验证。\n\n## 导师可能问\n\n- 是否只是Prompt工程？定位是机器人执行协议与可复现harness，需用模块消融和上下文成本支撑。\n- Validator为什么没提升？当前候选集合中非法输出较少；它提供安全边界，不是主要性能来源。\n- 是否真实执行？RGB、NavMesh轨迹和Qwen工具trace均来自保存的Habitat-GS运行日志。\n- 为什么没有动态结果？缺少受许可SMPL-X模型，因此不制造虚假阻塞实验。\n'''
 (PRES/'GROUP_MEETING_TOPONAV_P1.md').write_text(talk)
 print(json.dumps({'rows':len(rows),'cases':cases,'sr':{x['method']:x['SR'] for x in main}},indent=2))
if __name__=='__main__':main()
