from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT=Path('/root/autodl-tmp/.autodl/worldfield_nav'); OUT=ROOT/'paper_assets/figures'; OUT.mkdir(parents=True,exist_ok=True)
fig=plt.figure(figsize=(15,8),dpi=220,facecolor='#0f131c')
gs=fig.add_gridspec(2,2,height_ratios=[1.08,.92],hspace=.38,wspace=.22)
ax=fig.add_subplot(gs[0,:]); ax.set_facecolor('#18202d'); ax.axis('off'); ax.set_xlim(0,1); ax.set_ylim(0,1)
ax.text(.02,.93,'Same episodes · different completion semantics',color='white',fontsize=17,fontweight='bold',ha='left')
ax.text(.02,.875,'Arrival-only can advance the task after a false relation completion; relation-preserving recovery keeps the relation and reselects an admissible target.',color='#bbc7d8',fontsize=10,ha='left')
def box(x,y,w,h,label,val,color):
    p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012,rounding_size=.02',facecolor='#202b3a',edgecolor=color,linewidth=2)
    ax.add_patch(p); ax.text(x+.02,y+h*.62,label,color=color,fontsize=10,fontweight='bold'); ax.text(x+.02,y+h*.25,val,color='white',fontsize=19,fontweight='bold')
def arrow(x1,y1,x2,y2,text,color):
    a=FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=14,linewidth=2,color=color); ax.add_patch(a); ax.text((x1+x2)/2,(y1+y2)/2+.04,text,color=color,fontsize=9,ha='center')
box(.04,.54,.19,.19,'arrival reported','100%', '#ef5b5b'); arrow(.235,.635,.32,.635,'nominal', '#ef5b5b')
box(.33,.54,.19,.19,'relation valid','64.9%', '#ffbe55'); arrow(.525,.635,.61,.635,'false completion 35.1%', '#ef5b5b')
box(.62,.54,.19,.19,'task complete','0%', '#ef5b5b')
ax.text(.04,.48,'Arrival-only / held-out 100 episodes',color='#d9e1ec',fontsize=11,fontweight='bold')
box(.04,.15,.19,.19,'first attempt','64.9%', '#ffbe55'); arrow(.235,.245,.32,.245,'same relation', '#56d88a')
box(.33,.15,.19,.19,'reselected target','97.4%', '#56d88a'); arrow(.525,.245,.61,.245,'semantic success', '#56d88a')
box(.62,.15,.19,.19,'task complete','92%', '#56d88a')
ax.text(.04,.09,'Relation-preserving recovery / held-out 100 episodes',color='#d9e1ec',fontsize=11,fontweight='bold')
ax.text(.86,.57,'35.1%\nwrong-stage\nadvance',ha='center',va='center',color='#ef8c8c',fontsize=11,fontweight='bold')
ax.text(.86,.23,'92.6%\nrecovery\nsuccess',ha='center',va='center',color='#8df2b0',fontsize=11,fontweight='bold')

# Completion matrix
ax2=fig.add_subplot(gs[1,0]); ax2.set_facecolor('#18202d')
methods=['Arrival-only','Recovery','Oracle']; vals=[0,.92,.92]; colors=['#ef5b5b','#56d88a','#8c9eff']
ax2.barh(methods,vals,color=colors,height=.48); ax2.set_xlim(0,1); ax2.set_xlabel('episode semantic task success',color='#cbd5e1'); ax2.tick_params(colors='#cbd5e1'); ax2.xaxis.grid(True,alpha=.2); ax2.set_axisbelow(True); ax2.spines[['top','right','left']].set_visible(False); ax2.spines['bottom'].set_color('#718096')
for i,v in enumerate(vals): ax2.text(v+.025,i,f'{100*v:.1f}%',va='center',color='white',fontweight='bold',fontsize=12)
ax2.set_title('Held-out task outcome',color='white',loc='left',fontweight='bold')

# Metric interpretation panel
ax3=fig.add_subplot(gs[1,1]); ax3.set_facecolor('#18202d'); ax3.axis('off'); ax3.set_xlim(0,1); ax3.set_ylim(0,1)
ax3.text(.02,.9,'What the video demonstrates',color='white',fontsize=15,fontweight='bold')
items=[('1','arrival ≠ relation completion','#ffbe55'),('2','same-goal retry repeats failure','#ef5b5b'),('3','recovery preserves entity + relation','#56d88a'),('4','Habitat-GS supplies predicate ground truth','#8c9eff')]
for i,(n,t,c) in enumerate(items):
 y=.72-i*.17; ax3.text(.04,y,n,color=c,fontsize=16,fontweight='bold'); ax3.text(.12,y,t,color='#dce5f2',fontsize=11,va='center')
ax3.text(.02,.06,'Pilot diagnostic; not a learned-model gain.',color='#9aa9bc',fontsize=10,style='italic')
fig.suptitle('RelationNav pilot: why distance-only completion is not enough',color='white',fontsize=21,fontweight='bold',y=.985)
fig.savefig(OUT/'relationnav_pilot_story_figure.png',bbox_inches='tight',facecolor=fig.get_facecolor()); plt.close(fig)
print(OUT/'relationnav_pilot_story_figure.png')
