#!/usr/bin/env python3
"""Exp003 — deterministic field transport vs. action-dependent information injection.

No simulator is stepped and no network is trained.  The transport construction
sees only Phi_t and the action-induced future crop origins (GT robot pose).
Future field snapshots are loaded solely afterwards as supervision for R.
"""
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT=Path('/root/autodl-tmp/worldfield_nav')
SRC=ROOT/'outputs/exp002_counterfactual_ego_flow'
OUT=ROOT/'outputs/exp003_transport_injection'
ENC=ROOT/'tools/raw_rgb_to_mp4'
N=128; CELL=10./128; FPS=12
NAMES=['A_straight','B_left','C_right']

class Video:
    def __init__(self,p,w=768,h=256): self.p=subprocess.Popen([str(ENC),str(p),str(w),str(h),str(FPS)],stdin=subprocess.PIPE)
    def add(self,x): self.p.stdin.write(np.ascontiguousarray(x[...,:3].astype(np.uint8)).tobytes())
    def close(self): self.p.stdin.close(); assert self.p.wait()==0

def render(phi, mode='occupancy'):
    image=np.full((N,N,3),(18,20,24),np.uint8); seen=phi[2]>0; occ=phi[0]>.5
    if mode=='occupancy': image[seen]=(48,78,88); image[occ]=(40,220,195)
    elif mode=='injection':
        # channel 2 is a binary injection mask in this visualisation tensor.
        image[:]=(18,20,24); image[phi[2]>.5]=(255,135,50); image[phi[0]>.5]=(255,230,75)
    im=Image.fromarray(image).resize((256,256),Image.Resampling.NEAREST); d=ImageDraw.Draw(im); d.ellipse((124,124,132,132),fill=(255,55,55)); return np.asarray(im)

def transport(phi0, origin0, destination_origin, horizon):
    """Regrid old world cells into the future world-aligned local crop.

    This is the deterministic T(Phi_t,a): the input is never augmented with
    future RGB/depth/field values.  Visibility does not grow under transport.
    """
    out=np.zeros_like(phi0); out[3].fill(float(phi0.shape[1]+horizon))
    rows,cols=np.indices((N,N)); wx=destination_origin[0]+(cols+.5)*CELL; wz=destination_origin[1]+(N-rows-.5)*CELL
    src_col=np.floor((wx-origin0[0])/CELL).astype(np.int32); src_z=np.floor((wz-origin0[1])/CELL).astype(np.int32); src_row=N-1-src_z
    valid=(src_col>=0)&(src_col<N)&(src_row>=0)&(src_row<N)
    for c in range(3):
        out[c,valid]=phi0[c,src_row[valid],src_col[valid]]
    old_age=phi0[3,src_row[valid],src_col[valid]]; old_seen=phi0[2,src_row[valid],src_col[valid]]>.5
    out_age=np.full_like(old_age,float(phi0.shape[1]+horizon)); out_age[old_seen]=old_age[old_seen]+horizon; out[3,valid]=out_age
    return out

def iou(a,b):
    u=np.count_nonzero(a|b); return float(np.count_nonzero(a&b)/u) if u else 1.0

def montage(gt,pred,inj,labels=('GT future','transport only','information injection')):
    im=Image.new('RGB',(768,256)); im.paste(Image.fromarray(render(gt)),(0,0)); im.paste(Image.fromarray(render(pred)),(256,0)); im.paste(Image.fromarray(render(inj,'injection')),(512,0)); d=ImageDraw.Draw(im)
    for x,t in zip((10,266,522),labels): d.text((x,10),t,fill=(255,255,255))
    return np.asarray(im)

def plot(path,series,title):
    im=Image.new('RGB',(960,480),(18,20,24)); d=ImageDraw.Draw(im); d.text((24,20),title,fill='white')
    allv=np.concatenate([x for _,x in series]); lo=float(allv.min()); hi=max(float(allv.max()),lo+1e-7); cols=[(60,210,255),(255,155,65),(200,100,255)]
    for k,((name,x),col) in enumerate(zip(series,cols)):
        pts=[(48+i/(len(x)-1)*880,430-(float(v)-lo)/(hi-lo)*340) for i,v in enumerate(x)]
        d.line(pts,fill=col,width=2); d.text((60,58+k*23),name,fill=col)
    d.text((48,448),f'range {lo:.4g} .. {hi:.4g}',fill=(210,210,210)); im.save(path)

def main():
    if not ENC.is_file(): raise RuntimeError('local mp4 encoder missing')
    OUT.mkdir(parents=True,exist_ok=True); summaries={}; injections=[]; ious=[]
    for name in NAMES:
        x=np.load(SRC/f'exp002_{name}.npz',allow_pickle=False); gt=x['phi'].astype(np.float32); origins=x['field_origin'].astype(np.float32)
        phi0,origin0=gt[0],origins[0]; predicted=[]; residual=[]; injection=[]; occ_iou=[]; injected_area=[]; residual_l1=[]; old_residual=[]
        video=Video(OUT/f'{name}_transport_injection.mp4')
        for h in range(len(gt)):
            pred=transport(phi0,origin0,origins[h],h); future=gt[h]
            # R=GT - T.  Positive visibility residual is precisely newly
            # revealed information; it cannot be produced by transport.
            new_visible=(future[2]>.5)&~(pred[2]>.5)
            r=future-pred; inj=np.zeros_like(future); inj[0]=np.maximum(r[0],0); inj[2]=new_visible.astype(np.float32); inj[3]=np.maximum(-r[3],0)
            predicted.append(pred); residual.append(r); injection.append(inj); occ_iou.append(iou(future[0]>.5,pred[0]>.5)); injected_area.append(float(np.count_nonzero(new_visible)*CELL*CELL)); residual_l1.append(float(np.mean(np.abs(r[:3])))); old=(future[2]>.5)&(pred[2]>.5); old_residual.append(float(np.mean(np.abs(r[0][old]))) if np.any(old) else 0.)
            video.add(montage(future,pred,inj))
        video.close(); predicted=np.stack(predicted); residual=np.stack(residual); injection=np.stack(injection)
        np.savez_compressed(OUT/f'exp003_{name}.npz',transport_only=predicted,residual=residual,information_injection=injection,gt_phi=gt,field_origin=origins,timestamps=x['timestamps'],actions=x['actions'],metadata=json.dumps({'construction':'transport uses Phi_t plus GT action-induced future field origin only; GT future Phi only used after transport for residual','channels':['occupancy','height','visibility','age'],'residual':'R=Phi_GT-transport_only','positive_visibility_residual':'newly observed information injection'}))
        final=montage(gt[-1],predicted[-1],injection[-1]); Image.fromarray(final).save(OUT/f'{name}_final_transport_vs_injection.png')
        summaries[name]={'horizon_frames':int(len(gt)-1),'transport_occupancy_iou_mean':float(np.mean(occ_iou)),'transport_occupancy_iou_final':float(occ_iou[-1]),'information_injection_area_final_m2':float(injected_area[-1]),'information_injection_area_mean_m2':float(np.mean(injected_area)),'residual_l1_mean_OHV':float(np.mean(residual_l1)),'old_observed_occupancy_residual_mean':float(np.mean(old_residual))}
        injections.append((name,np.asarray(injected_area))); ious.append((name,np.asarray(occ_iou)))
    plot(OUT/'ABC_information_injection_area.png',injections,'Action-dependent new observed area (m²)')
    plot(OUT/'ABC_transport_occupancy_iou.png',ious,'Transport-only occupancy IoU against GT future field')
    metrics={'experiment':'003_transport_vs_information_injection','input_experiment':'002_counterfactual_ego_flow_rollout','transport_input':'shared Phi_t + GT action-induced robot pose/crop origin; no future observations','comparison_only':'future GT field used only to calculate R=GT-transport','field':{'channels':['O','H','V','A'],'world_plane':'X-Z fixed, Y height','resolution':[N,N],'cell_m':CELL},'branches':summaries,'interpretation':'positive visibility residual localizes action-dependent newly revealed information; old observed occupancy residual checks persistence under transport'}
    (OUT/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n'); print(json.dumps(metrics,indent=2))

if __name__=='__main__': main()
