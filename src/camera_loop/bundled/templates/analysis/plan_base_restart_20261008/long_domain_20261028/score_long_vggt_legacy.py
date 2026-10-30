"""Independent bounded-view VGGT, image quality and fixed contact strips, never DA3 selection."""
from pathlib import Path
import argparse,sys,json,time
import numpy as np
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parent;A=R.parent
sys.path[:0]=[str(A/'fair_common_20261025'),str(A/'memorizon_plugins'),'@WORKSPACE@/training_runs/joint_samplerfix_20261012',str(A/'fresh_teacher_20261021'),'@WORKSPACE@/training_runs/memory1500_resplat_20260905/python_deps']
import fair_common as fc

def read(path,ids):
 import cv2
 cap=cv2.VideoCapture(str(path));assert cap.isOpened(),path;out={}
 for i in range(max(ids)+1):
  ok,f=cap.read();assert ok,(path,i)
  if i in ids:out[i]=cv2.cvtColor(f,cv2.COLOR_BGR2RGB)
 cap.release();return out

def main():
 p=argparse.ArgumentParser()
 for k in ['batch','model','scene','out']:p.add_argument('--'+k,required=True)
 p.add_argument('--horizon',type=int,required=True);p.add_argument('--runs',nargs='+',required=True);a=p.parse_args();batch=Path(a.batch);out=Path(a.out);h=a.horizon
 row=next(s for s in json.load(open(batch/'index.json'))['scenes'] if s['name']==a.scene)
 with np.load(batch/'trajectories'/f'{a.scene}.npz') as z:plan=np.linalg.inv(z['w2c'].astype(np.float64));depth=z['anchor_depth'].astype(np.float32);K=z['intrinsics'][0]
 milestones=[x for x in [160,320,480] if x<=h]+[h];ids=sorted(set(np.linspace(0,h,41).round().astype(int).tolist()+milestones));fid=sorted(set(np.linspace(1,h,40).round().astype(int).tolist()+milestones));shows=np.linspace(0,h,7).round().astype(int).tolist();allids=set(ids+fid+shows)
 from score_vggt import load_vggt
 from score_vggt_fair import vggt_relposes,vggt_size
 from evaluate_recorded import metrics_models
 from scipy.spatial.transform import Rotation
 import torch
 model=load_vggt();emb,lp=metrics_models();results={};strips={}
 for spec in a.runs:
  arm,d=spec.split('=',1);d=Path(d);f=read(d/'review_full.mp4',allids);H,W=f[0].shape[:2]
  if a.model=='vmem':Kout,_=fc.k_vmem(K,(864,480),(W,H))
  else:Kout=fc.scale_crop_K(K,W/864,H/480)
  dep=fc.remap_depth(depth,K,Kout,(H,W));refs={i:fc.preprocess(a.model,np.asarray(Image.open(Path(row['frames_dir'])/f"frame_{row['original_start']+i+1:05d}.jpg").convert('RGB')),(W,H)) for i in set(fid+shows)}
  meas,scale=vggt_relposes(model,[f[i] for i in ids],dep,vggt_size((H,W)));target=plan[ids];pe=np.linalg.norm(meas[:,:3,3]-target[:,:3,3],axis=-1);rot=np.degrees(Rotation.from_matrix(meas[:,:3,:3].transpose(0,2,1)@target[:,:3,:3]).magnitude());ds=[];ls=[]
  with torch.no_grad():
   for start in range(0,len(fid),4):
    js=fid[start:start+4];g=np.stack([f[i] for i in js]);ref=np.stack([refs[i] for i in js]);ds.extend((emb(g)*emb(ref)).sum(-1).cpu().tolist());ls.extend(np.asarray(lp(g,ref)).tolist())
  horizons={}
  for end in sorted(set(milestones)):
   mask=(np.asarray(ids)>0)&(np.asarray(ids)<=end);ii=ids.index(end);im=np.asarray(fid)<=end
   horizons[str(end)]={'camera_mean':float(pe[mask].mean()),'camera_endpoint':float(pe[ii]),'rotation_mean':float(rot[mask].mean()),'rotation_endpoint':float(rot[ii]),'dino':float(np.asarray(ds)[im].mean()),'lpips':float(np.asarray(ls)[im].mean())}
  results[arm]=dict(horizons=horizons,camera_frame_ids=ids,position_errors=pe.tolist(),rotation_errors=rot.tolist(),measured_c2w=meas.tolist(),anchor_scale=scale,image_frame_ids=fid,dino_by_frame=ds,lpips_by_frame=ls,source=str(d),finite=bool(np.isfinite(meas).all()),path_length_ratio=float(np.linalg.norm(np.diff(meas[:,:3,3],axis=0),axis=-1).sum()/max(1e-8,np.linalg.norm(np.diff(target[:,:3,3],axis=0),axis=-1).sum())))
  assert results[arm]['finite'];strips.setdefault('Recording',[refs[i] for i in shows]);strips[arm]=[f[i] for i in shows]
  print(arm,horizons,flush=True)
 report=dict(model=a.model,scene=a.scene,seed=7,horizon=h,batch=str(batch),units='fixed anchor-depth gauge; NOT verified metric metres',camera_sampling='41 uniformly spaced views plus fixed milestones; same set within pair',arms=results,completed=time.time());out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(out)
 sheet=Image.new('RGB',(280*len(shows),185*len(strips)),'white');draw=ImageDraw.Draw(sheet)
 for r,(label,fs) in enumerate(strips.items()):
  for c,(i,f) in enumerate(zip(shows,fs)):sheet.paste(Image.fromarray(f).resize((280,158)),(c*280,r*185+25));draw.text((c*280+4,r*185+4),f'{label}: {i}',fill='black')
 sheet.save(out.with_suffix('.jpg'))
if __name__=='__main__':main()
