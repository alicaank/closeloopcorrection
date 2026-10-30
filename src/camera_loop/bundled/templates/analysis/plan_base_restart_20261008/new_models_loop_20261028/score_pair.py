"""Independent VGGT + DINO/LPIPS; fixed anchor-depth scale, no trajectory alignment."""
import argparse,sys,json,time,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
R=Path(__file__).resolve().parent;sys.path[:0]=[str(R.parent/'fair_common_20261025'),str(R.parent/'memorizon_plugins'),'@WORKSPACE@/training_runs/joint_samplerfix_20261012',str(R.parent/'fresh_teacher_20261021'),'@WORKSPACE@/training_runs/memory1500_resplat_20260905/python_deps']

def transform(x,model,depth=False):
 if model=='lingbot':
  im=Image.fromarray(np.asarray(x,dtype=np.float32)[:,16:848] if depth else np.asarray(x)[:,16:848])
  return np.asarray(im.resize((832,464),Image.Resampling.BILINEAR if depth else Image.Resampling.BICUBIC))
 if model=='worldwarp':return np.asarray(x)[:,72:792]
 im=Image.fromarray(np.asarray(x,dtype=np.float32) if depth else np.asarray(x))
 method=Image.Resampling.BILINEAR if depth else Image.Resampling.BILINEAR
 if model=='sana':
  sw,sh=im.size;scale=max(704/sh,1280/sw);rw=max(1280,round(sw*scale));rh=max(704,round(sh*scale));left=(rw-1280)//2;top=(rh-704)//2
  return np.asarray(im.resize((rw,rh),method if depth else Image.Resampling.LANCZOS))[top:top+704,left:left+1280]
 if model in ['dreamx','matrixgame']:return np.asarray(im.resize((1280,704),method))
 raise ValueError(model)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--batch',type=Path,required=True);ap.add_argument('--model',required=True);ap.add_argument('--scene',required=True);ap.add_argument('--seed',type=int,default=7);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--runs',nargs='+',required=True);ap.add_argument('--horizon',type=int);a=ap.parse_args()
 from score_vggt import load_vggt
 from score_vggt_fair import vggt_relposes,vggt_size
 from evaluate_recorded import metrics_models
 from scipy.spatial.transform import Rotation
 import torch
 entries=[s.split('=',1) for s in a.runs];row=next(x for x in json.load(open(a.batch/'index.json'))['scenes'] if x['name']==a.scene)
 with np.load(a.batch/'trajectories'/f'{a.scene}.npz') as z:plan=np.linalg.inv(z['w2c_overrun'].astype(np.float64));depth=transform(z['anchor_depth'],a.model,True)
 lengths=[]
 for arm,d in entries:
  with np.load(Path(d)/'frames.npz') as z:lengths.append(len(z['frames']))
 h=min(lengths)-1 if a.horizon is None else a.horizon
 assert h>0 and all(n>h for n in lengths)
 ids=sorted(set([0]+list(range(4,h+1,4))+[h]));fid=list(range(1,h+1,4));refs=np.stack([transform(np.asarray(Image.open(Path(row['frames_dir'])/f"frame_{row['original_start']+j+1:05d}.jpg").convert('RGB')),a.model) for j in fid])
 vggt=load_vggt();emb,lp=metrics_models();results={};review={};show_ids=np.unique(np.linspace(0,h,5).astype(int))
 for arm,d in entries:
  t=time.time()
  with np.load(Path(d)/'frames.npz') as z:frames=z['frames'][:h+1]
  assert np.isfinite(frames).all() and frames.dtype==np.uint8
  review[arm]=frames[show_ids].copy()
  measured,scale=vggt_relposes(vggt,[frames[i] for i in ids],depth,vggt_size(frames.shape[1:3]))
  target=plan[ids];pe=np.linalg.norm(measured[:, :3,3]-target[:,:3,3],axis=-1);rot=np.degrees(Rotation.from_matrix(measured[:,:3,:3].transpose(0,2,1)@target[:,:3,:3]).magnitude())
  ds=[];ls=[]
  with torch.no_grad():
   for start in range(0,len(fid),4):
    gen=frames[fid[start:start+4]];ref=refs[start:start+4];ds.extend((emb(gen)*emb(ref)).sum(-1).detach().cpu().tolist());ls.extend(np.asarray(lp(gen,ref)).tolist())
  entry={'camera_mean':float(pe[1:].mean()),'camera_endpoint':float(pe[-1]),'rotation_mean':float(rot[1:].mean()),'rotation_endpoint':float(rot[-1]),'dino':float(np.mean(ds)),'lpips':float(np.mean(ls)),'path_length_ratio':float(np.linalg.norm(np.diff(measured[:,:3,3],axis=0),axis=-1).sum()/max(np.linalg.norm(np.diff(target[:,:3,3],axis=0),axis=-1).sum(),1e-8)),'camera_frame_ids':ids,'position_errors':pe.tolist(),'rotation_errors':rot.tolist(),'measured_c2w':measured.tolist(),'anchor_scale':scale,'seconds':time.time()-t,'output_directory':d,'frame_sha256':hashlib.sha256(frames.tobytes()).hexdigest()}
  results[arm]=entry
  report={'model':a.model,'scene':a.scene,'seed':a.seed,'horizon':h,'units':'anchor-depth units; metric scale not claimed','batch':str(a.batch),'arms':results};a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2));print(json.dumps({'arm':arm,**{k:entry[k] for k in ['camera_mean','camera_endpoint','rotation_endpoint','dino','lpips']}}),flush=True)
 # Save a fixed, unselected strip for every scored arm alongside the recording.
 from PIL import ImageDraw
 gt=np.stack([transform(np.asarray(Image.open(Path(row['frames_dir'])/f"frame_{row['original_start']+int(j)+1:05d}.jpg").convert('RGB')),a.model) for j in show_ids])
 strips={'Recording':gt,**review};sheet=Image.new('RGB',(320*len(show_ids),205*len(strips)),'white');draw=ImageDraw.Draw(sheet)
 for row_id,(label,fr) in enumerate(strips.items()):
  for col,(j,f) in enumerate(zip(show_ids,fr)):
   sheet.paste(Image.fromarray(f).resize((320,180)),(320*col,205*row_id+25));draw.text((320*col+5,205*row_id+5),f'{label} | frame {j}',fill='black')
 sheet.save(a.out.with_suffix('.jpg'))

if __name__=='__main__':main()
