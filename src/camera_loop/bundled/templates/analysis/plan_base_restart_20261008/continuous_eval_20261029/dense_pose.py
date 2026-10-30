"""ViPE on consecutive real RGB. No official poses enter SLAM; alternating sparse views validate it."""
from pathlib import Path
import os,sys,json,argparse,time
os.environ['HF_HUB_OFFLINE']='1'
import numpy as np
import torch
from PIL import Image
R=Path(__file__).resolve().parent;sys.path.insert(0,'@WORKSPACE@/vipe');sys.path.insert(0,str(R.parent/'fair_common_20261025'))
from vipe.config import parse_typed_config
from vipe.slam.system import SLAMSystem
from vipe.streams.base import VideoStream,VideoFrame,FrameAttribute
from vipe.utils.cameras import CameraType
from vipe.utils.model_cache import ModelCache
import fair_common as fc

class DiskFrames(VideoStream):
 def __init__(self,d,meta):
  super().__init__();self.d=d;self.meta=meta;K=np.array(meta['K']);self.k=torch.tensor([K[0,0],K[1,1],K[0,2],K[1,2]],dtype=torch.float32)
 def frame_size(self):return (480,864)
 def name(self):return self.meta['scene']
 def fps(self):return float(self.meta['source_fps'])
 def __len__(self):return self.meta['frames']
 def attributes(self):return {FrameAttribute.INTRINSICS,FrameAttribute.CAMERA_TYPE}
 def __getitem__(self,i):
  im=np.asarray(Image.open(self.d/'frames'/f'frame_{i+1:05d}.jpg').convert('RGB')).copy()
  return VideoFrame(raw_frame_idx=i,rgb=torch.from_numpy(im).cuda().float()/255,intrinsics=self.k.clone(),camera_type=CameraType.PINHOLE)
 def __iter__(self):
  for i in range(len(self)):yield self[i]
class GuardedSLAM(SLAMSystem):
 def _add_keyframe(self,*args,**kwargs):
  phase=kwargs.get('phase');reserve=int(self.config.infill_chunk_size)+2 if phase==1 else 0
  if self.buffer.n_frames>=int(self.config.buffer)-reserve:raise RuntimeError('ViPE keyframe buffer capacity reached; no truncated trajectory accepted')
  return super()._add_keyframe(*args,**kwargs)

def fit_similarity(x,y):
 # y = scale * R * x + t; estimated only on even-numbered sparse validation views.
 xc=x-x.mean(0);yc=y-y.mean(0);u,s,vt=np.linalg.svd(yc.T@xc/len(x));sign=np.ones(3);sign[-1]=np.linalg.det(u@vt);rot=(u*sign)@vt;scale=float((s*sign).sum()/np.mean(np.sum(xc*xc,axis=1)));trans=y.mean(0)-scale*(rot@x.mean(0));return scale,rot,trans

def validate(poses,sparse):
 from scipy.spatial.transform import Rotation
 ids=sparse['dense_ids'];gt=sparse['c2w'];assert len(ids)>=20
 calib=np.arange(0,len(ids),2);test=np.arange(1,len(ids),2);pred=poses[ids]
 scale,rot,t=fit_similarity(pred[calib,:3,3],gt[calib,:3,3]);p=scale*(pred[:,:3,3]@rot.T)+t;pe=np.linalg.norm(p-gt[:,:3,3],axis=-1)
 re=np.degrees(Rotation.from_matrix((rot@pred[:,:3,:3]).transpose(0,2,1)@gt[:,:3,:3]).magnitude());extent=float(np.max(np.linalg.norm(gt[:,:3,3]-gt[0,:3,3],axis=-1)))
 rmse=float(np.sqrt(np.mean(pe[test]**2))/max(extent,1e-8));med=float(np.median(re[test]));p95=float(np.quantile(re[test],.95));passed=bool(np.isfinite(poses).all() and extent>1e-5 and scale>0 and rmse<=.05 and med<=3 and p95<=5)
 return dict(passed=passed,heldout_position_rmse_over_extent=rmse,heldout_rotation_median_deg=med,heldout_rotation_p95_deg=p95,official_extent=extent,fit_scale=scale,fit_rotation=rot.tolist(),fit_translation=t.tolist(),calibration_dense_ids=ids[calib].tolist(),validation_dense_ids=ids[test].tolist(),position_errors=pe.tolist(),rotation_errors=re.tolist(),thresholds={'rmse_over_extent':.05,'median_rotation_deg':3,'p95_rotation_deg':5},note='GlobalSim3 gauge fit uses only alternating official sparse views; remaining views validate. No official extrinsics supplied to ViPE.')

def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--dry-run',action='store_true');a=p.parse_args();d=R/'scenes'/a.scene;meta=json.load(open(d/'prepared.json'))
 overrides=['pipeline=default','pipeline.init.intrinsics=gt','pipeline.init.instance=null','pipeline.post.depth_align_model=null','pipeline.output.save_viz=false','pipeline.output.save_artifacts=false','pipeline.slam.buffer=384']
 cfg=parse_typed_config('default',hydra_args=overrides).pipeline.slam.to_dictconfig();assert not cfg.optimize_intrinsics
 if a.dry_run:print('ViPE configuration and inputs valid',meta['frames'],dict(cfg));return
 torch.manual_seed(0);t=time.time();slam=GuardedSLAM(torch.device('cuda'),cfg,ModelCache());out=slam.run([DiskFrames(d,meta)],camera_type=CameraType.PINHOLE)
 poses=out.trajectory.matrix().detach().cpu().double().numpy();assert poses.shape==(meta['frames'],4,4);poses=np.linalg.inv(poses[0])@poses
 assert int(slam.buffer.tstamp[0])==0
 # Anchor depth prior was estimated from frame0 alone; do not expose optimized/future-dependent depth to inference.
 disp=slam.buffer.disps_sens[0,0].detach().cpu().float().numpy();depth=np.full_like(disp,np.nan);ok=np.isfinite(disp)&(disp>0);depth[ok]=1/disp[ok]
 ki=slam.buffer.intrinsics[0,:4].detach().cpu().double().numpy()/8;Kl=np.array([[ki[0],0,ki[2]],[0,ki[1],ki[3]],[0,0,1]])
 depth=fc.remap_depth(depth,Kl,np.array(meta['K']),(480,864));valid=np.isfinite(depth)&(depth>0);assert valid.mean()>.85
 unit=float(np.median(depth[valid]));gauge=depth/unit;poses[:,:3,3]/=unit
 with np.load(d/'official_sparse.npz') as z:quality=validate(poses,{k:z[k] for k in z.files})
 np.savez_compressed(d/'dense_poses.npz',c2w=poses,anchor_depth=gauge,K=np.array(meta['K']),metres_per_anchor_unit=unit,source_frame_ids=np.arange(meta['frames'])+meta['source_start'])
 quality.update(scene=a.scene,frames=len(poses),seconds=time.time()-t,keyframes=int(slam.buffer.n_frames),anchor_depth_source='single-view UniDepth-L prior at frame0, frozen; never the optimized SLAM depth',metres_per_anchor_unit_estimate=unit,settings=list(overrides))
 (d/'pose_quality.json').write_text(json.dumps(quality,indent=2));print(json.dumps(quality),flush=True)
 # Process success means the estimation finished. Explicit quality.passed controls eligibility downstream.
if __name__=='__main__':main()
