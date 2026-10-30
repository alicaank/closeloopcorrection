"""LingBot native chunks with causal endpoint feedback. Baseline runs the same observer."""
import argparse,os,sys,json,inspect,textwrap,hashlib,time
from pathlib import Path
import numpy as np
from PIL import Image
R=Path(__file__).resolve().parent;PREP=R.parent/'model_readiness_20261027';REPO=PREP/'repos/lingbot'
sys.path.insert(0,str(REPO));sys.path.insert(0,str(R))
from common import rejoin,PoseWorker

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--batch',type=Path,required=True);ap.add_argument('--scene',required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--seed',type=int,default=7);ap.add_argument('--frames',type=int,default=77);ap.add_argument('--arms',default='baseline,correction');ap.add_argument('--native-parity',action='store_true');a=ap.parse_args()
 assert (a.frames-1)%4==0 and ((a.frames-1)//4+1)%4==0
 a.out.mkdir(parents=True,exist_ok=False)
 import torch
 import wan
 import wan.image2video as lib
 from wan.configs import WAN_CONFIGS
 from wan.utils.cam_utils import compute_relative_poses,get_plucker_embeddings
 from einops import rearrange
 row=next(s for s in json.load(open(a.batch/'index.json'))['scenes'] if s['name']==a.scene)
 with np.load(a.batch/'trajectories'/f'{a.scene}.npz') as z:
  plan=np.linalg.inv(z['w2c_overrun'].astype(np.float64))[:a.frames];K=z['intrinsics'][0].astype(np.float64);depth=z['anchor_depth'].astype(np.float32)
 assert len(plan)==a.frames
 img=Image.open(a.batch/row['image']).convert('RGB');assert img.size==(864,480)
 img=img.crop((16,0,848,480));K[0,2]-=16;depth=depth[:,16:848]
 ip=a.out/'input';ip.mkdir();np.save(ip/'poses.npy',plan);np.save(ip/'intrinsics.npy',np.array([[K[0,0],K[1,1],K[0,2],K[1,2]]],np.float32));img.save(ip/'anchor.png')
 cfg=WAN_CONFIGS['i2v-1.3B']
 pipe=wan.WanI2VCausal(config=cfg,checkpoint_dir=str(PREP/'weights/lingbot'),device_id=0,rank=0,t5_fsdp=False,dit_fsdp=False,use_sp=False,t5_cpu=True,convert_model_dtype=False,local_attn_size=18,sink_size=6,infer_mode='causal_fast',assets_dir=str(PREP/'weights/lingbot/assets'))
 original=lib.WanI2VCausal._generate_causal_fast;src=textwrap.dedent(inspect.getsource(original))
 old='c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True)'
 assert src.count(old)==1
 src=src.replace(old,'self._loop_observer.setup(c2ws_infer)\n    '+old)
 needle='\n        pred_latent_chunks = torch.cat(pred_latent_chunks, dim=1)'
 assert src.count(needle)==1
 injected='''
            if chunk_id + 1 < num_inference_chunk:
                self._loop_observer.observe(self, pred_latent_chunks, chunk_id, chunk_size, c2ws_plucker_emb_chunk, Ks, h, w, lat_h, lat_w)

'''
 src=src.replace(needle,injected+needle)
 scope=dict(lib.__dict__);exec(compile(src,str(R/'lingbot_hook_generated.py'),'exec'),scope);hook=scope['_generate_causal_fast'];(a.out/'hook_source.py').write_text(src)
 class Observer:
  def __init__(self,arm,worker):self.arm=arm;self.worker=worker;self.events=[]
  def setup(self,absolute):
   self.plan=absolute.detach().cpu().numpy().copy();self.commands=self.plan.copy()
   raw=compute_relative_poses(absolute.clone(),framewise=True,normalize_trans=False);self.norm=float(raw[:,:3,3].norm(dim=-1).max());self.norm=max(self.norm,1e-12)
  def observe(self,pipeline,prefix,chunk,cs,emb_chunks,Ks,h,w,lh,lw):
   # Decode is stateless at entry/exit in Wan2_1_VAE; preserve every RNG stream.
   t0=time.time()
   with torch.random.fork_rng(devices=[0]):
    decoded=pipeline.vae.decode([torch.cat(prefix,dim=1)])[0]
    frames=((decoded.float()+1)*127.5).round().clamp(0,255).byte().permute(1,2,3,0).cpu().numpy()
    del decoded;torch.cuda.empty_cache()
    actual_K=K.copy();actual_K[0,:]*=frames.shape[2]/832;actual_K[1,:]*=frames.shape[1]/480
    actual_depth=np.asarray(Image.fromarray(depth.astype(np.float32)).resize((frames.shape[2],frames.shape[1]),Image.Resampling.BILINEAR))
    measured,info=self.worker.measure(frames,actual_K,actual_depth)
    self.actual_K=actual_K
   lo=(chunk+1)*cs;hi=min(lo+cs,len(self.plan));assert len(frames)==4*(lo-1)+1
   before=self.commands.copy()
   if self.arm=='correction':self.commands[lo:hi]=rejoin(measured,self.commands[lo-1],self.plan[lo:hi])
   elif self.arm=='sham':self.commands[lo:hi]=rejoin(self.commands[lo-1],self.commands[lo-1],self.plan[lo:hi])
   if self.arm!='baseline':
    absolute=torch.tensor(self.commands,dtype=torch.float32)
    rel=compute_relative_poses(absolute,framewise=True,normalize_trans=False)
    # The next native relative command starts at the measured camera. Cached history stays intact.
    if self.arm=='correction':
     pair=torch.tensor(np.stack([measured,self.commands[lo]]),dtype=torch.float32)
     rel[lo]=compute_relative_poses(pair,framewise=True,normalize_trans=False)[1]
    rel[:,:3,3]/=self.norm
    with torch.autocast(device_type='cuda',enabled=False):
     pl=get_plucker_embeddings(rel.to(pipeline.device),Ks,h,w)
    pl=rearrange(pl,'f (h c1) (w c2) c -> 1 (c c1 c2) f h w',c1=h//lh,c2=w//lw).to(pipeline.param_dtype)
    pl=pl[:,:,lo:hi]
    if self.arm=='sham':assert torch.equal(pl,emb_chunks[chunk+1]),float((pl-emb_chunks[chunk+1]).abs().max())
    emb_chunks[chunk+1].copy_(pl)
   assert np.array_equal(self.commands[:lo],before[:lo])
   assert np.allclose(self.commands[hi-1],self.plan[hi-1])
   self.events.append({'chunk':chunk,'endpoint_frame':len(frames)-1,'first_future_frame':4*lo,'measured':measured.tolist(),'position_residual':float(np.linalg.norm(measured[:3,3]-before[lo-1,:3,3])),'correction_applied':self.arm=='correction','seconds':time.time()-t0,'measurement':info,'prefix_sha256':hashlib.sha256(frames.tobytes()).hexdigest()})
   (a.out/f'{self.arm}_events.json').write_text(json.dumps(self.events,indent=2))
 baseline_raw=None
 try:
  if a.native_parity:
   t=time.time();raw=pipe.generate(row['caption'],img,action_path=str(ip),chunk_size=4,max_area=480*832,frame_num=a.frames,seed=a.seed,offload_model=True)
   baseline_raw=((raw.float()+1)*127.5).round().clamp(0,255).byte().permute(1,2,3,0).cpu().numpy();del raw
   np.savez_compressed(a.out/'native_frames.npz',frames=baseline_raw)
  lib.WanI2VCausal._generate_causal_fast=hook
  for arm in a.arms.split(','):
   assert arm in ['baseline','correction','sham'];worker=PoseWorker(a.out/arm/'observer');obs=Observer(arm,worker);pipe._loop_observer=obs
   try:
    pipe.model.to(pipe.device);t=time.time()
    v=pipe.generate(row['caption'],img,action_path=str(ip),chunk_size=4,max_area=480*832,frame_num=a.frames,seed=a.seed,offload_model=True)
    frames=((v.float()+1)*127.5).round().clamp(0,255).byte().permute(1,2,3,0).cpu().numpy();del v
   finally:worker.close()
   np.savez_compressed(a.out/arm/'frames.npz',frames=frames)
   from wan.utils.utils import save_video
   save_video(tensor=torch.from_numpy(frames).permute(3,0,1,2)[None].float()/127.5-1,save_file=str(a.out/arm/'video.mp4'),fps=16,nrow=1,normalize=True,value_range=(-1,1))
   log={'model':'lingbot','arm':arm,'seed':a.seed,'scene':a.scene,'frames':len(frames),'seconds':time.time()-t,'K':obs.actual_K.tolist(),'input_K_before_native_resize':K.tolist(),'plan':plan.tolist(),'latent_commands':obs.commands.tolist(),'native_translation_normalizer':obs.norm,'events':obs.events,'memory_policy':'KV updates retain actual conditioning; no rewriting','camera_frame_ids':list(range(0,a.frames,4))}
   if arm=='baseline' and baseline_raw is not None:
    delta=np.abs(frames.astype(np.float32)-baseline_raw.astype(np.float32));log['native_parity']={'exact':bool(np.array_equal(frames,baseline_raw)),'pixel_mae':float(delta.mean()),'max':float(delta.max())}
   (a.out/arm/'run.json').write_text(json.dumps(log,indent=2));print('DONE',arm,len(frames),flush=True)
 finally:lib.WanI2VCausal._generate_causal_fast=original

if __name__=='__main__':main()
