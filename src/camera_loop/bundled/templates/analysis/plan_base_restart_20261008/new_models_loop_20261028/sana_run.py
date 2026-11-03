"""SANA native chunks with serialized, causal stage1/refiner/decode and CPU offload."""
import argparse,sys,os,json,inspect,textwrap,time,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
R=Path(__file__).resolve().parent;P=R.parent/'model_readiness_20261027';REPO=P/'repos/sana';sys.path[:0]=[str(REPO),str(R)]
os.environ.update(DISABLE_XFORMERS='1',SANA_WM_STREAMING_LAZY_VAE_DECODER='1',SANA_WM_STREAMING_SEQUENTIAL_OFFLOAD='1',SANA_WM_LOCAL_GEMMA2=str(P/'weights/gemma2'))
from common import PoseWorker,rejoin

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--batch',type=Path,required=True);ap.add_argument('--scene',required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--arm',choices=['native','baseline','sham','correction'],required=True);ap.add_argument('--frames',type=int,default=49);ap.add_argument('--seed',type=int,default=7);ap.add_argument('--scale_mode',choices=['anchor','legacy'],default='anchor');a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 import torch
 import inference_video_scripts.wm.inference_sana_wm_streaming as entry
 import inference_video_scripts.wm.inference_sana_wm as lib
 import inference_video_scripts.wm.streaming_pipeline as stream
 row=next(s for s in json.load(open(a.batch/'index.json'))['scenes'] if s['name']==a.scene)
 with np.load(a.batch/'trajectories'/f'{a.scene}.npz') as z:
  plan=np.linalg.inv(z['w2c_overrun'].astype(np.float64))[:a.frames];K=z['intrinsics'][0].astype(np.float32);depth=z['anchor_depth'].astype(np.float32)
 assert len(plan)==a.frames
 # Released training multiplies pose translations by metric-scale sidecars.
 # prepare_camera only changes the origin; it does NOT undo our anchor gauge.
 sys.path.insert(0,str(R.parent/'continuous_eval_20261029'))
 from anchor_scale import fit_scale,freeze_scale
 mpu=float(row['metres_per_anchor_unit'])
 if not np.isfinite(mpu) or mpu<=0:raise RuntimeError('Missing valid metric anchor conversion')
 scale,scale_receipt=fit_scale(depth,depth*mpu)
 scale,scale_receipt=freeze_scale(scale,scale_receipt,'sana',a.scene)
 scale_receipt['calibrated_native_units_per_reference_unit']=scale
 if a.scale_mode=='legacy':scale=1.0
 scale_receipt.update(applied_native_units_per_reference_unit=scale,scale_mode=a.scale_mode)
 plan[:,:3,3]*=scale;depth*=scale
 scale_receipt.update(model='sana',units='estimated metres',native_contract='sana_wm_zip_latent_data.py:355 metric sidecar; prepare_camera preserves translation magnitude',observer_pose_scale=scale)
 (a.out/'scale_calibration.json').write_text(json.dumps(scale_receipt,indent=2))
 # Use the released preprocessing exactly: native output is 704x1280.
 image=Image.open(a.batch/row['image']).convert('RGB')
 image,src_size,resized_size,crop=lib.resize_and_center_crop(image)
 image.save(a.out/'image.png')
 rw,rh=resized_size;left,top=crop;sw,sh=src_size
 depth=np.asarray(Image.fromarray(depth).resize((rw,rh),Image.Resampling.BILINEAR))[top:top+lib.TARGET_HEIGHT,left:left+lib.TARGET_WIDTH]
 K[0]*=rw/sw;K[1]*=rh/sh;K[0,2]-=left;K[1,2]-=top
 intr=np.repeat(np.array([[K[0,0],K[1,1],K[0,2],K[1,2]]],np.float32),a.frames,axis=0);np.save(a.out/'poses.npy',plan);np.save(a.out/'K.npy',intr);(a.out/'prompt.txt').write_text(row['caption'])
 state={};commands=plan.copy();prefix=[np.asarray(image)];events=[];worker=None
 def setup(pipe,ray,plucker,latent_T):
  state.update(pipe=pipe,ray=ray,plucker=plucker,latent_T=latent_T)
  # Keep FFN weights on CPU between their own forwards. This preserves dtype,
  # attention windows and temporal chunks, while freeing room for the native KV cache.
  import diffusion.refiner.diffusers_ltx2_refiner as refmod
  from sana_kv_offload import install
  install(refmod)
  transformer=pipe.refiner.transformer
  def move_video(device):
   for name in ('proj_in','caption_projection','time_embed','rope','norm_out','proj_out'):
    child=getattr(transformer,name,None)
    if isinstance(child,torch.nn.Module):child.to(device)
   refmod._move_tensor_attr(transformer,'scale_shift_table',device)
   for block in transformer.transformer_blocks:
    refmod._move_tensor_attr(block,'scale_shift_table',device)
    for name in ('norm1','attn1','norm2','attn2','norm3'):
     child=getattr(block,name,None)
     if isinstance(child,torch.nn.Module):child.to(device)
    block.ff.to('cpu')
   torch.cuda.empty_cache()
  pipe.refiner.move_video_modules=move_video
  for block in transformer.transformer_blocks:
   ff=block.ff;old=ff.forward;ff.to('cpu')
   def chunked(x,*args,_old=old,_ff=ff,**kw):
    _ff.to(x.device)
    try:
     if x.shape[-2]<=512:return _old(x,*args,**kw)
     return torch.cat([_old(part,*args,**kw) for part in x.split(512,dim=-2)],dim=-2)
    finally:_ff.to('cpu');torch.cuda.empty_cache()
   ff.forward=chunked
 def callback(pixel_np,frame_base,chunk):
  nonlocal worker
  assert pixel_np.ndim==4 and tuple(pixel_np.shape[1:])==prefix[0].shape,(pixel_np.shape,prefix[0].shape)
  prefix.extend(pixel_np);lo=len(prefix)-1
  np.savez_compressed(a.out/'frames.npz',frames=np.asarray(prefix,np.uint8))
  if a.arm=='native' or lo>=a.frames-1:return
  pipe=state['pipe']
  from camera_loop.resource_residency import observer_residency
  with observer_residency(pipe.model):
   # Free idle stage-1 host weights before bringing decoder weights back to CPU.
   pipe._move_vae_decoder_for_streaming('cpu');torch.cuda.empty_cache()
   try:
    if worker is None:worker=PoseWorker(a.out/'observer')
    with torch.random.fork_rng(devices=[0]):measured,info=worker.measure(np.asarray(prefix),K,depth)
   finally:
    # Close before generator weights return to CPU, including on failure.
    if worker is not None:worker.close();worker=None
  stride=int(pipe.config.vae.vae_stride[0]);latent_lo=(lo//stride)+1;latent_hi=min(latent_lo+int(pipe.refiner_settings.block_size),state['latent_T']);hi=(latent_hi-1)*stride+1
  assert lo%stride==0
  if a.arm in ['sham','correction']:
   start=measured if a.arm=='correction' else plan[lo]
   if a.arm=='correction':
    desired=rejoin(start,plan[lo],plan[lo+1:hi]);commands[lo+1:hi]=(commands[lo]@np.linalg.inv(start))@desired
   fresh=lib.prepare_camera(commands,intr,target_size=(lib.TARGET_HEIGHT,lib.TARGET_WIDTH),vae_stride=pipe.config.vae.vae_stride)
   ray=fresh['raymap'].unsqueeze(0).to(state['ray']);pl=fresh['chunk_plucker'].unsqueeze(0).to(state['plucker'])
   if state['ray'].shape[0]>1:ray=ray.repeat(state['ray'].shape[0],1,1);pl=pl.repeat(state['plucker'].shape[0],1,1,1,1)
   if a.arm=='sham':
    assert torch.equal(ray[:,latent_lo:latent_hi],state['ray'][:,latent_lo:latent_hi]);assert torch.equal(pl[:,:,latent_lo:latent_hi],state['plucker'][:,:,latent_lo:latent_hi])
   state['ray'][:,latent_lo:latent_hi].copy_(ray[:,latent_lo:latent_hi]);state['plucker'][:,:,latent_lo:latent_hi].copy_(pl[:,:,latent_lo:latent_hi])
  events.append({'chunk':chunk,'observed_frame':lo,'next_latent_range':[latent_lo,latent_hi],'measured':measured.tolist(),'measurement':info,'correction_applied':a.arm=='correction','prefix_sha256':hashlib.sha256(np.asarray(prefix).tobytes()).hexdigest()});(a.out/'events.json').write_text(json.dumps(events,indent=2))
 # Capture live camera tensors, without changing model kwargs or their original preparation.
 src=textwrap.dedent(inspect.getsource(lib.SanaWMPipeline.generate_streaming));needle='    latent_channels = first_latent.shape[1]';assert src.count(needle)==1;src=src.replace(needle,'    _setup(self, raymap_cfg, chunk_plucker_cfg, latent_T)\n'+needle)
 scope=dict(lib.__dict__,_setup=setup);exec(compile(src,str(R/'sana_generate_hook.py'),'exec'),scope);lib.SanaWMPipeline.generate_streaming=scope['generate_streaming']
 # Reuse published sequential stage computations. Only module placement/order changes.
 src=textwrap.dedent(inspect.getsource(stream._run_streaming_inference_sequential))
 if a.arm=='native':
  src=src.replace('        for _ in range(n_stage1_chunks):',"        refiner_runner._refiner.move_video_modules('cpu')\n        torch.cuda.empty_cache()\n        for _ in range(n_stage1_chunks):")
  src=src.replace('        refiner_t0 = time.perf_counter()',"        refiner_runner._refiner.move_video_modules(device)\n        refiner_t0 = time.perf_counter()")
  src=src.replace('        decoder = getattr(vae_streaming_decoder.vae, "decoder", None)',"        refiner_runner._refiner.move_video_modules('cpu')\n        torch.cuda.empty_cache()\n        decoder = getattr(vae_streaming_decoder.vae, \"decoder\", None)")
 else:
  # Retain exact published refiner and decoder loop bodies, interleaved one block at a time.
  rstart=src.index('        for k_ref in range(n_refiner):');rend=src.index('        if device.type == "cuda":',rstart)
  dstart=src.index('        for k_dec in range(n_decode):');dend=src.index('        if device.type == "cuda":',dstart)
  refbody=src[rstart:rend].split('\n',1)[1];decbody=src[dstart:dend].split('\n',1)[1]
  start=src.index('        for _ in range(n_stage1_chunks):');end=src.index('\n        if config.progress_callback is not None:',src.index('        decode_seconds =',dend))
  replacement='''        assert n_stage1_chunks == n_refiner, 'Native stage-1 and refiner chunk mapping requires audit'
        assert tuple(config.stage1_chunk_ends) == tuple(sink_size+(i+1)*block_size for i in range(n_refiner))
        decoder = getattr(vae_streaming_decoder.vae, 'decoder', vae_streaming_decoder.vae)
        vae_streaming_decoder.reset()
        stage1_seconds=refiner_seconds=decode_seconds=0.0
        for k_ref in range(n_refiner):
            refiner_runner._refiner.move_video_modules('cpu');decoder.to('cpu');torch.cuda.empty_cache()
            _state['pipe'].model.to(device)
            t0=time.perf_counter()
            chunk_idx,latent_view,start_f,end_f=next(stage1_chunk_iter)
            assert end_f==sink_size+(k_ref+1)*block_size
            latents_full[:,:,start_f:end_f].copy_(latent_view)
            torch.cuda.synchronize();stage1_seconds+=time.perf_counter()-t0
            _state['pipe'].model.to('cpu');torch.cuda.empty_cache()
            refiner_runner._refiner.move_video_modules(device)
            t0=time.perf_counter()
'''+refbody+'''            torch.cuda.synchronize();refiner_seconds+=time.perf_counter()-t0
            refiner_runner._refiner.move_video_modules('cpu');torch.cuda.empty_cache()
            decoder.to(device);k_dec=k_ref;t0=time.perf_counter()
'''+decbody+'''            torch.cuda.synchronize();decode_seconds+=time.perf_counter()-t0
'''
  src=src[:start]+replacement+src[end:]
 scope=dict(stream.__dict__,_state=state);exec(compile(src,str(R/'sana_stream_hook.py'),'exec'),scope);stream._run_streaming_inference_sequential=scope['_run_streaming_inference_sequential'];(a.out/'stream_hook_source.py').write_text(src)
 original=lib.SanaWMPipeline.generate_streaming
 def generate(self,*args,**kw):kw['decoded_chunk_callback']=callback;return original(self,*args,**kw)
 lib.SanaWMPipeline.generate_streaming=generate
 sys.argv=['sana','--config',str(REPO/'configs/sana_wm/sana_wm_streaming_1600m_720p.yaml'),'--streaming_root',str(P/'weights/sana_streaming'),'--image',str(a.out/'image.png'),'--prompt',str(a.out/'prompt.txt'),'--camera',str(a.out/'poses.npy'),'--intrinsics',str(a.out/'K.npy'),'--output_dir',str(a.out),'--num_frames',str(a.frames),'--seed',str(a.seed),'--no_compile','--offload_vae','--offload_refiner','--offload_text_encoder']
 t=time.time()
 try:entry.main()
 finally:
  if worker is not None:worker.close()
 assert len(prefix)==a.frames,len(prefix)
 (a.out/'run.json').write_text(json.dumps({'model':'sana','arm':a.arm,'scene':a.scene,'seed':a.seed,'seconds':time.time()-t,'events':events,'plan':plan.tolist(),'commands':commands.tolist(),'K':K.tolist(),'deviations':['Serialized native temporal chunks with whole-module CPU offload; same weights and precision. Refiner FFN weights offloaded between forwards and token-independent FFN evaluated in groups of 512 to fit 32 GiB.','Refiner sink/history/captured KV stored on CPU in original dtype, transferred per attention layer; no context truncation.', 'Input anchor prepended to emitted frames (native drops its sink frame).']},indent=2))
if __name__=='__main__':main()
