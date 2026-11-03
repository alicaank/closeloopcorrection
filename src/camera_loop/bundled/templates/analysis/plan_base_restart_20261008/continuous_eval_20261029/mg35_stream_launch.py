"""Matrix-Game runtime hooks: whole-module CPU offload, causal future-command feedback."""
import os,sys,runpy,json,inspect,textwrap,atexit,hashlib
from pathlib import Path
import numpy as np,torch
R=Path(__file__).resolve().parent;sys.path[:0]=[str(R),str(R.parent/'new_models_loop_20261028')];REPO=Path(os.environ['MG35_REPO']);sys.path.insert(0,str(REPO))
from common import rejoin,PoseWorker
from mg35_camera import native_w2c,corrected_commands
from diffsynth.diffusion.base_pipeline import BasePipeline
# Move whole modules without changing weights, arithmetic precision, or attention settings.
def staged(self,names):
 for name,model in self.named_children():
  if name not in names:model.to('cpu')
 torch.cuda.empty_cache()
 for name,model in self.named_children():
  if name in names:model.to(self.device)
BasePipeline.load_models_to_device=staged
import examples.wanvideo.pipeline.mosaic.validation as V
meta=json.load(open(os.environ['MG_LOOP_META']));out=Path(meta['out']);arm=meta['arm'];plan=np.asarray(meta['plan']);K=np.asarray(meta['K']);depth=np.load(meta['depth']);events=[];prefix=[];worker=None;all_commands=plan.copy()
V._annotate_frames_with_index=lambda frames,**kw:[np.asarray(f) for f in frames]
write=V._write_video
def save(path,frames,*a,**kw):
 if str(path).endswith('_history.mp4'):np.savez_compressed(out/'frames.npz',frames=np.asarray(frames,np.uint8))
 return write(path,frames,*a,**kw)
V._write_video=save

def capture(section,full,new):
 global prefix
 if section==0:prefix=[np.asarray(full[0])]
 prefix.extend(np.asarray(new));assert len(prefix)==1+84*(section+1),len(prefix)

def observe(section,pipe,camera):
 global worker
 if section==0 or arm=='native':return
 lo=84*section;assert len(prefix)==lo+1
 orig=camera['noisy_latent_indices_prope_extrinsic'];expected=native_w2c(plan[lo+1:lo+85])
 assert np.array_equal(orig.cpu().numpy(),expected),'Native pose frame mapping or serialization differs'
 clean=camera['clean_latent_indices_prope_extrinsic']
 assert np.array_equal(clean.cpu().numpy(),native_w2c(plan[lo-3:lo+1])),'Native clean-prefix mapping changed'
 if arm=='correction':
  # These are the poses actually used to generate the cached four clean RGB frames.
  camera['clean_latent_indices_prope_extrinsic']=torch.tensor(native_w2c(all_commands[lo-3:lo+1]),dtype=clean.dtype,device=clean.device)
 with torch.random.fork_rng(devices=[0]):
  pipe.load_models_to_device([])
  try:
   if worker is None:worker=PoseWorker(out/'observer')
   measured,info=worker.measure(np.asarray(prefix),K,depth)
  finally:
   if worker is not None:worker.close();worker=None
 commands=plan[lo+1:lo+85].copy()
 if arm in ['sham','correction']:
  start=measured if arm=='correction' else plan[lo]
  # Place the new motion in the cached commanded coordinate frame; no old pose changes.
  desired=rejoin(start,plan[lo],commands)
  if np.array_equal(start,plan[lo]) and np.array_equal(all_commands[lo],plan[lo]):commands=desired
  else:commands=(all_commands[lo]@np.linalg.inv(start))@desired
  all_commands[lo+1:lo+85]=commands
  updated=torch.tensor(native_w2c(commands),dtype=orig.dtype,device=orig.device)
  if arm=='sham':assert torch.equal(updated,orig),'Identity intervention changed native camera values'
  camera['noisy_latent_indices_prope_extrinsic']=updated
 events.append({'section':section,'observed_frame':lo,'measured':measured.tolist(),'commands':commands.tolist(),'measurement':info,'correction_applied':arm=='correction','prefix_sha256':hashlib.sha256(np.asarray(prefix).tobytes()).hexdigest()});(out/'events.json').write_text(json.dumps(events,indent=2))
src=textwrap.dedent(inspect.getsource(V._run_mosaic_validation_impl))
needle='            prope_camera_kwargs = _prope_camera_kwargs(camera_data)';assert src.count(needle)==1;src=src.replace(needle,'            _observe(section_idx, unwrapped_model.pipe, camera_data)\n'+needle)
needle='            section_visualization = decoded_full[clean_decode_drop_count:]';assert src.count(needle)==1;src=src.replace(needle,needle+'\n            _capture(section_idx, decoded_full, section_visualization)')
# Export the exact block RGB seen by the observer. Native final joint505-frame
# re-decode caused hostOOM after generation completed; it is presentation-only.
begin=src.index('        history_frames = _decode_latents_to_numpy_frames(')
end=src.index('        # Per-batch release of the big transient buffers built above.',begin)
replacement="""        handler = None
        _finish_stream(all_section_meta)
"""
src=src[:begin]+replacement+src[end:]
def finish_stream(section_meta):
 frames=np.asarray(prefix,np.uint8)
 assert len(frames)==len(plan),(len(frames),len(plan))
 np.savez_compressed(out/'frames.npz',frames=frames)
 (out/'stream_export.json').write_text(json.dumps({'frames':len(frames),'export':'already decoded native generation blocks; no whole-history VAE re-decode','sections':section_meta},indent=2,default=str))
scope=dict(V.__dict__,_observe=observe,_capture=capture,_finish_stream=finish_stream);exec(compile(src,str(R/'mg35_hook_generated.py'),'exec'),scope);V._run_mosaic_validation_impl=scope['_run_mosaic_validation_impl'];(out/'hook_source.py').write_text(src)
def close():
 if worker is not None:worker.close()
atexit.register(close)
sys.argv=[str(REPO/'examples/wanvideo/pipeline/run_pipeline.py')]+sys.argv[1:];runpy.run_path(sys.argv[0],run_name='__main__')
