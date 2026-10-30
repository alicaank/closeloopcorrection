"""Training-free GaussVerse interventions plugged into released Lyra-2 inference (lyra2 env).

Runs lyra2_custom_traj_inference unchanged except for monkeypatches selected by the GV_LYRA_ARM json env var:
  write : 'commanded' (stock) | 'predicted' (Lyra's own --da3_use_predicted_pose, continuation on) | 'gated'
          (predicted poses, but new-frame cache writes whose measured-vs-commanded position residual exceeds
          gate_tau anchor-depth units are skipped: F' uncertainty-gated write)
  replan: bool. F' replanning: chunk k+1 commands start from the MEASURED last pose of chunk k and rejoin the
          planned path linearly over replan_frac of the chunk (SE(3) interpolation of the world-frame correction).
          Requires write != 'commanded' (needs a measured pose).
  slots : 'full' | 'drop' (retrieval returns nothing -> slots padded with the anchor frame, Lyra's no-memory
          condition) | 'appearance' (retrieved frames and their source-camera Plucker kept, but the per-slot
          correspondence/depth channels on the target tail are replaced by the absent-slot pad: no geometric address)
  depth_slot_scale   : multiply cached depth used for slot correspondence warps (retrieval selection untouched)
  depth_buffer_scale : multiply the last-frame buffer depth used for the main warp (render path)
Everything is logged to <output_path>/gv_arm.json plus per-chunk replan/gate records in gv_events.jsonl.
"""
import json, os, sys, runpy, math
import numpy as np, torch
from scipy.spatial.transform import Rotation, Slerp

from lyra_2._src.inference import lyra2_ar_inference as ARI
from lyra_2._src.models import lyra2_model as LM
from lyra_2._ext.imaginaire.utils import log

ARM = dict(name='baseline', write='commanded', replan=False, replan_frac=1.0, slots='full',
           depth_slot_scale=1.0, depth_buffer_scale=1.0, gate_tau=0.15)
ARM.update(json.loads(os.environ.get('GV_LYRA_ARM', '{}')))
assert ARM['write'] in ('commanded', 'predicted', 'gated') and ARM['slots'] in ('full', 'drop', 'appearance')
assert ARM['write']=='commanded', 'This comparison must preserve native memory labels'
EVENTS = []


def _event(**kw):
    EVENTS.append(kw)
    log.info('[gv] ' + json.dumps(kw, default=float)[:400], rank0_only=True)


def _inv(m):
    return torch.linalg.inv(m.double()).to(m.dtype)


def se3_interp(delta_c2w, w):
    """Fraction w in [0,1] of a rigid world-frame correction (w=1 full correction, 0 identity)."""
    d = delta_c2w.detach().cpu().double().numpy()
    rot = Rotation.from_matrix(d[:3, :3])
    r = Slerp([0., 1.], Rotation.concatenate([Rotation.identity(), rot]))(w).as_matrix()
    out = np.eye(4); out[:3, :3] = r; out[:3, 3] = w * d[:3, 3]
    return torch.from_numpy(out).to(delta_c2w.dtype)


# ---------------------------------------------------------------- F' : replanning + write policy
P = ARI.Lyra2InferencePipeline
_orig_step = P.autoregressive_step
_orig_update = P._update_depth_cache
_orig_add = LM.Sparse3DCache.add
_orig_upd = LM.Sparse3DCache.update_by_frame_id
_orig_retrieve = LM.Sparse3DCache.retrieve
_orig_rgbd = LM.Sparse3DCache.get_rgbd_by_frame_id
_orig_controls = LM.Lyra2Model._apply_camera_controls


def gv_step(self, *, cam_w2c_chunk, intrinsics_chunk, **kw):
    from pathlib import Path
    import hashlib
    sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'new_models_loop_20261028'))
    from common import PoseWorker,rejoin
    from PIL import Image
    meta=json.load(open(os.environ['LONG_LYRA_META']))
    cam=cam_w2c_chunk
    # Native history starts with repeated anchor padding; it is not generated time.
    history=self.history_frames[:, :, self.start_index:]
    lo=int(history.shape[2])-1
    assert lo == int(self.cam_w2c.shape[1])-1, 'Generated history/camera timeline mismatch'
    if lo>0 and ARM['name']!='native':
        frames=((history[0].detach().cpu().permute(1,2,3,0).float()*.5+.5).clamp(0,1)*255).round().byte().numpy()
        h,w=frames.shape[1:3];mw=504;mh=round(h*mw/w)
        small=np.stack([np.asarray(Image.fromarray(f).resize((mw,mh),Image.Resampling.BICUBIC)) for f in frames])
        with np.load(meta['trajectory']) as z:
            plan=np.linalg.inv(z['w2c'].astype(np.float64));K=z['intrinsics'][0].astype(np.float64);depth=z['anchor_depth'].astype(np.float32)
        plan[:,:3,3]*=meta['pose_scale'];depth*=meta['pose_scale']
        K[0]*=mw/864;K[1]*=mh/480
        depth=np.asarray(Image.fromarray(depth).resize((mw,mh),Image.Resampling.BILINEAR))
        # Never change native history labels; external measurement is a read-only observer.
        if not hasattr(self,'_long_observer'):self._long_observer=PoseWorker(Path(meta['out'])/'observer')
        da3dev=next(self.local_da3_model.parameters()).device
        self.local_da3_model.to('cpu');ARI._offload_diffusion_to_cpu(self.model,True);torch.cuda.empty_cache()
        try:
            with torch.random.fork_rng(devices=[0]):measured,info=self._long_observer.measure(small,K,depth)
        finally:
            self.local_da3_model.to(da3dev);ARI._restore_diffusion_to_gpu(self.model,True)
        n=int(cam.shape[-3]);future=plan[lo+1:lo+n+1];assert len(future)==n
        if ARM['name']=='correction':
            previous=np.linalg.inv(self.cam_w2c[0,-1].detach().cpu().double().numpy())
            desired=rejoin(measured,plan[lo],future)
            commands=(previous@np.linalg.inv(measured))@desired
            upd=torch.as_tensor(np.linalg.inv(commands),dtype=cam.dtype,device=cam.device)
            cam=upd[None] if cam.dim()==4 else upd
        elif ARM['name']=='sham':
            assert np.array_equal(rejoin(plan[lo],plan[lo],future),future)
            assert torch.equal(cam,cam_w2c_chunk)
        _event(kind='external_feedback',frame=lo,applied=ARM['name']=='correction',measured=measured.tolist(),measurement=info,prefix_sha256=hashlib.sha256(frames.tobytes()).hexdigest())
    result=_orig_step(self,cam_w2c_chunk=cam,intrinsics_chunk=intrinsics_chunk,**kw)
    if kw.get('is_last_step') and hasattr(self,'_long_observer'):self._long_observer.close()
    return result

def gv_add(self, depth, w2c, K, latent_index, frame_id=None):
    if getattr(self, '_gv_defer', False):
        self._gv_pending.append(('add', int(frame_id if frame_id is not None else latent_index),
                                 dict(depth_B_1_H_W=depth, w2c_B_4_4=w2c, K_B_3_3=K, latent_index=latent_index, frame_id=frame_id)))
        return
    return _orig_add(self, depth, w2c, K, latent_index, frame_id)


def gv_upd(self, frame_id, depth_B_1_H_W, w2c_B_4_4, K_B_3_3):
    if getattr(self, '_gv_defer', False):
        self._gv_pending.append(('update', int(frame_id), dict(frame_id=frame_id, depth_B_1_H_W=depth_B_1_H_W, w2c_B_4_4=w2c_B_4_4, K_B_3_3=K_B_3_3)))
        return True
    return _orig_upd(self, frame_id, depth_B_1_H_W, w2c_B_4_4, K_B_3_3)


def gv_update(self, end_px_idx):
    before = self.cam_w2c.clone()
    cache = self.retrieval_cache
    gated = ARM['write'] == 'gated'
    if gated:
        cache._gv_defer, cache._gv_pending = True, []
    try:
        _orig_update(self, end_px_idx)
    finally:
        if gated:
            cache._gv_defer = False
    if gated:
        kept = skipped = 0; residuals = {}
        for kind, fid, kwargs in cache._gv_pending:
            if fid < before.shape[1]:
                r = float((_inv(before[0, fid])[:3, 3] - _inv(self.cam_w2c[0, fid].to(torch.float32))[:3, 3]).norm())
            else:
                r = 0.
            residuals[fid] = r
            if kind == 'update' or fid == 0 or r <= float(ARM['gate_tau']):
                (_orig_upd(cache, **kwargs) if kind == 'update' else _orig_add(cache, **kwargs)); kept += 1
            else:
                skipped += 1
        cache._gv_pending = []
        _event(kind='gate', ar_idx=int(self.ar_idx), kept=kept, skipped=skipped, tau=ARM['gate_tau'],
               residuals={str(k): round(v, 4) for k, v in residuals.items()})
    else:
        new = [f for f in range(1, min(before.shape[1], self.cam_w2c.shape[1])) if f >= end_px_idx - int(self.model.framepack_num_new_video_frames)]
        res = [float((_inv(before[0, f])[:3, 3] - _inv(self.cam_w2c[0, f].to(torch.float32))[:3, 3]).norm()) for f in new[::8]]
        _event(kind='measured_minus_commanded', ar_idx=int(self.ar_idx), position_residual_every8=[round(r, 4) for r in res])
    if float(ARM['depth_buffer_scale']) != 1.0 and self.buffer_depth_latest is not None:
        self.buffer_depth_latest = self.buffer_depth_latest * float(ARM['depth_buffer_scale'])


# ---------------------------------------------------------------- C1 : slot interventions
def gv_retrieve(self, *a, **kw):
    if ARM['slots'] == 'drop':
        _event(kind='retrieve', dropped=True)
        return []
    out = _orig_retrieve(self, *a, **kw)
    _event(kind='retrieve', frame_ids=[int(f) for _, f in out])
    return out


def gv_rgbd(self, frame_id):
    d, w, k = _orig_rgbd(self, frame_id)
    s = float(ARM['depth_slot_scale'])
    return (d * s if s != 1.0 else d), w, k


def gv_controls(self, cond_latent, selected_idx, video_indices, **kw):
    cond_latent, buffer_cond_latents = _orig_controls(self, cond_latent, selected_idx, video_indices, **kw)
    if ARM['slots'] == 'appearance' and buffer_cond_latents is not None:
        C = LM.LYRA2_CORRESPONDENCE_CHANNELS_PER_SLOT
        pad = torch.full_like(buffer_cond_latents, -1.0)
        for s in range(int(buffer_cond_latents.shape[1]) // C):
            pad[:, s * C + 3 * 64: (s + 1) * C] = 1.0                      # same values as the absent-slot pad
        buffer_cond_latents = pad
    return cond_latent, buffer_cond_latents


# ---------------------------------------------------------------- host-RAM: cached UMT5 prompt embedding
from lyra_2._src.inference import get_t5_emb as T5
_T5_CACHE = os.environ.get('GV_T5_CACHE', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prompt_umt5.pt'))


def gv_umt5(prompts, device='cuda', max_length=512):
    cache = torch.load(_T5_CACHE, map_location='cpu', weights_only=False)
    texts = [prompts] if isinstance(prompts, str) else list(prompts)
    if 'prompts' in cache:                                  # multi-caption cache (precompute_t5_captions.py)
        miss = [t for t in texts if t not in cache['prompts']]
        if miss:
            raise ValueError(f'caption not in UMT5 cache {_T5_CACHE}: {miss[0][:80]!r}')
        return torch.stack([cache['prompts'][t][:max_length] for t in texts]).to(device)
    if any(t != cache['prompt'] for t in texts):
        raise ValueError('cached UMT5 embedding exists only for the fixed GaussVerse prompt')
    emb = cache['embedding'][:max_length].to(device)
    return emb.unsqueeze(0).repeat(len(texts), 1, 1)


if os.path.exists(_T5_CACHE):
    T5.get_umt5_embedding = gv_umt5
    T5.get_umt5_embedding_offloaded = gv_umt5

P.autoregressive_step = gv_step
P._update_depth_cache = gv_update
LM.Sparse3DCache.add = gv_add
LM.Sparse3DCache.update_by_frame_id = gv_upd
LM.Sparse3DCache.retrieve = gv_retrieve
LM.Sparse3DCache.get_rgbd_by_frame_id = gv_rgbd
LM.Lyra2Model._apply_camera_controls = gv_controls

if __name__ == '__main__':
    argv = sys.argv[1:]
    if ARM['write'] in ('predicted', 'gated'):
        argv += ['--da3_use_predicted_pose', '--da3_predicted_pose_continuation']
    out = argv[argv.index('--output_path') + 1]
    os.makedirs(out, exist_ok=True)
    json.dump(dict(ARM, argv=argv), open(os.path.join(out, 'gv_arm.json'), 'w'), indent=1)
    sys.argv = ['lyra2_custom_traj_inference'] + argv
    if os.environ.get('GV_LYRA_BF16') == '1':                     # 10-26: under --sequential_cpu_offload the net stays fp32 in host RAM
        import torch, gc                                          # (~61 GB with the DMD LoRA -> host OOM on 62 GB); cast it to the
        import lyra_2._src.utils.model_loader as _ml              # model's own dtype right after loading, as the official non-offload
        _orig_load = _ml.load_model_from_checkpoint               # path does (lyra2_custom_traj_inference.py:308-312)
        def _load_cast(*a, **k):
            model, config = _orig_load(*a, **k)
            model.net = model.net.to(dtype=model.tensor_kwargs['dtype']); gc.collect()
            print(f"[gv] cast model.net to {model.tensor_kwargs['dtype']} on CPU (GV_LYRA_BF16=1)", flush=True)
            return model, config
        _ml.load_model_from_checkpoint = _load_cast
    try:
        # Hook the final native anchor depth, after MoGe alignment, before either
        # camera tensors or the initial reconstruction enter the pipeline.
        import importlib.util
        from pathlib import Path
        from anchor_scale import calibrate_lyra
        module = 'lyra_2._src.inference.lyra2_custom_traj_inference'
        source_path = importlib.util.find_spec(module).origin
        source = Path(source_path).read_text()
        marker = '        img_bchw = image_chw01.to(device=desired_device) * 2.0 - 1.0'
        assert source.count(marker) == 1, 'Native anchor hook moved; audit required'
        source = source.replace(marker, '        w2cs_T_44 = _calibrate_anchor(w2cs_T_44, depth_hw, mask_hw)\n' + marker)
        Path(out, 'native_scale_hook_source.py').write_text(source)
        exec(compile(source, source_path, 'exec'), dict(__name__='__main__', __file__=source_path,
             __package__='lyra_2._src.inference', _calibrate_anchor=calibrate_lyra))
    finally:
        with open(os.path.join(out, 'gv_events.jsonl'), 'a') as fh:
            for e in EVENTS:
                fh.write(json.dumps(e, default=float) + '\n')
