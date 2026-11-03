"""First-boundary shared-normalization diagnostic; not a deployed controller."""
import hashlib,json,os,pickle,random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from einops import rearrange
from camera_loop.conditioning_gauge import shared_translation_gauge
from runtime import append,digest,stage_memory_write


def tree_digest(value):
    if torch.is_tensor(value):return dict(shape=list(value.shape),dtype=str(value.dtype),sha256=digest(value))
    if isinstance(value,dict):return {str(k):tree_digest(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [tree_digest(v) for v in value]
    return value


def rng_digest(generator):
    h=hashlib.sha256(pickle.dumps((random.getstate(),np.random.get_state())))
    for t in [torch.get_rng_state(),*torch.cuda.get_rng_state_all(),generator.get_state()]:h.update(t.cpu().numpy().tobytes())
    return h.hexdigest()


def encode(relative,Ks,L):
    from wan.utils.cam_utils import get_plucker_embeddings
    pipe=L['self']
    with torch.autocast(device_type='cuda',enabled=False):
        p=get_plucker_embeddings(relative.to(pipe.device),Ks,L['h'],L['w'])
    return rearrange(p,'f (h c1) (w c2) c -> 1 (c c1 c2) f h w',
        c1=L['h']//L['lat_h'],c2=L['w']//L['lat_w']).to(pipe.param_dtype)


def select_relative(plan,measured,arm,accepted,original_norm):
    """CPU-testable arm selection; preserve the native float32 no-op path."""
    if arm not in ('A','I','G','M'):raise ValueError('Unknown diagnostic arm')
    raw=torch.as_tensor(plan,dtype=torch.float32)
    if arm in ('A','I') or not accepted:
        result=raw.clone()
        if original_norm>0:result[:,:3,3]/=original_norm
        return result,dict(denominator=original_norm,gauge_applied=False,memory_write_applied=False)
    candidate=shared_translation_gauge(raw.numpy(),np.asarray(measured))
    chosen=candidate.commanded_control if arm=='G' else candidate.measured_history
    return torch.tensor(chosen.copy(),dtype=torch.float32),dict(candidate.receipt,
        gauge_applied=candidate.receipt['changed_from_plan_normalizer'],memory_write_applied=arm=='M')


def before_clean(L):
    from wan.utils.cam_utils import compute_relative_poses
    import memory_hooks as mm
    pipe=L['self'];obs=pipe._loop_observer;ctx=obs.memory_context
    arm=os.environ['CAMERA_LINGBOT_GAUGE_ARM'];out=Path(ctx['out'])
    if L['chunk_id']!=0:return
    if arm not in ('A','I','G','M'):raise ValueError('Unknown diagnostic arm')
    end=L['chunk_size'];assert end==4 and L['num_inference_chunk']==5
    assert len(L['pred_latent_chunks'])==1
    ids=4*np.arange(end);old=obs.commands[:end].copy();measured=old;q={'passed':False}
    raw=compute_relative_poses(torch.tensor(obs.commands,dtype=torch.float32),framewise=True,normalize_trans=False)
    original_norm=obs.norm;assert original_norm==float(raw[:,:3,3].norm(dim=-1).max())
    receipt=dict(arm=arm,frame_ids=ids.tolist(),first_affected_rgb=13,
        original_norm=original_norm,first_generated_latent_sha256=digest(L['x0']),
        initial_noise_sha256=digest(L['noise']),rng_before=rng_digest(L['seed_g']))
    if arm!='A':
        py_state=random.getstate();np_state=np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[0]):
                pipe.model.cpu();torch.cuda.empty_cache()
                try:
                    decoded=pipe.vae.decode([torch.cat(L['pred_latent_chunks'],dim=1)])[0]
                    frames=mm.pixels(decoded,'cthw');del decoded;torch.cuda.empty_cache()
                    assert len(frames)==13
                    receipt['observed_rgb_sha256']=hashlib.sha256(frames.tobytes()).hexdigest()
                    K=ctx['K'].copy();K[0]*=frames.shape[2]/832;K[1]*=frames.shape[1]/480
                    depth=np.asarray(Image.fromarray(ctx['depth']).resize((frames.shape[2],frames.shape[1]),Image.Resampling.BILINEAR))
                    measured,q=mm.observer(out).measure(frames,K,depth,ids)
                    del frames
                finally:
                    mm.close_observer(out);pipe.model.to(pipe.device)
        finally:random.setstate(py_state);np.random.set_state(np_state)
    write=stage_memory_write(ids,old,measured,available_last_frame=12,quality=q)
    accepted=write.applied
    history=compute_relative_poses(torch.tensor(write.poses,dtype=torch.float32),framewise=True,normalize_trans=False)
    rel,selection=select_relative(raw.numpy(),history.numpy(),arm,accepted,original_norm)
    if arm!='A':
        encoded=encode(rel,L['Ks'],L)
        if arm=='I' or not accepted:
            assert torch.equal(encoded,L['c2ws_plucker_emb']), 'No-op native camera tensors differ'
        # At the first clean pass there is no earlier chunk to rebuild. The
        # released clean forward replaces this chunk's camera-dependent KV.
        L['kwargs']['dit_cond_dict']={'c2ws_plucker_emb':encoded[:,:,:end].chunk(1,dim=0)}
        for j,chunk in enumerate(L['c2ws_plucker_emb_chunk']):
            chunk.copy_(encoded[:,:,j*end:(j+1)*end])
        obs.norm=selection['denominator']
    receipt.update(selection,observation_accepted=bool(accepted),quality=q,
        measured_poses=np.asarray(measured).tolist(),raw_measured_relative=history.tolist(),
        future_camera_sha256=digest(L['c2ws_plucker_emb'][:,:,end:]),
        clean_camera_sha256=digest(L['kwargs']['dit_cond_dict']['c2ws_plucker_emb'][0]),
        rng_after=rng_digest(L['seed_g']))
    assert receipt['rng_before']==receipt['rng_after'],'Observation changed RNG'
    (out/'gauge_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    append(out,dict(model='lingbot',frame_ids=ids,applied=bool(accepted and arm=='M'),
        reason='first_boundary_diagnostic',stored=write.poses if arm=='M' else old,quality=q,
        gauge_applied=selection['gauge_applied'],diagnostic_arm=arm,
        mechanism='One first-boundary shared normalization; all later camera tensors held fixed'))


def after_clean(L):
    if L['chunk_id'] not in (0,L['num_inference_chunk']-1):return
    obs=L['self']._loop_observer;out=Path(obs.memory_context['out'])
    record=dict(chunk=int(L['chunk_id']),self_cache=tree_digest(L['self_kv_cache']),
        cross_cache=tree_digest(L['cross_kv_cache']),rng=rng_digest(L['seed_g']),
        native_generator_sha256=digest(L['seed_g'].get_state()))
    (out/f"cache_{L['chunk_id']:02d}.json").write_text(json.dumps(record,indent=2)+'\n')
