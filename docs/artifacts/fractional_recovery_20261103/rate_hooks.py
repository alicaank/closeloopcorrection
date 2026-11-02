"""Frozen experiment hooks: identical measured memory, optional future rate control."""
import hashlib,json,os,random,sys
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(R))
import components as c
from runtime import Observer
from memory_contract import stage_memory_write
from camera_loop.fractional_recovery import fractional_memory_continuation
ARM=os.environ.get('MEMORY_RATE_ARM');STATE={};TARGET=None

def outdir():return Path(os.environ['MEMORY_RATE_OUT'])
def event(row):
    p=outdir()/'rate_events.jsonl';p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f:f.write(json.dumps(dict(experiment_arm=ARM,**row),default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')

def observation(frames,K,depth,ids):
    shared=outdir().parent/'common_prefix';shared.mkdir(exist_ok=True)
    key=hashlib.sha256(''.join(c.digest(x) for x in [frames,K,depth,np.asarray(ids)]).encode()).hexdigest()
    p=shared/'observation.json'
    if p.exists():
        row=json.loads(p.read_text());assert row['key']==key,'Shared-prefix observation input mismatch'
    else:
        assert ARM=='M','Memory baseline must establish the common observation'
        worker=Observer(shared/'observer')
        try:poses,q=worker.measure(frames,K,depth,ids)
        finally:worker.close()
        row=dict(key=key,poses=poses.tolist(),quality=q,ids=np.asarray(ids).tolist())
        p.write_text(json.dumps(row,indent=2)+'\n')
    STATE['measurement_key']=key
    return np.asarray(row['poses']),row['quality']

class CachedObserver:
    def measure(self,frames,K,depth,ids):
        with c.preserve_rng():return observation(frames,K,depth,ids)
    def close(self):pass

def shared_measure(frames,K,depth,ids,old):
    poses,q=observation(frames,K,depth,ids)
    return stage_memory_write(np.asarray(ids),old,poses,available_last_frame=len(frames)-1,quality=q),q,STATE['measurement_key']

def capture_event(out,row):
    from runtime import append
    append(out,row)
    if row.get('model') in ['sana','lyra']:
        STATE['write']=row
        event(dict(kind='memory_write',record=row,observation_key=STATE.get('measurement_key')))

def install(model):
    import memory_hooks as mm
    assert ARM in ['M','F']
    mm.append=capture_event
    if model=='sana':
        c.setup(model,'M',outdir());c.measure=shared_measure;c.sana_install(mm)
        original_capture=mm.sana_capture
        def capture(L):
            original_capture(L)
            extra=L['self']._extra_model_kwargs;start,end=L['start_f'],L['end_f']
            ray=extra['camera_conditions'][:,start:end]
            event(dict(kind='future_native_camera',frame_ids=(8*np.arange(start,end)),
                       camera_conditions=ray.detach().float().cpu().numpy(),
                       actual_sha256=c.thash(ray)))
        mm.sana_capture=capture
        original_write=mm.sana_write
        def write(L):
            global TARGET
            TARGET=L['commands'].copy() if TARGET is None else TARGET
            result=original_write(L)
            row=STATE['write'];boundary=len(L['prefix'])-1
            assert boundary==24,'This protocol has exactly one 24-frame SANA boundary'
            if row['applied']:
                measured=np.asarray(row['stored']);ids=np.asarray(row['frame_ids'])
                r=fractional_memory_continuation(TARGET,measured,ids,native_stride=8,recovery_fraction=.5,speed_multiplier=2.)
                event(dict(kind='rate_candidate',receipt=r.receipt,measured_boundary=measured[-1],memory_labels_sha256=c.digest(measured)))
                if ARM=='F' and r.receipt['applied']:
                    L['commands'][boundary+1:]=r.commands[boundary+1:]
                    fresh=L['lib'].prepare_camera(L['commands'],L['intr'],target_size=(L['lib'].TARGET_HEIGHT,L['lib'].TARGET_WIDTH),vae_stride=L['pipe'].config.vae.vae_stride)
                    state=L['state'];ray=fresh['raymap'].unsqueeze(0).to(state['ray']);pl=fresh['chunk_plucker'].unsqueeze(0).to(state['plucker'])
                    if state['ray'].shape[0]>1:ray=ray.repeat(state['ray'].shape[0],1,1);pl=pl.repeat(state['plucker'].shape[0],1,1,1,1)
                    start=boundary//8+1
                    state['ray'][:,start:].copy_(ray[:,start:]);state['plucker'][:,:,start:].copy_(pl[:,:,start:])
                np.savez_compressed(outdir()/'issued_commands.npz',target=TARGET,commands=L['commands'],measured=measured,frame_ids=ids)
            event(dict(kind='future_policy',boundary=boundary,applied=bool(ARM=='F' and row['applied'] and r.receipt['applied']),memory_accepted=row['applied']))
            return result
        mm.sana_write=write
    elif model=='lyra_child':
        mm.observer=lambda out:CachedObserver()
        original=mm.lyra_memory
        def memory(pipe,end):
            original(pipe,end)
            row=STATE.get('write');assert row is not None
            if row['applied']:
                ids=np.asarray(row['frame_ids']);poses=np.linalg.inv(pipe.cam_w2c[0,ids].detach().cpu().double().numpy())
                STATE.update(committed=poses,ids=ids)
                event(dict(kind='committed_memory',native_history_sha256=c.thash(pipe.cam_w2c),poses=poses,frame_ids=ids))
        mm.lyra_memory=memory

def lyra_future(pipe,cam):
    global TARGET
    import torch
    history=pipe.history_frames[:,:,pipe.start_index:];boundary=int(history.shape[2])-1
    original=cam.clone();array=cam[0] if cam.ndim==4 else cam
    future=np.linalg.inv(array.detach().cpu().double().numpy())
    if boundary==0:
        anchor=np.linalg.inv(pipe.cam_w2c[0,:1].detach().cpu().double().numpy())
        TARGET=np.concatenate([anchor,future]);updated=cam
    else:
        assert boundary==80 and len(TARGET)==81 and len(future)==80,'Lyra single-boundary schedule changed'
        TARGET=np.concatenate([TARGET,future]);updated=cam
        row=STATE.get('write');assert row is not None
        if row['applied']:
            measured=STATE['committed'];ids=STATE['ids']
            r=fractional_memory_continuation(TARGET,measured,ids,native_stride=4,recovery_fraction=.5,speed_multiplier=2.)
            event(dict(kind='rate_candidate',receipt=r.receipt,measured_boundary=measured[-1],memory_labels_sha256=c.digest(measured)))
            if ARM=='F' and r.receipt['applied']:
                updated=cam.clone();u=updated[0] if updated.ndim==4 else updated
                # Preserve the supplied native w2c rotation bits exactly.
                t=-np.einsum('nij,nj->ni',array[:,:3,:3].detach().cpu().double().numpy(),r.commands[boundary+1:,:3,3])
                u[:,:3,3]=torch.as_tensor(t,dtype=u.dtype,device=u.device)
            commands=TARGET.copy()
            if ARM=='F' and r.receipt['applied']:commands[boundary+1:]=r.commands[boundary+1:]
            np.savez_compressed(outdir()/'issued_commands.npz',target=TARGET,commands=commands,measured=measured,frame_ids=ids)
        event(dict(kind='future_policy',boundary=boundary,applied=bool(ARM=='F' and row['applied'] and r.receipt['applied']),memory_accepted=row['applied']))
    assert torch.equal(updated[...,:3,:3],original[...,:3,:3]),'Future rotations changed'
    event(dict(kind='future_input',boundary=boundary,original_sha256=c.thash(original),issued_sha256=c.thash(updated),rotation_exact=True))
    return updated

def patch_source(model,source):
    if model=='lyra_child':
        needle='    result=_orig_step(self,cam_w2c_chunk=cam,intrinsics_chunk=intrinsics_chunk,**kw)'
        assert source.count(needle)==1
        source=source.replace(needle,"    cam=__import__('rate_hooks').lyra_future(self,cam)\n"+needle)
    return source
