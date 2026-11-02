"""Test-only factorization in the camera coordinates consumed by each model."""
import hashlib,json,os,random
from contextlib import contextmanager
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
ARM=None;OUT=None;MODEL=None
ARMS=('A','S','R','T','M','N')

def setup(model,arm,out):
    global ARM,OUT,MODEL
    assert arm in ARMS
    MODEL=model;ARM=arm;OUT=Path(out)
    os.environ['CAMERA_MEMORY_POLICY']='measured'

def digest(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str((a.shape,a.dtype)).encode()+memoryview(a).cast('B')).hexdigest()

def thash(t):
    import torch
    return digest(t.detach().contiguous().view(torch.uint8).cpu().numpy())

def append(record):
    OUT.mkdir(parents=True,exist_ok=True)
    record=dict(model=MODEL,arm=ARM,**record)
    with (OUT/'conditioning.jsonl').open('a') as f:
        f.write(json.dumps(record,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')

@contextmanager
def preserve_rng():
    import torch
    py=random.getstate();n=np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[0]):yield
    finally:random.setstate(py);np.random.set_state(n)

def measure(frames,K,depth,ids,old):
    from runtime import Observer
    from memory_contract import stage_memory_write
    shared=OUT.parent/'common_prefix';shared.mkdir(exist_ok=True)
    key=hashlib.sha256(''.join(digest(x) for x in [frames,K,depth,np.asarray(ids)]).encode()).hexdigest()
    p=shared/'observation.json'
    if p.exists():
        row=json.loads(p.read_text());assert row['key']==key,'Prefix RGB/K/depth/query mismatch'
        poses=np.asarray(row['poses']);quality=row['quality']
    else:
        assert ARM=='A','Baseline observation must exist before intervention'
        worker=Observer(shared/'observer')
        try:poses,quality=worker.measure(frames,K,depth,ids)
        finally:worker.close()
        row=dict(key=key,poses=poses,quality=quality,ids=np.asarray(ids),prefix_frames=len(frames))
        p.write_text(json.dumps(row,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item(),indent=2)+'\n')
    write=stage_memory_write(np.asarray(ids),old,poses,available_last_frame=len(frames)-1,quality=quality)
    return write,quality,key

def factor(old,new,arm):
    """old/new are c2w in the native relative/anchored representation, before rays.

    N is an artificial amplitude intervention, NOT a physical calibration.
    It bounds maximum camera-center magnitude to the commanded prefix envelope.
    """
    old=np.asarray(old);new=np.asarray(new);assert old.shape==new.shape
    limit=float(np.linalg.norm(old[...,:3,3],axis=-1).max())
    observed=float(np.linalg.norm(new[...,:3,3],axis=-1).max())
    alpha=min(1.,limit/observed) if observed>0 else 1.
    if arm in ['A','S']:out=old.copy()
    elif arm=='M':out=new.copy()
    elif arm=='R':out=old.copy();out[...,:3,:3]=new[...,:3,:3]
    elif arm=='T':out=old.copy();out[...,:3,3]=new[...,:3,3]
    elif arm=='N':out=new.copy();out[...,:3,3]*=alpha
    else:raise ValueError(arm)
    return out,dict(commanded_max_translation=limit,measured_max_translation=observed,
        measured_over_commanded=observed/limit if limit>0 else None,
        amplitude_multiplier=alpha if arm=='N' else 1.,
        output_max_translation=float(np.linalg.norm(out[...,:3,3],axis=-1).max()))

def record(old,new,out,stats,quality,key,**extra):
    append(dict(kind='memory_boundary',native_commanded_c2w=old,native_measured_c2w=new,
        native_written_c2w=out,range=stats,quality=quality,prefix_key=key,**extra))

def lingbot_future(L):
    append(dict(kind='future_input',chunk=int(L['chunk_id']),sha256=thash(L['current_c2ws_plucker_emb'])))

def lingbot_cache(L):
    import torch
    from PIL import Image
    from einops import rearrange
    from wan.utils.cam_utils import compute_relative_poses,get_plucker_embeddings
    from memory_hooks import pixels
    if L['chunk_id']+1>=L['num_inference_chunk']:return
    assert L['chunk_id']==0 and L['num_inference_chunk']==2
    pipe=L['self'];obs=pipe._loop_observer;ctx=obs.memory_context
    end=L['chunk_size'];ids=4*np.arange(end);old=obs.commands[:end].copy()
    with preserve_rng():
        pipe.model.cpu();torch.cuda.empty_cache()
        try:
            decoded=pipe.vae.decode([torch.cat(L['pred_latent_chunks'],dim=1)])[0]
            frames=pixels(decoded,'cthw');del decoded;torch.cuda.empty_cache()
            K=ctx['K'].copy();K[0]*=frames.shape[2]/832;K[1]*=frames.shape[1]/480
            depth=np.asarray(Image.fromarray(ctx['depth']).resize((frames.shape[2],frames.shape[1]),Image.Resampling.BILINEAR))
            write,q,key=measure(frames,K,depth,ids,old)
        finally:pipe.model.to(pipe.device)
    def relative(p):
        x=compute_relative_poses(torch.as_tensor(p,dtype=torch.float32),framewise=True,normalize_trans=False)
        x[:,:3,3]/=obs.norm
        return x.numpy()
    original=L['kwargs']['dit_cond_dict']['c2ws_plucker_emb'][0]
    a=relative(old);b=relative(write.poses);out,stats=factor(a,b,ARM)
    def encode(poses):
        with torch.autocast(device_type='cuda',enabled=False):
            pl=get_plucker_embeddings(torch.as_tensor(poses,device=pipe.device),L['Ks'][:end],L['h'],L['w'])
        return rearrange(pl,'f (h c1) (w c2) c -> 1 (c c1 c2) f h w',c1=L['h']//L['lat_h'],c2=L['w']//L['lat_w']).to(pipe.param_dtype)
    identity=encode(a);assert torch.equal(identity,original),'LingBot identity encoding mismatch'
    if ARM=='S' or (ARM not in ['A','S'] and write.applied):
        encoded=encode(out);L['kwargs']['dit_cond_dict']={'c2ws_plucker_emb':encoded.chunk(1,dim=0)}
    final=L['kwargs']['dit_cond_dict']['c2ws_plucker_emb'][0]
    record(a,b,out,stats,q,key,frame_ids=ids,write_accepted=write.applied,normalizer=obs.norm,
        original_sha256=thash(original),written_sha256=thash(final),identity_encoding_exact=True,
        representation='native framewise relative c2w; translation divided by fixed original plan normalizer')

def dreamx_future(L):
    append(dict(kind='future_input',chunk=int(L['current_start_frame'])//3,
        sha256=thash(L['y_camera_latents']['viewmats'])))

class DreamWriter:
    def __init__(self,args,commands,total,entry):
        self.args=args;self.commands=commands;self.total=total;self.entry=entry
        with np.load(args.batch/'trajectories'/f'{args.scene}.npz') as z:
            self.K=z['intrinsics'][0].astype(float);self.depth=z['anchor_depth'].astype(np.float32)
        self.ids=[0]+[1+4*i for i in range(total-1)]
    def close(self):pass
    def __call__(self,pipe,output,start,count,camera):
        import torch
        from PIL import Image
        end=start+count
        if end<self.args.prefix_latents or end>=self.total:return camera
        assert start==3 and end==6 and self.total==9
        ids=np.asarray(self.ids[start-1:end]);old=self.commands[ids].copy()
        latent_hash=thash(output[:,:end])
        with preserve_rng():
            pipe.generator.cpu();torch.cuda.empty_cache()
            try:
                decoded=pipe.vae.decode_to_pixel(output[:,:end]);pipe.vae.model.clear_cache()
                frames=((decoded[0].float()+1)*127.5).round().clamp(0,255).byte().permute(0,2,3,1).cpu().numpy()
                del decoded;torch.cuda.empty_cache()
                K=self.K.copy();K[0]*=frames.shape[2]/864;K[1]*=frames.shape[1]/480
                depth=np.asarray(Image.fromarray(self.depth).resize((frames.shape[2],frames.shape[1]),Image.Resampling.BILINEAR))
                write,q,key=measure(frames,K,depth,ids,old)
            finally:pipe.generator.to(output.device)
        def relative(p):
            rows=np.zeros((len(p),19));rows[:,1:5]=[969.6969696969696/1920,969.6969696969696/1080,.5,.5]
            rows[:,7:]=np.linalg.inv(p)[:,:3].reshape(len(p),12)
            return self.entry.get_relative_pose([self.entry.Camera(x.tolist()) for x in rows])[1:]
        def encode(p):
            vm=self.entry._invert_SE3(torch.as_tensor(p,device=output.device,dtype=camera['viewmats'].dtype))
            return vm.unsqueeze(1).expand(-1,pipe.frame_seq_length,-1,-1).reshape(1,-1,4,4)
        a=relative(old);b=relative(write.poses);out,stats=factor(a,b,ARM)
        identity=encode(a);assert torch.equal(identity,camera['viewmats']),'DreamX identity encoding mismatch'
        updated=camera
        if ARM=='S' or (ARM not in ['A','S'] and write.applied):updated=dict(camera,viewmats=encode(out))
        assert thash(output[:,:end])==latent_hash
        record(a,b,out,stats,q,key,frame_ids=ids,write_accepted=write.applied,
            original_sha256=thash(camera['viewmats']),written_sha256=thash(updated['viewmats']),identity_encoding_exact=True,
            representation='native c2w relative to preceding chunk endpoint, before bfloat16 PRoPE inversion')
        return updated

def sana_install(mm):
    """Reuse the qualified native replay, changing only the staged camera factors."""
    from memory_contract import MemoryWrite
    original_write=mm.sana_write;original_capture=mm.sana_capture
    mm.verify_replay=lambda:ARM=='S'
    def measured_write(out,frames,K,depth,ids,old):
        with preserve_rng():write,q,key=measure(frames,K,depth,ids,old)
        # First chunk includes fixed frame zero; native prepare_camera reanchors
        # there and preserves metric translation magnitude.
        assert ids[0]==0
        poses,stats=factor(old,write.poses,ARM)
        active=ARM not in ['A','S'] and write.applied
        staged=MemoryWrite(np.asarray(ids),poses if active else old.copy(),active,
            'diagnostic_'+ARM if write.applied else write.reason)
        record(old,write.poses,staged.poses,stats,q,key,frame_ids=np.asarray(ids),write_accepted=write.applied,
            representation='anchor-relative metric c2w; native raymap and chunk Plucker encoding')
        return staged,q
    def capture(L):
        extra=L['self']._extra_model_kwargs;start,end=L['start_f'],L['end_f']
        hashes={k:thash(v[:,:,start:end] if k=='chunk_plucker' else v[:,start:end])
            for k,v in extra.items() if k in ['camera_conditions','chunk_plucker']}
        assert set(hashes)=={'camera_conditions','chunk_plucker'}
        append(dict(kind='future_input',chunk=int(L['chunk_idx']),sha256=hashes))
        original_capture(L)
    def write(L):
        before={k:thash(L['state'][k]) for k in ['ray','plucker']}
        result=original_write(L)
        after={k:thash(L['state'][k]) for k in before}
        assert before==after,'SANA replay mutated shared future camera tensors'
        append(dict(kind='future_restored',sha256=after,exact=True))
        return result
    mm.measured_write=measured_write;mm.sana_capture=capture;mm.sana_write=write
