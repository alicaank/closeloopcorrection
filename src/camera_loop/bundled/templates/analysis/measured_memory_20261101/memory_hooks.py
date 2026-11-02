"""Native memory consumers: rebuilt KV, clean-context cameras, and geometry."""
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
from runtime import Observer,choose,append,policy,digest

OBSERVERS={}
OBSERVER_SESSIONS={}
SANA={}
WORLD={}
MATRIX={}


def observer(out):
    key=str(out)
    if key not in OBSERVERS:
        index=OBSERVER_SESSIONS.get(key,0)+1;OBSERVER_SESSIONS[key]=index
        path=Path(out)/'memory_observer'
        if index>1:path=path/f'session_{index:03d}'
        OBSERVERS[key]=Observer(path)
    return OBSERVERS[key]


def close_observer(out):
    worker=OBSERVERS.pop(str(out),None)
    if worker is not None:worker.close()


def pixels(decoded,layout='tfhw'):
    x=decoded.float()
    if layout=='cthw':x=x.permute(1,2,3,0)
    else:x=x.permute(0,2,3,1)
    return ((x+1)*127.5).round().clamp(0,255).byte().cpu().numpy()


def measured_write(out,frames,K,depth,ids,old):
    if policy()=='measured':poses,q=observer(out).measure(frames,K,depth,ids)
    else:poses,q=old,{'passed':True}
    return choose(old,poses,ids,len(frames)-1,q),q


def matrix_context(L):
    import torch
    out,lo,camera=L['out'],L['lo'],L['camera'];ids=np.arange(max(1,lo-83),lo+1)
    MATRIX.clear()
    old=L['all_commands'][ids].copy()
    if policy()=='commanded':return old[-1]
    with torch.random.fork_rng(devices=[0]):
        L['pipe'].load_models_to_device([])
        write,q=measured_write(out,np.asarray(L['prefix']),L['K'],L['depth'],ids,old)
    if write.applied:
        # Both the clean context and native frustum registration consume these
        # labels. Registration reconstructs world points from depth afterward.
        from mg35_camera import native_w2c
        clean=camera['clean_latent_indices_prope_extrinsic']
        encoded=native_w2c(write.poses)
        camera['clean_latent_indices_prope_extrinsic']=torch.as_tensor(encoded[-4:],device=clean.device,dtype=clean.dtype)
        MATRIX.update(end=lo,extrinsics=encoded)
    append(out,dict(model='matrixgame',frame_ids=ids,applied=write.applied,reason=write.reason,
        stored=write.poses,quality=q,mechanism='Clean-context PRoPE and source frustum registration; native depth-to-world cache rebuilt'))
    return write.poses[-1]


def matrix_registered(section,old,history):
    import torch
    if not section or not MATRIX:return old
    if MATRIX['end']!=84*section:raise RuntimeError('Matrix memory registration timeline mismatch')
    extrinsics=MATRIX['extrinsics']
    if old is None or old.shape!=extrinsics.shape or len(history)<len(extrinsics):
        raise RuntimeError('Matrix source registration contract changed')
    history[-len(extrinsics):].copy_(torch.as_tensor(extrinsics,device=history.device,dtype=history.dtype))
    return extrinsics.copy()


def lingbot_cache(L):
    import torch
    from einops import rearrange
    from wan.utils.cam_utils import compute_relative_poses,get_plucker_embeddings
    pipe=L['self'];obs=pipe._loop_observer;ctx=obs.memory_context
    start=L['chunk_id']*L['chunk_size'];end=start+L['chunk_size']
    if policy()=='commanded' or L['chunk_id']+1>=L['num_inference_chunk']:return
    old=obs.commands[max(0,start-1):end].copy();ids=4*np.arange(max(0,start-1),end)
    measured=old;q={'passed':True}
    if policy()=='measured':
        with torch.random.fork_rng(devices=[0]):
            pipe.model.cpu();torch.cuda.empty_cache()
            try:
                decoded=pipe.vae.decode([torch.cat(L['pred_latent_chunks'],dim=1)])[0]
                frames=pixels(decoded,'cthw');del decoded;torch.cuda.empty_cache()
                K=ctx['K'].copy();K[0]*=frames.shape[2]/832;K[1]*=frames.shape[1]/480
                depth=np.asarray(Image.fromarray(ctx['depth']).resize((frames.shape[2],frames.shape[1]),Image.Resampling.BILINEAR))
                measured,q=observer(ctx['out']).measure(frames,K,depth,ids)
            finally:pipe.model.to(pipe.device)
    write=choose(old,measured,ids,4*(end-1),q)
    if write.applied:
        relative=compute_relative_poses(torch.as_tensor(write.poses,dtype=torch.float32),framewise=True,normalize_trans=False)
        if start:relative=relative[1:]
        relative[:,:3,3]/=obs.norm
        Ks=L['Ks']
        if getattr(Ks,'ndim',0)==3 and len(Ks)>len(relative):Ks=Ks[start:end]
        with torch.autocast(device_type='cuda',enabled=False):pl=get_plucker_embeddings(relative.to(pipe.device),Ks,L['h'],L['w'])
        pl=rearrange(pl,'f (h c1) (w c2) c -> 1 (c c1 c2) f h w',c1=L['h']//L['lat_h'],c2=L['w']//L['lat_w']).to(pipe.param_dtype)
        L['kwargs']['dit_cond_dict']={'c2ws_plucker_emb':pl.chunk(1,dim=0)}
    append(ctx['out'],dict(model='lingbot',frame_ids=ids,applied=write.applied,reason=write.reason,
        stored=write.poses,quality=q,mechanism='Native clean-context cache forward; future input embeddings unchanged'))


def clone_tree(x,cpu=False):
    import torch
    if torch.is_tensor(x):return x.detach().to('cpu').clone() if cpu else x.clone()
    if isinstance(x,list):return [clone_tree(v,cpu) for v in x]
    if isinstance(x,tuple):return tuple(clone_tree(v,cpu) for v in x)
    if isinstance(x,dict):return {k:clone_tree(v,cpu) for k,v in x.items()}
    return x


def move_tree(x,device):
    import torch
    if torch.is_tensor(x):return x.to(device)
    if isinstance(x,list):return [move_tree(v,device) for v in x]
    if isinstance(x,tuple):return tuple(move_tree(v,device) for v in x)
    if isinstance(x,dict):return {k:move_tree(v,device) for k,v in x.items()}
    return x


def verify_replay():
    return policy()=='identity' and os.environ.get('CAMERA_MEMORY_VERIFY_REPLAY')=='1'


def assert_same_tree(a,b,path='cache'):
    import torch
    if torch.is_tensor(a):
        if not torch.is_tensor(b) or a.dtype!=b.dtype or a.shape!=b.shape or not torch.equal(a,b.to(a.device)):
            raise AssertionError('Identity replay changed '+path)
    elif isinstance(a,(list,tuple)):
        if type(a)!=type(b) or len(a)!=len(b):raise AssertionError(path+' structure changed')
        for i,(x,y) in enumerate(zip(a,b)):assert_same_tree(x,y,f'{path}[{i}]')
    elif isinstance(a,dict):
        if a.keys()!=b.keys():raise AssertionError(path+' keys changed')
        for k in a:assert_same_tree(a[k],b[k],path+'.'+str(k))
    elif a!=b:raise AssertionError(path+' value changed')


def sana_capture(L):
    # Before native clean-cache save: retain pre-save state for one replay.
    if policy()=='commanded':return
    SANA['capture']=dict(L)
    if policy()=='measured' or verify_replay():
        import torch
        SANA['prior']=clone_tree(L['chunk_kv_cache'],cpu=True)
        SANA['autocast']=(torch.is_autocast_enabled('cuda'),torch.get_autocast_dtype('cuda'))


def install_sana():
    import inspect,textwrap
    import diffusion.scheduler.self_forcing_flow_euler_sampler as native
    from overlays import native_source
    cls=native.SelfForcingFlowEulerCamCtrl
    if getattr(cls,'_measured_memory_installed',False):return
    source=native_source('sana',textwrap.dedent(inspect.getsource(cls.sample_chunks)))
    scope=dict(native.__dict__)
    exec(compile(source,inspect.getfile(cls),'exec'),scope)
    cls.sample_chunks=scope['sample_chunks'];cls._measured_memory_installed=True


def sana_write(L):
    import torch
    if policy()=='commanded' or 'capture' not in SANA:return L['commands'][len(L['prefix'])-1].copy()
    cap=SANA['capture'];start,end=cap['start_f'],cap['end_f'];pipe=L['pipe'];state=L['state']
    stride=int(pipe.config.vae.vae_stride[0]);first=max(0,start*stride-(stride-1));last=(end-1)*stride
    ids=np.arange(first,last+1);old=L['commands'][ids].copy();out=L['a'].out
    with torch.random.fork_rng(devices=[0]):
        try:write,q=measured_write(out,np.asarray(L['prefix']),L['K'],L['depth'],ids,old)
        finally:close_observer(out)
    if not write.applied and not verify_replay():
        SANA.pop('prior',None)
        append(out,dict(model='sana',frame_ids=ids,applied=False,reason=write.reason,quality=q));return old[-1].copy()
    labels=L['commands'].copy();labels[ids]=write.poses
    fresh=L['lib'].prepare_camera(labels,L['intr'],target_size=(L['lib'].TARGET_HEIGHT,L['lib'].TARGET_WIDTH),vae_stride=pipe.config.vae.vae_stride)
    ray=fresh['raymap'].unsqueeze(0).to(state['ray']);pl=fresh['chunk_plucker'].unsqueeze(0).to(state['plucker'])
    if state['ray'].shape[0]>1:ray=ray.repeat(state['ray'].shape[0],1,1);pl=pl.repeat(state['plucker'].shape[0],1,1,1,1)
    oldray=state['ray'][:,start:end].clone();oldpl=state['plucker'][:,:,start:end].clone()
    if verify_replay():
        assert torch.equal(ray[:,start:end],oldray),'Identity replay changed ray encoding'
        assert torch.equal(pl[:,:,start:end],oldpl),'Identity replay changed Plucker encoding'
    device=cap['device'];pipe.model.to(device)
    try:
        state['ray'][:,start:end].copy_(ray[:,start:end]);state['plucker'][:,:,start:end].copy_(pl[:,:,start:end])
        enabled,dtype=SANA['autocast']
        with torch.no_grad(),torch.random.fork_rng(devices=[0]),torch.autocast('cuda',enabled=enabled,dtype=dtype):
            _,updated=cap['self'].model(cap['latent_model_input'],cap['timestep'],cap['prompt_embeds'],
                start_f=cap['rope_start_f'],end_f=cap['rope_end_f'],frame_index=cap['frame_index'],
                save_kv_cache=True,kv_cache=move_tree(SANA.pop('prior'),device),mask=cap['mask'],
                data_info=cap['local_data_info'],**cap['self'].model_kwargs)
        if verify_replay():assert_same_tree(cap['kv_cache'][cap['chunk_idx']],updated)
        cap['kv_cache'][cap['chunk_idx']]=updated
    finally:
        state['ray'][:,start:end].copy_(oldray);state['plucker'][:,:,start:end].copy_(oldpl)
        pipe.model.to('cpu');torch.cuda.empty_cache()
    append(out,dict(model='sana',frame_ids=ids,applied=write.applied,reason=write.reason,stored=write.poses,quality=q,
        replay_identity_cache_exact=verify_replay(),
        mechanism='Stage-1 clean cache replay from pre-save state; refiner image-only cache and future cameras preserved'))
    return write.poses[-1].copy()


def lyra_memory(pipe,end):
    import torch
    if policy()=='commanded':return
    meta=json.loads(Path(os.environ['LONG_LYRA_META']).read_text());out=Path(meta['out'])
    cache=pipe.retrieval_cache
    if not getattr(cache,'_store_values',False):raise RuntimeError('Lyra memory rewrite requires stored depth/K values')
    lo=max(1,end-int(pipe.model.framepack_num_new_video_frames))
    # Native Plucker conditioning also reads dense history cameras, including
    # frames that are not selected into the sparse depth/retrieval cache.
    ids=list(range(lo,end))
    old=np.linalg.inv(pipe.cam_w2c[0,ids].detach().float().cpu().numpy()).astype(float)
    poses=old;q={'passed':True}
    if policy()=='measured':
        from lyra_2._src.inference import lyra2_ar_inference as ARI
        if hasattr(pipe,'_long_observer'):
            pipe._long_observer.close();del pipe._long_observer
        with torch.random.fork_rng(devices=[0]):
            da3dev=next(pipe.local_da3_model.parameters()).device
            ARI._offload_diffusion_to_cpu(pipe.model,True);pipe.local_da3_model.to('cpu');torch.cuda.empty_cache()
            try:
                frames=pixels(pipe.history_frames[0,:,pipe.start_index:],'cthw')
                with np.load(meta['trajectory']) as z:K=z['intrinsics'][0].astype(float);depth=z['anchor_depth']*meta['pose_scale']
                K[0]*=frames.shape[2]/864;K[1]*=frames.shape[1]/480
                depth=np.asarray(Image.fromarray(depth.astype(np.float32)).resize((frames.shape[2],frames.shape[1]),Image.Resampling.BILINEAR))
                poses,q=observer(out).measure(frames,K,depth,ids)
                poses=np.linalg.inv(pipe.cam_w2c[0,0].detach().float().cpu().numpy())@poses
            finally:
                close_observer(out)
                pipe.local_da3_model.to(da3dev)
                ARI._restore_diffusion_to_gpu(pipe.model,True)
    write=choose(old,poses,ids,end-1,q)
    if write.applied:
        for frame,pose in zip(ids,write.poses):
            w=torch.as_tensor(np.linalg.inv(pose),dtype=pipe.cam_w2c.dtype,device=pipe.cam_w2c.device)[None]
            if frame in cache._frame_ids:
                index=cache._frame_ids.index(frame)
                if not cache.update_by_frame_id(frame,cache._depths[index],w,cache._Ks[index]):raise RuntimeError('Lyra cache entry disappeared')
            pipe.cam_w2c[:,frame]=w
    append(out,dict(model='lyra',frame_ids=ids,applied=write.applied,reason=write.reason,stored=write.poses,quality=q,
        mechanism='Dense history camera labels plus Sparse3DCache world points recomputed from retained depth/K'))


def worldwarp_prepare(L):
    WORLD.clear();WORLD.update(prefix=np.asarray(L['prefix']),K=L['K'].copy(),depth=L['depth'].copy(),out=L['a'].out,
        plan=L['plan'],generator=L['self'],scale_state=L['scale_state'])


def worldwarp_memory(L):
    import torch
    old=L['rel_cache_poses']
    if policy()=='commanded' or L['is_first_chunk']:return old
    n=L['T_video'];frames=WORLD['prefix'];base=len(frames)-n
    if base<0:raise ValueError('WorldWarp source window is longer than generated prefix')
    # Rewrite the complete source window: smoothing also reads non-splat poses.
    local=np.arange(n);ids=local+base
    original=WORLD['plan'][ids].copy();poses=original;q={'passed':True}
    if policy()=='measured':
        modules=[WORLD['generator'].transformer,WORLD['generator'].vae,L['self'].ttt3r.model]
        devices=[next(m.parameters()).device for m in modules]
        with torch.random.fork_rng(devices=[0]):
            for m in modules:m.to('cpu')
            torch.cuda.empty_cache()
            try:poses,q=observer(WORLD['out']).measure(frames,WORLD['K'],WORLD['depth'],ids)
            finally:
                for m,d in zip(modules,devices):m.to(d)
    write=choose(original,poses,ids,len(frames)-1,q)
    updated=old
    if write.applied:
        relative=np.linalg.inv(write.poses[-1])@write.poses
        calibration=WORLD['scale_state']
        if calibration.get('status')!='identified':raise ValueError('WorldWarp memory units unavailable')
        scale=calibration['applied_native_units_per_reference_unit']
        if not np.isfinite(scale) or scale<=0:raise ValueError('Invalid WorldWarp memory units')
        relative[:,:3,3]*=scale
        updated=old.clone();updated[0]=torch.as_tensor(relative,device=old.device,dtype=old.dtype)
    append(WORLD['out'],dict(model='worldwarp',frame_ids=ids,applied=write.applied,reason=write.reason,
        stored=write.poses,quality=q,mechanism='Measured source poses before native depth-to-splat reconstruction; anchor is last source frame'))
    return updated


def astronex_memory(L,ctx,hook):
    """Observe finalized pixels before both clean conditional/unconditional writes."""
    import torch
    from models.prope import anchor_viewmats_to_frame
    chunk,window=L['vm_chunk'],L['prope_window_viewmats']
    if policy()=='commanded':return chunk,window
    start=L['current_start_frame'];end=start+L['current_num_frames']
    if end>=L['output'].shape[1]:return chunk,window
    # Use the existing incremental decoder once. The next boundary hook skips
    # decoding/measurement if it has already been handled here.
    done=dict(L,current_start_frame=end,cache_start_frame=end)
    with torch.random.fork_rng(devices=[0]):hook(done)
    v=ctx['videos'][ctx['i']];st=ctx['st'];pipe=L['self'];ids=4*np.arange(start,end)
    vm=L['viewmats'];old=np.linalg.inv(vm[0,start:end].detach().float().cpu().numpy()).astype(float)
    old[:,:3,3]/=ctx['scale'];measured=old;q={'passed':True}
    out=ctx['out']/ctx['arm']['name']/v['name']
    if policy()=='measured':
        device=next(pipe.generator.parameters()).device
        with torch.random.fork_rng(devices=[0]):
            pipe.generator.cpu();torch.cuda.empty_cache()
            try:
                frames=np.concatenate(st['frames']).copy();frames[0]=v['frame0']
                measured,q=observer(out).measure(frames,v['K'],v['anchor_depth'],ids)
            finally:pipe.generator.to(device)
    write=choose(old,measured,ids,4*(end-1),q)
    if write.applied:
        native=write.poses.copy();native[:,:3,3]*=ctx['scale']
        vm[0,start:end]=torch.as_tensor(np.linalg.inv(native),device=vm.device,dtype=vm.dtype)
        # The existing replan ran in the old stored frame; bridge its future
        # command once into the newly stored frame, without changing the plan.
        if ctx['flags']['replan'] and ctx['arm']['name']!='sham':
            for j in range(end,min(end+L['current_num_frames'],len(v['cmd']))):
                v['cmd'][j]=write.poses[-1]@np.linalg.inv(old[-1])@v['cmd'][j]
                c=v['cmd'][j].copy();c[:3,3]*=ctx['scale']
                vm[0,j]=torch.as_tensor(np.linalg.inv(c),device=vm.device,dtype=vm.dtype)
        if window is not None:
            window=anchor_viewmats_to_frame(vm[:,pipe._prope_mirror],L['anchor_idx'])
            if pipe.prope_pin_sink:
                sink=max(0,min(L['prope_sink_frames'],window.shape[1]))
                if sink and L['anchor_idx']>=sink:
                    window=window.clone();window[:,:sink]=window[:,L['anchor_idx']:L['anchor_idx']+1]
            chunk=window[:,-L['current_num_frames']:]
        else:chunk=vm[:,start:end]
    append(out,dict(model='astronex',frame_ids=ids,applied=write.applied,reason=write.reason,stored=write.poses,quality=q,
        mechanism='Clean conditional/unconditional KV writes and PRoPE mirror cameras updated before appearance-memory commit'))
    return chunk,window
