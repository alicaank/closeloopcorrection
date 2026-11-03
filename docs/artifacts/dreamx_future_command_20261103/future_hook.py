"""DreamX next-chunk command actuation. Native KV receives commanded cameras."""
import hashlib,json,random
from pathlib import Path
import numpy as np
from PIL import Image
from camera_loop.fractional_recovery import issue_fractional_future
from camera_loop.geometry import poses
from runtime import Observer


def digest(x):
    import torch
    return hashlib.sha256(x.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def encode(entry,commands,device,dtype):
    rows=np.zeros((len(commands),19));rows[:,7:]=np.linalg.inv(commands)[:,:3].reshape(-1,12)
    return entry.cam_params_to_prope_dict([entry.Camera(r.tolist()) for r in rows],device=device,dtype=dtype,chunk_relative=True)


def next_camera_chunk(entry,plan,issued,measured,ids,*,available_frame,anchor_frame,
                      end,total,feedback_index,arm,quality,camera,frame_seq_length):
    """CPU-testable causal proposal and next-slice commit; never mutate history."""
    import torch
    if arm not in ['A','S','F']:raise ValueError('Unknown experiment arm')
    if not 0<end<total or end%3 or total%3:raise ValueError('Native complete chunks required')
    aligned=np.array([0]+[1+4*i for i in range(total-1)])
    if available_frame!=4*(end-1) or anchor_frame!=aligned[end-1]:raise ValueError('Native RGB/camera timestamp mismatch')
    next_latent_end=min(end+3,total);next_end=int(aligned[next_latent_end-1])+1
    lo=end*frame_seq_length;hi=next_latent_end*frame_seq_length
    receipt=dict(applied=False,reason='observer_free_control' if arm=='A' else 'observe_only',memory_policy='commanded',
                 available_frame=available_frame,memory_anchor_frame=anchor_frame,
                 native_latent_range=[end,next_latent_end],native_camera_frames=aligned[end:next_latent_end].tolist(),
                 first_potentially_affected_rgb=available_frame+1)
    updated=np.asarray(issued).copy()
    if arm=='F':
        if quality.get('passed'):
            result=issue_fractional_future(plan,issued,measured,ids,next_end=next_end,
                available_frame=available_frame,feedback_index=feedback_index,nominal_boundaries=1,
                history_labels='commanded',native_stride=4,recovery_fraction=.5,speed_multiplier=2.,
                memory_anchor_frame=anchor_frame)
            updated=result.commands;receipt.update(result.receipt)
        else:receipt['reason']='observation_rejected'
    # Always derive this unissued chunk from committed commands. After an earlier
    # correction, even an abstention must use the committed native camera anchor.
    proposed=encode(entry,updated,camera['viewmats'].device,camera['viewmats'].dtype)
    if not torch.equal(proposed['K'],camera['K']):raise ValueError('Native intrinsics changed')
    before=digest(camera['viewmats'][:,lo:hi]);candidate=proposed['viewmats'][:,lo:hi]
    if arm in ['A','S'] or feedback_index==0:
        if not torch.equal(candidate,camera['viewmats'][:,lo:hi]):raise ValueError('No-op/nominal camera encoding mismatch')
    np.testing.assert_array_equal(updated[:available_frame+1],issued[:available_frame+1])
    np.testing.assert_array_equal(updated[next_end:],issued[next_end:])
    result=dict(camera);result['viewmats']=camera['viewmats'].clone()
    result['viewmats'][:,lo:hi].copy_(candidate)
    assert torch.equal(result['viewmats'][:,:lo],camera['viewmats'][:,:lo])
    assert torch.equal(result['viewmats'][:,hi:],camera['viewmats'][:,hi:])
    receipt.update(native_input_changed=before!=digest(candidate),native_input_before_sha256=before,
                   native_input_after_sha256=digest(candidate),
                   native_viewmats_after=candidate[0,::frame_seq_length].float().cpu().tolist(),
                   past_native_exact=True,unissued_tail_exact=True,
                   response_gain='not fitted; no inverse control')
    return updated,result,receipt


class FutureWriter:
    def __init__(self,args,commands,total,entry):
        self.args=args;self.plan=commands.copy();self.commands=commands.copy();self.total=total
        self.entry=entry;self.feedback_index=0;self.previous_anchor=None;self.events=[]
        self.ids=np.array([0]+[1+4*i for i in range(total-1)])
        with np.load(args.batch/'trajectories'/f'{args.scene}.npz') as z:
            self.K=z['intrinsics'][0].astype(float);self.depth=z['anchor_depth'].astype(np.float32)
        self.anchor_shape=self.depth.shape

    def __call__(self,pipe,output,end,camera):
        import torch
        if end<self.args.prefix_latents or end>=self.total:return camera
        b=4*(end-1);anchor=int(self.ids[end-1]);query=sorted({0,b-4,anchor,b,*([self.previous_anchor] if self.previous_anchor is not None else [])})
        old=self.commands[query].copy();measured=old;quality={'passed':False,'reason':'observer_disabled'}
        prefix_hash=digest(output[:,:end]);rng=digest(torch.cuda.get_rng_state())
        # A is the native observer-free control. S and F share the measurement path.
        if self.args.arm!='A':
            py_state=random.getstate();np_state=np.random.get_state();observer=None
            with torch.random.fork_rng(devices=[0]):
                pipe.generator.cpu();torch.cuda.empty_cache()
                try:
                    pixels=pipe.vae.decode_to_pixel(output[:,:end]);pipe.vae.model.clear_cache()
                    frames=((pixels[0].float()+1)*127.5).round().clamp(0,255).byte().permute(0,2,3,1).cpu().numpy()
                    del pixels;torch.cuda.empty_cache()
                    if frames.shape[0]!=b+1 or frames.shape[1:3]!=self.anchor_shape:raise ValueError('Prepared camera grid differs from decoded output')
                    observer=Observer(self.args.out/'observer'/f'boundary_{end:03d}')
                    measured,quality=observer.measure(frames,self.K,self.depth,query)
                    del frames
                finally:
                    try:
                        if observer is not None:observer.close()
                    finally:
                        try:pipe.generator.to(output.device)
                        finally:random.setstate(py_state);np.random.set_state(np_state)
        if quality.get('passed'):measured=poses(measured)
        old_commands=self.commands.copy()
        self.commands,updated,receipt=next_camera_chunk(self.entry,self.plan,self.commands,measured,np.asarray(query),
            available_frame=b,anchor_frame=anchor,end=end,total=self.total,feedback_index=self.feedback_index,
            arm=self.args.arm,quality=quality,camera=camera,frame_seq_length=pipe.frame_seq_length)
        assert digest(output[:,:end])==prefix_hash,'Observation changed latent prefix'
        assert digest(torch.cuda.get_rng_state())==rng,'Observation changed generator RNG'
        passive=None
        if self.previous_anchor is not None and quality.get('passed'):
            previous=query.index(self.previous_anchor);current=query.index(anchor)
            u=np.linalg.inv(old_commands[self.previous_anchor])@old_commands[anchor]
            m=np.linalg.inv(measured[previous])@measured[current]
            un=float(np.linalg.norm(u[:3,3]));mn=float(np.linalg.norm(m[:3,3]))
            passive=dict(commanded_anchor_interval=[self.previous_anchor,anchor],command_translation=u[:3,3].tolist(),observed_translation=m[:3,3].tolist(),execution_ratio=mn/un if un>0 else None,
                cosine=float(u[:3,3]@m[:3,3]/(un*mn)) if un>0 and mn>0 else None,
                interpretation='Passive same-window execution ratio, not differential gain or causal authority; not used to invert commands')
        event=dict(latent_end=end,feedback_index=self.feedback_index,query_frames=query,quality=quality,
                   observation=measured.tolist() if quality.get('passed') else None,receipt=receipt,
                   latent_prefix_sha256=prefix_hash,passive_execution=passive)
        self.events.append(event);self.previous_anchor=anchor;self.feedback_index+=1
        (self.args.out/'future_events.json').write_text(json.dumps(self.events,indent=2,allow_nan=False)+'\n')
        np.savez_compressed(self.args.out/'issued_commands.npz',plan=self.plan,commands=self.commands)
        return updated

    def close(self):pass
