import json
import numpy as np
from PIL import Image
from runtime import Observer,choose,append,digest,policy


class MemoryWriter:
    def __init__(self,args,commands,total,entry):
        self.args=args;self.commands=commands.copy();self.labels=commands.copy()
        self.total=total;self.entry=entry;self.observer=None
        with np.load(args.batch/'trajectories'/f'{args.scene}.npz') as z:
            self.K=z['intrinsics'][0].astype(float);self.depth=z['anchor_depth'].astype(np.float32)
        self.ids=[0]+[1+4*i for i in range(total-1)]

    def __call__(self,pipe,output,start,count,camera):
        import torch
        end=start+count
        if end<self.args.prefix_latents or end>=self.total or policy()=='commanded':return camera
        # One preceding native camera defines chunk-relative pose encoding.
        indices=self.ids[max(0,start-1):end]
        old=self.labels[indices].copy();measured=old;quality={'passed':True}
        latent_hash=digest(output[:,:end]);rng=digest(torch.cuda.get_rng_state())
        if policy()=='measured':
            with torch.random.fork_rng(devices=[0]):
                pipe.generator.cpu();torch.cuda.empty_cache()
                try:
                    pixels=pipe.vae.decode_to_pixel(output[:,:end]);pipe.vae.model.clear_cache()
                    frames=((pixels[0].float()+1)*127.5).round().clamp(0,255).byte().permute(0,2,3,1).cpu().numpy()
                    del pixels;torch.cuda.empty_cache()
                    K=self.K.copy();K[0]*=frames.shape[2]/864;K[1]*=frames.shape[1]/480
                    depth=np.asarray(Image.fromarray(self.depth).resize((frames.shape[2],frames.shape[1]),Image.Resampling.BILINEAR))
                    if self.observer is None:self.observer=Observer(self.args.out/'memory_observer')
                    measured,quality=self.observer.measure(frames,K,depth,indices)
                finally:pipe.generator.to(output.device)
        write=choose(old,measured,indices,4*(end-1),quality)
        updated=camera
        if write.applied:
            labels=self.labels.copy();labels[indices]=write.poses
            rows=np.zeros((len(labels),19),dtype=np.float64)
            rows[:,0]=np.arange(len(labels));rows[:,1:5]=[969.6969696969696/1920,969.6969696969696/1080,.5,.5]
            rows[:,7:]=np.linalg.inv(labels)[:,:3].reshape(len(labels),12)
            cams=[self.entry.Camera(row.tolist()) for row in rows]
            encoded=self.entry.cam_params_to_prope_dict(cams,device=output.device,chunk_relative=True)
            lo=start*pipe.frame_seq_length;hi=end*pipe.frame_seq_length
            updated={key:value[:,lo:hi].clone() for key,value in encoded.items()}
            self.labels=labels
        assert digest(output[:,:end])==latent_hash
        assert digest(torch.cuda.get_rng_state())==rng
        append(self.args.out,dict(model='dreamx',policy=policy(),latent_start=start,latent_end=end,
            frame_ids=indices,applied=write.applied,reason=write.reason,quality=quality,
            commanded=old,stored=write.poses,latent_prefix_sha256=latent_hash,
            camera_before_sha256=digest(camera['viewmats']),camera_after_sha256=digest(updated['viewmats']),
            mechanism='Native context-noise pass with measured camera; generated latents and future camera tensors unchanged'))
        return updated

    def close(self):
        if self.observer:self.observer.close()
