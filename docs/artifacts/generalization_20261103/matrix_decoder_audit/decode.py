"""Same frozen latents: qualify block replay, then isolate whole-history decode."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
R=Path(__file__).resolve().parent;W=Path('/home/hj/Alican')
REPO=W/'sota_repos/matrix_game/Matrix-Game-3.5'
sys.path.insert(0,str(REPO))
from diffsynth.pipelines.wan_video import WanVideoPipeline,ModelConfig
from examples.wanvideo.pipeline.mosaic.video_io import _decode_latents_to_numpy_frames
from lossless_archive import archive_rgb
VAE=REPO/'checkpoints/Wan2.2-TI2V-5B/Wan2.2_VAE.pth'
FF=W/'analysis/plan_base_restart_20261008/model_controls_20261024/bin/ffmpeg'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''):h.update(block)
    return h.hexdigest()


def load():
    pipe=WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16,device='cpu',
        model_configs=[ModelConfig(path=str(VAE))],tokenizer_config=None,redirect_common_files=False)
    assert pipe.vae.__class__.__name__=='WanVideoVAE38'
    assert pipe.dit is None and pipe.text_encoder is None
    assert all(p.dtype==torch.bfloat16 for p in pipe.vae.parameters())
    return pipe


def adjacent(frames, indices):
    rows=[]
    for i in indices:
        a=frames[i-1].astype(np.float32);b=frames[i].astype(np.float32)
        rows.append(dict(frame=i,rgb_mae=float(np.abs(b-a).mean()),mean_level=float(b.mean()),std_level=float(b.std())))
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path);p.add_argument('--check-load',action='store_true');a=p.parse_args()
    pipe=load()
    if a.check_load:
        proof=dict(passed=True,model_class=pipe.vae.__class__.__name__,parameters=sum(p.numel() for p in pipe.vae.parameters()),
            dtype='torch.bfloat16',device='cpu',diffusion_and_text_models_loaded=False,
            scope='Actual native VAE-only loading. No GPU decode or RGB equivalence claimed.')
        (R/'cpu_load.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof));return
    spec=json.loads(a.spec.read_text());out=Path(spec['out']);out.mkdir(exist_ok=False,parents=True)
    for path,h in spec['inputs'].items():assert sha(path)==h,path
    latent=torch.load(spec['latents'],map_location='cpu',weights_only=True)
    assert latent.shape==(1,48,43,44,80) and latent.dtype==torch.bfloat16
    with np.load(spec['original_rgb']) as z:original=z['frames']
    assert original.shape==(169,704,1280,3) and original.dtype==np.uint8
    pipe.device=torch.device('cuda:0')
    # Same exact-dtype whole-module placement used by the causal host; only
    # the VAE is loaded, since generation has already produced these latents.
    def place(names):
        pipe.vae.to(pipe.device if 'vae' in names else 'cpu');torch.cuda.empty_cache()
    pipe.load_models_to_device=place
    started=time.time()
    with torch.no_grad():
        first=_decode_latents_to_numpy_frames(pipe,latent[:,:,:22],pipe.device,tiled=True)
        second=_decode_latents_to_numpy_frames(pipe,latent[:,:,21:43],pipe.device,tiled=True)
    replay=np.concatenate([first,second[1:]],axis=0)
    assert replay.shape==original.shape
    exact=bool(np.array_equal(original,replay))
    control=dict(exact_rgb_equal=exact,original_rgb_sha256=hashlib.sha256(original.tobytes()).hexdigest(),
                 replay_rgb_sha256=hashlib.sha256(replay.tobytes()).hexdigest(),
                 mae=float(np.mean([np.abs(x.astype(np.int16)-y.astype(np.int16)).mean() for x,y in zip(original,replay)])))
    result=dict(case=spec['case'],arm=spec['arm'],inputs=spec['inputs'],control=control,
                native_tiling=dict(tile_size=[30,52],tile_stride=[15,26]),units='8-bit RGB levels, 0–255')
    (out/'control.json').write_text(json.dumps(result,indent=2)+'\n')
    if not exact:raise RuntimeError('Reconstructed block export differs; decoder attribution is unqualified')
    del first,second,replay
    with torch.no_grad():joint=_decode_latents_to_numpy_frames(pipe,latent,pipe.device,tiled=True)
    assert joint.shape==original.shape
    result.update(prior_85_frames_exact=bool(np.array_equal(original[:85],joint[:85])),
        joint_vs_stream_mae_per_frame=[float(np.abs(x.astype(np.int16)-y.astype(np.int16)).mean()) for x,y in zip(original,joint)],
        stream_adjacency=adjacent(original,range(77,95)),joint_adjacency=adjacent(joint,range(77,95)),
        interpretation='Offline decoder ablation on identical generated latents. Existing online scores and emitted RGB are unchanged. Pixel seams are not camera-velocity measurements.',
        seconds=time.time()-started)
    result['archive']=archive_rgb(joint,out/'joint.lossless.mkv',FF)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(case=spec['case'],arm=spec['arm'],control_exact=exact,prior_85_frames_exact=result['prior_85_frames_exact'])))


if __name__=='__main__':main()
