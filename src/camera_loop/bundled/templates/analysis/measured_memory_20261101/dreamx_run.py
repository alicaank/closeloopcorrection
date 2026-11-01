"""DreamX measured-memory writes at the native clean-context cache pass."""
import argparse
import hashlib
import inspect
import json
import sys
import textwrap
import time
from pathlib import Path

import numpy as np

ROOT = Path('@WORKSPACE@')
PREP = ROOT / 'analysis/plan_base_restart_20261008/model_readiness_20261027'
REPO = PREP / 'repos/dreamx'
sys.path.insert(0, str(REPO))


def perturb(plan, delta, boundary=17):
    """Scale translation after the last camera anchor of the shared prefix."""
    result = plan.copy()
    result[boundary + 1:, :3, 3] = (plan[boundary, :3, 3]
        + (1 + delta) * (plan[boundary + 1:, :3, 3] - plan[boundary, :3, 3]))
    np.testing.assert_array_equal(result[:boundary + 1], plan[:boundary + 1])
    np.testing.assert_array_equal(result[:, :3, :3], plan[:, :3, :3])
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--batch', type=Path, required=True)
    p.add_argument('--scene', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--delta', type=float, default=0.)
    p.add_argument('--memory-policy', choices=['commanded','identity','measured'], default='measured')
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--prefix-latents', type=int, choices=[6,15], required=True)
    a = p.parse_args()
    assert a.delta in [0., -.8, -.4, -.2, .1, .2, .3, .4, .8]
    import os
    os.environ['CAMERA_MEMORY_POLICY']=a.memory_policy
    from dreamx_hook import MemoryWriter
    total_latents=a.prefix_latents+3
    boundary=4*a.prefix_latents-7
    frame_count=4*(total_latents-1)+1
    a.out.mkdir(parents=True, exist_ok=False)
    import torch
    import inference_ar_forcing as entry
    import pipeline.pipeline_causal_camera as lib
    row = next(x for x in json.load(open(a.batch / 'index.json'))['scenes'] if x['name'] == a.scene)
    with np.load(a.batch / 'trajectories' / f'{a.scene}.npz') as z:
        plan = np.linalg.inv(z['w2c_overrun'].astype(float))[:frame_count]
    commands = perturb(plan, a.delta, boundary)
    np.save(a.out / 'commands.npy', commands)
    (a.out / 'input.json').write_text(json.dumps([dict(image_path=str(a.batch / row['image']),
        caption=row['caption'], camera_c2w_path=str(a.out / 'commands.npy'), task_id=a.scene)]))
    receipts = []
    writer=MemoryWriter(a,commands,total_latents,entry)

    def digest(t):
        return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()

    def tree_digest(obj):
        h = hashlib.sha256()
        def walk(x):
            if torch.is_tensor(x):
                h.update(str((tuple(x.shape), x.dtype)).encode()); h.update(digest(x).encode())
            elif isinstance(x, dict):
                for k in sorted(x): h.update(str(k).encode()); walk(x[k])
            elif isinstance(x, (list, tuple)):
                for v in x: walk(v)
            else: h.update(repr(x).encode())
        walk(obj)
        return h.hexdigest()

    def record_input(noise, camera):
        np.savez_compressed(a.out / 'native_camera.npz',
            viewmats=camera['viewmats'][0, ::880].float().cpu().numpy(),
            K=camera['K'][0, ::880].float().cpu().numpy())
        receipts.append(dict(kind='input', noise_sha256=digest(noise),
            conditioning_prefix_sha256=digest(camera['viewmats'][:, :a.prefix_latents * 880]),
            conditioning_sha256=digest(camera['viewmats'])))

    def record_boundary(pipe, output, end):
        receipts.append(dict(kind='boundary', latent_end=int(end),
            latent_prefix_sha256=digest(output[:, :end]),
            cuda_rng_sha256=digest(torch.cuda.get_rng_state()),
            cpu_rng_sha256=digest(torch.get_rng_state()),
            cache_sha256=tree_digest([pipe.kv_cache1, pipe.crossattn_cache]) if end == a.prefix_latents else None))
        (a.out / 'receipts.json').write_text(json.dumps(receipts, indent=2))

    src = textwrap.dedent(inspect.getsource(lib.CausalCameraInferencePipeline.inference))
    needle = '    batch_size, num_frames, num_channels, height, width = noise.shape'
    assert src.count(needle) == 1
    src = src.replace(needle, '    _record_input(noise, y_camera)\n' + needle)
    needle = '        # Rerun with context noise to update KV cache'
    assert src.count(needle)==1
    src=src.replace(needle, '        y_camera_latents = _memory_camera(self, output, current_start_frame, current_num_frames, y_camera_latents)\n'+needle)
    needle = '        current_start_frame += current_num_frames'
    assert src.count(needle) == 1
    src = src.replace(needle, needle + '\n        _record_boundary(self, output, current_start_frame)')
    needle = '\n    video = self.vae.decode_to_pixel(output)'
    assert src.count(needle) == 1
    src = src.replace(needle, '\n    self.generator.cpu()\n    torch.cuda.empty_cache()' + needle)
    scope = dict(lib.__dict__, _record_input=record_input, _record_boundary=record_boundary, _memory_camera=writer)
    exec(compile(src, str(a.out / 'hook_source.py'), 'exec'), scope)
    lib.CausalCameraInferencePipeline.inference = scope['inference']
    (a.out / 'hook_source.py').write_text(src)
    write = entry.write_video

    def save(path, video, **kwargs):
        np.savez_compressed(a.out / 'frames.npz', frames=video.cpu().numpy().astype(np.uint8))
        return write(path, video, **kwargs)

    entry.write_video = save
    sys.argv = ['inference_ar_forcing.py', '--config_path', str(REPO / 'configs/dreamx-ar/causal_camera_forcing_5b.yaml'),
        '--model_name', str(ROOT / 'FrameCrafter/models/Wan-AI/Wan2.2-TI2V-5B'),
        '--transformer_path', str(REPO / 'configs/dreamx-ar'),
        '--base_checkpoint_path', str(PREP / 'weights/dreamx/model.safetensors'),
        '--data_path', str(a.out / 'input.json'), '--output_folder', str(a.out),
        '--num_output_frames', str(total_latents), '--fps', '16', '--seed', str(a.seed),
        '--color_correction_strength', '1.0', '--chunk_relative']
    start = time.time()
    try:entry.main()
    finally:writer.close()
    (a.out / 'run.json').write_text(json.dumps(dict(model='dreamx', scene=a.scene, seed=a.seed,
        delta=a.delta, seconds=time.time() - start, plan=plan.tolist(), commands=commands.tolist(),
        camera_frame_ids=[0]+[1+4*i for i in range(total_latents-1)], shared_prefix_frames=4*(a.prefix_latents-1)+1,
        observation_endpoint=boundary+12, prefix_latents=a.prefix_latents, protocol='measured_memory_v1', memory_policy=a.memory_policy, receipts=receipts), indent=2))


if __name__ == '__main__':
    main()
