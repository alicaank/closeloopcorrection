"""Matched DreamX observation with separate legacy and measured-memory actuators.

Observe once before the context-noise pass. M changes only that pass's camera;
L commits only the next chunk's camera after that pass. A/S are native/observer
controls. The calibrated input grid is shared by all arms.
"""
import hashlib
import json
import random
import numpy as np
from legacy_curve import rejoin
from memory_contract import stage_memory_write
from runtime import Observer


def digest(t):
    import torch
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def encode(entry, commands, device, dtype):
    rows = np.zeros((len(commands), 19))
    rows[:, 7:] = np.linalg.inv(commands)[:, :3].reshape(-1, 12)
    return entry.cam_params_to_prope_dict(
        [entry.Camera(r.tolist()) for r in rows], device=device, dtype=dtype, chunk_relative=True)


def query_frames(start, end, total):
    if not (0 <= start < end < total and start % 3 == 0 and end - start == 3 and total % 3 == 0):
        raise ValueError('Expected one complete nonterminal native chunk')
    ids = np.array([0] + [1 + 4*i for i in range(total - 1)])
    return np.unique(np.r_[0, ids[max(0, start - 1):end]])


def stage_boundary(entry, plan, issued, labels, measured, *, start, end, total,
                   arm, quality, camera, context, frame_seq_length):
    """Pure proposal: returned future tensor is committed only after native KV update."""
    import torch
    if arm not in ['A', 'S', 'L', 'M']:
        raise ValueError('Unknown matched arm')
    ids = np.array([0] + [1 + 4*i for i in range(total - 1)])
    query = query_frames(start, end, total)
    if len(plan) != 4*(total-1)+1 or np.shape(issued) != np.shape(plan) or np.shape(labels) != np.shape(plan):
        raise ValueError('Full native timeline required')
    lo, hi = start*frame_seq_length, end*frame_seq_length
    if any(not torch.equal(context[k], camera[k][:, lo:hi]) for k in ['K', 'viewmats']):
        raise ValueError('Current native context does not match issued camera slice')
    write = stage_memory_write(query, labels[query], measured,
        available_last_frame=4*(end-1), quality=quality)
    accepted = write.reason in ['identity', 'measured']
    commands = np.asarray(issued).copy()
    stored = np.asarray(labels).copy()
    next_camera = camera
    next_context = context
    receipt = dict(arm=arm, applied=False, accepted_observation=accepted,
        observation_reason=write.reason, native_latent_range=[start, end],
        query_frames=query.tolist(), available_last_frame=4*(end-1),
        memory_anchor_frame=int(ids[end-1]), first_potentially_affected_rgb=4*(end-1)+1,
        context_changed=False, future_changed=False)
    if arm == 'M' and write.applied:
        stored[query] = write.poses
        proposed = encode(entry, stored, camera['viewmats'].device, camera['viewmats'].dtype)
        if not torch.equal(proposed['K'], camera['K']):
            raise ValueError('Native intrinsics changed')
        next_context = {k: v[:, lo:hi].clone() for k, v in proposed.items()}
        receipt['context_changed'] = not torch.equal(next_context['viewmats'], context['viewmats'])
        receipt['applied'] = True
    elif arm == 'L' and accepted:
        future = ids[end:min(end+3, total)]
        anchor = int(ids[end-1])
        m = write.poses[np.flatnonzero(query == anchor)[0]]
        commands[future] = rejoin(m, issued[anchor], plan[future])
        # Temporary measured anchor is only for native next-chunk encoding.
        conditioning = commands.copy()
        conditioning[anchor] = m
        proposed = encode(entry, conditioning, camera['viewmats'].device, camera['viewmats'].dtype)
        if not torch.equal(proposed['K'], camera['K']):
            raise ValueError('Native intrinsics changed')
        nlo, nhi = end*frame_seq_length, min(end+3, total)*frame_seq_length
        next_camera = dict(camera)
        next_camera['viewmats'] = camera['viewmats'].clone()
        next_camera['viewmats'][:, nlo:nhi].copy_(proposed['viewmats'][:, nlo:nhi])
        receipt['future_changed'] = not torch.equal(next_camera['viewmats'], camera['viewmats'])
        receipt['applied'] = receipt['future_changed']
        receipt['future_camera_frames'] = future.tolist()
        np.testing.assert_array_equal(commands[:4*(end-1)+1], issued[:4*(end-1)+1])
        np.testing.assert_array_equal(commands[future[-1]+1:], issued[future[-1]+1:])
        np.testing.assert_allclose(commands[future[-1]], plan[future[-1]], atol=1e-12, rtol=0)
        assert torch.equal(next_camera['viewmats'][:, :nlo], camera['viewmats'][:, :nlo])
        assert torch.equal(next_camera['viewmats'][:, nhi:], camera['viewmats'][:, nhi:])
    receipt.update(context_before_sha256=digest(context['viewmats']),
        context_after_sha256=digest(next_context['viewmats']),
        future_before_sha256=digest(camera['viewmats'][:, hi:]),
        future_after_sha256=digest(next_camera['viewmats'][:, hi:]))
    return commands, stored, next_context, next_camera, receipt


class BoundaryWriter:
    def __init__(self, args, commands, total, entry):
        self.args, self.total, self.entry = args, total, entry
        self.plan = commands.copy()
        self.commands = commands.copy()
        self.labels = commands.copy()
        self.pending = None
        self.events = []
        with np.load(args.batch/'trajectories'/f'{args.scene}.npz') as z:
            self.K = z['intrinsics'][0].astype(float)
            self.depth = z['anchor_depth'].astype(np.float32)
        self.anchor_shape = self.depth.shape

    def before_cache(self, pipe, output, start, count, context, camera):
        import torch
        end = start + count
        if end < self.args.prefix_latents or end >= self.total:
            return context
        if self.pending is not None:
            raise RuntimeError('Previous boundary was not committed')
        query = query_frames(start, end, self.total)
        measured = self.labels[query].copy()
        quality = dict(passed=False, reason='observer_free_control')
        prefix_hash = digest(output[:, :end])
        cpu_rng, cuda_rng = digest(torch.get_rng_state()), digest(torch.cuda.get_rng_state())
        rgb_hash = None
        if self.args.arm != 'A':
            py_state, np_state = random.getstate(), np.random.get_state()
            observer = None
            with torch.random.fork_rng(devices=[0]):
                pipe.generator.cpu()
                torch.cuda.empty_cache()
                try:
                    pixels = pipe.vae.decode_to_pixel(output[:, :end])
                    pipe.vae.model.clear_cache()
                    frames = ((pixels[0].float()+1)*127.5).round().clamp(0,255).byte().permute(0,2,3,1).cpu().numpy()
                    del pixels
                    torch.cuda.empty_cache()
                    if frames.shape[:3] != (4*(end-1)+1, *self.anchor_shape):
                        raise ValueError('Decoded RGB differs from calibrated input grid')
                    rgb_hash = hashlib.sha256(frames.tobytes()).hexdigest()
                    observer = Observer(self.args.out/'observer'/f'boundary_{end:03d}')
                    measured, quality = observer.measure(frames, self.K, self.depth, query)
                    del frames
                finally:
                    try:
                        if observer is not None:
                            observer.close()
                    finally:
                        try:
                            pipe.generator.to(output.device)
                        finally:
                            random.setstate(py_state)
                            np.random.set_state(np_state)
        self.commands, self.labels, updated, future, receipt = stage_boundary(
            self.entry, self.plan, self.commands, self.labels, measured,
            start=start, end=end, total=self.total, arm=self.args.arm, quality=quality,
            camera=camera, context=context, frame_seq_length=pipe.frame_seq_length)
        assert digest(output[:, :end]) == prefix_hash, 'Observer changed generated latents'
        assert digest(torch.get_rng_state()) == cpu_rng, 'Observer changed CPU RNG'
        assert digest(torch.cuda.get_rng_state()) == cuda_rng, 'Observer changed CUDA RNG'
        self.pending = (end, future)
        self.events.append(dict(latent_end=end, quality=quality, receipt=receipt,
            observation=measured.tolist() if quality.get('passed') else None,
            latent_prefix_sha256=prefix_hash, decoded_prefix_sha256=rgb_hash))
        (self.args.out/'boundary_events.json').write_text(json.dumps(self.events, indent=2, allow_nan=False)+'\n')
        np.savez_compressed(self.args.out/'issued_commands.npz', plan=self.plan,
            commands=self.commands, memory_labels=self.labels)
        return updated

    def after_cache(self, end, camera):
        if self.pending is None:
            return camera
        expected, updated = self.pending
        if end != expected:
            raise ValueError('Native cache boundary changed')
        self.pending = None
        return updated

    def close(self):
        pass
