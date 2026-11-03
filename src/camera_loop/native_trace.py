"""Explicit, read-only intermediate tracing for native identity diagnostics.

Tracing copies tensors to CPU and synchronizes device work. It is not enabled
by production adapters and does not promise identical timing/kernel scheduling.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import pickle
import random

import numpy as np
import torch


def digest(value):
    """Describe supported values without retaining tensors or consuming RNG."""
    if isinstance(value, torch.Tensor):
        raw = value.detach().reshape(-1).contiguous().view(torch.uint8).cpu().numpy()
        return dict(kind='tensor', shape=list(value.shape), dtype=str(value.dtype),
                    sha256=hashlib.sha256(raw.tobytes()).hexdigest())
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError('Object arrays cannot be reproducibly traced')
        return dict(kind='array', shape=list(value.shape), dtype=str(value.dtype),
                    sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())
    if isinstance(value, torch.Generator):
        return dict(kind='generator', state=digest(value.get_state()))
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise TypeError('Trace dictionaries require string keys')
        return {k: digest(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [digest(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    # Explicit coverage limit, never repr(object) with unstable addresses.
    return dict(untraced_type=type(value).__module__ + '.' + type(value).__qualname__)


def rng_digest():
    result = dict(
        python=hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),
        numpy=hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),
        torch_cpu=digest(torch.random.get_rng_state()),
    )
    # Do not initialize CUDA as a side effect of a CPU trace.
    if torch.cuda.is_initialized():
        result['torch_cuda'] = [digest(s) for s in torch.cuda.get_rng_state_all()]
    return result


class NativeTrace:
    def __init__(self, path):
        self.path = Path(path)
        self.stream = self.path.open('x')
        self.sequence = 0
        self.restores = []

    def record(self, event, **values):
        row = dict(sequence=self.sequence, event=event, values=digest(values), rng=rng_digest())
        self.stream.write(json.dumps(row, allow_nan=False, sort_keys=True) + '\n')
        self.stream.flush()
        self.sequence += 1

    def wrap(self, obj, name, event, *, inputs=None, outputs=None, state=None):
        original = getattr(obj, name)
        had_instance_attr = name in vars(obj)
        instance_value = vars(obj).get(name)

        def wrapped(*args, **kwargs):
            before = inputs(args, kwargs) if inputs else dict(args=args, kwargs=kwargs)
            self.record(event + '.before', inputs=before, state=state() if state else None)
            result = original(*args, **kwargs)
            self.record(event + '.after', output=outputs(result) if outputs else result,
                        state=state() if state else None)
            return result

        setattr(obj, name, wrapped)

        def restore():
            if had_instance_attr:
                setattr(obj, name, instance_value)
            else:
                delattr(obj, name)
        self.restores.append(restore)

    def module(self, module, event):
        def before(_module, args, kwargs):
            self.record(event + '.before', args=args, kwargs=kwargs)

        def after(_module, args, kwargs, output):
            self.record(event + '.after', output=output)

        self.restores.append(module.register_forward_pre_hook(before, with_kwargs=True).remove)
        self.restores.append(module.register_forward_hook(after, with_kwargs=True).remove)

    def close(self):
        try:
            for restore in reversed(self.restores):
                restore()
        finally:
            self.restores.clear()
            self.stream.close()


@contextmanager
def worldplay_trace(roller, path):
    """Trace an unchanged-command Roller; save hashes, never raw tensor copies."""
    trace = NativeTrace(path)
    pipe = roller.wp.pipe

    def state():
        return dict(ctx=getattr(pipe, 'ctx', None),
                    decode_state=getattr(pipe, '_decode_state', None),
                    points_local=getattr(pipe, 'points_local', None))

    try:
        trace.wrap(roller, 'wp_rows', 'native_inputs', state=state)
        trace.wrap(pipe, 'prepare_latents', 'noise', state=state)
        trace.module(pipe.transformer, 'transformer')
        trace.wrap(pipe, 'decode_next_latent', 'decode', state=state)
        if roller.response is not None:
            trace.wrap(roller.response, 'observe', 'observer',
                       inputs=lambda args, kwargs: dict(index=args[1], next_end=args[2],
                           frames=len(args[0]), frame_sha256=[
                               hashlib.sha256(np.asarray(x).tobytes()).hexdigest() for x in args[0]]),
                       outputs=lambda result: dict(future=result[0], measured=result[2]), state=state)
        trace.record('installed')
        yield trace
        trace.record('completed')
    finally:
        trace.close()
