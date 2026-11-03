"""Opt-in device placement during SANA observation; no policy or dtype change."""
from contextlib import contextmanager
import itertools
import json
import os
from pathlib import Path
import time


def release_host_allocator():
    """Opt-in reclamation of freed glibc pages; never evict live tensor storage."""
    if os.environ.get('CAMERA_HOST_ALLOCATOR_TRIM') != '1':
        return None
    import ctypes
    import gc
    gc.collect()
    try:
        trim = ctypes.CDLL(None).malloc_trim
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
        return bool(trim(0))
    except (AttributeError, OSError):
        return False


@contextmanager
def observer_residency(model, *, enabled=None, device='cuda:0'):
    """Temporarily move idle CPU weights to GPU, then restore their device.

    Callers must close their observer *inside* this scope so its host weights
    are released before the generator returns to CPU. This is an experimental
    resource option, disabled by default, not a camera-control parameter.
    """
    if enabled is None:
        enabled = os.environ.get('CAMERA_SANA_OBSERVER_GPU_RESIDENCY') == '1'
    if not enabled:
        yield
        return
    import torch
    tensors = list(itertools.chain(model.parameters(), model.buffers()))
    devices = {t.device for t in tensors}
    if len(devices) != 1:
        raise RuntimeError('Observer residency requires one original model device')
    original = devices.pop()
    target = torch.device(device)
    # Do not retain parameter/buffer references across module.to(): replaced
    # buffer objects would otherwise keep their original CPU storage alive.
    tensor_bytes = sum(t.numel() * t.element_size() for t in tensors)
    del tensors
    if original == target:
        yield
        return
    if original.type != 'cpu' or target.type != 'cuda':
        raise RuntimeError('Observer residency only supports CPU to CUDA placement')

    def record(phase):
        path = os.environ.get('CAMERA_RESIDENCY_LOG')
        if not path:
            return
        mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        free, total = torch.cuda.mem_get_info(target)
        entry = dict(time=time.time(), phase=phase, pid=os.getpid(), tensor_bytes=tensor_bytes,
                     host_available_bytes=int(mem['MemAvailable'].split()[0])*1024,
                     gpu_free_bytes=free, gpu_total_bytes=total)
        status = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
        entry['process_rss_bytes'] = int(status['VmRSS'].split()[0])*1024
        entry['host_allocator_trim_enabled'] = os.environ.get('CAMERA_HOST_ALLOCATOR_TRIM') == '1'
        with Path(path).open('a') as f:
            f.write(json.dumps(entry)+'\n')

    record('before')
    try:
        # Device only: retain original dtype, values and module mode. No RNG
        # operations or model forwards are introduced by this context.
        model.to(target)
        if os.environ.get('CAMERA_HOST_ALLOCATOR_TRIM') == '1':
            record('before_host_trim')
            release_host_allocator()
            record('after_host_trim')
        record('observer_enter')
        yield
    finally:
        # Also restore partially transferred modules if .to(target) raises.
        release_host_allocator()
        model.to(original)
        torch.cuda.empty_cache()
        release_host_allocator()
        record('restored')
