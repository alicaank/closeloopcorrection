"""Exact-dtype CPU residency for SANA refiner caches; one layer on device."""
import functools
import os


def release_history_temporaries(source):
    """Release consumed caches without changing tensor arithmetic or history."""
    before='    with profiler.section("history_update"):\n        for layer_idx in range(self._n_layers):'
    after='            raw_k = None\n            raw_v = None'
    if source.count(before)!=1 or source.count(after)!=1:
        raise RuntimeError('SANA native history lifetime contract changed')
    source=source.replace(before,
        '    kv_prefix_per_layer = None\n'
        '    prefix = None\n'
        '    hk = None\n'
        '    captured_kv_post = None\n'+before)
    return source.replace(after,after+'\n'
        '            block_kv_post[layer_idx] = None\n'
        '            raw_pair = None\n'
        '            old = None\n'
        '            new_k = None\n'
        '            new_v = None')


def install(refmod):
    if getattr(refmod, '_camera_cpu_kv_installed', False):
        return
    import torch
    # Preconcat applies GPU rotary embeddings outside the attention wrapper.
    os.environ['SANA_WM_REFINER_PRECONCAT_PREFIX'] = '0'
    original_attention = refmod._streaming_self_attention

    def capture(tensor):
        # Independent storage even when input is already on CPU.
        return tensor.detach().to(device='cpu', copy=True)

    def store(tensor, dtype):
        return tensor.to(device='cpu', dtype=dtype or tensor.dtype)

    def transfer(value, device):
        if isinstance(value, torch.Tensor):
            return value.to(device=device)
        if isinstance(value, tuple):
            return tuple(transfer(x, device) for x in value)
        return value

    @functools.wraps(original_attention)
    def attention(*, attn, hidden_states, **kwargs):
        prefix = getattr(attn, '_tf_kv_prefix', None)
        if not isinstance(prefix, dict):
            return original_attention(attn=attn, hidden_states=hidden_states, **kwargs)
        attn._tf_kv_prefix = {k: transfer(v, hidden_states.device) for k, v in prefix.items()}
        try:
            return original_attention(attn=attn, hidden_states=hidden_states, **kwargs)
        finally:
            attn._tf_kv_prefix = prefix

    refmod._capture_kv_tensor = capture
    refmod._store_kv_tensor = store
    refmod._streaming_self_attention = attention
    import inspect,textwrap
    method=inspect.unwrap(refmod.RefinerChunkRunner.refine_block)
    source=release_history_temporaries(textwrap.dedent(inspect.getsource(method)))
    scope=dict(refmod.__dict__)
    exec(compile(source,inspect.getfile(method),'exec'),scope)
    refmod.RefinerChunkRunner.refine_block=scope['refine_block']
    refmod._camera_cpu_kv_installed = True
