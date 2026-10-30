"""Exact-dtype CPU residency for SANA refiner caches; one layer on device."""
import functools
import os


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
    refmod._camera_cpu_kv_installed = True
