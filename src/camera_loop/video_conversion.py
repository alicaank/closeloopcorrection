"""Bound temporary float storage without changing legacy RGB conversion math."""
import numpy as np


def history_rgb(decoded, *, formula, chunk_frames=8):
    """Convert CTHW normalized history to THWC uint8 in bounded temporal blocks.

    Endpoint and memory observers historically use different operation orders.
    Keep each order: algebraically equal formulas can round differently.
    The output still includes every frame; no observer subsampling is added.
    """
    if decoded.ndim != 4 or decoded.shape[0] != 3:
        raise ValueError('Expected three-channel CTHW history')
    if type(chunk_frames) is not int or chunk_frames < 1:
        raise ValueError('chunk_frames must be a positive integer')
    if formula not in ('endpoint', 'memory'):
        raise ValueError('Unknown legacy conversion formula')
    _, count, height, width = decoded.shape
    result = np.empty((count, height, width, 3), dtype=np.uint8)
    for start in range(0, count, chunk_frames):
        block = decoded[:, start:start + chunk_frames].detach()
        if formula == 'endpoint':
            x = block.cpu().permute(1, 2, 3, 0).float()
            rgb = ((x * .5 + .5).clamp(0, 1) * 255).round().byte()
        else:
            x = block.float().permute(1, 2, 3, 0)
            rgb = ((x + 1) * 127.5).round().clamp(0, 255).byte().cpu()
        result[start:start + chunk_frames] = rgb.numpy()
        del x, rgb, block
    return result
