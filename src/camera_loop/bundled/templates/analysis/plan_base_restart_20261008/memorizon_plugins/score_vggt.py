"""Handoff extraction: exact scorer function, unrelated training imports omitted."""
import sys
sys.path.insert(0, '@WORKSPACE@/VGGT-World')
CKPT = '@WORKSPACE@/growworld/third_party/assets/vggt-1b/model.safetensors'

def load_vggt():
    from vggt.models.vggt import VGGT
    from safetensors.torch import load_file
    m = VGGT(enable_track=False, enable_point=False); res = m.load_state_dict(load_file(CKPT), strict=False)   # VGGT-World class (needs CUDA)
    miss = [k for k in res.missing_keys if not k.startswith('fm.')]; assert not miss, f'missing VGGT keys: {miss[:5]}'
    return m.cuda().eval()
