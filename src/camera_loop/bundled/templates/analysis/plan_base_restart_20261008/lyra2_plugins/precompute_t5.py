"""Embed the fixed GaussVerse prompt once with Lyra's UMT5 encoder (own process, ~21 GB host RAM) and cache it.

The main Lyra process then skips the text encoder (gv_lyra_plugins patches get_umt5_embedding*), keeping host RSS
near 36 GB instead of 58 GB (which OOM-killed the 62 GB host). Lyra's own code path and weights are used, so the
embedding is bit-for-bit what the stock pipeline would compute.
"""
import json, sys, torch
sys.path.insert(0, '@WORKSPACE@/lyra/Lyra-2')
from lyra_2._src.inference.get_t5_emb import UMT5EncoderModel
PROMPT = json.load(open('@WORKSPACE@/analysis/plan_base_restart_20261008/lyra2_plugins/batch_dev10/index.json'))['prompt']
enc = UMT5EncoderModel(device='cpu', load_on_cpu=True)
enc.model.to('cuda'); enc.device = 'cuda'
with torch.no_grad():
    emb = enc([PROMPT], device='cuda')          # [1, 512, 4096]
out = '@WORKSPACE@/analysis/plan_base_restart_20261008/lyra2_plugins/prompt_umt5.pt'
torch.save(dict(prompt=PROMPT, embedding=emb[0].detach().to('cpu', torch.float32), shape=list(emb.shape), dtype=str(emb.dtype)), out)
print('saved', out, tuple(emb.shape), emb.dtype, float(emb.float().abs().mean()))
