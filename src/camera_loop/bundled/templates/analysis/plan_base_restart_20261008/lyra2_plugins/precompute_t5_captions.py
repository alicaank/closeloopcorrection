"""Embed every dataset caption of a fair batch with Lyra's own UMT5 encoder, in its own process, and cache them.

Why: loading the 11 GB UMT5 inside the main Lyra process (14B DiT held in host RAM for sequential offload) pushed host
RSS to ~58 GB and the kernel OOM-killed the 10-25/26 fair runs at 'loading text_encoder'. gv_lyra_plugins.gv_umt5 reads
this cache (dict caption -> embedding), so the main process never loads the encoder; embeddings are the stock ones.
usage: precompute_t5_captions.py <batch_dir> <out.pt>
"""
import glob, json, os, sys, torch
sys.path.insert(0, '@WORKSPACE@/lyra/Lyra-2')
from lyra_2._src.inference.get_t5_emb import UMT5EncoderModel
batch, out = sys.argv[1], sys.argv[2]
caps = sorted({c for f in glob.glob(os.path.join(batch, 'captions', '*.json')) for c in json.load(open(f)).values()})
print(len(caps), 'unique captions', flush=True)
enc = UMT5EncoderModel(device='cpu', load_on_cpu=True)
enc.model.to('cuda'); enc.device = 'cuda'
emb = {}
with torch.no_grad():
    for c in caps:
        e = enc([c], device='cuda')                        # [1, 512, 4096]
        emb[c] = e[0].detach().to('cpu', torch.float32)
torch.save(dict(prompts=emb, shape=[512, 4096]), out + '.tmp'); os.replace(out + '.tmp', out)
print('saved', out, len(emb), flush=True)
