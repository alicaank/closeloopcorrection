"""Handoff extraction: exact scorer function, unrelated training imports omitted."""
import numpy as np, torch
from PIL import Image

def metrics_models():
    import lpips, torchvision.transforms as T
    lp = lpips.LPIPS(net='alex', verbose=False).eval().cuda()
    dino = torch.hub.load('@CACHE_ROOT@/torch/hub/facebookresearch_dinov2_main', 'dinov2_vits14', source='local', pretrained=False)
    dino.load_state_dict(torch.load('@CACHE_ROOT@/torch/hub/checkpoints/dinov2_vits14_pretrain.pth')); dino.eval().cuda()
    pre = T.Compose([T.Resize((224, 392)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    @torch.no_grad()
    def emb(x): return torch.nn.functional.normalize(dino(torch.stack([pre(Image.fromarray(i)) for i in x]).cuda()), dim=-1)
    @torch.no_grad()
    def lp_pair(a, b):
        ta = torch.from_numpy(np.asarray(a)).permute(0, 3, 1, 2).float().cuda() / 127.5 - 1
        tb = torch.from_numpy(np.asarray(b)).permute(0, 3, 1, 2).float().cuda() / 127.5 - 1
        return lp(ta, tb).flatten().cpu().numpy()
    return emb, lp_pair
