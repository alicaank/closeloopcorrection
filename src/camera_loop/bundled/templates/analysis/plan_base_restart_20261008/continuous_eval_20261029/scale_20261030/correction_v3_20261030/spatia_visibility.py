"""Bound native visibility temporaries by camera view, preserving point partitions."""
def bounded_visibility(native,points,w2c,intrinsics,image_size,*args,views_per_batch=1,**kwargs):
    import torch
    if views_per_batch<1:raise ValueError('views_per_batch must be positive')
    parts=[]
    for start in range(0,len(w2c),views_per_batch):
        end=min(start+views_per_batch,len(w2c))
        K=intrinsics[start:end] if intrinsics.ndim==2 else intrinsics
        parts.append(native(points,w2c[start:end],K,image_size,*args,**kwargs))
    return torch.cat(parts,dim=0)

def install(module,views_per_batch):
    native=module.frustum_frontmost_mask_triton
    def bounded(*args,**kwargs):return bounded_visibility(native,*args,views_per_batch=views_per_batch,**kwargs)
    module.frustum_frontmost_mask_triton=bounded
