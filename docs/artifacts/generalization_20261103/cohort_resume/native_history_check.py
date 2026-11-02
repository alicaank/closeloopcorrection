"""Exact native cache-update arithmetic before/after temporary-reference release."""
import ast,contextlib,copy,importlib.util,json,textwrap,types
from pathlib import Path
import torch
R=Path(__file__).resolve().parent;W=Path('/home/hj/Alican')
p=W/'analysis/plan_base_restart_20261008/model_readiness_20261027/repos/sana/diffusion/refiner/diffusers_ltx2_refiner.py'
native=p.read_text();tree=ast.parse(native)
cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='RefinerChunkRunner')
method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='refine_block')
source=textwrap.dedent('\n'.join(native.splitlines()[method.lineno-1:method.end_lineno]))
spec=importlib.util.spec_from_file_location('resource_overlay',R/'source/camera_sana_resource.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
patched=mod.release_history_temporaries(source);compile(patched,str(p),'exec')
def build(text):
    body=ast.parse(text).body[0].body
    update=next(x for x in body if isinstance(x,ast.With) and isinstance(x.body[0],ast.For)
                and isinstance(x.items[0].context_expr,ast.Call) and x.items[0].context_expr.args
                and isinstance(x.items[0].context_expr.args[0],ast.Constant) and x.items[0].context_expr.args[0].value=='history_update')
    function=ast.parse('def update(self,block_kv_post,active_len):\n pass').body[0]
    function.body=[update];module=ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]))
    scope=dict(torch=torch,profiler=types.SimpleNamespace(section=lambda name:contextlib.nullcontext()),
               _store_kv_tensor=lambda x,dtype:x.to(dtype=dtype or x.dtype))
    exec(compile(module,str(p),'exec'),scope);return scope['update']
original=build(source);changed=build(patched);torch.manual_seed(7);count=0
for dtype in [torch.float32,torch.float16,torch.bfloat16]:
 for old_frames in [0,2,5,8]:
  for active in [1,3,9]:
   for limit in [1,4,8]:
    for mask in [[True,True,True],[True,False,True]]:
     state=types.SimpleNamespace(_n_layers=3,_history_layer_mask=mask,_kv_cache_storage_dtype=dtype,
        _max_history_frames=limit,_tokens_per_frame=2,_history_frames=old_frames,
        _history_kv_post=[(torch.randn(1,2*old_frames,4).to(dtype),torch.randn(1,2*old_frames,4).to(dtype)) if old_frames else None for _ in range(3)])
     blocks=[(torch.randn(1,2*active,4).to(dtype),torch.randn(1,2*active,4).to(dtype)) if keep else None for keep in mask]
     other=copy.deepcopy(state);original(state,copy.deepcopy(blocks),active);changed(other,copy.deepcopy(blocks),active)
     assert state._history_frames==other._history_frames
     for a,b in zip(state._history_kv_post,other._history_kv_post):
      if a is None:assert b is None
      else:
       for x,y in zip(a,b):assert x.dtype==y.dtype and x.shape==y.shape and torch.equal(x,y)
     count+=1
(R/'native_history_validation.json').write_text(json.dumps(dict(passed=True,cases=count,
    source=str(p),scope='Actual released history-update block: bitwise tensor/shape/dtype/frame-count equality across storage dtypes, masks, history lengths, incoming chunk lengths and clipping. This does not replace GPU prefix parity.'),indent=2)+'\n')
print('Actual native history arithmetic matches in',count,'cases')
