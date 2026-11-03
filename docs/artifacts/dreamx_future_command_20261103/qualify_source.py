"""Compile exactly the driver's native source overlay without loading weights."""
import ast,inspect,json,textwrap,types,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent
native=Path('/home/hj/Alican/analysis/plan_base_restart_20261008/model_readiness_20261027/repos/dreamx/pipeline/pipeline_causal_camera.py')
tree=ast.parse(native.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CausalCameraInferencePipeline');fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='inference')
lines=native.read_text().splitlines(True);source=textwrap.dedent(''.join(lines[fn.lineno-1:fn.end_lineno]))
main=next(n for n in ast.parse((R/'run.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main')
ops=[];collect=False
for n in main.body:
 if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='src' for t in n.targets):collect=True
 if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='scope' for t in n.targets):break
 if collect:ops.append(n)
ns=dict(inspect=types.SimpleNamespace(getsource=lambda fn:source),textwrap=textwrap,lib=types.SimpleNamespace(CausalCameraInferencePipeline=types.SimpleNamespace(inference=None)))
exec(compile(ast.Module(body=ops,type_ignores=[]),str(R/'run.py'),'exec'),ns)
result=ns['src'];compile(result,str(R/'native_hook_preview.py'),'exec')
assert result.count('_future_camera(')==1
assert result.index('# Rerun with context noise')<result.index('_future_camera(')
assert '_memory_camera' not in result
(R/'native_hook_preview.py').write_text(result)
(R/'source_validation.json').write_text(json.dumps(dict(passed=True,native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),overlay_sha256=hashlib.sha256(result.encode()).hexdigest(),future_callback_after_native_cache_pass=True,no_memory_camera_rewrite=True),indent=2)+'\n')
print('Actual native overlay compiled; future callback follows unchanged context-cache pass')
