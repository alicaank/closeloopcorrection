import ast,hashlib,json,sys,textwrap
from pathlib import Path
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'source'))
from overlays import native_source,driver_source
P=Path('/home/hj/Alican/analysis/plan_base_restart_20261008/model_readiness_20261027/repos/lingbot/wan/image2video.py')
raw=P.read_text();tree=ast.parse(raw)
klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='WanI2VCausal')
fn=next(n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name=='_generate_causal_fast')
source=textwrap.dedent(''.join(raw.splitlines(True)[fn.lineno-1:fn.end_lineno]))
modified=native_source('lingbot',source);compile(modified,str(P),'exec')
assert modified.index("lingbot_cache(locals())")<modified.index('# Update kv cache')
assert modified.index("after_clean(locals())")>modified.index('self.model(x=[x0], t=timestep,')
for model in ['lingbot']:compile(driver_source(model),model,'exec')
(R/'native_hook_preview.py').write_text(modified)
result=dict(passed=True,native_sha256=hashlib.sha256(raw.encode()).hexdigest(),
    overlay_sha256=hashlib.sha256(modified.encode()).hexdigest(),
    callbacks='Before and after unchanged native clean-context pass',gpu_qualified=False)
(R/'source_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
