"""Select the guarded worker in a new adapter process, without changing shared source files."""
import ast,json,sys
from pathlib import Path
R=Path(__file__).resolve().parent

def install():
    manifest=json.loads((R/'source_manifest.json').read_text());original=Path(manifest['common.py']['original']);sys.path.insert(0,str(original.parent))
    import common
    if getattr(common.PoseWorker,'_v3_checked',False):return
    source=(R/'snapshots/common.py').read_text();tree=ast.parse(source);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PoseWorker')
    code=ast.unparse(cls);needle="str(root / 'spatia_da3_worker.py')";assert code.count(needle)==1
    code=code.replace(needle,repr(str(R/'observer_worker.py')))
    scope=dict(common.__dict__);exec(compile(code,str(original),'exec'),scope)
    common.PoseWorker.__init__=scope['PoseWorker'].__init__;common.PoseWorker._v3_checked=True
