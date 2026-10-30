"""Isolated runner: camera-view batching and optional observer-quality policy.
Original running adapter and checkpoint files are never modified.
"""
import argparse,json,os,sys
from pathlib import Path
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser(add_help=False);p.add_argument('--visibility-views-per-batch',type=int,default=1);p.add_argument('--observer-quality',choices=['off','report','enforce'],default='off')
a,rest=p.parse_known_args();sys.argv=[sys.argv[0]]+rest
if a.visibility_views_per_batch<1:raise ValueError('Invalid camera batch size')
if '--out' in rest:
    out=Path(rest[rest.index('--out')+1]);out.mkdir(parents=True,exist_ok=True)
    receipt=out/'v3_protocol.json'
    if receipt.exists():raise FileExistsError('Use a fresh v3 output directory')
    receipt.write_text(json.dumps(dict(version='correction_v3_20261030',visibility_views_per_batch=a.visibility_views_per_batch,point_partition='unchanged',observer_quality=a.observer_quality),indent=2))
meta=json.loads((R/'source_manifest.json').read_text())['spatia_run.py'];source=(R/'snapshots/spatia_run.py').read_text()
needle='    import inference as SI\n';assert source.count(needle)==1
source=source.replace(needle,needle+'    from spatia_visibility import install\n    install(SI, V3_VIEWS)\n')
if a.observer_quality!='off':
    os.environ['GV_OBSERVER_QUALITY']=a.observer_quality
    needle="str(HERE / 'spatia_da3_worker.py')";assert source.count(needle)==1
    source=source.replace(needle,'V3_WORKER')
# Abort the process after an arm exception: do not run another arm in a potentially
# exhausted CUDA context. A retry uses a fresh process and a fresh output directory.
needle="                torch.cuda.empty_cache()\n            atomic_json"
assert source.count(needle)==1
source=source.replace(needle,"                atomic_json(a.out / '_proc' / f'{tag}.json', proc)\n                raise\n            atomic_json")
exec(compile(source,meta['original'],'exec'),dict(__name__='__main__',__file__=meta['original'],V3_VIEWS=a.visibility_views_per_batch,V3_WORKER=str(R/'observer_worker.py')))
