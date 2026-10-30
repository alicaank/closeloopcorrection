"""Versioned DA3 worker with persisted quality diagnostics and explicit rejection."""
import json,sys
from pathlib import Path
R=Path(__file__).resolve().parent
meta=json.loads((R/'source_manifest.json').read_text())['spatia_da3_worker.py']
source=(R/'snapshots/spatia_da3_worker.py').read_text()
old='from realized_pose import load_da3, realized_poses'
assert source.count(old)==1
source=source.replace(old,'from guarded_observer import realized_poses\nfrom realized_pose import load_da3')
source=source.replace("sec=round(time.time() - t0, 2)))","sec=round(time.time() - t0, 2), quality=info.get('quality')))")
source=source.replace("error=f'{type(e).__name__}: {e}'))", "error=f'{type(e).__name__}: {e}', quality=getattr(e,'quality',None)))")
exec(compile(source,meta['original'],'exec'),dict(__name__='__main__',__file__=meta['original']))
