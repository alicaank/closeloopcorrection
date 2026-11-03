"""Full original-baseline parity and matched prefix gates for Matrix retry."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'source'))
from camera_loop.lossless_archive import read_rgb
p=argparse.ArgumentParser();p.add_argument('--pair',required=True);p.add_argument('--resource-prefix',action='store_true');a=p.parse_args()
m=json.loads((R/'manifest.json').read_text())
if a.resource_prefix:
 ref=Path(m['resource_reference']['path']);h=hashlib.sha256()
 with ref.open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 assert h.hexdigest()==m['resource_reference']['sha256']
 old=read_rgb(ref)
 with np.load(R/'runs'/a.pair/'commanded/frames.npz') as z:new=z['frames']
 passed=len(new)==337 and np.array_equal(old,new)
 result=dict(passed=bool(passed),new_frames=len(new),reference_frames=len(old),required_frames=337,
             reference_sha256=h.hexdigest(),scope='Full 337-frame commanded output byte equality to original completed baseline; not partial-prefix qualification')
 out=R/'resource_prefix.json'
else:
 score=R/'scores'/a.pair/'score.json';d=json.loads(score.read_text());passed=d['comparison_valid'] is True and d['horizon']==336
 result=dict(passed=bool(passed),score=str(score),prefix_exact=d['prefix_rgb_exact'],horizon=d['horizon'])
 out=score.parent/'attribution_gate.json'
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if not passed:raise SystemExit('Qualification gate failed')
