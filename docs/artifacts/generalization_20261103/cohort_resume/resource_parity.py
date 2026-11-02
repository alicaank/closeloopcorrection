"""Qualify the resource-only overlay against the preserved previous RGB prefix."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'source'))
from score_memory_pair import load_frames
m=json.loads((R/'freeze.json').read_text());q=m['resource_parity']
old=Path(q['reference'])
assert hashlib.sha256(old.read_bytes()).hexdigest()==q['sha256']
with np.load(old) as z:a=z['frames']
b,_=load_frames(R/'runs/sana_02267acf_s7/commanded','02267acf')
assert len(a)>0 and len(b)>=len(a)
same=a.shape==b[:len(a)].shape and np.array_equal(a,b[:len(a)])
d=dict(passed=bool(same),reference=str(old),reference_sha256=q['sha256'],prefix_frames=len(a),new_frames=len(b),
       max_absolute_rgb_difference=int(np.abs(a.astype(np.int16)-b[:len(a)].astype(np.int16)).max()) if a.shape==b[:len(a)].shape else None,
       scope='Resource-only SANA overlay, same seed, inputs and schedule; all available prior RGB frames.')
(R/'resource_parity.json').write_text(json.dumps(d,indent=2)+'\n')
print(json.dumps(d))
if not same:raise SystemExit('Resource overlay RGB parity failed')
