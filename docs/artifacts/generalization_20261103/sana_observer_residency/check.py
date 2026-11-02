"""Resource-prefix and attribution gates for the separate SANA retry."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--pair',required=True);p.add_argument('--resource-prefix',action='store_true');a=p.parse_args()
m=json.loads((R/'manifest.json').read_text())
if a.resource_prefix:
    ref=Path(m['resource_reference']['path'])
    assert hashlib.sha256(ref.read_bytes()).hexdigest()==m['resource_reference']['sha256']
    with np.load(ref) as z:old=z['frames']
    with np.load(R/'runs'/a.pair/'commanded/frames.npz') as z:new=z['frames']
    passed=len(new)==121 and np.array_equal(old,new[:len(old)])
    result=dict(passed=bool(passed),new_frames=len(new),reference_frames=len(old),
        required_frames=121,reference_sha256=m['resource_reference']['sha256'],
        scope='Full commanded rollout completion plus exact equality of the preserved 97-frame prefix; paired intervention still requires scoring.')
    out=R/'resource_prefix.json'
else:
    score=R/'scores'/a.pair/'score.json';d=json.loads(score.read_text())
    passed=d['comparison_valid'] is True and d['horizon']==120
    result=dict(passed=bool(passed),score=str(score),score_sha256=hashlib.sha256(score.read_bytes()).hexdigest(),
                prefix_exact=d['prefix_rgb_exact'],horizon=d['horizon'])
    out=R/'scores'/a.pair/'attribution_gate.json'
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if not passed:raise SystemExit('Qualification gate failed')
