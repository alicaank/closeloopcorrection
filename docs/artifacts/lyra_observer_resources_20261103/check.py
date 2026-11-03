"""Partial legacy parity qualifies resources; all evaluated pairs remain 321 frames."""
import argparse,json,sys
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'source'))
from score_memory_pair import load_frames
p=argparse.ArgumentParser();p.add_argument('--pair',required=True);p.add_argument('--resource-prefix',action='store_true');a=p.parse_args()
if a.resource_prefix:
    spec=json.loads((R/'specs'/f'{a.pair}.json').read_text())
    reference=json.loads((R/'specs/resource_reference.json').read_text())
    old,_=load_frames(Path(reference['commanded']),spec['scene'])
    new,_=load_frames(Path(spec['commanded']),spec['scene'])
    passed=len(old)==161 and len(new)==321 and np.array_equal(old,new[:161])
    result=dict(passed=bool(passed),reference_frames=len(old),new_frames=len(new),compared_frames=161,
        scope='Byte equality of original 161-frame baseline to repaired 321-frame prefix; no original full-horizon output exists. This gate does not qualify later-frame parity.')
    out=R/'resource_prefix.json'
else:
    score=R/'scores'/a.pair/'score.json';d=json.loads(score.read_text())
    passed=d['comparison_valid'] is True and d['horizon']==320
    result=dict(passed=bool(passed),score=str(score),prefix_exact=d['prefix_rgb_exact'],horizon=d['horizon'])
    out=score.parent/'attribution_gate.json'
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if not passed:raise SystemExit('Qualification gate failed')
