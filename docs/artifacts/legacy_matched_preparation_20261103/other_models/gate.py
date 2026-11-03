"""Lossless full sham identity and matched scientific-arm prefix gates."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from evidence import verify_binding
p = argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--case',required=True)
p.add_argument('--kind',choices=['identity','legacy','triplet'],required=True)
a = p.parse_args()
R = a.root.resolve()
sys.path.insert(0,str(R/'source'))
from score_memory_pair import load_frames,output_K
m = json.loads((R/'manifest.json').read_text())
c = m['protocol']['cases'][a.case]
arms = {'identity':['B','S'],'legacy':['B','L'],'triplet':['B','L','M']}[a.kind]
first = c['frames'] if a.kind=='identity' else c['first_affected_frame']
with np.load(Path(c['batch'])/'trajectories'/f"{c['scene']}.npz") as z:
    reference_K = z['intrinsics'][0].astype(float)
    reference_shape = z['anchor_depth'].shape
baseline = None
baseline_K = None
checks = {}
for arm in arms:
    binding = json.loads((R/'bindings'/f'{a.case}_{arm}.json').read_text())
    verify_binding(binding)
    directory = Path(binding['directory'])
    frames,fmt = load_frames(directory,c['scene'])
    assert fmt.startswith('lossless') and len(frames)==c['frames']
    K = output_K(c['model'],directory,reference_K,reference_shape,frames.shape[1:3])
    if baseline is None:
        baseline = frames[:first].copy()
        baseline_K = K.copy()
    checks[arm] = dict(prefix_exact=bool(np.array_equal(baseline,frames[:first])),
        intrinsics_match=bool(np.allclose(K,baseline_K,atol=1e-5,rtol=0)),frames=len(frames))
    del frames
passed = all(v['prefix_exact'] and v['intrinsics_match'] for v in checks.values())
result = dict(passed=passed,case=a.case,kind=a.kind,frames_checked=first,checks=checks,
    scope='Full raw-RGB sham identity' if a.kind=='identity' else 'Exact pre-intervention RGB and matched intrinsics',
    score_interpretation_allowed=passed)
(R/'gates'/f'{a.case}_{a.kind}.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
if not passed: raise SystemExit('Matched comparison gate failed; scoring withheld')
