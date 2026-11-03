"""Qualify raw RGB and native receipts before matched actuator scoring."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parent
sys.path[:0] = [str(R/'source'), str(R)]
from score_memory_pair import load_frames

p = argparse.ArgumentParser()
p.add_argument('--case', required=True)
p.add_argument('--kind', choices=['identity','attribution'], required=True)
a = p.parse_args()
m = json.loads((R/'manifest.json').read_text())
case = m['protocol']['cases'][a.case]
root = R/'runs'/a.case
arms = ['A','S'] if a.kind == 'identity' else ['S','L','M']
first = 69 if a.kind == 'identity' else 21
reference = None
reference_receipts = None
checks = {}
for arm in arms:
    frames, fmt = load_frames(root/arm, case['scene'])
    assert fmt.startswith('lossless') and frames.shape == (69,704,1280,3)
    meta = json.loads((root/arm/'run.json').read_text())
    assert (meta['model'],meta['scene'],meta['seed'],meta['arm']) == ('dreamx',case['scene'],case['seed'],arm)
    assert meta['memory_policy'] == ('measured' if arm=='M' else 'commanded')
    receipts = json.loads((root/arm/'receipts.json').read_text())
    selected = receipts if a.kind == 'identity' else [r for r in receipts if r['kind']=='input' or (r['latent_end']<6 or r['latent_end']==6 and r['phase']=='before_cache')]
    assert len(receipts)==13 and len(selected)==(13 if a.kind=='identity' else 4)
    if reference is None:
        reference = frames[:first].copy()
        reference_receipts = selected
    checks[arm] = dict(prefix_exact=bool(np.array_equal(reference,frames[:first])),
        native_receipts_exact=selected==reference_receipts)
    del frames
    events = json.loads((root/arm/'boundary_events.json').read_text())
    assert [e['latent_end'] for e in events] == [6,9,12,15]
    if arm in ['A','S','L']:
        assert not any(e['receipt']['context_changed'] for e in events)
    if arm in ['A','S','M']:
        assert not any(e['receipt']['future_changed'] for e in events)
    if arm in ['A','S']:
        assert not any(e['receipt']['applied'] for e in events)
    checks[arm]['actuator_separation'] = True
passed = all(c['prefix_exact'] and c['native_receipts_exact'] for c in checks.values())
result = dict(passed=passed,case=a.case,kind=a.kind,checks=checks,frames_checked=first,
    first_potentially_affected_rgb=21,complete_frames=69,scores_not_computed=True)
(R/'gates'/f'{a.case}_{a.kind}.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
if not passed:
    raise SystemExit('Matched-host qualification failed; scores withheld')
