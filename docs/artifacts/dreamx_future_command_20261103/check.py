"""Native no-op and shared-prefix attribution, using lossless rendered RGB."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;sys.path[:0]=[str(R/'source'),str(R)]
from score_memory_pair import load_frames
p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--kind',choices=['identity','attribution'],required=True);a=p.parse_args()
m=json.loads((R/'manifest.json').read_text());case=m['protocol']['cases'][a.case];root=R/'runs'/a.case
left,right=('A','S') if a.kind=='identity' else ('S','F');first=69 if a.kind=='identity' else 33
x,xf=load_frames(root/left,case['scene']);y,yf=load_frames(root/right,case['scene'])
passed=x.shape==y.shape==(69,704,1280,3) and np.array_equal(x[:first],y[:first])
lmeta=json.loads((root/left/'run.json').read_text());rmeta=json.loads((root/right/'run.json').read_text())
assert all(d['model']=='dreamx' and d['scene']==case['scene'] and d['seed']==case['seed'] and d['memory_policy']=='commanded' for d in [lmeta,rmeta])
lreceipt=json.loads((root/left/'receipts.json').read_text());rreceipt=json.loads((root/right/'receipts.json').read_text())
assert lreceipt[0]==rreceipt[0],'Initial native input/noise differs'
for le,re in zip(lreceipt[1:],rreceipt[1:]):
 if a.kind=='identity' or le['latent_end']<=9:
  assert le==re,'Shared latent/cache/RNG boundary receipt differs'
events=json.loads((root/right/'future_events.json').read_text())
assert len(events)==4 and events[0]['latent_end']==6
assert not events[0]['receipt']['native_input_changed'],'First feedback command changed'
if a.kind=='identity':assert not any(e['receipt']['native_input_changed'] for e in events)
result=dict(passed=bool(passed),kind=a.kind,case=a.case,left=left,right=right,frames_checked=first,
            complete_frames=len(x),prefix_rgb_exact=bool(np.array_equal(x[:first],y[:first])),initial_native_inputs_exact=True,shared_boundary_receipts_exact=True,
            source_formats=[xf,yf],memory_policy='commanded',first_feedback_nominal=True,
            scores_not_computed=True)
path=R/'gates'/f'{a.case}_{a.kind}.json';path.parent.mkdir(exist_ok=True);path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if not passed:raise SystemExit('Native qualification failed')
