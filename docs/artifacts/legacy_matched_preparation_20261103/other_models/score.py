"""Common offline camera scoring, first legacy pair, then complete triplet."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from evidence import sha,verify_binding
p = argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--case',required=True)
p.add_argument('--kind',choices=['legacy','triplet'],required=True)
a = p.parse_args()
R = a.root.resolve()
sys.path.insert(0,str(R/'source'))
from score_memory_pair import load_frames,output_K,estimate,metrics,fc
from camera_loop.evaluation_cache import cached_estimate
m = json.loads((R/'manifest.json').read_text())
c = m['protocol']['cases'][a.case]
assert json.loads((R/'gates'/f"{m['protocol']['first_case']}_identity.json").read_text())['passed']
assert json.loads((R/'gates'/f'{a.case}_{a.kind}.json').read_text())['passed']
arms = ['B','L'] if a.kind=='legacy' else ['B','L','M']
out = R/'scores'/a.case/a.kind
out.mkdir(parents=True,exist_ok=False)
with np.load(Path(c['batch'])/'trajectories'/f"{c['scene']}.npz") as z:
    target = np.linalg.inv(z['w2c_overrun'].astype(float))[:c['frames']]
    K = z['intrinsics'][0].astype(float)
    depth = z['anchor_depth'].astype(np.float32)
score = dict(protocol='legacy_rejoin_matched_native_host_v1',case=c,kind=a.kind,
    horizon=c['frames']-1,first_affected_frame=c['first_affected_frame'],comparison_valid=True,
    reference_units='Fixed reference anchor-depth units; not verified metres',
    reference_provenance='Stored poses may be VIPE-derived; not independent ground truth',
    observer='Independent offline VIPE; online DA3 poses do not enter scoring',
    arms={},visual_review='pending',attribution_gate=str(R/'gates'/f'{a.case}_{a.kind}.json'))
ts = sorted({0,c['first_affected_frame']-1,c['first_affected_frame'],(c['frames']-1)//2,3*(c['frames']-1)//4,c['frames']-1})
row = next(x for x in json.loads((Path(c['batch'])/'index.json').read_text())['scenes'] if x['name']==c['scene'])
refs = [Path(row['frames_dir'])/f"frame_{row.get('original_start',0)+i+1:05d}.jpg" for i in ts]
tiles = [[Image.open(p).convert('RGB').resize((240,132)) for p in refs]]
score['reference_samples'] = [dict(frame=i,path=str(p),sha256=sha(p)) for i,p in zip(ts,refs)]
for arm in arms:
    binding = json.loads((R/'bindings'/f'{a.case}_{arm}.json').read_text())
    verify_binding(binding)
    directory = Path(binding['directory'])
    frames,fmt = load_frames(directory,c['scene'])
    assert len(frames)==c['frames'] and fmt.startswith('lossless')
    ko = output_K(c['model'],directory,K,depth.shape,frames.shape[1:3])
    rd = fc.remap_depth(depth,K,ko,frames.shape[1:3])
    po,meta = cached_estimate(frames,ko,rd,out/(arm+'.npz'),cache_dir=R/'scores/observer_cache',
        evaluator_id='legacy-matched:'+sha(R/'manifest.json'),estimate=estimate)
    valid = bool(meta['registration']['valid'])
    memory_events = [json.loads(line) for p in directory.rglob('memory_events.jsonl') for line in p.read_text().splitlines()]
    score['arms'][arm] = dict(position_valid=valid,registration=meta['registration'],
        post_intervention=metrics(po,target,c['first_affected_frame'],valid),
        whole_rollout=metrics(po,target,1,valid),K=ko.tolist(),vipe=meta,
        frame_sha256=hashlib.sha256(frames.tobytes()).hexdigest(),binding=binding,
        accepted_memory_writes=sum(bool(e.get('applied')) for e in memory_events),
        rejected_memory_writes=[e.get('reason') for e in memory_events if not e.get('applied')])
    tiles.append([Image.fromarray(frames[i]).resize((240,132)) for i in ts])
    del frames
    (out/'partial.json').write_text(json.dumps(score,indent=2,allow_nan=False)+'\n')
score['comparisons'] = {}
for control,intervention in ([('B','L')] if a.kind=='legacy' else [('B','L'),('B','M'),('L','M')]):
    delta = {}
    for key in ['position_mean','position_endpoint','rotation_mean_deg','rotation_endpoint_deg']:
        x,y = [score['arms'][arm]['post_intervention'][key] for arm in [control,intervention]]
        delta[key] = dict(control=x,intervention=y,difference=None if x is None or y is None else y-x,
            reduction_percent=None if x is None or y is None or abs(x)<1e-12 else 100*(x-y)/x)
    score['comparisons'][control+'_'+intervention] = delta
canvas = Image.new('RGB',(len(ts)*240,len(tiles)*158),'white')
draw = ImageDraw.Draw(canvas)
for ri,(name,row) in enumerate(zip(['Reference (source view)',*arms],tiles)):
    for ci,(i,img) in enumerate(zip(ts,row)):
        canvas.paste(img,(ci*240,ri*158+23))
        draw.text((ci*240+3,ri*158+3),f'{name}: {i}',fill='black')
canvas.save(out/'contact_sheet.jpg',quality=90)
(out/'score.json').write_text(json.dumps(score,indent=2,allow_nan=False)+'\n')
print(json.dumps(score['comparisons']))
