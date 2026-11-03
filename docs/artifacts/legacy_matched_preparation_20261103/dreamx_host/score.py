"""Independent VIPE evaluation of the qualified baseline/legacy/memory triplet."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
R = Path(__file__).resolve().parent
sys.path[:0] = [str(R/'source'),str(R)]
from score_memory_pair import load_frames, estimate, metrics
from camera_loop.evaluation_cache import cached_estimate

p = argparse.ArgumentParser()
p.add_argument('--case', required=True)
a = p.parse_args()
m = json.loads((R/'manifest.json').read_text())
c = m['protocol']['cases'][a.case]
for kind in ['identity','attribution']:
    assert json.loads((R/'gates'/f'{a.case}_{kind}.json').read_text())['passed']
root = R/'runs'/a.case
out = R/'scores'/a.case
out.mkdir(parents=True,exist_ok=False)
with np.load(Path(c['batch'])/'trajectories'/f"{c['scene']}.npz") as z:
    target = np.linalg.inv(z['w2c_overrun'].astype(float))
    K = z['intrinsics'][0].astype(float)
    depth = z['anchor_depth'].astype(np.float32)
score = dict(protocol='dreamx_legacy_memory_matched_camera_grid_v1',case=c,horizon=68,
    first_affected_frame=21,comparison_valid=True,
    reference_units='Fixed reference anchor-depth units; not verified metres',
    reference_provenance='Stored poses may themselves be VIPE-derived; not independent ground truth',
    observer='Offline VIPE; distinct from online DA3, no online poses used for scoring',
    scope='Corrected camera grid and common observer schedule; not exact historical reproduction',
    arms={},visual_review='pending')
timestamps = [0,20,21,32,44,68]
index = json.loads((Path(c['batch'])/'index.json').read_text())
row = next(r for r in index['scenes'] if r['name']==c['scene'])
tiles = [[Image.open(Path(row['frames_dir'])/f'frame_{i+1:05d}.png').resize((240,132)) for i in timestamps]]
for arm in ['S','L','M']:
    frames,fmt = load_frames(root/arm,c['scene'])
    assert frames.shape==(69,*depth.shape,3) and fmt.startswith('lossless')
    po,meta = cached_estimate(frames,K,depth,out/(arm+'.npz'),
        cache_dir=R/'scores/observer_cache',
        evaluator_id='legacy-matched:'+hashlib.sha256((R/'manifest.json').read_bytes()).hexdigest(),estimate=estimate)
    events = json.loads((root/arm/'boundary_events.json').read_text())
    valid = bool(meta['registration']['valid'])
    score['arms'][arm] = dict(position_valid=valid,registration=meta['registration'],
        post_intervention=metrics(po,target,21,valid),whole_rollout=metrics(po,target,1,valid),
        frame_sha256=hashlib.sha256(frames.tobytes()).hexdigest(),K=K.tolist(),vipe=meta,
        feedback=dict(boundaries=len(events),accepted_observations=sum(e['receipt']['accepted_observation'] for e in events),
            changed_context_chunks=sum(e['receipt']['context_changed'] for e in events),
            changed_future_chunks=sum(e['receipt']['future_changed'] for e in events)))
    tiles.append([Image.fromarray(frames[i]).resize((240,132)) for i in timestamps])
    del frames
    (out/'partial.json').write_text(json.dumps(score,indent=2,allow_nan=False)+'\n')
score['comparisons'] = {}
for control,intervention in [('S','L'),('S','M'),('L','M')]:
    delta = {}
    for key in ['position_mean','position_endpoint','rotation_mean_deg','rotation_endpoint_deg']:
        x,y = [score['arms'][arm]['post_intervention'][key] for arm in [control,intervention]]
        delta[key] = dict(control=x,intervention=y,difference=None if x is None or y is None else y-x,
            reduction_percent=None if x is None or y is None or abs(x)<1e-12 else 100*(x-y)/x)
    score['comparisons'][control+'_'+intervention] = delta
canvas = Image.new('RGB',(1440,4*158),'white')
draw = ImageDraw.Draw(canvas)
for ri,(name,row) in enumerate(zip(['Reference','S baseline','L legacy rejoin','M measured memory'],tiles)):
    for ci,(frame,img) in enumerate(zip(timestamps,row)):
        canvas.paste(img,(ci*240,ri*158+23))
        draw.text((ci*240+3,ri*158+3),f'{name}: {frame}',fill='black')
canvas.save(out/'contact_sheet.jpg',quality=90)
(out/'score.json').write_text(json.dumps(score,indent=2,allow_nan=False)+'\n')
print(json.dumps(score['comparisons']))
