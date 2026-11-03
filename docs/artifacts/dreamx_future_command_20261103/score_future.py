"""Fixed-time S/F camera scores with independent offline VIPE; no online poses."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parent;sys.path[:0]=[str(R/'source'),str(R)]
from score_memory_pair import load_frames,estimate,metrics
from camera_loop.evaluation_cache import cached_estimate
p=argparse.ArgumentParser();p.add_argument('--case',required=True);a=p.parse_args()
m=json.loads((R/'manifest.json').read_text());c=m['protocol']['cases'][a.case]
for kind in ['identity','attribution']:assert json.loads((R/'gates'/f'{a.case}_{kind}.json').read_text())['passed']
root=R/'runs'/a.case;out=R/'scores'/a.case;out.mkdir(parents=True,exist_ok=False)
with np.load(Path(c['batch'])/'trajectories'/f"{c['scene']}.npz") as z:
 target=np.linalg.inv(z['w2c_overrun'].astype(float));K=z['intrinsics'][0].astype(float);depth=z['anchor_depth'].astype(np.float32)
score=dict(protocol='dreamx_future_fractional_v1_camera_grid',case=c,horizon=68,first_affected_frame=33,
           comparison_valid=True,memory_policy='commanded',reference_units='Original fixed anchor-depth units, not verified metres',
           scope='Separate fixed-K input contract; S versus F command-only pilot. No memory actuation or learned inverse gain.',arms={})
frames_at=[0,20,32,33,50,68];tiles=[]
index=json.loads((Path(c['batch'])/'index.json').read_text());row=next(r for r in index['scenes'] if r['name']==c['scene'])
tiles.append([Image.open(Path(row['frames_dir'])/f'frame_{i+1:05d}.png').resize((240,132)) for i in frames_at])
for arm in ['S','F']:
 frames,fmt=load_frames(root/arm,c['scene']);assert frames.shape==(69,*depth.shape,3)
 po,meta=cached_estimate(frames,K,depth,out/(arm+'.npz'),cache_dir=R/'scores/observer_cache',evaluator_id='dreamx-future:'+hashlib.sha256((R/'manifest.json').read_bytes()).hexdigest(),estimate=estimate)
 events=json.loads((root/arm/'future_events.json').read_text())
 valid=bool(meta['registration']['valid']);score['arms'][arm]=dict(position_valid=valid,registration=meta['registration'],post_intervention=metrics(po,target,33,valid),whole_rollout=metrics(po,target,1,valid),frame_sha256=hashlib.sha256(frames.tobytes()).hexdigest(),K=K.tolist(),vipe=meta,
     feedback=dict(boundaries=len(events),accepted_observations=sum(e['quality']['passed'] for e in events),requested_corrections=sum(e['receipt']['applied'] for e in events),changed_native_chunks=sum(e['receipt']['native_input_changed'] for e in events)),visual_review='pending')
 tiles.append([Image.fromarray(frames[i]).resize((240,132)) for i in frames_at]);(out/'partial.json').write_text(json.dumps(score,indent=2,allow_nan=False)+'\n')
score['delta']={}
for key in ['position_mean','position_endpoint','rotation_mean_deg','rotation_endpoint_deg']:
 s,f=[score['arms'][a]['post_intervention'][key] for a in ['S','F']]
 score['delta'][key]=dict(S=s,F=f,F_minus_S=None if s is None or f is None else f-s,reduction_percent=None if s is None or f is None or abs(s)<1e-12 else 100*(s-f)/s)
canvas=Image.new('RGB',(1440,3*158),'white');draw=ImageDraw.Draw(canvas)
for row,name in zip(tiles,['Reference','S observer-only','F future correction']):
 ri=['Reference','S observer-only','F future correction'].index(name)
 for ci,(frame,img) in enumerate(zip(frames_at,row)):
  canvas.paste(img,(ci*240,ri*158+23));draw.text((ci*240+3,ri*158+3),f'{name}: {frame}',fill='black')
canvas.save(out/'contact_sheet.jpg',quality=90);(out/'score.json').write_text(json.dumps(score,indent=2,allow_nan=False)+'\n');print(json.dumps(score['delta']))
