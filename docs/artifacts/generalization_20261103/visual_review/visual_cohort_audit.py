"""Read-only timestamp-matched review sheets and reference-motion context.

Uses completed pairs only and decodes thumbnails to avoid competing for host
memory with the live generation queue. Stored references are not asserted to
be independent ground-truth geometry. No generated result selects the frames.
"""
from pathlib import Path
import hashlib,json,subprocess
import numpy as np
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parent;W=R.parents[1];Q=R/'cohort_resume'
OUT=R/'visual_review';OUT.mkdir(exist_ok=True)
FF=W/'analysis/plan_base_restart_20261008/model_controls_20261024/bin/ffmpeg'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
frozen=read(Q/'freeze.json');state=read(Q/'status.json');batch=W/frozen['protocol']['batch']
index={x['name']:x for x in read(batch/'index.json')['scenes']}
motion=[]
for scene in frozen['protocol']['scenes']:
    src=batch/'trajectories'/f'{scene}.npz'
    with np.load(src) as z:poses=np.linalg.inv(z['w2c_overrun' if 'w2c_overrun' in z else 'w2c'].astype(float))
    for model,schedule in frozen['protocol']['schedules'].items():
        n=schedule['frames'];p=poses[:n];steps=np.linalg.norm(np.diff(p[:,:3,3],axis=0),axis=1)
        rel=np.einsum('nji,njk->nik',p[:-1,:3,:3],p[1:,:3,:3])
        angles=np.degrees(np.arccos(np.clip((np.trace(rel,axis1=1,axis2=2)-1)/2,-1,1)))
        motion.append(dict(scene=scene,model=model,frames=n,source_sha256=sha(src),
            position_units='reference anchor-depth units, not established metric metres',
            path_length=float(steps.sum()),net_displacement=float(np.linalg.norm(p[-1,:3,3]-p[0,:3,3])),
            median_step=float(np.median(steps)),maximum_step=float(steps.max()),rotation_path_deg=float(angles.sum())))
review=[]
for pair in frozen['pairs']:
    key=pair['key'];ps=state['pairs'].get(key,{})
    if ps.get('state') not in ('completed','confounded','reused'):continue
    scorepath=Path(ps['score']);score=read(scorepath);spec=score['spec'];n=score['horizon']+1
    ids=sorted({0,spec['first_affected_frame']-1,spec['first_affected_frame'],n//2,(3*n)//4,n-1})
    width,height=288,160;pad=24
    sheet=Image.new('RGB',(len(ids)*width,3*(height+pad)+40),'#151922');draw=ImageDraw.Draw(sheet)
    draw.text((6,4),f'{key}: source reference / commanded / measured; frame indices shown',fill='white')
    entries=[]
    row=index[spec['scene']]
    for j,i in enumerate(ids):
        p=Path(row['frames_dir'])/f"frame_{row.get('original_start',0)+i+1:05d}.jpg"
        with Image.open(p) as im:im=im.convert('RGB');thumb=im.resize((width,height))
        sheet.paste(thumb,(j*width,40+pad));draw.text((j*width+4,42),f'reference {i}',fill='white')
        entries.append(dict(frame=i,path=str(p),sha256=sha(p)))
    archives={}
    for ri,arm in enumerate(('commanded','measured'),1):
        archive=Path(spec[arm])/'frames.lossless.mkv'
        select='+'.join(f'eq(n\\,{i})' for i in ids)
        data=subprocess.check_output([str(FF),'-v','error','-threads','1','-i',str(archive),'-vf',f'select={select},scale={width}:{height}',
            '-vsync','0','-threads','1','-f','rawvideo','-pix_fmt','rgb24','pipe:1'])
        rgb=np.frombuffer(data,np.uint8).reshape(-1,height,width,3)
        assert len(rgb)==len(ids),(key,arm,len(rgb),ids)
        for j,(i,frame) in enumerate(zip(ids,rgb)):
            y=40+ri*(height+pad);sheet.paste(Image.fromarray(frame),(j*width,y+pad));draw.text((j*width+4,y+2),f'{arm} {i}',fill='white')
        archives[arm]=dict(path=str(archive),sha256=sha(archive))
    target=OUT/f'{key}.jpg';sheet.save(target,quality=88)
    review.append(dict(key=key,score=str(scorepath),score_sha256=sha(scorepath),frames=ids,archives=archives,
        reference_frames=entries,sheet=str(target),sheet_sha256=sha(target),review_status='awaiting_visual_inspection'))
out=dict(scope='Timestamp-matched sampled visual inspection, not full-video review or a quality metric. Full-frame thumbnails may retain native crop/FOV differences. References are source RGB; stored poses may be VIPE-derived.',
         freeze_sha256=sha(Q/'freeze.json'),script_sha256=sha(Path(__file__)),motion=motion,pairs=review)
(OUT/'manifest.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(sheets=len(review),motion_rows=len(motion),output=str(OUT))))
