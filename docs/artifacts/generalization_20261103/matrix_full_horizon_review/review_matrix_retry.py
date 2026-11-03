"""Read-only sampled RGB review of qualified original-horizon MatrixGame retries."""
from pathlib import Path
import hashlib,json,subprocess
import numpy as np
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parent;W=R.parents[1];Q=R/'matrix_observer_lifetime';O=Q/'visual_review';O.mkdir(exist_ok=True)
FF=W/'analysis/plan_base_restart_20261008/model_controls_20261024/bin/ffmpeg'
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
m=read(Q/'manifest.json');items=[]
for spec in m['protocol']['pairs']:
 key=spec['key'];scorepath=Q/'scores'/key/'score.json';gatepath=scorepath.with_name('attribution_gate.json')
 if not gatepath.exists():continue
 score=read(scorepath);gate=read(gatepath)
 assert gate['passed'] and score['comparison_valid'] and score['prefix_rgb_exact']
 assert score['horizon']==336 and score['spec']==spec
 ids=[0,84,85,168,252,336];width,height=256,144;pad=20
 sheet=Image.new('RGB',(width*6,3*(height+pad)+32),'#151922');draw=ImageDraw.Draw(sheet)
 draw.text((5,3),f'{key}: reference / commanded / measured; sampled review only',fill='white')
 index=read(Path(spec['batch'])/'index.json');src=next(x for x in index['scenes'] if x['name']==spec['scene'])
 refs=[]
 for j,i in enumerate(ids):
  p=Path(src['frames_dir'])/f"frame_{src.get('original_start',0)+i+1:05d}.jpg"
  with Image.open(p) as im:sheet.paste(im.convert('RGB').resize((width,height)),(j*width,32+pad))
  draw.text((j*width+3,34),f'reference {i}',fill='white');refs.append({'frame':i,'path':str(p),'sha256':sha(p)})
 archives={}
 for row,arm in enumerate(['commanded','measured'],1):
  p=Path(spec[arm])/'frames.lossless.mkv';select='+'.join(f'eq(n\\,{i})' for i in ids)
  data=subprocess.check_output([str(FF),'-v','error','-threads','1','-i',str(p),'-vf',f'select={select},scale={width}:{height}','-vsync','0','-threads','1','-f','rawvideo','-pix_fmt','rgb24','pipe:1'])
  rgb=np.frombuffer(data,np.uint8).reshape(-1,height,width,3);assert len(rgb)==6
  y=32+row*(height+pad)
  for j,(i,frame) in enumerate(zip(ids,rgb)):
   sheet.paste(Image.fromarray(frame),(j*width,y+pad));draw.text((j*width+3,y+2),f'{arm} {i}',fill='white')
  archives[arm]={'path':str(p),'sha256':sha(p)}
 target=O/f'{key}.jpg';sheet.save(target,quality=90)
 item={'key':key,'frames':ids,'sheet':str(target),'sheet_sha256':sha(target),'score':str(scorepath),'score_sha256':sha(scorepath),'gate':str(gatepath),'gate_sha256':sha(gatepath),'archives':archives,'reference_frames':refs,'accepted_writes':score['accepted_memory_writes'],'delta':score['delta'],'review_status':'awaiting_visual_inspection'}
 rp=O/f'{key}.review.json'
 if rp.exists():
  review=read(rp);assert review['sheet_sha256']==item['sheet_sha256'] and review['score_sha256']==item['score_sha256']
  item.update(review_status='sampled_visual_review_complete',classification=review['classification'],inspection_record=str(rp),inspection_sha256=sha(rp))
 items.append(item)
(O/'manifest.json').write_text(json.dumps({'scope':'Original 337-frame retry; sampled RGB review, not a full-video quality metric. Stored reference poses may be VIPE-derived, not independent GT.','queue_manifest_sha256':sha(Q/'manifest.json'),'script_sha256':sha(Path(__file__)),'pairs':items},indent=2)+'\n')
print('Prepared',len(items),'review sheets in',O)
