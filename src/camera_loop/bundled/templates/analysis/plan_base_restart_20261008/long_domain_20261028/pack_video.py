"""Uniform CRF12 scoring video; raw outputs kept until successful scoring."""
from pathlib import Path
import argparse,sys,json,hashlib,subprocess
import numpy as np
from PIL import Image
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R.parent/'fair_common_20261025'));import fair_common as fc

def pack(model,source,out,batch,scene,n):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 if (out/'video.json').exists():return
 source=Path(source);row=next(s for s in json.load(open(Path(batch)/'index.json'))['scenes'] if s['name']==scene)
 raw=np.load(source/'frames.npz')['frames'] if (source/'frames.npz').exists() else None
 png=sorted((source/f'{scene}_frames').glob('frame_*.png'))
 if raw is not None:assert len(raw)>=n;frames=(raw[i] for i in range(n))
 elif png:assert len(png)>=n;frames=(np.asarray(Image.open(p).convert('RGB')) for p in png[:n])
 else:
  import cv2
  cap=cv2.VideoCapture(str(source));assert cap.isOpened(),source
  def stream():
   for i in range(n):
    ok,f=cap.read();assert ok,(source,i,n);yield cv2.cvtColor(f,cv2.COLOR_BGR2RGB)
   cap.release()
  frames=stream()
 tmp=out/'review_full.tmp.mp4';dst=out/'review_full.mp4';sha=hashlib.sha256();samples=[];ids=np.unique(np.linspace(0,n-1,9).astype(int));proc=None
 for i,f in enumerate(frames):
  if i==0:
   h,w=f.shape[:2];f=fc.preprocess(model,np.asarray(Image.open(Path(batch)/row['image']).convert('RGB')),(w,h))
   cmd=[fc.FFMPEG,'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r','16','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','12','-pix_fmt','yuv420p',str(tmp)]
   proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
  sha.update(f.tobytes());proc.stdin.write(f.tobytes())
  if i in ids:samples.append(f.copy())
 proc.stdin.close();assert proc.wait()==0
 import imageio_ffmpeg
 count,_=imageio_ffmpeg.count_frames_and_secs(str(tmp));assert count==n
 tmp.replace(dst);np.savez_compressed(out/'review_frames.npz',frames=samples,frame_ids=ids)
 (out/'video.json').write_text(json.dumps(dict(model=model,scene=scene,frames=n,hw=[h,w],source=str(source),source_raw_sha256=sha.hexdigest(),crf=12,frame0='processed conditioning image',scoring='All arms scored on this CRF12 video; prior native mp4 input implies extra encode, identical within pair'),indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in ['model','source','out','batch','scene']:p.add_argument('--'+k,required=True)
 p.add_argument('--frames',type=int,required=True);a=p.parse_args();pack(a.model,a.source,a.out,a.batch,a.scene,a.frames)
