"""Verified RGB preservation for newly generated experiment frame artifacts."""
import hashlib,json,subprocess
from pathlib import Path
import numpy as np


def rgb_digest(frames):
    return hashlib.sha256(memoryview(np.ascontiguousarray(frames)).cast('B')).hexdigest()


def read_rgb(path):
    import cv2
    cap=cv2.VideoCapture(str(path));frames=[]
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    finally:cap.release()
    if not frames:raise ValueError('Empty or undecodable lossless archive')
    return np.stack(frames)


def archive_rgb(frames,path,ffmpeg):
    frames=np.asarray(frames);path=Path(path)
    if frames.dtype!=np.uint8 or frames.ndim!=4 or frames.shape[-1]!=3:raise ValueError('Expected uint8 RGB frames')
    if path.exists():raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True);h,w=frames.shape[1:3]
    child=subprocess.Popen([str(ffmpeg),'-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r','16','-i','pipe:0','-an','-c:v','ffv1','-level','3','-pix_fmt','bgr0','-threads','2',str(path)],stdin=subprocess.PIPE)
    try:
        for f in frames:child.stdin.write(f.tobytes())
        child.stdin.close()
        if child.wait()!=0:raise RuntimeError('Lossless encoding failed')
    except BaseException:
        if child.poll() is None:child.terminate();child.wait()
        raise
    restored=read_rgb(path)
    if restored.shape!=frames.shape or rgb_digest(restored)!=rgb_digest(frames):raise ValueError('Lossless archive pixel verification failed; originals must be retained')
    receipt=dict(codec='ffv1',shape=list(frames.shape),dtype='uint8',rgb_sha256=rgb_digest(frames),pixel_exact=True)
    path.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt
