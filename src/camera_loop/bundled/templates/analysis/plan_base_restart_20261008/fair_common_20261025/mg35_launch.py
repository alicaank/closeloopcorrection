"""Launch Matrix-Game-3.5's run_pipeline.py unchanged, except two output-side hooks (generation itself is untouched):
1. _annotate_frames_with_index -> identity: the official history video burns a 52 px frame counter into every frame, which
   would contaminate every image metric.
2. _write_video: for the *_history.mp4 (the pure generated rollout = official result.mp4) also save the decoded uint8 frames
   losslessly as <stem>_raw.npy, so the fair mp4 is encoded once from raw frames (crf 12) instead of from a crf-23 re-encode.
Called by mg35_fair.py through `accelerate launch mg35_launch.py <run_pipeline args>` with cwd = the MG3.5 repo."""
import os, runpy, sys
import numpy as np
REPO = os.environ['MG35_REPO']; sys.path.insert(0, REPO)
import examples.wanvideo.pipeline.mosaic.validation as V
V._annotate_frames_with_index = lambda frames, start_index=0, labels=None, font_size=52: [np.asarray(f) for f in frames]
_write = V._write_video
def _write_video(path, frames, *a, **k):
    if str(path).endswith('_history.mp4'):
        arr = np.stack([np.asarray(f, np.uint8) for f in frames]); np.save(str(path)[:-4] + '_raw.npy', arr)
        print(f'[mg35_launch] raw history frames {arr.shape} -> {str(path)[:-4]}_raw.npy', flush=True)
    return _write(path, frames, *a, **k)
V._write_video = _write_video
sys.argv = [os.path.join(REPO, 'examples/wanvideo/pipeline/run_pipeline.py')] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
