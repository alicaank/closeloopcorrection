"""Lossless retention of owned pilot RGB after exact archive round trip."""
import argparse,json,sys
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'source'))
from camera_loop.lossless_archive import archive_rgb
p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--arm',choices=['A','S','F'],required=True);a=p.parse_args()
m=json.loads((R/'manifest.json').read_text());assert a.case in m['protocol']['cases']
src=R/'runs'/a.case/a.arm;assert src.resolve().is_relative_to(R/'runs')
raw=src/'frames.npz'
with np.load(raw) as z:frames=z['frames']
assert frames.shape==(69,704,1280,3) and frames.dtype==np.uint8
receipt=archive_rgb(frames,src/'frames.lossless.mkv',Path('/home/hj/Alican/analysis/plan_base_restart_20261008/model_controls_20261024/bin/ffmpeg'))
receipt.update(replaced_file=str(raw),original_bytes=raw.stat().st_size)
(src/'retention.json').write_text(json.dumps(receipt,indent=2)+'\n');raw.unlink()
