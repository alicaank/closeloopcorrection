"""Decode all RGB frames to verify zero-write fallback equality, using bounded RAM."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import time

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
FF = WORKSPACE / 'analysis/plan_base_restart_20261008/model_controls_20261024/bin/ffmpeg'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument('--review-dir', type=Path, default=ROOT / 'visual_review')
args = parser.parse_args()
manifest = json.loads((args.review_dir / 'manifest.json').read_text())
for pair in manifest['pairs']:
    score = json.loads(Path(pair['score']).read_text())
    if score['accepted_memory_writes'] != 0:
        continue
    arms = {}
    for arm, archive in pair['archives'].items():
        path = Path(archive['path'])
        assert sha(path) == archive['sha256']
        # framemd5 records dimensions, frame count, timestamps and every decoded
        # RGB frame's digest without retaining whole videos in host memory.
        cmd = [str(FF), '-v', 'error', '-threads', '1', '-i', str(path),
               '-map', '0:v:0', '-an', '-vsync', '0', '-pix_fmt', 'rgb24',
               '-threads', '1', '-f', 'framemd5', '-hash', 'sha256', 'pipe:1']
        raw = subprocess.check_output(cmd).decode()
        frames = [line.strip() for line in raw.splitlines() if line and not line.startswith('#')]
        headers = [line for line in raw.splitlines() if line.startswith(('#dimensions', '#tb', '#media_type'))]
        assert len(frames) == score['horizon'] + 1
        arms[arm] = dict(archive_sha256=archive['sha256'], frame_count=len(frames),
                         metadata=headers, rgb_frame_sha256=[line.split(',')[-1].strip() for line in frames],
                         frame_records=frames)
    a, b = arms['commanded'], arms['measured']
    equal = a['metadata'] == b['metadata'] and a['frame_records'] == b['frame_records']
    assert equal, pair['key']
    out = dict(key=pair['key'], audited_at=time.time(), score_sha256=pair['score_sha256'],
               script_sha256=sha(Path(__file__)), accepted_writes=0,
               method='All native-resolution frames decoded as RGB24; per-frame SHA256 and dimensions/timestamps compared.',
               full_rgb_exact=equal, arms=arms)
    (args.review_dir / (pair['key'] + '.abstention_rgb.json')).write_text(json.dumps(out, indent=2) + '\n')
    print(pair['key'], 'all', a['frame_count'], 'RGB frames exact', flush=True)
