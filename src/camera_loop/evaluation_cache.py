"""Content-addressed observer results; commands/targets never enter estimation.

The caller must bind evaluator_id to its frozen code/config/weights receipt.
Pose metrics are computed afterward against each experiment's own target.
"""
import hashlib,json,os,shutil,tempfile
from pathlib import Path
import numpy as np


def evaluation_key(frames, intrinsics, reference_depth, *, evaluator_id):
    if not isinstance(evaluator_id,str) or not evaluator_id.strip():
        raise ValueError('A frozen evaluator contract is required')
    h=hashlib.sha256(evaluator_id.encode())
    for name,value in [('frames',frames),('intrinsics',intrinsics),('reference_depth',reference_depth)]:
        a=np.ascontiguousarray(value)
        if a.dtype.hasobject:raise ValueError('Object arrays cannot identify an evaluation')
        h.update(json.dumps([name,a.dtype.str,a.shape],separators=(',',':')).encode())
        h.update(memoryview(a).cast('B'))
    return h.hexdigest()


def cached_estimate(frames,K,depth,output,*,cache_dir,evaluator_id,estimate):
    """Reuse exact observer inputs, preserving all estimator sidecar artifacts."""
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    key=evaluation_key(frames,K,depth,evaluator_id=evaluator_id)
    root=Path(cache_dir);root.mkdir(parents=True,exist_ok=True);entry=root/key
    # Queues serialize GPU work; flock also protects independent cache clients.
    import fcntl
    with (root/(key+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        hit=entry.is_dir()
        if not hit:
            poses,meta=estimate(frames,K,depth,output)
            with tempfile.TemporaryDirectory(prefix=key+'.',dir=root) as td:
                tmp=Path(td)/'entry';tmp.mkdir()
                for p in output.parent.glob(output.stem+'.*'):
                    if p.is_file():shutil.copyfile(p,tmp/('result'+p.name[len(output.stem):]))
                np.savez_compressed(tmp/'registered.npz',poses=poses)
                (tmp/'receipt.json').write_text(json.dumps(dict(key=key,evaluator_id=evaluator_id,metadata=meta),allow_nan=False))
                os.rename(tmp,entry)
        else:
            receipt=json.loads((entry/'receipt.json').read_text())
            if receipt['key']!=key or receipt['evaluator_id']!=evaluator_id:raise ValueError('Cache contract mismatch')
            output.parent.mkdir(parents=True,exist_ok=True)
            destinations=[(p,output.parent/(output.stem+p.name[len('result'):])) for p in entry.glob('result.*')]
            if any(q.exists() for p,q in destinations):raise FileExistsError('Existing estimator sidecar preserved')
            for p,q in destinations:shutil.copyfile(p,q)
            with np.load(entry/'registered.npz') as z:poses=z['poses'].copy()
            meta=receipt['metadata']
    return poses,dict(meta,evaluation_cache=dict(key=key,hit=hit,evaluator_id=evaluator_id))
