"""Qualification of existing complete outputs; no source directory is modified."""
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def receipt_paths(model, directory):
    d = Path(directory)
    if model == 'lingbot':
        return {model: d.parent/f'memory_policy_{model}.json'}
    result = {model: d/f'memory_policy_{model}.json'}
    if model == 'matrixgame': result['matrix_child'] = d/'raw/memory_policy_matrix_child.json'
    if model == 'lyra': result['lyra_child'] = d/'memory_policy_lyra_child.json'
    return result


def eligibility(case, arm, directory, expected_hashes):
    """Metadata/source qualification only; raw RGB still needs decoding/checking."""
    d = Path(directory)
    try:
        meta = json.loads((d/'run.json').read_text())
        if (meta.get('model'), meta.get('scene'), meta.get('seed'), meta.get('arm')) != (
                case['model'], case['scene'], case['seed'], 'baseline'):
            return False, 'reference_run_identity_mismatch'
        expected_policy = 'measured' if arm == 'M' else 'commanded'
        for kind,p in receipt_paths(case['model'], d).items():
            receipt = json.loads(p.read_text())
            if (receipt.get('model') != kind or receipt.get('memory_policy') != expected_policy
                    or receipt.get('completed') is not True):
                return False, 'reference_policy_or_completion_mismatch'
            if receipt.get('overlay_sha256') != expected_hashes[kind]:
                return False, 'reference_generated_source_mismatch'
        return True, 'metadata_and_parent_source_match'
    except (FileNotFoundError, KeyError, ValueError):
        return False, 'reference_missing_or_incomplete'


def artifact_files(model, directory):
    d = Path(directory)
    files = [d/'run.json', *receipt_paths(model,d).values()]
    if (d/'frames.npz').is_file(): files.append(d/'frames.npz')
    elif (d/'frames.lossless.mkv').is_file(): files.append(d/'frames.lossless.mkv')
    else:
        meta = json.loads((d/'run.json').read_text())
        files += sorted((d/(meta['scene']+'_frames')).glob('frame_*.png'))
        if len(files) == 1+len(receipt_paths(model,d)): raise ValueError('Missing lossless RGB')
    if (d/'retention.json').is_file(): files.append(d/'retention.json')
    return files


def verify_binding(binding):
    for path,h in binding['files'].items():
        if sha(path) != h: raise ValueError('Bound output changed: '+path)
