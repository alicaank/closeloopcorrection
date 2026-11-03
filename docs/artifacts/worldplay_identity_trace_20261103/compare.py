"""Compare intermediate traces separately from the required RGB identity gate."""
import hashlib
import json
from collections import Counter
from pathlib import Path

R = Path(__file__).resolve().parent


def difference(a, b, path=''):
    if type(a) is not type(b):
        return dict(path=path, a=a, b=b)
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return dict(path=path, a_keys=list(a), b_keys=list(b))
        for k in a:
            d = difference(a[k], b[k], path + '/' + k)
            if d:
                return d
    elif isinstance(a, list):
        if len(a) != len(b):
            return dict(path=path, a_length=len(a), b_length=len(b))
        for i, (u, v) in enumerate(zip(a, b)):
            d = difference(u, v, path + '/' + str(i))
            if d:
                return d
    elif a != b:
        return dict(path=path, a=a, b=b)
    return None


def main():
    metadata, traces, hashes = {}, {}, {}
    for arm in ['A', 'S']:
        base = R / 'runs/worldplay' / arm
        p = base / '0238de14_s7.json'
        metadata[arm] = json.loads(p.read_text())['online_response']
        hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        p = base / '0238de14_s7.online_units.response/native_trace.jsonl'
        traces[arm] = [json.loads(s) for s in p.read_text().splitlines()]
        hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        counts = Counter(x['event'] for x in traces[arm])
        for event, n in [('native_inputs.before', 20), ('noise.after', 1),
                         ('transformer.after', 80), ('decode.after', 80),
                         ('observer.after', 20), ('completed', 1)]:
            assert counts[event] == n, (arm, event, counts[event], n)
        assert len(metadata[arm]['prefix_hashes']) == 20
        assert metadata[arm]['prefix_hashes'][-1]['frame_id'] == 316
        assert not any(e['applied'] for e in metadata[arm]['events'])
    assert [x['event'] for x in traces['A']] == [x['event'] for x in traces['S']]
    first_values = None
    first_rng = {}
    first_per_event = {}
    for a, b in zip(traces['A'], traces['S']):
        d = difference(a['values'], b['values'])
        if d:
            entry = dict(sequence=a['sequence'], event=a['event'], difference=d)
            if first_values is None:
                first_values = entry
            first_per_event.setdefault(a['event'], entry)
        for stream in set(a['rng']) | set(b['rng']):
            d = difference(a['rng'].get(stream), b['rng'].get(stream))
            if d:
                first_rng.setdefault(stream, dict(sequence=a['sequence'], event=a['event'], difference=d))
    a, b = metadata['A'], metadata['S']
    gate = dict(exact_commands=a['native_commands'] == b['native_commands'],
                exact_prefixes=a['prefix_hashes'] == b['prefix_hashes'],
                mismatch_boundaries=[u['frame_id'] for u, v in zip(a['prefix_hashes'], b['prefix_hashes']) if u != v])
    prior = R.parent / 'online_response_v2_resume_20261103/runs/worldplay/A/0238de14_s7.json'
    old = json.loads(prior.read_text())['online_response']
    hashes[str(prior)] = hashlib.sha256(prior.read_bytes()).hexdigest()
    report = dict(gate=gate, correction_allowed=False,
                  historical_A_prefix_parity=a['prefix_hashes'] == old['prefix_hashes'],
                  first_value_difference=first_values, first_difference_by_event=first_per_event,
                  first_rng_difference_by_stream=first_rng, input_sha256=hashes,
                  limits=['Trace synchronizes and copies tensors; it can change scheduling.',
                          'Pre-seeding RNG differences alone are not a causal explanation.',
                          'Opaque objects are tagged untraced_type; module-private caches and weights are not exhaustively hashed.',
                          'No correction arm is launched even if this diagnostic pair matches.'])
    (R / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
