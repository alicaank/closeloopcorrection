# SANA observer-transition resource retry

The preceding observer-residency attempt failed the host-memory guard at
**2,000,998,400 available bytes**. It saved 97 RGB frames, byte-identical to the
earlier preserved 97-frame baseline archive (SHA256
`a73bf0362b8408ba73d5b1d2c32fbf0adfcfd3db6f1543939beec5fd11f4d4fb`).
Only three observer-placement cycles completed, at boundaries 24, 48 and 72;
there is no fourth `before` residency record. This is partial-prefix equivalence,
not completion or qualification of the 121-frame run. The failed attempt and
its original outputs remain unchanged.

## Targeted placement change

Previously the endpoint callback saved its RGB prefix, moved the decoder to
CPU, and then moved idle stage-1 weights to GPU for observation. The new order
moves stage-1 weights first, before decoder CPU offload. This targets a possible
host-memory peak when both models temporarily occupy CPU memory. The observed
failure interval supports this hypothesis but does not prove the exact allocation
site; no stack was captured by the old guard.

The measured-memory hook keeps its own observation/replay placement scope and
runs before the endpoint placement scope. On the updated driver it therefore
runs while the decoder remains on GPU. This avoids nesting the cache replay's
CPU restoration inside the endpoint observer scope. The legacy driver marker
remains supported. GPU capacity for the changed overlap still requires validation.
The order of memory observation/replay before endpoint observation is retained.

Model weights, dtype, native cache length, denoising operations, camera policy
and requested timestamps are unchanged. The shared observer-residency option
remains opt-in. The new launcher registers a diagnostic stack-dump signal; a
guarded generation stop requests the main Python stack before termination.
This adds evidence for a further failure without relaxing memory reserves.

## Qualification and scope

The separate frozen attempt contains **all eight original SANA pairs**, each
at **121 RGB frames**: 16 generations and 49 total jobs. Its first commanded
run must finish all 121 frames and exactly match the preserved 97-frame prefix.
The measured run must then finish and pass paired-prefix attribution. A failed
resource or qualification gate holds the attempt. No shorter clip replaces an
original evaluation case, and earlier failures are not erased.

All **153 repository CPU tests pass**, including actual bundled-callback
ordering and restoration when decoder offload or measurement fails. The native
SANA overlay compiles. An initial validation run caught stale bundled hashes;
the manifest was updated and that failed test log was retained. These checks
do not establish GPU parity, memory recovery, or efficacy.

Workspace: `analysis/memory_generalization_20261103/sana_observer_transition/`.
Manifest SHA256:
`3a28b857d13a5be4147811d57f2afbe76e94ddbcbf20a09587b3331a7755bb81`.
The queue process was confirmed alive waiting for the standard GPU locks.

[Protocol](artifacts/sana_observer_transition_20261103/protocol.json),
[complete delta from the failed attempt](artifacts/sana_observer_transition_20261103/review_delta.diff),
[CPU evidence](artifacts/sana_observer_transition_20261103/cpu_validation.json),
and [queue snapshot](artifacts/sana_observer_transition_20261103/queue_snapshot.json)
record this resource change separately from the frozen scientific method.
