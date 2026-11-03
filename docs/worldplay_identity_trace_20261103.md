# WorldPlay identity diagnostic with intermediate receipts

The prior full-horizon A/S pair used identical commands but its decoded prefixes
first diverged between frames 45 and 60. That prevents admitting its correction
arm. [The completed review](online_response_review_20261103.md) records the
evidence and excludes camera-based memory selection as the initial cause.

## Frozen diagnostic

`analysis/worldplay_identity_trace_20261103/` repeats A and S on `0238de14`,
seed 7, 20 chunks / 317 RGB frames, at the original 544x960 resolution. It retains
the original four denoising steps, guidance 1, memory behavior, context window,
native command mapping, observer, and command-response gates. No correction
arm runs in this diagnostic, even if A/S matches. The previous failed run is
preserved.

The launcher wraps the frozen host at runtime with the opt-in
`camera_loop.native_trace.worldplay_trace` context. Production adapters are
unchanged. It saves SHA256 descriptors rather than raw intermediate tensors:

- Incoming native camera/intrinsic/action tensors and exposed pipeline context.
- Prepared initial noise and explicit generator state.
- Every transformer call's inputs and outputs.
- Every latent decode's output and exposed decoder state.
- Per-frame pre-export RGB hashes at each observer boundary, plus context before
  and after observation.
- Python, NumPy, Torch CPU and initialized CUDA RNG states at each trace event.

Hooks return no replacement outputs. The wrapper returns each original result
and removes wrappers/hooks on exit, including exceptions. Hashing detaches and
copies tensors to CPU without changing dtype; BF16 and noncontiguous tensors
are supported. It does not reseed or enable deterministic kernels.

Tracing synchronizes device work and changes allocation/timing. CPU output/RNG
parity is verified, but GPU parity is not established. If the traced pair
matches, that alone does not fix or explain the original failure. The report
compares traced A with the original A prefix receipts as a separate diagnostic.
Opaque objects are explicitly marked `untraced_type`; module-private caches
and weight contents are not exhaustively hashed. Checkpoint size/mtime and
source hashes remain bound by the inherited launch manifest.

## Qualification and resources

The comparison verifies completion of all 20 observation boundaries, 80
transformer calls, 80 latent decodes and one initial-noise preparation per arm.
It reports the first differing value and the first difference per event and
RNG stream. Initial RNG differences before the host's seeding are not themselves
a causal explanation. Final A/S native-command and decoded-prefix equality
remain separate mandatory checks; no arbitrary RGB tolerance replaces them.

The queue preserves the 8 GiB launch, 4 GiB runtime disk and 2 GiB host-memory
guards, the original five shared GPU locks, and an eight-hour job timeout. It
waits for disk space before taking locks. The first status verification found
about 6.5 GiB free and a live `waiting_for_disk` process; no diagnostic generation
had begun.
Other running/queued experiments were not modified. These are dated launch
facts, not a live status feed.

Validation passed all 158 repository CPU tests, including three new tracing
tests; frozen tracing tests, native WorldPlay host CLI import, and frozen-module
binding also passed. All 775 parent source/input hashes were checked before
freezing; the new manifest binds 780 source/input paths. Its SHA256 is
`a85495a8ea5a0bf23918c106fcc6e3dd1a92b2b12cd386ae947296b4b599a6f8`.

- [Launch receipt](artifacts/worldplay_identity_trace_20261103/launch.json)
- [CPU qualification](artifacts/worldplay_identity_trace_20261103/cpu_validation.json)
- [Repository test log](artifacts/worldplay_identity_trace_20261103/repository_tests.log)
- [Frozen manifest](artifacts/worldplay_identity_trace_20261103/manifest.json)
- [Diagnostic launcher](artifacts/worldplay_identity_trace_20261103/trace_launcher.py)
- [Comparison](artifacts/worldplay_identity_trace_20261103/compare.py)
- [Queue](artifacts/worldplay_identity_trace_20261103/run_queue.py)
- [Preparation](artifacts/worldplay_identity_trace_20261103/prepare.py)

No new GPU identity outcome, physical-unit calibration or correction benefit
is established by this preparation.
