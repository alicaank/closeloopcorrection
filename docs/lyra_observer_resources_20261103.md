# Lyra full-horizon resource retry

The original `02267acf`, seed-7 commanded run stopped at the host-memory guard
after two of four chunks. Minimum available RAM was **1,879,060,480 bytes**.
The failure stack includes a CUDA-to-host copy, but does not identify its Python
caller. No complete RGB baseline was saved. This remains an execution failure
with missing paired evidence, not a scientific null or a camera-control failure.

Source inspection found two avoidable allocations in the external observation
path: the endpoint worker remained resident across native generation, and the
entire growing history was converted through multiple full-size float buffers.
The measured-memory observer's history conversion also used full-size float
buffers. These are plausible contributors; the original trace does not prove
that either alone caused the failure.

The retry changes resource lifetimes and temporary allocation only:

- Endpoint observer construction occurs after native offload, inside RNG
  preservation. The worker closes after each measurement, including failure,
  before native model restoration and continuation.
- Both observer RGB conversions process temporal blocks of eight frames into
  one complete uint8 output. Every original frame and the original resize
  operation remain. Each observer retains its own historical arithmetic order:
  equivalent-looking formulas can round differently.
- Endpoint RGB scratch arrays are released before native continuation. Native
  weights, precision, history/cache contents, targets and decoder operations
  remain unchanged. New runs record per-process RSS samples every five seconds.

The reusable converter is [video_conversion.py](../src/camera_loop/video_conversion.py).
CPU tests compare its output byte-for-byte with both legacy formulas across
float32/float16/BF16, noncontiguous histories, clipping and rounding boundaries.
Callback tests execute the bundled callback with mocked IPC and native models,
checking worker recreation, ordering, and start/measurement failures. All
**148 repository tests pass**; both source overlays compile. These checks do
not establish GPU parity or full-horizon memory sufficiency.

## Qualification and preserved experiment scope

All **eight original Lyra pairs** remain scheduled on the same four scenes and
two seeds, each at **321 RGB frames**. There are 17 generations and 51 jobs:
16 evaluated generations plus one untouched 161-frame commanded reference used
only for resource qualification. The first case must:

1. Complete the original source at 161 frames on `02267acf`, seed 7.
2. Complete the repaired commanded source at 321 frames, matching all 161
   reference RGB frames exactly.
3. Complete the repaired measured arm at 321 frames and pass the ordinary exact
   pre-intervention prefix check before interpreting the paired score.

The reference run does not replace or shorten an efficacy case. Because the
failed original run saved no complete RGB output, later-frame parity against
the original 321-frame execution remains unproven even if this prefix gate
passes. Required qualification failures hold the supplement; original failures,
abstentions and scientific losses are retained.

The queue is frozen in
`analysis/memory_generalization_20261103/lyra_observer_resources/` and was verified
alive waiting for the standard GPU lock. Manifest SHA256:
`3f5c1831d98114571b25a21839fcd19880ad94e73d9a34afac4e183047909d19`.
GPU completion and parity remain pending. The
[40-case coverage ledger](generalization_coverage_20261103.json) records this
supplement separately from the original failure.

[Source diff](artifacts/lyra_observer_resources_20261103/source_change.diff),
[protocol](artifacts/lyra_observer_resources_20261103/protocol.json),
[CPU receipt](artifacts/lyra_observer_resources_20261103/cpu_validation.json),
and [qualification gate](artifacts/lyra_observer_resources_20261103/check.py)
preserve the implementation and scope. These are workspace-specific audit
copies, not standalone portable launchers.
