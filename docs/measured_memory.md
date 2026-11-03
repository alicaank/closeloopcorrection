# Measured-pose memory integration

This is an optional adapter contract. The main Method is command rejoin; the
reported VMem Method result includes these writes. Other memory experiments are
not universal defaults. See [retained ablations](ablations.md).

The runtime in `analysis/measured_memory_20261101/` adds a separate memory
policy to seven native adapters. It observes already generated frames and uses
accepted poses when native history is consumed. It does not run calibration
probes, train weights, or infer a causal control gain from an execution ratio.

The [latest mechanism audit](https://github.com/alicaank/closeloopcorrection/blob/b467679a28243241c3dcf5156f1c980540d879cd/docs/control_mechanism_audit_20261103.md) identifies
severe LingBot visual collapse after accepted writes in two seeds. Measured
translation conditioning reaches 70–106 times the original plan's maximum
normalized step. Existing observer/SE(3) checks do not reject this situation;
the lower pose scores are not successful correction. The frozen runtime below
does not yet implement an input-range gate or a validated collapse detector.

`camera_loop.memory.stage_memory_write` validates timestamps, observer acceptance,
finite rigid poses, and the fixed initial anchor. Rejection preserves the existing
poses. An identity write preserves their original dtype and values. This function
only stages a write: the native adapter must commit the corresponding cache or
geometry before claiming that memory uses measured labels.

| Adapter | Memory consumer | Write mechanism |
|---|---|---|
| DreamX | Native clean-context KV pass | Encode the finalized chunk's measured cameras through native Camera/PRoPE conversion before the existing cache forward. Preserve future camera tensors. |
| LingBot | Native clean-context KV pass | Convert measured poses into native framewise, normalized ray embeddings; replace the current clean pass's conditioning. |
| SANA-WM | Stage-1 clean KV cache | Save the pre-write cache, observe the decoded/refined chunk, replay the clean pass with measured camera/ray tensors, and replace that chunk's complete cache. Restore the shared camera tensors afterward. |
| Matrix-Game 3.5 | Clean context, camera-history pool, source frustum cache | Measure the previous 84-frame section; update the four clean context poses and full section's registration extrinsics before native depth-to-world reconstruction. |
| Lyra-2 | Sparse3DCache and dense camera history | Update all completed chunk cameras used by temporal Plucker conditioning, and recompute existing sparse entries through native `update_by_frame_id` using retained depth and intrinsics. The terminal camera also supplies future bridging. |
| Astronex | Conditional/unconditional KV, PRoPE mirror, appearance-memory commit | Decode the completed chunk before both clean cache forwards; write measured cameras and reconstruct the native anchored PRoPE window. |
| WorldWarp | Source depth-to-splat reconstruction | Replace the complete source-window poses, relative to its measured last frame, before native splat construction. Native pose optimization may subsequently refine them. |

These hooks do not pretend that KV caches contain an editable pose metadata field.
Pose-dependent activations are recomputed at the native consumer. SANA's second
stage retains its image-conditioning cache; only its pose-conditioned first stage
is replayed. WorldWarp performs a geometric source-pose substitution, not a KV
rewrite. Source checks fail if the audited native hook locations change.

## Running locally

Use the model's existing Python environment and dependencies. The new launcher
defaults to `measured`; legacy entrypoints retain their existing behavior.

```bash
python analysis/measured_memory_20261101/launch.py lingbot \
  --memory-policy measured -- \
  --batch BATCH --scene SCENE --out NEW_OUTPUT --frames 29 --arms baseline

python analysis/measured_memory_20261101/launch.py dreamx \
  --memory-policy measured -- \
  --batch BATCH --scene SCENE --out NEW_OUTPUT --latents 42 --seed 7
```

Models: `dreamx`, `lingbot`, `sana`, `matrixgame`, `lyra`, `astronex`, `worldwarp`.
Arguments after `--` belong to the existing driver, except DreamX's rollout route
which accepts `--latents` and has no probe-amplitude argument. SANA requires
`--arm baseline` or `correction` for serial per-chunk callbacks. Astronex retains
its `LONG_BATCH`, `--arms`, and `--videos` interface.

The repository now includes the seven source-only runtime files in its
hash-checked template bundle. `camera-loop prepare` materializes them in a fresh
workspace under `analysis/measured_memory_20261101/`. Run that workspace's
`launch.py` with the model's native interpreter after configuring its dependencies
and assets. For example:

```bash
/opt/envs/lingbot/bin/python \
  /data/camera-work/analysis/measured_memory_20261101/launch.py \
  lingbot --memory-policy measured -- \
  --batch /data/eval-batch --scene SCENE --out /data/results/NEW_OUTPUT \
  --frames 29 --arms baseline
```

`--check-source` checks audited driver insertion points without loading weights.
The relocated runtime uses a bundled copy of the portable memory staging
contract, so every model environment does not need an editable package install.
The native observer still needs its configured interpreter and model assets.
[Source provenance](https://github.com/alicaank/closeloopcorrection/blob/b467679a28243241c3dcf5156f1c980540d879cd/docs/measured_memory_source_provenance.json) records the original
hashes and export transformations. Local GPU qualification does not establish
GPU portability of a newly materialized workspace.

- `commanded`: existing native memory behavior.
- `identity`: intercept the memory site using its existing labels; do not observe
  or rebuild memory solely for this control.
- `measured`: independently guarded DA3 measurement in anchor-depth units, then
  a native memory write if accepted.

Use `baseline` to isolate memory actuation. `correction` also enables the existing
driver's future-command controller; it is not a new adaptive controller. Where
an absolute camera bridge is needed, it uses the successfully stored boundary
pose, including the commanded fallback after a rejected write. Issued command
history remains separate from measured memory labels.

`memory_events.jsonl` records frame IDs, acceptance/rejection, stored poses, and
the mechanism. `memory_observer/measurement_*.json` retains observer quality
diagnostics. No write uses frames beyond the generated prefix. The observer
requires its explicit quality gate to pass, but this is not a substitute for a
validated scene-collapse detector or a tracking-efficacy evaluation.

On the local 62-GiB host, SANA's observer is closed before replay and its endpoint
observer is serialized with the memory observer. Keeping both resident caused a
kernel-confirmed host OOM. Lyra similarly releases the external memory observer
before restoring diffusion. Restarted observers use separate session directories
so later chunks do not overwrite earlier measurement receipts.

## Validation and scope

Engineering checks and GPU receipts are under
`analysis/measured_memory_20261101/`. Each `smoke.py` run snapshots its runtime
sources and uses new output directories; prior experimental sources and weights
are not edited. See `validation_summary.json` there for the latest completed
tests and limitations.

All seven adapters completed a native GPU measured-memory write test. The CPU
suite passes 19 tests covering staging, source contracts, and memory consumers.
These are short engineering runs, not a multi-scene tracking evaluation.

DreamX's initial 33-frame test passed exact commanded/identity pixels, latent,
RNG, and cache receipts. A measured write passed the quality gate; the first 21
RGB frames remained exact, with changes starting in the next chunk. LingBot's
29-frame test also applied a measured write and preserved its first 13 RGB
frames. A changed continuation demonstrates actuation, not improved tracking.

SANA additionally passed a forced identity replay: the recomputed native cache
was tensor-exact and all 49 RGB frames matched the no-replay identity run. This
test uses `smoke.py --model sana --policies identity --verify-replay --out NEW_OUT`;
the replay verification switch is a test-only assertion, not a deployed control
parameter. Its measured run preserved the first 25 RGB frames. Matrix-Game's
full-section registration run preserved its first 85 RGB frames against the
earlier context-only arm. Astronex's commanded/identity exported RGBs matched;
its raw latent/RNG/cache parity was not measured.

Lyra's full-history run accepted all 80 generated camera labels, updated existing
sparse geometry, and completed 161 frames. Its first 81 RGB frames were exact
against the earlier sparse-cache-only measured arm. WorldWarp accepted all 81
source-window cameras and completed the subsequent chunk after native geometry
reconstruction. Neither comparison establishes improved camera tracking.

The explicit launcher is now included in newly materialized workspaces. The
package's existing `camera-loop run` and `run-batch` routes do not automatically
opt into these new hooks. The portable staging contract is available as
`from camera_loop import MemoryWrite, stage_memory_write, bridge_future`.

## Paired camera evaluation queue

`analysis/measured_memory_20261101/queue_memory_eval.py --out NEW_DIRECTORY`
runs the seven development pairs serially under the existing GPU locks. It reuses
verified generated outputs and creates the four missing commanded controls.
The frozen protocol records sources, scenes, seeds, intervention boundaries,
and output paths. Existing outputs are never overwritten; failures are recorded
while independent model pairs continue.

`score_memory_pair.py` runs a separate VIPE estimator on both arms, using only
RGB and intrinsics for SLAM and anchor depth for scale registration. Primary
metrics cover the continuation after the first memory intervention. A rejected
registration leaves position scores unavailable. Raw prefix mismatch marks a
comparison confounded; lossy Astronex export cannot establish raw prefix parity.
Reference batch poses may themselves be VIPE-derived, so this is observer
separation, not an independent ground-truth claim.

The active first queue is `analysis/measured_memory_20261101/efficacy_queue_20261101/`.
Its `status.json` records job progress; `RESULTS.md` and `results.json` are updated
by `report_memory_eval.py --watch`. These short, single-scene/seed comparisons
do not establish generalization, statistical significance, or visual quality.

For a self-contained paper-writing reference, including the distinction between
execution ratio and corrective authority, see [the method reference](paper_method_reference.md).
