# General camera correction: decision rule and next evidence

The research objective is a shared causal method that improves camera tracking
across a representative set of scenes and supported frozen generators. It does
not require every scene, seed or model to improve. Individual scientific losses
remain in the evaluation; they do not automatically veto the method or trigger
a model-specific exception. Conversely, a few large gains cannot establish
generality when all active interventions come from one scene.

Current development snapshot: **16/40 original repeated-memory pairs are scored
and visually reviewed**, all eight LingBot and all eight DreamX pairs. The resumed
main queue has finished with holds. SANA's separate full-horizon residency retry
also failed at the host-memory guard after saving an exact 97-frame prefix.
A [transition-order retry](sana_observer_transition_20261103.md) is now queued.
The MatrixGame observer-lifetime retry is running, and its decoder-only audit
has completed; those results remain separate from the original attempts. SANA, MatrixGame and Lyra retain their original resource failures/holds.
The [original-attempt accounting](primary_cohort_accounting_20261103.md) records
12 active pairs, four exact-RGB abstentions and the 24 cases missing paired scores. See the
[coverage ledger](generalization_coverage_20261103.json) and
[visual audit](multiboundary_visual_audit_20261103.md). The execution narrative
below preserves earlier attempts and is not a live status feed.

## Keep the core method and its extensions separate

The primary candidate remains [measured-memory v1](method_v1.md): infer executed
poses from generated RGB, register geometry using the anchor contract, reconcile
the supported native memory, and retain the native future target policy.
Generator weights remain fixed. No probe generations or fitted per-model gains
are used by that deployed policy. The actual memory consumer and reconstruction
remain adapter-specific; this is not a universal edit to arbitrary KV tensors.

The [fractional-recovery experiment](fractional_recovery_20261103.md) tests an
additional future-command policy. Its two-seed tradeoff—worse mean position,
better endpoint and rotation—does not invalidate memory-only correction. It
does not justify choosing endpoint error after seeing the results, either.
Keep mean fixed-time tracking error primary, with endpoint, rotation, quality,
runtime and failure outcomes alongside it.

Separate three categories:

- Incorrect units, stale cache geometry or changed pre-intervention pixels are
  implementation/attribution problems. Fix and version them before interpreting
  their scores as causal evidence.
- A valid intervention that worsens tracking is a scientific negative result.
  Retain it in the predeclared cohort, without redesigning the entire method
  around that scene.
- Resource failures and unavailable observer registration are missing evidence.
  Report their frequency and causes; never count them as improvements or silently
  drop them from the denominator of attempted cases.

## What the existing prespecified cohort actually establishes

The [reconciled evidence](generalization_evidence_20261103.json) audits all 20
planned pairs in `analysis/measured_memory_v1_20261101/`: five models, two
scenes, two seeds. Eighteen pairs completed; Lyra has one runtime failure and
one subsequent held pair. Ten completed pairs accepted a memory write; eight
abstained. Every active pair is on **0238de14**, and each contains one write.

For identical generated RGB, intrinsics and the fixed scene reference, the
report reuses the commanded-arm estimate while preserving the original measured
score for provenance. This removes evaluator-only differences from identical
policy outputs. Invalid reference registration remains unavailable. W/L/T
below means strict numeric wins/losses/ties, not statistical significance.

| Model | Complete / planned | Active / abstained | Mean-position W/L/T/unavailable | Mean-rotation W/L/T/unavailable | Active mean-position reduction |
| --- | --- | --- | --- | --- | --- |
| LingBot | 4/4 | 2/2 | 2/0/1/1 | 1/1/2/0 | +24.82% to +43.43% |
| SANA | 4/4 | 2/2 | 2/0/0/2 | 0/2/2/0 | +81.39% to +84.81% |
| DreamX | 4/4 | 2/2 | 0/2/0/2 | 0/2/2/0 | −1.64% to −0.35% |
| MatrixGame | 4/4 | 2/2 | 0/2/1/1 | 2/0/2/0 | −69.98% to −34.59% |
| Lyra | 2/4 | 2/0 | 2/0/0/0 | 0/2/0/0 | +66.08% to +74.23% |

These active ranges are descriptive results on one development scene. They are
not five-model generalization estimates, and no raw error average is pooled
across incompatible native horizons or units. The linked evidence carries the
existing sampled visual reviews, including LingBot's ghosting and MatrixGame's
unresolved causal-host seam. Lyra's small absolute angular changes must also be
read in absolute degrees, not inflated into a large practical penalty solely
through percentage change. No new full-video visual review is implied here.

The original attempt at the separate 40-pair repeated-memory cohort stopped at a host-memory guard after
one completed LingBot pair. That pair's lower pose error accompanies visual
collapse and remains a policy failure. The queue's PID is gone and its recorded
state is terminal `held`; it is not a running experiment. This is distinct from
the 18 completed short pairs above.

## Broaden evidence without demanding perfection

Preserve the 40-pair cohort's four selected scenes, two seeds and five models.
Do not replace difficult scenes using observed scores or observer acceptance.
Restore the execution path under a new frozen attempt, retaining prior outputs
and source hashes. The next resource qualification retries the first failed
SANA case on its original 121-frame horizon; it does not replace the broader
cohort with SANA alone.

For the eventual broader evaluation:

1. Freeze the algorithm, shared settings, model eligibility, scene selection,
   target timestamps and primary metric before scoring.
2. Report paired outcomes within each model, then aggregate with equal model
   weight using predeclared dimensionless summaries. Show means and distributions,
   not only win rates; avoid unstable percentage ratios near zero baseline error.
3. Keep all attempted cases visible, including abstentions, observer failures,
   visual collapse and runtime failures. Report quality separately from pose.
4. Estimate uncertainty across scenes and seeds rather than treating correlated
   frames as independent samples. Two development scenes are insufficient for a
   strong population claim. Use a separate held-out split after development.
5. Judge the primary method by aggregate benefit and transparent limitations.
   A negative scene need not cause rejection; a systematic quality regression
   or a result dominated by one model must remain explicit in the claim.

The [matched visual and reference-motion audit](multiboundary_visual_audit_20261103.md)
confirms the retained LingBot collapse and records target motion at each native
horizon. Scene labels must not hide motion changes later in a longer rollout.

## More general extensions worth testing

These are hypotheses, not implemented or validated improvements. Each should
be a separate version/ablation against the frozen core.

**Joint history reconciliation.** Use overlapping observed frames to infer one
coherent retained pose/depth history, then rebuild every affected native memory
consumer. Use image evidence to correct past geometry; do not fit past poses to
the desired trajectory. Preserve a fixed anchor and supported units. Compare
full relative-history correction with boundary-only correction under equivalent
future requests. This isolates repairing internal geometric inconsistency from
merely changing the next effective camera command. Pose-graph optimization by
itself is established machinery, not a novelty claim.

**Correction in the consumed representation.** Compare candidate corrections in
the actual ray maps, relative cameras or normalized camera tokens, relative to
the plan/history envelope. A common positive geometry scale must be applied
consistently to every affected history and future consumer. The hypothesis is
that such a constraint is more transferable than fixed translation multipliers.
Conditioning distance is only a proposed support heuristic; it does not prove
training support or differential command authority. Candidate ranking can use
deterministic encodings without extra generation probes.

**Adaptive recovery duration.** Seek the shortest supported return to the
original timestamped trajectory instead of always removing a fixed residual
fraction. Treat continuity as a measured tradeoff, rather than assuming a video
generator must obey the previous pilot's hard physical acceleration model.
The test must allow useful return after overshoot and measure repeated-boundary
behavior. Geometric feasibility is not a prediction that the generator will
execute that command; inverse adaptation still needs separate identification.

## Novelty must concern the mechanism

Training-free general camera control already exists: CamTrol constructs
camera-conditioned layouts and noisy latent priors.
[Primary paper](https://arxiv.org/abs/2406.10126).
Warp-as-History constructs target-aligned camera-warped pseudo-history for a
frozen model and also studies a separately trained LoRA extension.
[Primary paper](https://arxiv.org/abs/2605.15182).
Closing the Loop retrieves historical latents and biases attention using
engine-provided camera poses and metric depth.
[Primary paper](https://arxiv.org/abs/2607.21848).

A candidate distinction is **causal estimation of executed geometry from
generated RGB and consistent reconstruction of existing native geometric memory
across heterogeneous frozen generators**. The observations, assumptions and
actual intervention differ from supplying known engine geometry or constructing
pseudo-history from the desired target. That is a mechanism to compare directly,
not evidence that it is first. This focused source check is not an exhaustive
literature review. We should not require an elaborate new controller if the
simpler measured-memory method already demonstrates broad benefit.

## Current implementation and execution progress

The SANA callback closes its auxiliary endpoint observer after each measurement,
including failure, before the next native chunk. That first resource retry
still stopped at the host-memory guard, at a later chunk than the original
failure. It is an execution failure, not a scientific negative or a completed
121-frame experiment. Its partial outputs and terminal state remain preserved.

The next resource overlay releases consumed references to old per-layer KV
history before constructing replacement history, and releases each consumed
chunk buffer after copying it. It changes no cache length, precision, tensor
arithmetic, model forward call or camera policy. Two regressions check lifetime
release and fail-closed source matching. The history-release revision passed **130 CPU tests**;
216 cases using the actual native history-update block have bitwise-identical
tensors, shapes, dtypes and frame counts. These checks do not prove GPU parity.

The new frozen attempt is
`analysis/memory_generalization_20261103/cohort_resume/`. It retains all original
**40 pairs: five models × four scenes × two seeds**, with their original native
horizons. The completed original LingBot pair is reused with content hashes;
its visual collapse remains a policy failure. The other 78 generation jobs are
scheduled. The first new SANA baseline must match every available RGB frame of
the preserved preceding resource retry before its measured arm runs. A resource
prefix mismatch or host-memory guard holds SANA while other models continue.
Disk and source-integrity guards still hold the whole queue. Scientific losses
remain results and do not trigger model holds.

The second SANA baseline also hit the host-memory guard after exporting 97
frames, so the 121-frame resource qualification failed and SANA is held. All 97
completed RGB frames match the prior retry exactly (maximum byte difference 0;
the NPZ archives also have identical SHA256). This establishes partial-prefix
equivalence only, not full-horizon qualification. The queue continued to DreamX:
its commanded arm and lossless retention completed, and its measured arm was
confirmed running. The remaining model/scene scope is preserved. See the
[dated execution snapshot](generalization_execution_20261103.json); it is not a
live status feed. The [audit artifacts](artifacts/generalization_20261103/README.md)
include the preparation/queue/parity scripts and CPU evidence. Broad evaluation
and the proposed method extensions remain unfinished.

Additional SHA256-identical completed output archives were hardlinked while
preserving every path and byte: the first cleanup recovered 0.434 GiB and the
second 2.135 GiB. Free space was 6.6 GiB at launch. At that earlier check no additional DL3DV
scenes were removed. A later user-authorized cleanup removed resized image
payloads and completion markers from 53 unused scenes, releasing 2.32 GiB.
Camera/crop metadata, active inputs, test data and protected model dependencies
were retained; checkpoint hashes and retained trees were verified unchanged.
The [cleanup receipt](artifacts/generalization_20261103/dl3dv_cleanup_receipt.json)
records the exact manifest hash and post-deletion checks. Historical replay of
those unused resized scenes now requires reconstructing their image payloads. Workspace receipts
are `analysis/memory_generalization_20261103/deduplication*.json` and
`storage_check_20261103.json` in that directory.

### Observer-time residency supplement (queued, not GPU-qualified)

The next opt-in resource change moves idle SANA stage-1 weights from CPU to GPU
before **either** observer starts, and closes that observer before restoring the
original CPU placement. It moves devices only, without dtype conversion or model
forwards. The context restores placement on measurement and transfer failures and
rejects mixed original devices. It drops temporary parameter/buffer references
before transfer so they do not retain old CPU buffer storage. A per-callback log
records host/GPU memory before, during and after placement.

`CAMERA_SANA_OBSERVER_GPU_RESIDENCY=1` enables this experimental resource option;
it is disabled by default. It is not an online control gain or calibration
constant. **135 CPU tests pass**, including observer construction/closure ordering
and restoration on exceptions. The native SANA overlay compiles. GPU capacity,
full-horizon RGB equivalence and measured-arm completion remain unverified.

The separate `sana_observer_residency/` attempt is queued behind the running
cohort's shared GPU locks. It contains **all eight original SANA pairs** (four
scenes × two seeds; 16 generations). The first case must complete 121 frames,
match the preserved 97-frame baseline prefix, complete the measured arm and pass
paired-prefix attribution before proceeding. All scientific outcomes remain
reportable. This supplement does not replace the original 40-pair denominator,
shorten horizons or alter the currently running frozen sources.

The first longer DreamX pair has now completed and received sampled visual
inspection: four writes, exact prefix, a 1.76% mean-position reduction and a
0.0143° mean-rotation reduction. Both outputs look similar and drift relative to
the reference. No regeneration floor or population confidence interval was
estimated; treat this as a small descriptive result. Details are in the
[visual audit](multiboundary_visual_audit_20261103.md).

### MatrixGame decoder attribution (queued)

The saved latent history permits a [decoder-only audit](matrix_decoder_audit_20261103.md)
of all four original MatrixGame short pairs, both arms, without new generation.
All eight GPU block replays now reproduce the original causal RGB exactly.
Whole-history decoding of the same latents reduces the 84→85 RGB jump by
46.6–91.5%, and inspected seam frames show a suppressed contrast/saturation pulse.
This establishes a decoder/export contribution. Six exports also change earlier
emitted pixels slightly, so the offline joint output cannot replace causal
observations or scores. A causal host repair remains unqualified. The ongoing
full-horizon cohort source and original scores are unchanged.

### Full-cohort coverage accounting

The [40-pair coverage snapshot](generalization_coverage_20261103.json) keeps each
original scene/seed/model case visible. The latest linked snapshot has 16 scored
and reviewed pairs, including one reused LingBot pair. The main queue finished
with SANA, MatrixGame and Lyra held after one original execution failure each.
SANA's residency retry failed and its transition-order retry is queued; the retries appear
beside original failures/holds rather than replacing them. Four completed pairs
accepted no writes and have unavailable position metrics; every decoded RGB
frame is exact between their arms. Coverage is not a count of successful control outcomes.

### MatrixGame observer-lifetime supplement

The new MatrixGame measured arm also reached the host-memory guard, after its
third block. Its 337-frame commanded arm completed, but the pair is incomplete.
The main queue now holds MatrixGame and proceeds with other models. A separate
[observer-lifetime retry](matrix_observer_lifetime_20261103.md) preserves all
eight original Matrix pairs and requires full-baseline RGB equality before
its first measured run. Closing both auxiliary workers between callbacks is
CPU-tested; GPU recovery is pending. The 40-pair accounting retains the failed
attempt and does not substitute a shorter horizon.
