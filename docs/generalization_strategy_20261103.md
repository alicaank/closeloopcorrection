# General camera correction: decision rule and next evidence

The research objective is a shared causal method that improves camera tracking
across a representative set of scenes and supported frozen generators. It does
not require every scene, seed or model to improve. Individual scientific losses
remain in the evaluation; they do not automatically veto the method or trigger
a model-specific exception. Conversely, a few large gains cannot establish
generality when all active interventions come from one scene.

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

The separate 40-pair repeated-memory cohort stopped at a host-memory guard after
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

The SANA bundled callback now closes its auxiliary endpoint observer after each
measurement, including failure, before the next native chunk. Previously the
worker moved weights off the GPU but retained them in host RAM. This is a
concrete resource-lifetime issue; it is not yet proven to explain the entire
previous memory-guard event. Memory policy, future targets and native precision
are unchanged. Two tests execute the actual callback with a stub worker and
verify closure and recreation. The full current suite passes 128 tests.

The fresh attempt is under
`analysis/memory_generalization_20261103/sana_lifetime_qualification/` with
source/input hashing, the existing GPU locks, host/RAM guards and resource
telemetry. Two generation arms, verified lossless retention and a paired score
are scheduled. GPU completion and prefix parity remain pending. The exact old
attempt is retained. Broader evaluation and the proposed method extensions
remain unfinished; this document does not declare the research goal achieved.

[Execution snapshot](generalization_execution_20261103.json) and
[audit artifacts](artifacts/generalization_20261103/README.md) bind this progress
to the current source and the preserved cohort. Seven additional pairs of
SHA256-identical completed frame archives were hardlinked with all paths and
bytes preserved, reclaiming 0.434 GiB for the retry; no unique data or model
assets were removed. The workspace receipt is
`analysis/memory_generalization_20261103/deduplication.json`.
