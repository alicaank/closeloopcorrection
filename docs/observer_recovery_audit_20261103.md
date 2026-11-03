# Observation rejection and response learning: what the completed pilot exercised

The shared-controller pilot does not isolate a fix for legacy rejoin's failures.
It changes both observation acceptance and recovery behavior. A fresh CPU audit
of all eight prespecified cases, both B/C arms, additionally establishes that
**only two C runs actually issued non-unit response gains**. The other six pairs
have byte-identical RGB archives, identical saved commands and identical
intrinsics/reference plans. Their independent observer-score differences cannot
be attributed to control.

[Replay script](artifacts/observer_recovery_audit_20261103/audit.py) ·
[Measurements, canonical identical-input deltas and 192 input hashes](artifacts/observer_recovery_audit_20261103/audit.json).
The script reconstructs every response-fitting eligibility decision and accepted
fit from recorded issued transforms and measured endpoints using the frozen
implementation. All 136 logged gain vectors reproduce with **zero numerical
difference**. It runs in the original workspace using `baselines/venv_vmem/bin/python`.
This is not new generator inference or a replay of final quantized camera tensors.

## Acceptance and learning are separate

| Case | Accepted observations B / C | C chunks with clipping | C chunks using non-unit gain | B/C RGB |
| --- | --- | ---: | ---: | --- |
| LingBot e43058cb/s7 | 2/10 / 2/10 | 1/10 | 0 | Exact |
| LingBot 74310829/s7 | 1/10 / 1/10 | 1/10 | 0 | Exact |
| LingBot d3f042a2/s107 | 7/10 / 7/10 | 2/10 | 3 | Different |
| LingBot a4e9bdb6/s7 | 7/10 / 8/10 | 1/10 | 7 | Different |
| LingBot e43058cb/s107 | 2/10 / 2/10 | 1/10 | 0 | Exact |
| SANA 0f12b97e/s7 | 6/6 / 6/6 | 6/6 | 0 | Exact |
| SANA a4e9bdb6/s7 | 5/6 / 5/6 | 6/6 | 0 | Exact |
| SANA 0f12b97e/s107 | 6/6 / 6/6 | 6/6 | 0 | Exact |

LingBot B accepts 19/50 observations and C accepts 20/50. SANA accepts 17/18 in
each arm. These rates describe different generated trajectories; they do not
measure a detector's sensitivity or specificity on the legacy failure videos.

The C estimator requires consecutive accepted observations, an unclipped preceding
chunk, sufficient aligned motion, and at least three accepted samples per axis
before changing that axis's gain. Both e43058cb seeds collect one rotation-axis
sample, which never activates a gain. The 74310829 case has no accepted pair.
Every SANA chunk clips, so all otherwise consecutive accepted pairs are excluded.
Thus the SANA comparison did not exercise adaptive response compensation.

The active LingBot runs end with translation gains of 1.2331 on one axis
(`d3f042a2/s107`) and 1.1054/1.1588 on two axes (`a4e9bdb6/s7`). Their recorded
mean-position errors are 0.509845 → 0.506936 and 0.463830 → 0.426052. These remain
descriptive paired results, not proof of an identified differential actuation
gain or a general improvement. Passive closed-loop observations can mix response,
history and observation error.

## Identical inputs must not become apparent ablation effects

SHA256 equality covers each complete compressed `frames.npz` archive, so the
six unchanged pairs contain exactly the same RGB payload. The audit also checks
saved camera commands, intrinsics and plans. Scores share the same reference
batch and evaluation protocol within each pair.

For example, LingBot e43058cb/s107's separately evaluated identical videos differ
by 0.4262° in mean rotation and 1.0194° at the endpoint. That is an evaluator
difference, not a benefit of gain learning. The audit retains both raw scores and
provides a canonical zero C-minus-B delta for identical inputs, using B's existing
evaluation and preserving unavailable metrics. It does not overwrite historical
scores or infer a universal observer noise floor.

## What this means for the next controlled test

The [legacy zigzag audit](zigzag_reassessment_20261103.md) establishes continued
finite-pose acceptance after visible LingBot degradation. The present audit does
not establish that the stronger gate would detect those same corrupted images:
it ran on different B/C continuations. Direct inspection of the existing
`d3f042a2/s107` and `e43058cb/s7` B/C sheets shows recognizable rooms, but substantial
departure from the reference route/layout. Avoiding noise collapse is insufficient
for successful camera control.

A follow-up should vary acceptance and recovery independently:

| Factor | Control | Intervention |
| --- | --- | --- |
| Observation acceptance | Finite-pose acceptance | Causal overlap/depth/image-support rejection |
| Residual recovery | Full rejoin over one native chunk | Half the residual recovered over the next chunk |

All four cells must run the same causal observer calls and use the same candidate
endpoint estimates. The acceptance factor changes only whether those candidates
are used. The recovery factor changes only the residual schedule; it must permit
catch-up and a justified return after overshoot. In the translation-only uniform
case, with residual `e = M - P` and `n` next-chunk steps, the requested step is
`v - rho * e/n`, with `rho=1` versus `rho=0.5`. Recompute from the next accepted
state rather than adding the previous residual again. This is a shared experimental
factor, not a fitted model-specific constant or an established safe envelope.

The original legacy arm also remains as an external comparator: if the common
observer differs from its historical observer, none of the four cells is silently
called an exact legacy replay. Native/sham and prefix-equality checks remain
required. Keep successful legacy cases as well as failures, the moving and
near-stationary strata, original horizons and fixed-time primary metrics.
Report active corrections, observation outages, gain activation, unresolved lag,
native command magnitude and visual damage alongside pose scores. Do not score
stalling or corrupted imagery as success.

This matrix is a design for the next isolated experiment, **not a launched or
GPU-qualified experiment**. The already-frozen representative comparisons continue
unchanged. Spatia's forward commands with backward estimated progress require a
separate continuation/geometry test; the acceptance/recovery matrix does not
resolve that mechanism by itself.
