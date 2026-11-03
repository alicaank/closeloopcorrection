# Legacy rejoin failure investigation — 2026-11-03

Legacy rejoin remains a credible main candidate. Nine of the ten historical
cohort/seed rows improve mean position, but the failure mechanisms are not
interchangeable. The most consequential new finding is a correction to our own
analysis: **DreamX's previously reported 20–24 command reversals disappear when
the actual measured chunk anchor is included.** VMem also needs separate
command-only and command-plus-memory reporting.

This CPU investigation verifies all 80 source score hashes, reproduces their
published scalar metrics, and checks the saved position/rotation error arrays
against the timestamped reference poses. It replays all ten DreamX correction
command sequences and all 16 LingBot baseline/correction pairs. Eight existing
contact sheets covering ten selected cases were inspected directly. This is
not a fresh GPU experiment, a full-video review of every pair, or an independent
ground-truth pose measurement.

[Reproducible audit](artifacts/legacy_rejoin_failures_20261103/audit.py),
[all cases, input hashes and native replays](artifacts/legacy_rejoin_failures_20261103/audit.json),
[visual observations](artifacts/legacy_rejoin_failures_20261103/visual_review.json),
and [complete regression table](artifacts/legacy_rejoin_failures_20261103/regressions.md).
The script uses the original workspace paths; it does not require model inference.

## Scope and failure accounting

There are **21 mean-position regressions among 79 numerically valid pairs**;
SANA `c1496c1b` has unavailable position. Another **eight pairs improve mean
position but worsen the endpoint**. Numerical validity does not establish
trustworthy geometry after visual collapse. These are descriptive counts across
heterogeneous datasets, horizons and observers, not a pooled generalization rate.

| Historical cohort | Seed | Position regressions / valid pairs | Change in cohort mean position |
|---|---:|---:|---:|
| DreamX, RE10K160 | 7 | 1/10 | −32.5% |
| LingBot, RE10K160 | 7 | 2/10 | −46.1% |
| LingBot, RE10K160 | 107 | 2/6 | −29.6% |
| MatrixGame, RE10K160 | 7 | 3/10 | −7.4% |
| MatrixGame, DL3DV480 | 7 | 4/7 | −43.0% |
| Astronex, MinWM76 | 7 | 1/5 | −10.7% |
| VMem, RE10K40, **combined command + measured memory** | 7 | 2/10 | −25.1% |
| Lyra, VIPE v3, 480 | 7 | 1/5 | −54.5% |
| SANA, VIPE v3, 160 | 7 | 2/9 | −52.3% |
| Spatia, VIPE v3, 480 | 7 | 3/7 | +6.1% |

Positive changes mean worse error. Each mean is within its own cohort. In
particular, MatrixGame's DL3DV mean improves substantially even though four of
seven cases regress: large improvements on a few high-error cases dominate the
mean. Both the mean and the per-case distribution belong in the paper.

## DreamX: large recovery requests, not the claimed command reversal cycle

The old diagnostic differenced consecutive entries in `run.json.commands`.
However, `dreamx_run.py` temporarily replaces the preceding camera pose with the
measured pose before encoding each future chunk. Only the future tensor slice is
committed. The stored absolute labels omit that temporary anchor.

For the first step, the correct intended displacement is `D_1 − M`, not
`D_1 − U_previous`. Subsequent steps are `D_i − D_(i−1)`. The new audit replays
the original `common.rejoin` and checks every stored command to 1e−12, then
executes the released `Camera` and float32 `get_relative_pose` functions. Their
relative-pose construction agrees with the explicit transform within 5.9e−8.
This checks intended native-step geometry before BF16 tensor quantization; it
does not replay archived conditioning tensors or infer physical camera motion.

Across **all ten runs**, the saved-label diagnostic gives 20–24 adjacent
translation reversals; measured-anchor reconstruction gives **zero**. Therefore
the earlier “bounded control removed 24 reversals” comparison does not establish
a benefit in the actual command stream. Generated trajectories may still exhibit
instability, but that requires separate evidence.

The failing `e43058cb/s7` still has genuinely large intended steps: median
**5.80×**, p90 **17.32×**, maximum **23.43×** the corresponding planned step.
The old 36.80× maximum included the spurious boundary-label jump. Position
error rises **0.3863 → 0.4509**, while rotation improves **36.09° → 16.05°**.
The sampled images preserve a plausible dining room but both arms fail to follow
the reference into the kitchen. Correction also worsens the endpoint by 118.2%.

**Supported diagnosis:** aggressive short-horizon recovery, remaining three-frame
observation delay, and a translation/rotation tradeoff on a difficult scene.
Which of observer error, command response and history causes the regression is
not isolated. Passive path ratios do not justify an inverse gain, and these logs
do not establish a relay-like plant or differential gain.

## LingBot: real visual collapse after large native commands

Fresh replay of the 16 paired runs verifies the initial-prefix hashes,
normalizers, camera timestamps and rejoin endpoints. Unlike the erroneous
DreamX diagnostic, this replay already uses the measured pose for the first
native relative step.

The `d3f042a2` seed comparison is especially informative:

| Same scene | Maximum normalized step | Mean position change | Visual observation |
|---|---:|---:|---|
| seed 7 | 2.85 | −86.8% | Earlier review remains recognizable |
| seed 107 | 17.07 | +91.7% | Degradation visible at frame 48; noise collapse by frame 64 |

For seed 107, the step after frame 44 reaches 3.95; the next correction after
frame 60 reaches 17.07. Whole-run maxima alone are not trigger evidence, but this
chronology supports excessive command magnitude as a contributor. `e43058cb/s7`
and `74310829/s7` also visibly degrade after correction; their maxima are 15.26
and 10.42. There is no demonstrated universal collapse threshold at 4.

The failing seed's raw observer scale changes by at most 1.105× between adjacent
calls. Thus a raw-scale jump is not necessary for collapse. The legacy observer
continues accepting finite poses from corrupted imagery; SE(3) validity alone
does not establish scene tracking. Pose estimates after collapse must carry a
failure flag, even when a reported scalar error improves.

[Inspected collapse chronology](artifacts/legacy_rejoin_failures_20261103/lingbot_collapse_onsets.jpg).

This historical RE10K failure is distinct from the near-stationary DL3DV
`02267acf` measured-memory normalization problem. For any encoder change, preserve
a consistent scale for history and future commands; renormalizing only a future
chunk changes the semantics. Test it as a separate adapter version.

## SANA: a true tradeoff and a badly disagreed observation

`0f12b97e` is a tradeoff: mean position **0.05885 → 0.06585**, rotation
**4.24° → 1.41°**, and LPIPS **0.542 → 0.482**. DA3 versus VIPE translation
disagreement at exact common frames grows from **0.00238 at frame 24** to
**0.14909 at frame 144**, after applying the declared observer-unit conversion.

`e43058cb` worsens position **0.2557 → 0.4108** and rotation
**15.02° → 21.67°**. Both sampled videos remain room-like but depart from the
reference layout/path. DA3 and VIPE disagree by **1.196 anchor units and 79.27°
at frame 144**. This is not a small translation-gain identification case. Neither
observer is ground truth, so it does not prove which observer is wrong; it does
show that a controller diagnosis based on one pose trace is underdetermined.

These failures do not contradict the earlier observation that moving back toward
an overshot target can be correct. Physical reversal alone is not a failure;
persistent oscillation or worsening tracking/visual quality is the concern.

## MatrixGame, Lyra and Spatia: separate measurement and continuation effects

**MatrixGame:** several baseline errors are already small. On DL3DV `0238de14`,
mean position is **0.01574 → 0.02622**, rotation **0.840° → 1.214°**; LPIPS and
DINO improve slightly. Dense frames 334–339 show a contrast change at 337 in
both arms, with the playground still recognizable. The separate
[same-latent decoder experiment](matrix_decoder_audit_20261103.md) establishes a
decoder/export contribution in later 169-frame runs. It does not establish that
the older 480-frame pose regression is entirely caused by decoding. Whole-history
decoding also cannot silently replace causal observations that were already used.
Small regressions need repeatability evidence; “observer noise” is a hypothesis,
not a measured explanation in this archive.

**Lyra:** garden `61b27b61` worsens position **0.01892 → 0.04653** and mean
rotation **0.147° → 0.677°**. The endpoint rotation actually improves slightly.
Sampled images retain the garden but show changing tree/rock layout. These
historical angles differ from the later 0.04°→0.12–0.19° memory experiment.
Report absolute angles and registration sensitivity; do not import a universal
0.057° sensitivity floor from the unrelated DreamX refit. VIPE v3 accepts this
anchor registration, whereas the older v2 sheet marked position unavailable.

**Spatia:** `a126efe8` mean position **0.03041 → 0.09610**, endpoint
**0.09525 → 0.25981**; LPIPS and DINO slightly improve. The hall remains
recognizable in the sampled frames, with layout drift. Recurring continuation
seams were measured in both arms. They are real image discontinuities, but
physical camera reversal versus geometry inconsistency remains unresolved.
The memory-label/handoff mechanism needs controlled isolation; this archive
does not establish it as the cause. `334ff324` and `0238de14` also regress.

## VMem: published combined correction hides command-only failure

The [40-run command replay](vmem_legacy_failure_replay_20261103.md) verifies the
latest-memory bridge and corrects the protocol description: these runs used
`scale_mode=window`, not `global_demo`. Adding command correction to measured
memory helps 5/10 scenes, with a 0.01632 reduction in aggregate mean position.
The replay also separates the mixed-coordinate logged command residual from
physical tracking error and documents substantial DA3/VGGT disagreement.

The website's historical correction arm is `replan_measured`, which changes both
future commands and memory labels. All four saved arms are available for the same
ten scenes/seeds:

| Arm | Mean position | Change from baseline | Cases with lower position |
|---|---:|---:|---:|
| Baseline | 0.67215 | — | — |
| Command-only rejoin | 1.30199 | **+93.7%** | 2/10 |
| Measured memory only | 0.52000 | −22.6% | 7/10 |
| Rejoin + measured memory | 0.50369 | −25.1% | 8/10 |

On `74310829`, command-only error is **5.287**, compared with **1.064** baseline
and **1.565** combined. On `d3f042a2` the values are **1.145**, **0.335** and
**0.503**. Memory consistency materially affects the outcome; the published 25.1%
gain cannot be attributed to pure command correction. This is historical
same-cohort arm evidence, not a fresh sham-qualified experiment.

These are outbound-and-return 40-view cycles. A tiny final error can coexist with
severely wrong intermediate views. On `74310829`, combined correction shows a
stretched patio-wall close-up instead of the intended kitchen, yet returns close
to the anchor. Endpoint-only scoring would hide this failure. Similarly, the
1298% endpoint regression on `0f12b97e` is **0.000240 → 0.003353**, not a huge
absolute miss.

## Astronex: stationary reference and historical calibration limitation

`minwm_001231` has **exactly zero reference translation** over the scored horizon.
Its stored path ratios of 17–18 million are denominator artifacts, not enormous
execution gains. Mean position **0.04193 → 0.07538** and endpoint worsening remain
descriptive pose-estimator results, but require stationary-scene repeatability
and dynamic-object checks. Both sampled newsroom videos show moving people;
there is no noise collapse in these samples. The historical calibration uses
recorded reference motion and does not establish the newer causal scale method.

## Consequences for the next experiment

1. **Keep legacy rejoin as the control and candidate.** Individual failures do
   not negate broad improvement. Report cohort means, paired distributions,
   endpoint errors, visual failure counts and observer-invalid counts together.
2. **Measure the commands actually encoded.** Log the temporary measured anchor,
   native relative transforms, normalization and timestamps. Never diagnose a
   command cycle from persistent absolute labels alone.
3. **Isolate recovery magnitude from catch-up ability.** A longer return or slew
   limit must retain explicit recovery headroom and permit returning after an
   overshoot. A non-reversal cap that structurally prevents catch-up cannot test
   whether legacy recovery can be made safer. Keep the reference target fixed.
4. **Gate unreliable observations before learning response.** Shared-frame
   consistency and scene survival are more relevant than finite SE(3) or raw
   scale stability. Treat passive execution ratios as predictors, not inverse
   differential gains. Deployed calibration remains causal and probe-free.
5. **Use model-specific actuator contracts, not fitted correction constants.**
   Preserve VMem's four-arm comparison; isolate MatrixGame/Spatia handoffs from
   controller changes; qualify LingBot's history/future gauge together.

The already queued 40-case matched legacy/memory study is unchanged. Its four
DL3DV scenes do **not** cover RE10K `e43058cb`, LingBot `d3f042a2`, the VMem cycle
failures, Lyra garden, Spatia hall, or the stationary Astronex case. It therefore
cannot by itself close these diagnoses. A subsequent targeted failure panel
should retain these cases and add seeds, without selecting the final method on
that panel alone. No new GPU jobs were launched in this investigation.

## Corrections to earlier interpretation

The November 1 failure taxonomy's DreamX boundary-reversal explanation and the
claim that bounded B removed actual reversals are superseded by this native
replay. Its passive-ratio-derived inverse-gain recommendation is not established.
The six entries in that taxonomy mix measured symptoms and causal hypotheses;
they must not all be described as proven mechanisms. Existing raw results are
retained. Sections E/F's distinction between passive prediction and differential
authority remains necessary, and final method selection remains open.
