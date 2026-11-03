# Framing correction and next experiment

The comments change the interpretation and next experiment. The 36 component
runs cover one near-stationary scene. They cannot replace the moving-target
evidence or establish a general failure of measured-memory correction. Raw
scores and frozen runs are retained. This review recomputes the quantities in
[the evidence file](framing_review_evidence_20261103.json), rather than changing
earlier results to fit a new interpretation.

## Scene regime and LingBot normalization

All distances below use the existing reference anchor-depth gauge.

| Target interval | 02267acf maximum RGB step / net travel | 0238de14 maximum RGB step / net travel |
| --- | --- | --- |
| 29 frames | 0.000238 / 0.00286 | 0.001700 / 0.03444 |
| 49 frames | 0.000337 / 0.00744 | 0.001700 / 0.06476 |
| 77 frames | 0.000420 / 0.01338 | 0.001700 / 0.10162 |
| 161 frames | 0.001910 / 0.12276 | 0.002285 / 0.25428 |

The short component windows are near-stationary. The longer 161-frame Lyra
window on 02267acf includes substantial later motion and should not receive the
same whole-window label.

LingBot divides framewise translation by the maximum planned native step.
On the component scene, that denominator is 0.00073686; measured native steps
reach about 81–106, with seed-7 measured/commanded-prefix amplitude ratio 142.49.
On the saved moving-target pairs, the denominator is 0.0065074 and the largest
measured normalized steps are 8.37 and 7.50. The older e43058cb pair also remains
recognizable in sampled RGB. This supports a scene-conditioned normalization
failure, not a general inability to consume measured translations. “Saturation”
here means extreme conditioning relative to the native plan envelope; no hard
floating-point saturation or measured training-support boundary was established.

The next encoder candidate uses one denominator covering both planned and
observed-history steps. It must apply that denominator to **all retained camera
state and future conditioning**, rebuilding affected native caches together.
Rescaling history alone, as in diagnostic N, changes its relation to future
commands and is not that fix. The native encoder removes a common positive
translation scale, so an arbitrary physical-to-native scalar search is not the
priority for this failure.

This candidate is not yet a validated deployment repair. A maximum-envelope
normalizer does not prove support for extreme history/future speed ratios.
An abstention gate needs declared operational support and evidence; the
denominator cannot manufacture that evidence. If a consistent retained-state
rebuild or required support check cannot be performed, retain the previous
state and record the abstention.

## Return after overshoot and the stronger moving-target evidence

After a generated prefix overshoots, measured history places the original next
target behind the represented camera. Returning toward it can reduce tracking
error. Backward progress alone is not a failure. A one-boundary test cannot
establish oscillation; repeated overshoot, error growth and visual damage are
separate outcomes requiring longer runs.

The completed [bounded-continuation pilot](memory_rate_20261102.md) reduces the
boundary return but raises position error in all four SANA/Lyra comparisons.
That is a tradeoff introduced by its non-reversal objective, not evidence
against measured-memory correction. It remains an ablation, not the next
default tracking controller.

Its speed ceiling supplies no catch-up headroom when incoming speed and nominal
speed already equal the plan maximum. Its forward halfspace also forbids useful
return after an overshoot. These structural restrictions bias this ablation
against tracking recovery. They do not prove that F can never outperform M for
an arbitrary generator or a variable-speed plan. The replacement experiment
uses [fractional recovery](fractional_recovery_20261103.md), explicit above-plan
speed headroom and permission to return; it does not repeat the no-return arm.

Existing moving-target 0238de14 pairs are more relevant to tracking efficacy:

| Model | Position reduction, seeds 7 / 107 | Rotation, commanded → measured |
| --- | --- | --- |
| SANA | 84.81% / 81.39% | 0.491° → 1.169°; 0.529° → 1.038° |
| Lyra | 74.23% / 66.08% | 0.044° → 0.122°; 0.027° → 0.188° |
| LingBot | 43.43% / 24.82% | 2.333° → 2.254°; 1.421° → 3.493° |

SANA has a material angular tradeoff in these pairs. Lyra's absolute changes are
only 0.078° and 0.160°; report them as small, with unresolved significance,
rather than a demonstrated meaningful rotation penalty. The separate 0.057°
rotation-refit example shows translation-estimation sensitivity in another
condition. It does **not** calibrate a universal rotation-noise floor for Lyra.
These are two-seed development results, not held-out generalization estimates.
In particular, LingBot's two seeds can qualify an encoder change and its native
state rebuild, but cannot establish that a 25–43% effect exceeds seed variation.
The reported baseline differences as large as 0.88 elsewhere need their exact
metric/window verified before use as a variance estimate. A larger paired seed
cohort and uncertainty intervals are required for an efficacy claim; two seeds
do not supply a calibrated noise floor or a power calculation.

## DreamX needs a future-command experiment

The tested context-noise KV re-pass has little effect: about 12.44 times the
commanded prefix translation amplitude changes pixels by roughly 0.4/255 in
seed 7. This is weak actuation through that memory interface. It does not prove
that every possible history rewrite is inert.

The approximately 0.12 number discussed in the review is a passive execution
ratio for one scene/seed under rejoin, not differential command authority. It
cannot identify an inverse controller. There is also separate archived
future-command evidence: e43058cb, seed 7, later history, endpoint 65 has
projected gains 0.1434 (PnP) and 0.1217 (VIPE). The earlier follow-up still failed
its overall screen because other seed/history conditions disagreed. That
distinct, similarly rounded VIPE number must not be substituted for the passive
ratio, or treated as a universal per-step constant. The next matched DreamX
control test should hold memory fixed and intervene on future conditioning,
checking the actual camera/focal contract. Do not add a weak memory channel
merely to force a common actuator across models, or enable inverse scaling from
this single number.

Keep the first feedback command at the original plan amplitude. Subsequent
accepted lag observations can drive fractional recovery and provide natural
command variation. Variation alone does not establish identifiable authority:
the estimator still needs excitation, observer qualification and prediction
checks. The new issuance API enforces the nominal first boundary, but its CPU
tests do not qualify a DreamX native GPU integration.

## MatrixGame and Spatia attribution

MatrixGame's current causal host exports separately decoded native blocks and
skips the released final whole-history VAE re-decode. Its seam also appears in
zero-write controls. Compare block decoding with whole-history decoding of the
**same retained latents** before attributing the seam, or a model-level null
result, to the generator. Keep the causal output as the online score stream;
joint re-decoding is a separately labeled diagnostic because it can alter past
pixels. If the necessary latents were not retained, a new capture is needed;
an RGB-only comparison cannot isolate the decoder cause.

Spatia has observed boundary artifacts and persistent estimated backward
progress. Physical reversal versus geometry inconsistency remains unresolved.
The paper should not call this established genuine camera discontinuity.

## Concrete next-test specification

Use moving scene **0238de14**, seeds **7 and 107**, as the primary matched
development condition. Retain 02267acf's short intervals as a separately
reported stress condition. Compare exact shared prefixes and accepted
observations, original fixed-time targets, native-input receipts, RGB quality,
and absolute pose errors. Do not count return motion alone as failure.

| Interface | First qualification / matched comparison |
| --- | --- |
| LingBot | Original normalization and identity replay; then a shared plan/history denominator applied to commanded-history and measured-history controls with identical future tensors. Rebuild every retained state affected by the gauge change; fail closed if that cannot be demonstrated. |
| SANA | Retain native/observer-sham versus measured-memory evidence. Fresh M/F pairs compare identical measured memory with original future targets versus fractional recovery: request 0.5 of the boundary residual per chunk, speed budget 2 times the plan maximum, returns allowed. These are explicit experimental settings, not fitted response gains. |
| Lyra | Native/observer-sham versus measured memory; report sub-degree angular differences in absolute units and establish observer repeatability before significance claims. |
| DreamX | Fixed-memory baseline/sham versus future correction: first feedback command remains exactly nominal; later observed lag drives bounded fractional recovery. Qualify actual focal/translation inputs and conditional response before inverse adaptation. |
| MatrixGame | Same-latent block-versus-joint decode diagnostic, separated from causal policy scores. |
| Spatia | Baseline/correction boundary geometry and image evidence; retain unresolved attribution until independent tests distinguish the alternatives. |

First qualify the changed encoder/actuator at one boundary. Then freeze a run
with at least three intervention boundaries to assess repeated overshoot and
oscillation. The moving-scene figures above are reanalysis of existing matched
pairs. The new SANA M/F experiment has its own frozen manifest and
[execution record](fractional_recovery_20261103.md); it does not constitute
execution of the other model rows. The earlier eight-run SANA/Lyra pilot and
its 16 post-hoc observer checks retain their original manifests.

### DreamX native timing preflight

The [future-only preflight](dreamx_future_preflight_20261103.md) now qualifies
an explicit memory-anchor bridge with the released encoder on four scenes.
The last native camera can precede the latest emitted RGB by three frames;
recovery and native relative encoding must use their respective timestamps.
The generic issuance API supports this distinction. Focal-contract alignment
and native GPU qualification remain pending; this preflight does not queue an
unqualified future-only efficacy run or change the frozen memory cohort.
