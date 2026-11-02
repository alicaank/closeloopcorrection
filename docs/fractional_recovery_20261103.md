# Fractional residual recovery after measured-memory correction

Status: completed development experiment: four fresh SANA generations on moving
scene 0238de14, seeds 7 and 107, and two paired VIPE scores. All native input and
unit checks pass; the current 126-test CPU suite passes. Mean position error
increases in both pairs, while endpoint and rotation errors decrease. This
one-boundary tradeoff does not promote the policy or establish stability. The
separate frozen measured-memory v1 policy remains unchanged.

## Why change the previous F arm?

The [old bounded-continuation ablation](memory_rate_20261102.md) constrained
forward progress to be nonnegative and capped speed at the larger of incoming
speed and maximum planned speed. At constant maximum planned speed this leaves
no catch-up headroom for a lagging prefix. After overshoot, the forward
constraint prohibits useful return. Its four position-error losses cannot
establish that generators resist correction. Variable-speed plans and nonlinear
generator responses prevent turning that structural observation into a theorem
that the old F always loses.

The replacement makes the intended recovery explicit: request a fraction of the
boundary residual during the next chunk, provide above-plan speed headroom, and
allow either motion direction. It still limits speed and acceleration. Bounds
may prevent the requested recovery; receipts preserve that shortfall.

## Command geometry

The API is `camera_loop.fractional_recovery.fractional_memory_continuation`.
Inputs are camera-to-world poses in one already qualified geometry gauge. The
function does not infer metric scale, fit a response gain, inspect future RGB,
or write native memory itself. The adapter must first qualify the observation
and commit the measured history consistently.

Let `p_i` be original planned centers, `m_b` the accepted measured boundary,
`H` the next chunk's RGB duration, and `d` the native camera interval. Estimate
incoming velocity from the most recent measured interval of at least `d`
frames. Define:

```text
e = p_b - m_b                          boundary residual
r = alpha * e                         requested recovery this chunk
t_i = p_i - p_(i-1)                    original plan increments
B = beta * max_i ||t_i||               nominal command speed budget
V = max(B, ||v_b||)                    transient speed ceiling
A = V / d                             per-frame velocity-change ceiling
x_b = m_b
```

For each future frame, compute the recovery already represented by the command
path, excluding original planned travel, and distribute the remaining request:

```text
c_(i-1) = (x_(i-1) - m_b) - (p_(i-1) - p_b)
w_i = t_i + (r - c_(i-1)) / (b + H - i + 1)
z_i = norm_cap(w_i, B)
v_i = v_(i-1) + norm_cap(z_i - v_(i-1), A)
x_i = x_(i-1) + v_i
```

There is no forward-halfspace projection. If the constraints do not bind,
endpoint command residual is `(1-alpha) * e`. Redistribution can also recover
early clipping later in the chunk when bounds permit. An already overspeed
prefix is slewed toward `B`; it is not instantaneously truncated. Convexity of
the speed ball preserves `||v_i|| <= V` and slew is bounded by `A`.

The receipt reports requested recovery `r`, realized **command** recovery
`e - (p_(b+H)-x_(b+H))`, their vector difference, and every speed/slew clip.
The scalar realized fraction is projection onto `e`; it is not clamped or
interpreted as plant response. A subsequent boundary uses the new observed
residual. Do not add the previous shortfall to that residual again.

Exact identity history is an exact no-op. Without planned translation the
policy cannot invent a translational speed budget; incoming motion is slewed
down if present. Past command records and original future rotations are
preserved. Rotation jumps relative to measured history can therefore remain.
All quantities scale consistently under a common positive geometry-unit change.

The pilot declares `alpha=0.5`, `beta=2`, and SANA's native interval `d=8`.
These are fixed experimental design choices, **not model-specific calibrated
gains**. They do not guarantee support in every model's camera conditioning or
establish optimal settings. Native input qualification is still required.
M's unchanged absolute future target asks for the original endpoint, whereas
F deliberately leaves half the boundary residual in its unconstrained command
geometry. Thus removing the old structural prohibition does not guarantee a
lower immediate position error than M. The motivation for spreading recovery
also needs repeated-boundary evaluation, not just this one-boundary score.

## Future-only issuance and DreamX bootstrap

`issue_fractional_future` preserves issued past commands and the unissued tail
outside the next chunk. With measured history it issues the desired future
poses directly. With commanded history it bridges the reference correction
through `issued[b] @ inverse(measured[b])` before native encoding. Reference
rotation targets remain unchanged, though this frame bridge can change native
rotation entries.

For the proposed DreamX future-only arm, set `nominal_boundaries=1`. The first
feedback decision retains the original planned future exactly and reports
`nominal_bootstrap`; the host must bypass re-encoding for this no-intervention
decision. Already-modified future commands are rejected rather than incorrectly
reported as a no-op. Only later accepted observations drive recovery. The host owns the
monotonically increasing feedback index and causal observation availability.

This implements the issuance contract and its CPU regression tests. It does
not qualify DreamX's actual native focal/translation tensors or provide an
integrated GPU-tested DreamX controller. Natural lag-driven variation is not
automatically sufficient excitation: authority identification still requires
qualified observations, command/history separation and predictive checks.
Neither the passive execution ratio near 0.12 nor a separately archived probe
with a similar rounded value is installed as a gain. Inverse control stays off.

## Frozen moving-scene comparison

Workspace: `analysis/fractional_recovery_20261103/`. Manifest SHA256:
`42324612a975eae9f2a2825b9539c0a61df7899d6c6df894f879e02e3f378ff6`.

Each of seeds 7 and 107 produces fresh M and F outputs: 49 RGB frames, shared
prefix through frame 24, first affected output frame 25. M commits accepted
measured memory and retains original future targets. F commits exactly the same
memory and changes only future translation using the above policy. Four
generations and two paired VIPE scoring jobs execute serially. The observer
uses only the emitted prefix for feedback; offline evaluation is separate.

Required checks are exact prefix RGB, matching observation keys and memory
write records, fresh paired anchor calibration, preserved past/rotations, and
reproduction of actual native ray/Pluecker input hashes. Evaluate both arms
against the original target at original timestamps. Visually compare reference
RGB and retain failed/abstained runs. Pose reference is observer-derived,
previously checked against sparse official poses; it is not independent dense
metric ground truth. Shared-prefix disagreements in full-video evaluation are
diagnostic, not a calibrated uncertainty floor.

This experiment has no fresh unmodified native baseline. It isolates the
additional future policy relative to measured memory. Existing A/M comparisons
remain separately versioned. Two seeds cannot estimate population efficacy or
resolve LingBot's reported seed variability. One boundary cannot test repeated
overshoot or a reversal limit cycle. Other model qualifications in the
[framing review](framing_review_20261103.md) remain outstanding.

## Validation and results

Before freezing: all 125 repository CPU tests passed, the native SANA overlay
compiled, and the released SANA camera encoder was exercised at its actual
temporal/spatial stride `(8,32,32)` on a 704 by 1280 grid. Ten new tests cover
lag catch-up, allowed return, saturation accounting, incoming overspeed,
unit equivariance, identity/rotation preservation, a static plan, invalid
frame ordering, nominal first issuance, bridging and causal bounds.

The completed GPU comparison is below. Position uses fixed reference
anchor-depth units; means cover frames 25–48 and endpoints are frame 48.

| Seed | Mean position M → F | Endpoint position M → F | Mean rotation M → F |
| --- | --- | --- | --- |
| 7 | 0.023009 → 0.061150 | 0.038119 → 0.018703 | 1.205° → 0.656° |
| 107 | 0.031748 → 0.072164 | 0.036357 → 0.024833 | 1.050° → 0.769° |

Both command paths realize exactly the requested 0.5 recovery fraction in
float64, with zero speed clips and 9/10 slew clips. Actual bf16 ray and Pluecker
input hashes reproduce exactly. Sampled native speeds pass their bounds with
explicit rounding allowances. The declared native/reference conversion is
5.947927951876295 for both pairs; original target conversion error is zero.
There is no trajectory fit. Prefix RGB and accepted measured history are exact
matches between M and F; past commands and rotation targets are preserved.

The mean position penalties are 165.8% and 127.3%; endpoint improvements are
50.9% and 31.7%. Mean rotation improves 45.6% and 26.8%. These are paired
observations from two development seeds, not estimated population effects.
Both prefixes overshoot. M's absolute target creates a sharp command return;
F first decelerates the incoming overspeed, then returns. VIPE estimates this
broad timing pattern. The new F can return, but spending more of the chunk away
from the target raises its mean error despite a better endpoint. It has not
established better overall tracking.

Manual inspection of reference/output contact sheets finds recognizable scene
content in both arms, with conspicuous doubled stair edges in M at frame 25
and less apparent ghosting in F at that sample. This is sampled qualitative
review, not a full-video quality score. Full-video VIPE estimates of the
identical RGB prefix differ at the boundary by 0.00220 / 0.00324 anchor units.
Those discrepancies are retained as observer diagnostics, not a calibrated
uncertainty floor. No new independent post-hoc DA3 audit was run for this cohort.

See the [visual gallery](fractional_recovery_gallery_20261103.md),
[machine-readable evidence](fractional_recovery_evidence_20261103.json), and
[frozen source excerpts](artifacts/fractional_recovery_20261103/README.md).
All six queue jobs completed without failure. Frozen sources were verified
before each job and after completion. A subsequent API guard rejects
pre-modified future commands during nominal bootstrap, which otherwise could
report a false no-op to the host. Its extra regression raises the current suite
from 125 to 126 tests; the exact pilot source/test snapshots and before/after
hashes remain in the evidence. The SANA continuation function is unchanged.

## Next decision

Keep the old no-return F as a historical objective ablation. Keep fractional
recovery experimental: this moving-scene test validates command feasibility
and attribution, but not a mean-tracking improvement. A separately frozen run
with at least three feedback boundaries should retain original timestamps and
report mean/endpoint error, repeated overshoot and return, and visual validity.
That distinguishes delayed recovery from sustained oscillation without treating
return itself as failure. Do not tune the fraction to these two seeds and then
present them as held-out evidence.

For DreamX, qualify unchanged first-boundary native tensors and the later
future-only write before GPU efficacy claims; online inverse authority remains
disabled. LingBot's shared plan/history normalization and consistent cache
rebuild still need their own qualification, followed by more paired seeds for
efficacy. MatrixGame's decoder attribution and Spatia's observer/physical-motion
ambiguity remain as documented in the framing review.
