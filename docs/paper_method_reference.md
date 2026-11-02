# Closed-loop camera correction: theory, implementation, and paper reference

This is a working technical reference for writing the paper, dated 2026-11-01.
It describes the current code, the proposed extensions, and the evidence needed
to connect them. It is not a claim that every component below has been combined
into one validated system. Numerical results are versioned in
[the development evidence snapshot](paper_evidence_20261101.json).

The selected primary candidate is now [measured-memory v1](method_v1.md).
Its [frozen protocol](method_v1_protocol.json) fixes the memory-only algorithm
and expanded development cohort. The adaptive and future-command designs below
remain separate components/ablations, not part of that candidate.

The [initial mechanism audit](control_mechanism_audit_20261103.md) supersedes
optimistic interpretations of the recent LingBot memory result: both matched
seeds collapse visually despite lower camera errors. It also corrects VMem's
command-log interpretation and records Spatia boundary artifacts whose physical
reversal versus geometry inconsistency remains unresolved. The candidate is frozen for reproducibility,
not validated for broader deployment; the next version needs memory-input and
visual-quality qualification as well as coordinate algebra.

The [cross-model follow-up](memory_components_20261102.md) separates native
rotation and translation interventions in LingBot, DreamX and SANA and reviews
the other implemented memory writers. It distinguishes corrupted imagery,
small intervention effects, backward estimated recovery, and native boundary
appearance seams. Its reference comparisons, observer limitations, component
controls and figures should accompany any paper interpretation of v1 scores.
No component result establishes a universal physical unit mapping or a causal
gain from one ordinary rollout pair.

The [November 3 framing correction](framing_review_20261103.md) distinguishes
near-stationary stress tests from moving-target efficacy, valid return after
overshoot from instability, weak memory replay from future-command authority,
and host decoding artifacts from model-level failures. Use that interpretation
with the original immutable scores and the completed bounded-continuation pilot.

## 1. What the method does, and which version is implemented

We study inference-time camera tracking in a frozen, camera-conditioned video
generator. The desired camera trajectory is an input. Generated images may
realize a different trajectory. A causal observer estimates realized poses;
an adapter can change subsequent camera conditioning and, where implemented,
rebuild history under accepted measured poses. Generator weights are fixed.

A suitable descriptive working claim is:

> We investigate training-free camera tracking through causal geometric
> observation, explicit camera-unit and memory-frame contracts, bounded future
> command correction, and model-specific implementations of measured-pose
> history writes.

This is a description of the research program, not a priority claim or a claim
of consistent improvement across generators. The code has distinct paths:

| Path | Implemented behavior | What its experiments establish |
|---|---|---|
| Portable `CorrectionSession` / `AlignedLoop` | Timestamped boundary rejoin, native camera conversion, commanded/measured-history declaration, quality and causality checks | CPU algebra and contract tests; native hosts must supply an observer and perform actual writes |
| Experimental `AdaptiveController`, arms B/C | Stateful bounded recovery, transported velocities, evidence handling; C adds a conservative diagonal response fit | Shared-controller pilot behavior; it is not the complete authority-estimation design below |
| Measured-memory runtime | Seven explicit native memory consumers; `baseline` isolates memory intervention from the residual command controller | Local GPU write tests and short commanded-versus-measured development comparisons |
| Proposed uncertainty/authority controller | Separate passive prediction and corrective authority, calibrated uncertainty deadband, delay propagation, constrained rate recovery | Design specification; full implementation and efficacy are not established |

The seven memory experiments do **not** test the entire proposed adaptive
controller. Existing bundled research drivers also have their own controller
implementations; packaging them does not make their equations identical.

## 2. Problem, state, and interfaces

Let a fixed generator with parameters θ produce chunk k:

```text
(I_k, h_(k+1)) = G_θ(h_k, E(U_k), text, image, noise_k)
M_k           = O(I_0, ..., I_k; intrinsics, anchor depth)
```

`h_k` includes latent history, KV caches, image memories, geometric maps, and
other native state. `E` is the released camera encoder, including units, axes,
relative transforms, rays, Plücker coordinates, or PRoPE. `O` is the online
observer. The model's motion response can depend on history, scene, command,
and randomness; a scalar gain is generally only a local approximation.

Keep the following quantities separate:

| Symbol | Meaning | Rule |
|---|---|---|
| `P_t` | Original desired pose at output timestamp t | Immutable evaluation target |
| `U_t` | Absolute pose command issued for that timestamp | Record after all corrections and native encoding; do not rewrite the command log |
| `L_t` | Pose label actually represented by stored history | Changes only after the corresponding native cache/geometry write succeeds |
| `M_t` | Accepted observed pose of generated RGB | Estimate with provenance; never equate a prediction with a measurement |
| `Xhat_t` | Predicted current state | Can propagate during an observation outage; has separate uncertainty |
| `u_k` | Actual native conditioning increment | Include native normalization, quantization, sampling, and clipping |
| `b_k`, `q_k` | Latest generated RGB timestamp and observed camera timestamp | The delay is `b_k - q_k`; they need not be equal |

The desired outcome is lower position and rotation error at the **original
timestamps**, while preserving scene identity, appearance, and temporal
continuity. Changing the target time axis to conceal lag is a different task.
No controller can change already emitted RGB. Models without exposed camera
conditioning and a usable generation boundary are outside this interface.

## 3. Camera convention, units, and preprocessing

Canonical poses are OpenCV camera-to-world matrices in SE(3):

```text
T = [ R  t ]       x_reference = R x_camera + t
    [ 0  1 ]
```

Frame 0 anchors the coordinate system. Translation and z-depth share a fixed
reference unit; these units are not automatically metric metres. Native inputs
may use w2c, OpenGL camera axes, per-window normalization, or metric sidecars.

For the supported camera-basis conversion matrix B and positive unit scale k,
`CameraAlignment.encode` scales translation by k, right-multiplies c2w by B,
and inverts if the model expects w2c. Decoding reverses those steps. Compose
relative transforms **after** converting both absolute endpoints into the
native basis. Ray/Plücker construction stays in the model's native encoder.

Three different operations must not be called the same calibration:

1. **Model-unit alignment:** map reference units into the model's conditioning
   units using its declared contract. For corresponding z-depths,
   `k = median(d_native / d_reference)`. Metric and normalization contracts may
   supply k through metadata instead.
2. **Observer registration:** reanchor each reconstruction and convert its
   arbitrary scale back into fixed reference units. For raw poses V and
   corresponding depths, `M_i = inverse(V_0) @ V_i`, then scale translation by
   `median(d_reference / d_observer)`.
3. **Motion-response estimation:** estimate the relationship between issued
   commands and realized increments. Correct units do not imply unit response.

Neither model-unit alignment nor observer registration fits the generated
trajectory to the desired trajectory. Such a fit would remove errors that the
evaluation is intended to measure. Bind model-unit receipts to checkpoint,
anchor, and normalization context. The portable receipt rejects rebinding and
relative scale drift above 3%; this is an implementation threshold, not an
accuracy theorem. A changing native normalization requires a new consistent
context for both past and future camera representations.

Intrinsics, image crop/resize, and depth pixels must refer to the same rays.
`PixelTransform` uses pixel-center resize coordinates; legacy native adapters
retain their audited preprocessing, which need not use the same convention.
Depth must be z-depth rather than ray distance or inverse depth unless converted.

## 4. Causal observation and reliability

Only generated frames available at a boundary enter online DA3. The desired
trajectory is not provided as pose supervision. The measured-memory worker
uses a fixed anchor, at most 16 views per estimator call, and a tail of up to
8 requested views. It can make multiple calls to cover a longer camera history.
Its outputs are registered to reference/adapter units and must pass the
guarded observer's explicit quality report.

The portable depth gate requires at least 256 valid pixels, 20% coverage, and
a p90/p10 depth-ratio spread no greater than 4. Proper SE(3) checks reject
nonfinite matrices, invalid rotations, and malformed homogeneous coordinates.
Empty/all-missing pose sets fail. These checks establish necessary geometric
support; they do not establish accurate current motion or scene preservation.

The **adaptive pilot's** `assess_window` adds separate evidence:

- Agreement on at least three non-anchor frames shared by consecutive windows.
- p90 shared-frame position disagreement normalized by median reference depth,
  and p90 rotation disagreement.
- Textured pixels with projected depth/RGB agreement and spatial coverage.

Its current default policy uses position disagreement 0.03 depth units, rotation
3°, at least 64 supported pixels and 15% sampled support, relative depth
residual 0.15, RGB residual 0.18, and texture 0.01. These are declared heuristic
thresholds, not calibrated probabilistic confidence intervals. Bootstrap
windows have reduced confidence because repeated-frame evidence is unavailable.

Do not silently attribute this entire adaptive evidence policy to the separate
measured-memory worker. An observer's own depth/photometric consistency is also
not an independent test of its pose accuracy. A low-motion or collapsed output
may look internally consistent. Scene-collapse detection remains a separate
validation requirement.

Repeated estimates of identical frames measure **observer re-estimation
disagreement**. They are not regeneration noise, and a p90 disagreement is not
automatically a Gaussian σ. Future uncertainty-aware designs must calibrate
their use of these quantities rather than relabel them.

## 5. Implemented boundary-rejoin controller

For a boundary b, measured pose M_b, target P_b, and future timestamp f_i:

```text
a_i = (f_i - b) / (f_last - b)
δt  = t(M_b) - t(P_b)
δR  = R(M_b) R(P_b)^T
t(D_i) = t(P_fi) + (1 - a_i) δt
R(D_i) = Slerp(δR, I, a_i) R(P_fi)
```

This is the full-gain rule. Separate gains attenuate translation and rotation
residuals. The implementation uses actual output timestamps, so inserting
intermediate camera samples does not change the correction at shared times.
Uniform sampling recovers the original sample-count fade.

The last desired pose equals the original chunk target. That endpoint constraint
can demand an excessive catch-up speed after under-execution. The construction
does not guarantee bounded acceleration, generator responsiveness, scene
preservation, or closed-loop stability. Timestamp correctness fixes a sampling
bug; it does not remove this control-design limitation.

### The memory-coordinate bridge

For desired physical future pose D_i and actual stored boundary label L_b:

```text
C_i = L_b inverse(M_b) D_i
inverse(L_b) C_i = inverse(M_b) D_i
```

The second line proves equality of the requested relative motion. If memory
really represents M_b, then L_b = M_b and C_i = D_i. If it still represents an
old command, the bridge is needed. Rejected memory writes must use the old
stored label, not the proposed measured one. Reduced-gain `CorrectionSession`
uses its gain-adjusted endpoint in the bridge as well, preserving the intended
zero-gain behavior.

This is a boundary-coordinate identity. It does not prove consistency of an
entire history containing different errors at different timestamps. A single
rigid bridge cannot repair independently misregistered past views.

## 6. Implemented bounded adaptive pilot

`AdaptiveController` retains the original plan, predicted/observed state, issued
native increments, command history, and optional host-rebuilt boundary label.
Arm B uses identity response. Arm C additionally divides requested motion by
supported diagonal response gains. This experimental C behavior exists; it
must not be described as the future authority-gated estimator below.

The pilot converts each relative native pose to a six-vector consisting of its
translation and SO(3) rotation vector. This is **not** the full coupled SE(3)
logarithm. Dividing by the actual frame interval gives native input velocity.
Translation and angular-speed limits default to the maxima of the original
native plan. Prior velocity is rotated into the next camera basis before slew
comparison; differences between untransported body-frame vectors would be
misleading.

At each future step it chooses a return horizon at least `return_frames` and
large enough for twice the residual magnitude divided by the available speed
budget. Its recovery fraction is

```text
α = confidence × feedback_gain × (1 - exp(-Δf / H))
```

It then constructs a partially recovered desired pose, requests the corresponding
relative motion, divides by response gains in C, clips speed and speed change,
records the issued motion, and propagates the predicted state. It does not
force the residual to vanish at every chunk endpoint. A zero motion budget
stays constrained. Rejected evidence contributes no new residual innovation or
response fit; propagation continues from the retained state.

Default `RecoveryConfig`: return horizon 64 frames, feedback gain 0.5,
acceleration horizon 12 frames, response range [0.75, 1.5], prior weight 4,
minimum three samples, and directional cosine 0.8. These are shared pilot
settings, not model-specific tuned constants or a validated universal envelope.

`DirectionalResponse` maintains per-component exponentially discounted sums:

```text
xx ← 0.9 xx + confidence × x²
xy ← 0.9 xy + confidence × x y
g  = clip((xy + prior) / (xx + prior), response_min, response_max)
```

Updates require supported magnitude, directional consistency, positive matching
components, and acceptable prediction residual. Fitting pauses after clipped
commands or rejected observations. Unsupported components remain at gain 1.
This through-origin, sign-selective diagonal fit has no intercept or authority
confidence interval and cannot estimate cross-axis coupling. It is not evidence
that causal response has been identified. The full deadband, calibrated
uncertainty propagation, and passive/authority split are not implemented here.

The pilot represents delayed camera timestamps but does not estimate the motion
between a delayed camera anchor and the latest decoded RGB. DreamX's three-frame
offset must not be erased by assigning the old estimate a newer timestamp.

## 7. Measured-memory writing is a second intervention

Changing `history_labels="measured"` is a declaration, not a memory operation.
The native history representation must actually consume the updated camera.
For pose-conditioned KV caches this typically means a clean-context forward;
for geometric memory it means recomputing world points, projections, or splats.

`stage_memory_write` stages a proposed write in canonical c2w coordinates:

1. Require nonnegative, ordered, unique integer frame IDs with matching poses,
   all no later than the available generated frame.
2. Require explicit observer acceptance; reject invalid measured SE(3).
3. Preserve frame 0 when included in the write.
4. On rejection, return a copy of the old labels. Exact identity preserves old
   dtype and values and returns `applied=False, reason="identity"`.
5. On a changed accepted proposal, return measured labels for the host to commit.

The staging helper does not mutate a cache, implement universal rollback, or
certify that a subsequent native rebuild succeeded. A host failure during a
rebuild is a failed run, not an automatically safe old-cache fallback. Logs must
distinguish accepted changed writes, identity no-ops, rejected measurements, and
runtime failures.

### Native consumers currently implemented

| Model | Native write site | Recomputed or committed state | Timing/qualification detail |
|---|---|---|---|
| DreamX | Existing clean-context/context-noise KV forward | Native Camera/PRoPE encoding of finalized chunk cameras | Camera IDs `[0,1,5,9,...]`; observed camera anchor can precede decoded endpoint by three frames |
| LingBot | Existing clean-context KV update | Framewise poses and normalized native ray/Plücker embeddings for the current pass | Native path normalizer stays fixed; future conditioning tensors retained |
| SANA-WM | Stage-1 clean-cache replay after decoded/refined chunk | Replay from a CPU copy of the pre-write cache with measured camera/ray inputs | Restore shared camera tensors afterward; refiner image-only cache remains native |
| Astronex | Both conditional and unconditional clean forwards | View matrices, anchored PRoPE mirror, subsequent appearance-memory commit | Incremental decode before cache write; avoid duplicate boundary callbacks; camera IDs `4j` |
| Lyra-2 | After native depth-cache update | All completed dense camera labels plus existing Sparse3DCache world points from retained depth/K | 80 new camera labels in the tested chunk; sparse records are updated only where present |
| Matrix-Game 3.5 | Before source registration/next section | Four clean-context poses, all 84 previous-section camera-history entries, native world-point/frustum reconstruction | Native registration explicitly consumes supplied extrinsics |
| WorldWarp | After source depth/pose estimation, before splat construction | Complete 81-frame source pose window in its measured terminal-frame gauge | Smoothing and splat sources share one gauge; native optimization can subsequently refine poses |

The representation-specific operations are adapter code. The observation and
write contract are shared. Thus “general” should mean a common contract and
decision rule with explicit native encoders/writers, not a model-agnostic edit
to arbitrary hidden-state tensors.

VMem already has a native measured-memory path in separate experiments; the
shared controller can receive the label actually stored by that writer. Spatia
has a separate measured-extrinsics reconstruction path. These are not additional
newly qualified members of this seven-model memory queue. Memorizon and
WorldPlay still require separate memory/projection and control audits.

### What memory-only means in these comparisons

The external target trajectory and residual-correction arm remain fixed. Some
hosts require future camera matrices to be bridged into the newly stored
boundary frame; that coordinate conversion is part of a consistent memory
intervention. It is not evidence that every model's native future tensor is
bit-identical. DreamX and LingBot specifically preserve their future tensors.

Memory writing observes generated rollout frames and issues no calibration
probe. Runtime model scales come from declared camera/unit contracts and anchor
geometry, not a fitted per-model tracking gain.

## 8. Native memory execution and resource management

The runtime uses audited source overlays without editing original native source
files. Each insertion checks the expected marker count and fails if the source
contract changes. Launch receipts identify the policy and overlay hash. GPU
smoke jobs snapshot runtime sources and check hashes before execution.

Observation can mutate a generator indirectly through RNG, VAE state, caches,
precision, or device movement. Where implemented, RNG isolation and prefix/cache
hashes test these effects. Identity interception by itself may skip the observer
or rebuild, so it is not universally an observer-sham qualification.

SANA has a stronger forced-identity replay test: recomputing the clean cache
with unchanged cameras reproduces native cache tensors exactly and preserves
all 49 RGB frames. This does not test arbitrary measured cameras, but it rules
out that particular replay path changing results when its inputs are unchanged.

On the local 32-GiB GPU / 62-GiB host, SANA initially hit host OOM with multiple
resident DA3 workers plus replay state. The runtime now serializes observers,
closes the memory observer before replay, and restores the endpoint observer
later. Lyra closes the external memory observer before restoring diffusion.
Restarted observers use separate session directories, preserving measurements.
These are implementation requirements and overheads, not algorithmic speedups.

The queue uses existing GPU locks, fresh output directories, a 10-GiB disk
reserve, per-job timeouts, and explicit failure records. It never deletes
checkpoints to make room. Short smoke lengths and a successful write do not
qualify long-horizon memory growth or all offload configurations.

## 9. Proposed on-the-fly response calibration: the required distinctions

This section is a design target, not the current seven-model deployment.

### Execution ratio is not corrective authority

For scalar command magnitude u and local response `y = a(h) + g u`,

```text
execution ratio = y/u = a(h)/u + g
corrective slope = ∂y/∂u = g   (holding relevant history fixed)
```

A history-driven model can have a substantial execution ratio and nearly zero
differential command response. A large chunk displacement improves signal size;
it does not by itself identify the slope. One accepted pair cannot identify
both intercept and slope, and a constant-speed rollout leaves them confounded.

Keep a **passive motion predictor** for delay propagation distinct from an
**authority estimate** used to authorize inverse control. A possible passive
model is `Δm_k = a + G Δu_k + ε_k`, with commands and measured increments aligned
to matching intervals and coordinates. Increments can be expressed in a shared
local tangent frame; translation and rotation need separate units/weighting.

For a scalar channel, forgetting-factor RLS with `φ_k = [1, Δu_k]^T` could use:

```text
K_k = P_(k-1) φ_k / (λ + φ_k^T P_(k-1) φ_k)
θ_k = θ_(k-1) + K_k (Δm_k - φ_k^T θ_(k-1))
P_k = (P_(k-1) - K_k φ_k^T P_(k-1)) / λ
```

Here θ contains intercept and slope; P is an estimator matrix, unrelated to
the target pose symbol. Initialize with a declared prior and uncertainty, not
an assertion of identified gain. Autocorrelation, heteroscedastic observation
error, and history-dependent drift make naive RLS covariance an unreliable
confidence interval without further calibration.

Natural command variation must sufficiently excite the supported directions.
An informative design matrix is necessary, but even nonzero command variance
is insufficient for a causal interpretation when history or scene changes
co-vary with commands. Use held-out prediction checks and frozen-history
diagnostics in development. Without identification, label the estimate as a
predictive association and abstain from inverse scaling.

Crucially, a reliably measured near-zero or opposite-sign response is evidence
about low or reversed authority. Do not discard it because it disagrees with
the intended command. If it is below resolution, retain a censored/uncertain
observation or an explicit unresolved status. Observability acceptance and
desired-direction agreement are different decisions. The pilot C's
sign-selective fit does not implement this authority estimator.

Suggested states are `unidentified`, `identified`, and `low_or_incompatible`,
with separate reasons for insufficient excitation, unreliable observations,
and supported low response. There is currently no validated universal scalar
gain or closed-loop stability theorem for these generators.

### Delay propagation, deadband, and constrained recovery

If an accepted observation is at q < b, predict from q to b through the actual
issued-but-unobserved commands, including response uncertainty. Then form the
tracking residual against P_b. Do not compare an old pose directly to P_b and
call the difference current error.

A calibrated uncertainty deadband should suppress innovations indistinguishable
from observation error. Translation and rotation need separate uncertainty
scales. “3σ” is meaningful only after defining and validating σ; the current
overlap p90 statistic is not that definition.

Spread recovery across chunks within a declared native speed/slew envelope.
For a one-dimensional catch-up illustration, planned advance s and candidate
multiplier m give predicted advance `yhat(m) = a + g m s`. Catch-up is
`yhat(m) - s`, not `yhat(m)` itself. If the best supported prediction within
the envelope has nonpositive catch-up, increasing the recovery horizon cannot
remove lag in that model. Report predicted infeasibility and retain lag.
Without identified authority, this is not a proof of physical unreachability.

Under-execution may require speeding up; over-execution may require slowing
down. A multiplier constrained only to `[1, m_max]` cannot handle both. Any
final method must define its permissible slowing/stopping and reversal policy,
and handle zero-translation plans without dividing by speed. Limits based on
the original plan are reproducible experimental bounds, not proven safe
generator-response regions.

Anti-windup must distinguish newly observed residual from a previously clipped
requested correction. Carry bounded outstanding lag once; do not re-add an
unexecuted correction as an extra integral term. A formal guarantee would
require assumptions about plant response, observation error/delay, cache
consistency, and constraint feasibility that current experiments do not supply.

## 10. Conceptual boundary algorithm and implementation map

```text
Given immutable P, issued commands U, committed history labels L:
  generate and decode a native chunk using its existing memory
  observe only available RGB; register to the fixed reference
  record quality, timestamps, and observation delay

  if this adapter supports measured-memory writes:
      stage accepted measured labels for generated history
      recompute the native pose-dependent cache/geometry
      commit L only with the successful native write

  choose controller variant explicitly:
      baseline: retain original desired future path
      rejoin: timestamped endpoint rejoin
      adaptive B/C: implemented bounded-recovery policy
      proposed authority controller: not yet a qualified deployment

  bridge the desired future motion using the actual committed boundary label
  encode with native units, axes, sampling, and conditioning layout
  record actual issued conditioning and generate the next chunk

After generation: run offline VIPE; never feed its future-informed poses back
```

Operation order varies by host: some need temporary decoding before their
existing cache write, others need replay after an earlier write. An integration
must preserve that host's causal schedule rather than blindly apply this
pseudocode as a universal callback order.

| Component | Repository source |
|---|---|
| Units and conventions | [alignment.py](../src/camera_loop/alignment.py), [geometry.py](../src/camera_loop/geometry.py) |
| Original controller and host API | [controller.py](../src/camera_loop/controller.py), [hooks.py](../src/camera_loop/hooks.py) |
| Bounded adaptive pilot | [adaptive.py](../src/camera_loop/adaptive.py) |
| Evidence and basic geometry gates | [evidence.py](../src/camera_loop/evidence.py), [quality.py](../src/camera_loop/quality.py) |
| Portable memory staging/bridge | [memory.py](../src/camera_loop/memory.py) |
| Bundled measured-memory launcher | [launch.py](../src/camera_loop/bundled/templates/analysis/measured_memory_20261101/launch.py) |
| Native memory consumers | [memory_hooks.py](../src/camera_loop/bundled/templates/analysis/measured_memory_20261101/memory_hooks.py), [dreamx_hook.py](../src/camera_loop/bundled/templates/analysis/measured_memory_20261101/dreamx_hook.py) |
| Audited insertion points | [overlays.py](../src/camera_loop/bundled/templates/analysis/measured_memory_20261101/overlays.py) |
| Observer IPC and receipts | [runtime.py](../src/camera_loop/bundled/templates/analysis/measured_memory_20261101/runtime.py) |
| Local-to-bundle provenance | [source hashes and transformations](measured_memory_source_provenance.json) |

The bundle carries source-only copies with relocated paths and a standalone
copy of the portable memory contract. Fresh materialization and source checks
are CPU-qualified separately from the locally tested GPU runtime. Standard
`camera-loop run` retains its existing route; measured-memory execution uses
the explicit launcher described in [measured_memory.md](measured_memory.md).

## 11. Development probes and identifiability

Probes are a **validation instrument**, not part of the deployed memory loop.
Development experiments branch from identical generated history and change
camera input while holding scene, noise/seed, prefix, and measurement protocol
fixed. Report signed projected response along the actual command perturbation,
cross-axis response, and rotation, with estimator stability diagnostics.

The DreamX audit on e43058cb / seed 107 used a six-latent shared prefix and
endpoint frame 29. At command perturbations up to ±40%, projected translation
responses were at most about 4% of the command change with inconsistent signs
across amplitudes/observers. At ±80%, the reported magnitude ratios were about
3–14%, also with an observer sign disagreement on the negative branch. A
0.057° rotation refit could flip a fitted translation-gain sign within the
pixel-residual limit. These establish unresolved command-aligned authority in
that protocol, not literally zero response or a universal dead zone.

The shared-prefix drift column measures re-estimation of bit-identical images
in changed observer windows. Its 3D p90 norm is not the same statistic as a
signed command-direction projection. Do not compare these as interchangeable
σ estimates. The corrected audit also reports holdout prediction-error
reductions that must not be dismissed using the earlier mistaken “all below
noise” statement.

The ±80% geometric rejection could be a real reconstruction failure or a
baseline-dependent threshold effect; it has not been resolved. A dead-zone /
over-response plant and observer sign instability are competing explanations
for reversal cycles. Neither establishes that no scalar proportional gain can
ever be stable. Vary duration as well as amplitude in the next diagnostic:
the earlier probes already affected a whole continuation chunk.

History-label sensitivity is a separate test. A changed continuation after a
memory edit demonstrates an actuator effect. Only matched tracking and quality
evaluation establishes whether the measured write was useful.

## 12. Evaluation protocol and current evidence

The current memory queue compares commanded versus measured history with the
same model, scene, seed, baseline driver, and horizon. It checks run metadata,
intrinsics, output shape, and raw pre-intervention equality where available.
The primary score covers frames starting at the first potentially affected
frame; whole-rollout scores are retained separately.

| Model | Scene / seed | Tested RGB horizon | First scored affected frame |
|---|---|---|---|
| DreamX | e43058cb / 107 | 33 frames | 21 |
| LingBot | e43058cb / 7 | 29 frames | 13 |
| SANA-WM | e43058cb / 7 | 49 frames | 25 |
| Matrix-Game | e43058cb / 7 | 169 frames | 85 |
| Lyra | 02267acf / 7 | 161 frames | 81 |
| WorldWarp | e43058cb / 7 | 161 frames | 81 |
| Astronex | minwm_018300 / 7 | Native export; recorded frame count in score | 29 |

VIPE receives consecutive generated RGB and fixed preprocessed intrinsics. It
can use future frames because it is an offline evaluator. Its pose sequence
never enters generation. Position is registered using unsaturated optimized
frame-0 depth, matching reference pixels, and first-pose anchoring. Unsupported
scale registration leaves position metrics unavailable; rotation can remain.

```text
position error_t = || t(T_eval,t) - t(P_t) ||₂
rotation error_t = angle(R(T_eval,t)^T R(P_t)) in degrees
reduction (%)   = 100 (error_commanded - error_measured) / error_commanded
```

The reduction is undefined when the baseline error is zero or either position
registration is unsupported. A stationary target cannot define a translation
path-length ratio; report absolute path lengths and a null ratio instead.

Using VIPE rather than online DA3 separates the estimators. It does not create
independent ground truth: some batch reference trajectories are VIPE-derived.
The seven-model queue currently measures camera errors, not DINO/LPIPS, temporal
quality, or a validated collapse rate. Those remain required before a full
quality-preserving-control claim.

The first completed development pairs include improvement on LingBot and SANA,
a small DreamX change, and degradation on Matrix-Game. Astronex has rejected
position registration and only lossy-export prefix evidence. Preserve all of
these in the paper's development record; do not select only positive models.
Exact numbers and pending/completed states are in the dated evidence snapshot.

Completed initial development queue (2026-11-01). All means are over the
post-intervention continuation; a numerical score is not necessarily an
attributable intervention result:

| Model | Position mean, commanded → measured | Position reduction | Rotation mean, commanded → measured |
|---|---|---|---|
| LingBot | 0.47358 → 0.29166 | +38.41% | 5.83447° → 3.91365° |
| DreamX | 0.10437 → 0.10347 | +0.86% | 14.49684° → 14.49738° |
| SANA | 0.07306 → 0.05322 | +27.15% | 6.15310° → 4.16651° |
| Matrix-Game | 0.10766 → 0.14900 | −38.40% | 0.52379° → 3.05172° |
| Lyra | 0.02084 → 0.01841 | +11.68% | 0.05080° → 0.07540° |
| Astronex | unavailable in both arms | undefined | 14.56347° → 14.88234° |
| WorldWarp | 0.09174 → 0.07960 | +13.24%, confounded | 40.91971° → 40.78924° |

The five completed lossless comparisons preserve their pre-intervention RGB
exactly. Astronex's lossy exports do not; raw prefix parity is unverified. Lyra
improves translation in this sample while worsening its small rotation error.
No significance threshold was established for the small DreamX difference.
WorldWarp differs from frame 0, before the memory intervention, with prefix MAE
4.29365/255. Its frame arrays are decoded from native chunk videos; storing them
in NPZ does not make their provenance raw lossless generation. The observed
gain is confounded and is not counted as evidence for measured-memory efficacy.
WorldWarp already estimates source poses natively; our write replaces those
with external estimates rather than introducing pose estimation for the first
time. Qualification must address both provenance and prefix consistency.

Engineering qualification is separate: seven native memory write paths ran on
GPU; 19 memory/contract CPU tests and five evaluation-metric tests passed in the
local research workflow. DreamX's original supervisor exited 143 after saved
outputs; artifact checks, rather than a clean supervisor exit, support its
parity result. SANA's initial host-OOM attempt and later successful fix remain
in the record. Astronex raw latent/cache/RNG parity is not established.

## 13. Experiments needed for a paper claim

Predeclare the final algorithm and evaluate it without per-scene tuning. A
useful factorial design includes native baseline, observer sham, command-only
correction, memory-only correction, and their combination where supported.
Keep camera-unit alignment fixed across these arms. Test calibration changes,
observer policy changes, and memory writes separately before combining them.

Use multiple development and held-out scenes, more than one seed, motion types
including rotation-only/low-parallax cases, and longer rollouts. Report matched
per-scene differences and uncertainty across scenes/seeds rather than treating
many correlated frames as independent samples. Invalid observations, rejected
scale registrations, collapses, OOMs, and completion failures stay in the
denominator with explicit counts. Give conditional valid-position metrics and
coverage together. Compare wall time and GPU cost, including extra observation,
cache replay, offload, and any baseline search budget.

Required diagnostics include accepted/rejected writes, raw prefix parity,
actual native command norms and slew, observed-versus-predicted increments,
reversal count, retained lag, scale drift, quality/collapse rate, and appearance
metrics. A visually static collapse must not win by having a smooth camera
estimate. A geometry rebuild with a noisy pose may worsen native memory;
Matrix-Game's current result makes this an empirical issue rather than a
hypothetical caveat.

For the proposed response controller, require sufficient excitation and
out-of-window prediction checks, preserve zero/opposite responses, and test
whether authority estimates remain stable under plausible observer refits.
Development probes should support design decisions; held-out deployment should
use the rollout itself without probe generations or fitted per-model gains.

## 14. Claim ledger and writing guidance

| Statement | Current status |
|---|---|
| No generator training or weight update is required | Implemented for these inference paths |
| Deployment can write measured history without calibration probes | Implemented in the seven-model memory runtime |
| A common camera/memory contract can support different native representations | Implemented with explicit model-specific writers |
| `history_labels="measured"` alone updates KV/geometric memory | False; native rebuild/commit is required |
| Measured memory improves every tested generator | Not established; the tested Matrix-Game causal host has mixed scores and an unresolved decoder/export seam, while the near-stationary LingBot writer fails visually |
| Chunk execution ratio identifies inverse-control gain | False without additional excitation, history, and observability assumptions |
| The complete uncertainty/authority controller is deployed and validated | Not established |
| The method is stable for arbitrary frozen generators | Not established; no applicable stability proof supplied |
| The method generalizes to unseen scenes/checkpoints | Requires held-out evaluation |
| This is the first training-free/inference-time camera-control method | Unverified; requires a primary-source literature comparison |

Novelty should be argued at the mechanism and evidence level: which state is
observed, which representation is corrected, what is learned online, how camera
units and memory coordinates are kept consistent, and what compute is required.
Solving the same task as another paper neither proves nor disproves novelty.
Compare training requirements, model access, feedback timing, future-command
versus history intervention, online calibration/probes, and evaluation budgets.

Do not carry unverified titles, identifiers, “none report this” statements, or
“first” claims from brainstorming into the bibliography. A literature search
with checked primary sources is still required. This document cites internal
implementation/evidence rather than claiming that search has been completed.

For a paper draft, organize the method around the camera/memory contract, causal
observation, the exact selected controller, and native write consistency. Place
the response-identifiability study and negative cases in experiments/analysis.
Put source pins, thresholds, native schedules, preprocessing, runtime/memory
costs, and failure accounting in the reproducibility appendix. Use only the
components actually exercised by the reported experiment in its algorithm box.

The [bounded-continuation pilot](memory_rate_20261102.md) records the next
experimental policy after the cross-model memory-component audit. It compares
the same measured-memory write with and without bounded future translation in
SANA and Lyra. Its equations, native geometry prerequisites, abstentions, source
freeze and evaluation scope are documented separately. It does not implement
online differential-gain identification or establish closed-loop stability.

The [fractional-recovery follow-up](fractional_recovery_20261103.md) addresses
that ablation's missing catch-up headroom and forbidden return. It requests an
explicit residual fraction per chunk, permits return and accounts for recovery
left unserved by speed/slew bounds. This is command geometry, not an identified
plant gain. Its future-only wrapper supports nominal first-boundary issuance;
DreamX native integration and authority qualification remain pending.

The [generalization strategy](generalization_strategy_20261103.md) records the
current decision criterion and complete short-cohort accounting. A general
method may have scene-level losses; the paper should establish aggregate benefit
on a frozen, representative evaluation and retain failures, rather than demand
universal improvement or tune exceptions to individual development scenes.
