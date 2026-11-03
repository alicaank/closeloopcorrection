# Method: theory, implementation and paper reference

The Method observes a frozen camera-conditioned generator and rejoins its
future camera commands to a fixed target path. It does not train the generator,
repair emitted RGB, or require online response-gain learning or extra probes.
[Results](results.md), [failures](failures.md) and [ablations](ablations.md) describe
what has actually been measured. Full historical design notes remain in the
[archive](research_archive.md).

## State and causal inputs

Let `P_i` be the desired camera-to-world pose at output frame `i`, expressed in a
fixed initial-reference coordinate system. At boundary `b`, the generator has
emitted frames `I_0 ... I_b` and retains native memory `h_b`. An observer estimates
`M_b` from those emitted images, intrinsics and initial-anchor depth. Future
reference images and desired poses must not be supplied as realized-pose labels.

The adapter declares camera direction (c2w/w2c), axes, native units, frame/latent
sampling, conditioning layout and whether memory represents commanded or
measured cameras. These are interface contracts, not fitted per-model gains.
Models without supported camera inputs cannot acquire camera control through
this wrapper alone.

## Register geometry before comparing poses

An observer reconstruction can have an arbitrary rigid frame and scale. Anchor
its first pose and fit scale only from corresponding initial-anchor depths:

```
s = median(reference_depth[p] / observer_depth[p])
M_i = inverse(observer_pose_0) @ observer_pose_i
translation(M_i) *= s
```

Corresponding depths must share pixel rays and conventions. Resize/crop changes
must be applied to K and depth together. A new observer window may need fresh
registration; reusing its previous arbitrary raw scale is not generally valid.

Observer registration is distinct from native command-unit conversion. If the
native model supplies compatible anchor geometry, estimate
`k = median(native_depth / reference_depth)` and convert both commands and
history consistently. Otherwise use a declared native normalization or require
an audited provider. Astronex/WorldPlay do not gain physical calibration simply
because a default factor or reference-motion fit was removed. See
[unit contracts](online_units.md).

## Rejoin the planned continuation

For future frame `j > b`, let
`a_j = (j - b) / (j_end - b)`. The portable controller's full-strength curve is:

```
e_t = translation(M_b) - translation(P_b)
E_R = rotation(M_b) @ transpose(rotation(P_b))
translation(D_j) = translation(P_j) + (1 - a_j) * e_t
rotation(D_j) = Slerp(E_R, Identity, a_j) @ rotation(P_j)
```

The residual fades with elapsed output-frame time, including irregular sampling.
At the final future sample `D_j_end = P_j_end`. A lag can request catch-up; an
overshoot can request a return. Neither behavior guarantees the generator
executes the requested motion. Short recovery horizons can request large steps.

Optional translation/rotation attenuation is applied to the measured residual
before frame bridging. The default is full-strength rejoin. These are explicit
controller settings, not measured plant gains. The Method does not infer
corrective authority from a passive execution ratio.

## Bridge to the memory frame, then encode

If the retained latest camera label is commanded `L_b`, send:

```
U_j = L_b @ inverse(M_b) @ D_j
inverse(L_b) @ U_j = inverse(M_b) @ D_j
```

This preserves the desired relative transition from the measured boundary in
the frame represented by commanded memory. If the host has actually rebuilt
its memory with measured labels, use `D_j` directly. Declaring measured memory
without performing its native cache/geometry update is not sufficient.

Convert native units/axes and c2w/w2c consistently before building relative
chains, rays or other conditioning. Relative chains include the actual boundary
anchor. Differencing stored absolute labels alone can invent boundary reversals,
as the DreamX replay demonstrated. The host owns its released encoder and
latent/RGB timestamp mapping.

## Runtime contracts and implementation

- [controller.py](../src/camera_loop/controller.py): `Measurement`, timestamped
  `rejoin`, and `CorrectionSession.correct`.
- [alignment.py](../src/camera_loop/alignment.py): geometry conversion and
  calibration identity binding.
- [hooks.py](../src/camera_loop/hooks.py): `ModelHooks` and `AlignedLoop` bridge
  callbacks to the portable controller.
- [quality.py](../src/camera_loop/quality.py) and
  [evidence.py](../src/camera_loop/evidence.py): depth, pose, overlap and image
  evidence checks; not a universal scene-survival certificate.
- [memory.py](../src/camera_loop/memory.py): staged memory writes and bridges;
  native consumers are documented [separately](measured_memory.md).

The portable API requires the measured frame to equal the available boundary,
requires strictly increasing future timestamps, and rejects writes to generated
history. Sham bypasses correction exactly. `Measurement.validate` requires a
passed quality report and valid geometry. The host must compute meaningful
quality evidence; a manually supplied `passed=True` is not an observation gate.

Bundled native launchers retain model-specific schedules, history handling and
observer behavior. They are not all the portable API with different model names.
Some historical hosts only check worker success and finite poses. Such hosts can
continue correcting collapsed images; their old results cannot be relabeled as
results of stronger safeguards. New safeguards need separately qualified runs.

## Evaluation and paper claims

Use matched baseline/sham/Method runs with the same scene, seed, weights,
conditioning, native grid and horizon. Verify sham and pre-intervention identity
where supported. Changes to observation gates, camera normalization, cache
rebuild or decoding require declared adapter versions.

The offline evaluator receives completed generated RGB and calibrated intrinsics.
It registers to the initial anchor without fitting trajectories to desired
motion. Report mean and endpoint position, rotation, reference-image similarity,
visual failures and unavailable measurements. Position is in fixed anchor-depth
units unless metres are independently established. Reference trajectories may
themselves be estimator-derived; a separate observer is not ground truth.

Historical evidence supports average improvement in many cohorts, with important
failures and tradeoffs. VMem's reported Method arm combines commands and measured
memory, and command-only performs worse in that cohort. Newer bounded controllers,
response fitting and memory-only studies are separate ablations. No universal
stability, cross-model superiority or paper-priority claim follows from packaging
these components together.
