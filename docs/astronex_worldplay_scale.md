# Astronex and WorldPlay: what can be calibrated online

Checked 2026-11-02 against local Astronex commit `27584bf1a89da01ef35a03b48d83bc818d2adf2b` and WorldPlay commit `1588e1336e842b03b0a7860c654ebd7c46bb065e`, their public source, and their papers. This is a source audit and design proposal, not a newly enabled generator integration.

## Findings

| Interface | Observed behavior | What it establishes |
|---|---|---|
| Astronex `post_train/data/dataset.py::build_viewmats_and_Ks` | Converts stored w2c poses into coordinates relative to the first camera | Origin/orientation alignment; input translation scale is preserved |
| Astronex `utils/camera_trajectory.py` | Constructs native commands with a 0.08 translation increment and 3-degree rotation increment | Native navigation command convention; no metres-per-unit claim |
| WorldPlay `trainer/dataset/ar_camera_hunyuan_w_mem_dataset.py::camera_center_normalization` | Re-anchors poses to the first camera | Preserves input scale; no depth-derived scalar |
| WorldPlay `hyvideo/generate.py::pose_to_input` | Inverts supplied c2w poses, normalizes K, and computes discrete action IDs | Camera encoding, not metric calibration |
| WorldPlay native pose-string parser | Uses a 0.08 translation increment | Native navigation step, not physical distance |

The WorldPlay training loader inspected here belongs to the released HY-video training path. Our historical WorldPlay runs used the Wan 5B variant. Wan imports the shared inference pose utilities, but the inspected 8B training code alone does not establish every 5B training-data convention.

The Astro training loader and minWM geometry helper contain no scene-depth normalization in the inspected functions. The paper identifies Control2V as camera-control training data. The available minWM generated-video preencoder stores command-derived pose labels without a metric-depth conversion. This supports a command-coordinate interpretation for that path; it does not prove that every training corpus shares one physical scale. [Astronex paper](https://arxiv.org/html/2609.20034v1), [pinned loader](https://github.com/Astronex-Robotics/Astronex-World/blob/27584bf1a89da01ef35a03b48d83bc818d2adf2b/post_train/data/dataset.py).

WorldPlay explicitly motivates its combination of discrete actions and continuous camera poses by scene-scale variation. That design is consistent with useful navigation without a universal metric camera unit. It is not evidence that a chosen scalar converts those units into metres. [WorldPlay paper, section 3.2](https://arxiv.org/html/2512.14614v1#S3.SS2), [pinned loader](https://github.com/Tencent-Hunyuan/HY-WorldPlay/blob/1588e1336e842b03b0a7860c654ebd7c46bb065e/trainer/dataset/ar_camera_hunyuan_w_mem_dataset.py).

Our old `analysis/plan_base_restart_20261008/long_domain_20261028/prepare_minwm.py` explicitly measures recorded reference frames and regresses their observed translations against native command labels. The resulting `native_units_per_anchor_unit` is therefore a reference-motion fit. It is unavailable to the intended initial-image-only deployment and is not an independent physical unit calibration. The newer bundled launcher correctly stopped consuming it.

## Two distinct calibration questions

Let `c` denote native command displacement, `m` measured displacement in the fixed initial-anchor reference gauge, `k` native coordinate units per reference unit, and `g` execution gain. Under a simplified scalar response model:

```
m = (g / k) * c
```

Observed command/motion pairs can identify the combined response `h = g/k` under suitable excitation and model assumptions. They cannot separately identify `g` and `k`: multiplying both by the same positive number produces identical observations. A monocular metric-depth estimate can anchor the observer's geometry, but it does not establish the generator's native-unit meaning.

This refines the previous implementation decision: missing physical unit evidence should block claims of calibrated translation and measured-translation memory writes. It does not rule out a separately designed command-response controller operating through the native interface.

### Physical unit calibration

A legitimate runtime geometry provider requires either:

- A documented native translation-to-metres contract for the exact checkpoint, combined with a fresh metric estimate on the input image; or
- Actual native depth/geometry tied to the camera encoder's coordinate system, paired at corresponding input pixels with reference depth.

Neither was established for these two checkpoints by this audit. Running an unrelated depth model and calling its output “native depth” does not satisfy the requirement. Estimating a global scale from generated commanded poses likewise assumes the command was executed, which is precisely what the controller is supposed to test.

### Operational command-response calibration: proposed next experiment

1. Keep the desired path and observer measurements in a fixed initial-anchor gauge. Fix the evaluation target before any output-dependent calibration.
2. Start ordinary rollout chunks using the model's declared native command interface. If a bootstrap displacement is required, read the native navigation convention from the adapter; do not label that bootstrap physically calibrated. Preserve requested directions, rotations, and timing. Count these chunks in the evaluated rollout.
3. Observe emitted frames only. Record actual issued native displacements, delayed observed reference-gauge displacements, prior motion, discrete actions, and memory policy. Account for frame-to-latent indexing and express corresponding increments in consistent axes.
4. Fit a local forward-response model from these ordinary chunks. For example, `delta_m = h * delta_c + b`, with additional history terms where supported. Use multiple naturally varying commands, numerical rank/conditioning checks, observer-quality gates, and prediction on subsequent chunks before trusting an inverse. One large motion pair or high direction cosine is insufficient to separate command sensitivity from history-driven motion.
5. Use the resulting predictive model for bounded future command corrections. A nonzero average `motion/command` ratio does not prove a nonzero differential response. If command variation is insufficient, the fit is unstable, or response stays near zero, abstain; a probe-free method cannot promise identifiability on every trajectory.
6. Re-estimate only the response model during rollout. Keep the observer/evaluation gauge fixed. Do not rescale measured trajectories to make them match commands.

This can remain training-free and use no extra probe generations. It is an adaptive command-control route, not a substitute physical unit conversion. Its success must be tested on these models; no tracking improvement is claimed here.

## Memory consequences

Do not use inverse response gain as the conversion for measured translation labels. That would absorb execution errors into the memory coordinate system, and changing the factor at each boundary could make retained history inconsistent. Until a physical/native coordinate mapping or separately validated memory-coordinate protocol exists, keep translation labels in their native commanded gauge.

Rotation-only measured-label updates do not need a translation scale, but still require correct rotation conventions, frame correspondence, observer quality, and appropriate cache rewriting. Astronex's KV pathway and WorldPlay's pose/FOV/action pathway require separate qualification; neither is automatically enabled by this audit.

## Numerical validation and current status

`analysis/astronex_worldplay_scale_20261102/check_native_units.py` extracts and executes the actual native camera functions on CPU without initializing their model pipelines. **19 checks passed**:

- Both training pose-normalization functions preserve arbitrary input scales 0.25, 1, and 7 while correctly re-anchoring rotations.
- WorldPlay's inference pose encoder preserves those scales.
- WorldPlay's discrete action labels are identical for positive 0.25x and 7x forward commands above the movement threshold; continuous camera values still differ.
- The native WorldPlay parser uses the stated 0.08 step.
- A constructed counterexample verifies unit/gain ambiguity and rank deficiency for constant commands with an unknown motion offset.

These are interface and identifiability checks, not empirical generator-response measurements. The existing fail-closed native-geometry launchers remain in place. The command-only bootstrap/controller mode described above was subsequently implemented as a separate opt-in pilot; see the follow-up below. GPU qualification remains pending.

## Implementation follow-up

The command-response route is now implemented as a separate opt-in mode and
prepared for a frozen GPU pilot. See [online response](online_response.md) for
the exact estimator, guards, A/S/C controls, queue, and pending validation.
The source-audit conclusions about unidentified physical units and measured
translation memory remain unchanged.
