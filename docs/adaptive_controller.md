# Shared adaptive-controller pilot

`AdaptiveController` is the new stateful experimental controller. `CorrectionSession` remains available with its existing boundary-rejoin behavior. The pilot connects the same new controller to LingBot's normalized relative cameras and SANA's absolute cameras/rays.

The controller retains the original timestamped plan, a separate predicted/observed state, stored command labels, and the issued native relative transforms. It writes future commands only. The controller never rewrites generator memory, and predictions are never presented as measured poses. A host with an existing measured-memory policy can explicitly supply the label under which it has already rebuilt its memory.

## Controller behavior

`AdaptiveController(plan, frame_ids, alignment, arm="B", initial_count=...)` starts after an existing generated prefix. `frame_ids` are decoded output frames, not implicit latent indices. The declared `CameraAlignment` supplies units, camera basis and direction. If an adapter already normalizes its relative inputs, pass those exact values through `initial_native_relative`; keep the same normalization for subsequent calls.

`next_chunk(lo, hi, measured, evidence)` requires an observation report at `frame_ids[lo - 1]`. Its return value contains native relative matrices and an event record. Hosts requiring absolute poses use `controller.commands[lo:hi]` and their native encoder. Test the encoder's history prefix for exact equality before committing updated future tensors.

For measured-memory hosts, pass `stored_boundary=actual_memory_c2w` in canonical reference coordinates **after** the native memory rebuild. Future commands are encoded relative to that actual label, which may differ from the feedback observer's estimate. The issued-command history remains immutable. Omitting the argument preserves commanded-memory behavior. Supplying a measured pose without rebuilding the corresponding host memory is not a valid integration.

If native camera timestamps lag decoded output, pass `available_last_frame`. The controller checks that the next camera timestamp is strictly later than all generated RGB and logs the observation delay. DreamX uses camera anchors `[0, 1, 5, ...]` and can have three already-decoded frames after its latest camera anchor. This interface represents that delay; it does not estimate the unobserved motion during those three frames or guarantee smooth decoded motion.

The default speed envelope comes from the original native planned inputs divided by their actual frame intervals. The controller bounds native translation/angular speed and their changes, transporting the prior velocity into the next camera basis. It increases the recovery time with the size of the residual and available motion budget, and never forces the error to disappear at a chunk endpoint. A zero planned motion budget stays constrained. Limits can be declared explicitly, but must cover the already-issued initialization. These bounds are a pilot operating envelope, not a proven safe range for every checkpoint.

For rejected observations, prediction continues from the last accepted state through issued commands. No new residual correction or response fitting is added. This may preserve lag and cannot guarantee accurate tracking through a long observation outage.

## Evidence and B/C arms

The pilot observer receives only causal RGB, intrinsics and anchor depth. It keeps up to 16 views, preserving four non-anchor views from its last accepted window. After anchor-depth registration, it checks:

- Proper poses and support for a scalar anchor-depth conversion.
- Pose agreement on at least three identical frames shared across windows.
- Spatially distributed textured pixels supported by projected depth and RGB agreement between recent views.

The projected-image check uses the observer's own geometry; it is supporting evidence, not an independent accuracy measurement. Thresholds are explicitly recorded by `EvidencePolicy`. They have not been calibrated as probabilistic uncertainty. Initial windows have reduced confidence because repeated-frame evidence is unavailable.

B uses this evidence, bounded recovery and identity model response. C uses the same controller plus a conservative diagonal response model. It fits only sufficiently excited, directionally consistent components after several accepted observations; response fitting pauses after clipped commands and rejected measurements. The allowed gain interval is predeclared. This is not a full camera-response Jacobian.

These are new B/C definitions. Earlier LingBot B/C results used different controllers and cannot serve as a matched control for all the new components together. The present B/C comparison isolates adding the directional response model within the shared policy.

## Queued experiments

The local isolated queue is `analysis/shared_controller_pilot_v2/queue_status.json`. Its manifest and CPU receipt bind a frozen copy of the package, adapters and input data hashes. Version 1 preserves the failed smoke-loader attempt; version 2 converts stored half-precision depth before resizing it.

The queue runs a real observer smoke check, then native/sham qualification for each model, followed by eight B/C scene/seed pairs:

| Model | Scene and seed | Frames generated / evaluated |
| --- | --- | --- |
| LingBot | e43058cb / 7, 107 | 173 / 160 |
| LingBot | 74310829 / 7 | 173 / 160 |
| LingBot | d3f042a2 / 107 | 173 / 160 |
| LingBot | a4e9bdb6 / 7 | 173 / 160 |
| SANA | 0f12b97e / 7, 107 | 169 / 160 |
| SANA | a4e9bdb6 / 7 | 169 / 160 |

Parity failure holds that model's comparisons. VIPE evaluates original timestamps in fixed anchor units with native map validity and saturation checks; unsupported positional registration remains unavailable. Raw video, observation decisions, commands, controller events, first-prefix checks and scoring provenance are retained. The queue is serial, uses the existing GPU locks and keeps an explicit disk reserve. It does not delete checkpoints or modify completed results.

CPU validation passed 56 package tests, including the relocated evaluator's nine internal tests, and direct checks against both native camera encoders. GPU qualification and measured efficacy remain separate results.

## Continuation and measured-memory follow-up

`analysis/zigzag_followup_v1` adds local DreamX and VMem adapters and a separate Spatia diagnostic. These compare the existing controller with shared arm B (identity response); they do not repeat the earlier B-versus-C experiment. The comparison changes the feedback policy, including its observer window, so it does not isolate each individual controller component.

VMem retains its native measured-pose memory writer in both correction arms. An additional evidence window supplies shared-controller feedback, while future commands use the actual label stored by the original writer. Evidence rejection stops new feedback corrections; it does not silently change VMem's existing memory-writing policy. DreamX changes future PRoPE tensors only and records the actual quantized matrices. Native quantization can perturb the full-precision controller bounds.

Spatia records per-round versus final joint-decode RGB, history re-encoding differences, and whether frozen history latents remain unchanged during denoising. Temporal grouping and causal VAE context differ between a re-encoded history and a saved latent tail; a nonzero difference alone is not evidence of a bug.
