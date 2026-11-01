# Shared controller audit and next design

The common problem is controlling an uncertain generator through uncertain camera measurements while its history is already cached. Scale calibration is one part of that contract. A controller can use correct units and still request an abrupt reversal, trust a drifting observer, or express a future camera in a different frame from its memory.

The shared pilot now implements `AdaptiveController` and causal window evidence alongside the existing `CorrectionSession`. It adds bounded native motion, velocity transport, soft recovery across chunk boundaries, repeated-frame pose checks, depth/photometric support and conservative directional response fitting. LingBot and SANA have isolated pilot adapters; the older research launchers retain their original controllers. GPU efficacy remains unverified.

In this new pilot, B uses the shared evidence and bounded recovery with identity response. C uses exactly the same policy plus supported diagonal response adaptation. These B/C definitions differ from the older pilot; comparisons to old results must keep their protocol labels. Recovery and reliability thresholds are predeclared experimental settings, not calibrated universal safety guarantees. Native speed limits follow the original planned conditioning envelope, including its model normalization. Uncommanded directions with zero speed budget remain constrained.

## Reproduced bugs fixed in this working tree

| Issue | Reproduction and consequence | Patch coverage |
| --- | --- | --- |
| Rejoin uses sample count instead of elapsed frames | At boundary 10, samples 11, 12, 20 faded a translation residual of 2 to 1.333, 0.667, 0. Expected time-based residuals are 1.8, 1.6, 0. Adding intermediate samples changed commands at the same timestamps. | `CorrectionSession` uses elapsed output-frame IDs for translation and rotation; low-level `rejoin` accepts explicit progress. Uniform schedules retain their previous fade. Research adapter rejoin functions are not automatically migrated. |
| Missing observations pass pose quality | Both `pose_quality([])` and `pose_quality([None, None])` returned `passed=True`. This establishes a quality-check bug, not evidence that such a report reached a particular generator. | Portable and bundled geometry-quality helpers require at least one observed pose. Omitted context remains allowed. Endpoint validity is checked separately by `Measurement`. |
| Sham depends on observer success | A rejected measurement raised before the sham branch, preventing an observer-failure control run. | Portable sham bypasses measured-pose/quality rejection, logs the rejected quality report, and still enforces timestamps and future-only commands. Host observer exceptions and mutable state still require host isolation. |
| Frame validation has integer edge cases | `np.diff` on unsigned IDs `[3,2,1]` wraps and passes the increasing-order check; `False` and `0.0` could serve as available-frame IDs. | Direct ordering comparisons and explicit integer validation. |
| Rotation-only target gets a huge translation ratio | Dividing measured travel by `max(target_travel, 1e-8)` creates a numeric ratio when target translation is zero. | Bundled VIPE pair/long scoring returns a null ratio with a reason and retains both absolute path lengths. The threshold is numerical, not an observer noise estimate. |
| Optimizer saturation can become scale evidence | Constant native depths 0.1 or 1000 passed the packaged VIPE registration, producing scales 10 or 0.001 against unit reference depth. Those depths are the pinned optimizer's inverse-depth clamp endpoints. | Packaged VIPE registration excludes saturated values before coverage and scale checks. The scoring protocol is named `vipe_dense_anchor_depth_no_saturation_v2`. It rejects unsupported registration; it does not manufacture positional scores. |

Code: [controller](../src/camera_loop/controller.py), [quality](../src/camera_loop/quality.py), [contract tests](../tests/test_controller_contract.py), [packaged VIPE geometry](../src/camera_loop/bundled/templates/analysis/plan_base_restart_20261008/vipe_eval_20261030/camera.py).

Validation: all 45 top-level CPU unit tests pass, including a relocated evaluator subprocess with nine geometry/scoring tests and compilation of all 11 bundled launcher paths. The regression cases for time-based fading, empty observations, sham rejection and frame validation were first run against the previous implementation and failed. Saturated-depth acceptance was also reproduced before its patch. Run `CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python -m unittest discover -s tests -v` in the package environment to repeat the checks. No new GPU inference was run for these fixes.

The fixes are local source changes. Existing materialized workspaces, completed results and queue snapshots retain their original code and scores. The packaged evaluator still does not incorporate the later experimental native-map-mask diagnostic or the full unavailable-position reporting path. Its fail-closed scale handling must not be confused with that later diagnostic protocol.

## One controller, explicit state and adapter contracts

Maintain these quantities separately at every decoded output timestamp:

| Quantity | Meaning | Mutation policy |
| --- | --- | --- |
| `P_t` | Original timestamped target | Immutable throughout evaluation. |
| `M_t` and its uncertainty | Accepted estimate of actual generated camera state | Update only with causal geometric evidence. Preserve observation provenance and latency. |
| `U_t` | Camera label actually stored for the generated image/cache | Retain unless the adapter explicitly rebuilds every affected cache and geometric representation. |
| `u_t` | Conditioning actually consumed by the model after native encoding | Record after units, basis, normalization, sampling and clipping. Use this for response estimation. |

With commanded history and full correction gain, a boundary bridge must satisfy

```text
inverse(U_t) @ C_i = inverse(M_t) @ D_i
```

Here `D_i` is a desired future physical pose and `C_i` is the pose expressed for the existing memory. CPU tests now exercise this identity through relative-chain encoding in c2w/w2c, OpenCV/OpenGL and three scales, with nonidentity stored and measured boundaries.

This identity is necessary at the boundary; it does not prove that an entire historical map is consistent. If different past views have different errors, a single rigid transform cannot repair all of them. An adapter must expose which history supports each prediction and test relative reprojection for that history. Reject incompatible evidence or explicitly rebuild the affected representation; silently changing pose labels on an unchanged KV cache is not a generic solution.

For the Memorizon plus Gaussian-memory direction, preserve Memorizon's bank and its original conditioning provenance. Gaussian observations can carry measured poses, confidence and source-frame IDs separately. Reading Gaussian evidence must use the appropriate transform into the current conditioning frame. Treating an uncertain measured pose as exact ground truth in both memories would propagate the same observer error twice.

## Accept observations from evidence, not from target agreement

The current anchor-depth and proper-SE(3) checks cannot establish current camera accuracy. In addition, compare the estimated poses of identical frames shared by consecutive causal observer windows after registration. Use supported image tracks/reprojection, spatial coverage, occlusion and depth-fit residuals. Report which degrees of freedom have adequate support; rotation and translation can have different reliability.

A large difference from the command is not itself a rejection criterion: detecting that difference is the point of feedback. Conversely, a small command error is not evidence that a textureless or collapsed video has a valid reconstruction. Initial-anchor scale fit should not substitute for current overlapping geometry.

Estimate an observation noise floor from repeated/overlapping measurements and matched no-correction controls. Attenuate innovations that are not distinguishable from that uncertainty. On rejection, stop incorporating new innovations and stop fitting the response model. Propagate the last accepted state through issued commands with growing uncertainty and bounded motion. Do not label an old pose as a fresh endpoint or suddenly switch to incompatible nominal absolute commands.

## Choose feasible future motion and allow residual lag

The existing boundary rejoin still requires reaching the chunk endpoint. The timestamp fix does not solve that design limitation. The next planner should choose a return duration over multiple chunks while retaining the original target times in its error objective.

Optimize a short sequence of future motion increments for target tracking, smooth native input changes and uncertainty. Constrain the actual encoded conditioning through the adapter, including the boundary from the last issued command. Extend the recovery duration when those constraints would bind. Carry the remaining tracking error into the next update; do not accumulate an additional integral correction that was already clipped away.

Compare physical velocities in a common coordinate frame and divide by actual elapsed time. Differences between vectors expressed in successive rotating camera frames are not automatically physical acceleration. Native-input slew can also be constrained, but it is a different quantity and should be labelled as such.

Bounds must come from the checkpoint's native camera encoding and validated response range. A native magnitude of 1, a fixed execution ratio of 1.05, or an inverse-gain floor is not a universal safety threshold. A larger model-specific recovery time can be a consequence of the same shared policy without introducing scene-specific controller branches.

## Fit only the response that is observable

Two scalar translation/rotation gains assume the model follows the requested axis. They cannot describe cross-axis response or reliable movement in one direction and failure in another. A regularized local response model is a possible extension, but a full 6-by-6 inverse from a few weakly excited chunks would be unstable.

Start from the nominal response. Use actual post-encoding commands and accepted measured increments in matching coordinates and intervals. Require directional consistency and sufficient excitation; check prediction residuals on later windows. Update only supported directions and retain prior uncertainty elsewhere. Suspend learning when observations conflict, commands are persistently constrained, or the model's appearance has collapsed. Do not convert a low-confidence small measured motion into unlimited inverse-gain amplification.

## Treat observation as a transaction and qualify adapters

Sham should run the same observation, decoding and offloading path as correction while preserving native commands. Check RNG state and model state: explicit generators, VAE caches, KV caches, training/eval mode, scheduler state and mutable camera buffers. Fingerprint the actual sampled noise, conditioning and raw chunk outputs to locate the first divergence. Compare native/native replay before interpreting a native/sham mismatch; the backend itself may be nondeterministic.

This matters beyond one model. The latest WorldWarp all-global-RNG isolation run still differs from the saved native reference by raw pixel MAE 2.2008 over 161 frames, while baseline/sham and the first baseline/correction prefix match. The RNG-only explanation is insufficient. WorldWarp remains held; no corrected efficacy claim follows from that comparison.

The completed LingBot rotation-gain-floor pilot is also not evidence of a general cure. On `e43058cb`, C's mean rotation error is about 24.36 degrees versus 24.12 previously, and B is about 15.94. On `74310829`, C is about 37.82 degrees versus 36.08 previously, with position unavailable. These are single-seed diagnostic comparisons with separately rerun evaluation, so tiny changes are not established improvements or regressions. They support testing observability and command feasibility rather than assuming another scalar limit resolves the failure.

## Next controlled comparison

1. Qualify one adapter against the shared coordinate, timestamp, history and sham contracts. Then use the identical controller with a second structurally different adapter. Keep per-model conversion declarations, not per-scene correction logic.
2. Compare the current correction with evidence-weighted feedback, then add feasible multi-chunk recovery as a separate factor. Retain baseline/sham and the same scenes, seeds, target times and model sampling budgets.
3. Include held-out motion regimes: nearly stationary, rotation-only, translation, mixed motion, weak texture, disocclusion and continuation boundaries. Use stored observations for CPU replay first; distinguish replayed controller behavior from generated-video outcomes.
4. Score translation, rotation, timing lag, image quality and geometric/temporal consistency separately. Retain registration failures and visual collapse in the denominator. Do not align each generated trajectory to its target or time-warp away lag.

A useful result is improved tracking and consistency without increased visual collapse across models and unseen scenes. Passing these CPU contracts establishes implementation behavior, not that the proposed controller improves generated videos.
