# VMem legacy failure replay — 2026-11-03

The historical VMem result supports the combined intervention on average, but
does not establish that command correction alone works. A CPU replay of all
40 saved runs matches every command exactly. The latest-memory coordinate bridge
is algebraically correct. Large recovery requests, context-dependent camera
normalization, and observer disagreement remain plausible contributors to failure;
these archives do not separate their causal effects.

[Replay script](artifacts/legacy_rejoin_failures_20261103/vmem_replay.py) and
[results with 107 input hashes](artifacts/legacy_rejoin_failures_20261103/vmem_replay.json).
Run the script using `baselines/venv_vmem/bin/python` in the original workspace.
No model generation is needed. The original scores and runs are unchanged.

## Actual archived protocol

All 40 run metadata files and recorded launch arguments specify
**`--scale_mode window --nms cfg --first_step 4`**, with 40 generated views,
seed 7, and 50 sampling steps. They do **not** use the driver's current
`global_demo` default or its fixed `0.025 * 22.867` demo-step factor.
References to frozen demo normalization do not describe this cohort.

The inspected native implementation centers the retrieved context and future
poses together. Its scale is `camera_scale / ||centered_first_context|| + 0.01`,
except near zero where it uses `camera_scale`; the inspected config sets
`camera_scale=2.0` and enables NMS. This is native window normalization, not a
measured causal physical-unit calibration. `cfg` delegates retrieval policy to
the config; the generation-time config snapshot and selected context IDs were
not archived. Current inspected source hashes are not frozen runtime receipts.

The queue entry for `989b9344/replan_measured` was left marked `running`.
Its completion log, full metadata and scored 41-view trajectory exist. The audit
records that discrepancy rather than treating the stale state as live work.
All 40 completion log lines identify the correct arm, scene and horizon.

## What the four arms establish

| Arm | Mean position error | Change from baseline |
|---|---:|---:|
| Baseline | 0.67215 | — |
| Command correction, commanded memory | 1.30199 | +93.71% |
| Measured memory, unchanged future commands | 0.52000 | −22.64% |
| Command correction plus measured memory | 0.50369 | −25.06% |

These are historical VGGT measurements in anchor-depth units on ten scenes,
one seed, with an outbound/return reference. Adding command correction to
measured memory lowers mean error in **5/10 scenes** and lowers the aggregate
mean by **0.01632** (about 3.14% relative to memory-only). This is descriptive
paired evidence, not a confidence interval or a newly qualified factorial trial.
All four saved online pose prefixes are identical through frame 4 in every scene;
that check does not establish raw-RGB or full observer-sham identity.

## Coordinate bridge: correct locally, incomplete evidence about all history

Let `M` be the measured boundary pose, `L` the latest stored memory label, and
`D_i` the physical rejoin target. In the command-only arm the driver sends
`U_i = L inv(M) D_i`, so

`inv(L) U_i = inv(M) D_i`.

The audit verifies this equality at every corrected boundary, reproduces all
stored commands with **zero maximum difference**, and verifies that each
physical rejoin endpoint equals its planned endpoint. All earlier retrieved
labels need not share the same alignment as the latest one. Since retrieved IDs,
window scales and conditioning tensors were not recorded, this check cannot
establish full-history conditioning consistency or native input amplitudes.

The old logged `||M_i − U_i||` residual compares poses in different coordinate
systems in the bridged command-only arm. It is not physical tracking error.
For `74310829` at frame 40, that diagnostic is about **7.607**, while online
`||M_i − P_i||` is **0.610**. Offline VGGT reports **7.627** against the plan.
The last two values expose observer disagreement; their numerical relationship
does not validate the mixed-coordinate diagnostic. The published scores use the
plan, so the four-arm table is unaffected.

## Failure evidence

| Scene / arm | Mean VGGT position error | Mean online DA3 position error | Maximum intended step / plan step |
|---|---:|---:|---:|
| `74310829` baseline | 1.0638 | 0.2165 | — |
| `74310829` command-only | 5.2871 | 0.8306 | 39.45× |
| `74310829` combined | 1.5655 | 0.2275 | 14.62× |
| `d3f042a2` baseline | 0.3350 | 0.3229 | — |
| `d3f042a2` command-only | 1.1448 | 1.5076 | 13.31× |
| `d3f042a2` combined | 0.5030 | 0.5136 | 7.94× |

Step ratios reconstruct the physical curve with its measured boundary anchor,
**before window normalization**. They are not observed motion, normalized native
amplitudes, response gains, or proof of a saturation threshold. Recovery occurs
over only four generated views. The common endpoint constraint can therefore
produce large requests when accumulated error exceeds the planned displacement.

Three existing four-arm sheets were inspected at frames 0, 7, 13, 20, 27, 33, 40:

- [`74310829`](artifacts/legacy_rejoin_failures_20261103/vmem_74310829_four_arms.jpg):
  the reference enters the kitchen. All generated arms miss that path. Command-only
  develops distorted close-up surfaces; baseline and both memory arms return near
  the initial patio view. A good return endpoint does not repair the missing path.
- [`d3f042a2`](artifacts/legacy_rejoin_failures_20261103/vmem_d3f042a2_four_arms.jpg):
  the reference approaches the kitchen counter. Command-only ends at ceiling/light
  close-ups; the other arms remain broadly kitchen-like and return near the start.
  Both memory arms have intermediate layout changes and take the wrong view.
- [`e43058cb`](artifacts/legacy_rejoin_failures_20261103/vmem_e43058cb_four_arms.jpg):
  a control example where command-only improves mean position (0.5178→0.4388),
  yet all arms miss the reference kitchen traversal. Command-only fails to return
  to the initial dining view. Pose improvement alone does not qualify visual success.

These are sampled visual observations, not a full-video review or independent
camera ground truth. Online DA3 and offline VGGT use the declared anchor gauge
but different windows and estimators. For `74310829`, their mean translation
disagreement is 4.558 in command-only and 1.407 in combined. For `d3f042a2`,
both observers agree on the direction of regression. A universal observer-scale
explanation is therefore unsupported, as is an inverse gain fitted from these
passive rollouts.

## Memory writes affect more than labels

The driver writes measured poses into `pipe.c2ws` **before** calling
`construct_and_store_scene`. The native pipeline consumes these poses for scene
reconstruction, appends surfels for new frames, and later uses surfel visibility
and pose distances to select context. Therefore the memory arm changes camera
conditioning and potentially reconstruction/retrieval. The archive does not
isolate a pure label-only mechanism or verify consistency of retained surfels.

## Next discriminating test

Keep the original four arms and failing scenes. First qualify an observer-sham
run and record selected memory IDs, all context/future poses before and after
normalization, applied scale, measured-boundary residual and scene-validity flags.
Then compare legacy four-view recovery with a longer recovery horizon that still
permits catch-up and return after overshoot. Keep normalization and memory policy
fixed within each comparison. A separate normalization ablation can test whether
retrieval-dependent scaling contributes; changing both together would confound it.

Retain aggregate benefit, per-scene failures, rotation, visual fidelity and full
path error in the report. Failure on some scenes does not disqualify a useful
general method; unexplained protocol differences should not determine its design.
