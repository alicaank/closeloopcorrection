# Online command-response pilot

Implemented 2026-11-02 as an opt-in `--online-response` mode in the bundled
Astronex and WorldPlay launchers. The native-geometry provider requirement still
applies to the separate physical-unit mode. This experiment learns operational
command response; it never claims to have identified native units in metres.

## Runtime behavior

The target path and observer anchor depth remain fixed throughout the rollout.
The adapter reads its released native navigation step at runtime and uses it to
bound the maximum planned translation step by that native step. A tiny first
step cannot inflate later commands.
This is a declared bootstrap command mapping, not a measured conversion. Pure
rotation plans do not invent a translation scale. There are no separate probe
generations and no scale fitted from a recorded reference video.

At every decoded chunk boundary, a separate DA3 worker observes only selected
emitted RGB, K, and initial-anchor depth. The worker has no command or target
inputs. Depth registration uses the actual DA3 input preprocessing intrinsics.
Geometric support and repeated-frame disagreement gate the measurement. The
latter is a noise proxy, not a calibrated statistical confidence interval.
Python, NumPy, and Torch RNG states are isolated from observer work.

For each anchor-coordinate translation axis, the online model is

```
measured_velocity = h * issued_native_velocity + beta * previous_measured_velocity + b
```

Velocities are normalized by the original plan's reference/native speed
envelopes. Each sample is a chunk-average velocity. Only duration-matched samples are
fitted, and the model is inverted once for the next matching chunk, never
recursively once per latent. Weighted least squares is recomputed with
forgetting 0.9 and at most 32 observations. The additional history and offset
terms prevent constant motion from automatically becoming command sensitivity.
At least four samples, command variation, a full-rank well-conditioned design,
a supported positive slope, and two subsequent successful forward predictions
are required before that axis can drive inverse correction. A failed prediction
or rejected observation immediately disables its use. Constant commands may
remain unidentifiable indefinitely; more boundaries alone are not sufficient.

Only future native translations are modified. Reference tracking residuals are
spread over 64 decoded frames. Commands obey a speed ceiling of twice the
original native-plan maximum and a 12-frame acceleration horizon. Native
rotations retain their planned values. After the first intervention, slew and
speed bounds remain active even when an observation is rejected or an inverse
fit loses identifiability. Measured translations are never written
to memory; previous issued cameras are immutable. These common, declared pilot
thresholds are experimental choices, not established universal stability bounds.

The implementation is `online_response.py`; native hooks and observer IPC are
in `native_response_runtime.py` and `response_observer.py`. `ResponseConfig`
records all shared thresholds. Receipts record every fit, abstention reason,
forward innovation, active axis, clipping event, and generated-prefix hash.

## Frozen queued experiment

Directory: `analysis/online_response_v2_20261102/`. The superseded v1 snapshot
was stopped before any job began and is preserved.

- Models: Astronex and WorldPlay Wan 5B.
- Development scene: `0238de14`; seeds 7 and 107.
- Output: 20 native chunks per model. Astronex: 160 latent cameras / 637 RGB
  frames; WorldPlay: 80 latent cameras / 317 RGB frames. Each A/S/C comparison
  has the same horizon; across models chunk counts match, RGB counts do not.
- At least eight future chunks with active inverse control are needed for the
  sustained-correction duration check. Constant commands can still cause
  abstention. A duration-qualified result is not automatically an improvement.
- Seed 7: A baseline, S identity sham, C online response, for each model.
- Seed 107: matched A/C, for each model. Total: ten generation jobs.
- A and S observe and fit exactly as C does, but leave native cameras unchanged.
  Their decoded prefixes must match exactly before C proceeds. A/C must share
  the first decoded chunk. These checks qualify the experimental adapter,
  not parity with an uninstrumented upstream executable.
- Four independent VIPE A/C evaluation jobs follow generation groups. They use
  anchor-depth registration without fitting output trajectories to targets.
  Exact RGB/K/depth inputs share a content-addressed observer estimate under
  the frozen evaluator contract. WorldPlay scoring uses its actual resize/crop intrinsics. Invalid registration
  keeps position scores unavailable. DINO/LPIPS evaluation can read recorded
  reference images; the generation observer cannot.
- Reports explicitly distinguish no adaptation, adaptation applied, and
  independently measured tracking outcomes. No correction efficacy claim is
  made merely because the controller fit coefficients.

The queue takes the same locks as the existing Spatia/continuous-evaluation
queues and waits for the GPU to become idle. It checks source/input hashes and
checkpoint size/mtime before each job. Launch reserve is 8 GiB; runtime guards
stop the owned job below 4 GiB disk or 2 GiB available host memory. No checkpoint
or existing experiment is removed. The manifest and CPU qualification receipt
bind the launched sources; later repository edits do not change this snapshot.

This is a one-scene, two-seed pilot. It does not establish cross-scene
calibration or improvement, and GPU outcomes remain pending while queued.
