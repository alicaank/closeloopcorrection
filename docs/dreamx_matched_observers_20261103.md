# DreamX matched-image observer diagnostic

This diagnostic is queued under the existing shared GPU locks. No GPU result is
available at this snapshot. It follows the observation-pipeline disagreement in
the [issuance audit](dreamx_future_control_audit_20261103.md), with an explicit
correction to the interpretation: the original DA3 and VIPE inputs have different
RGB preprocessing as well as different temporal context.

## Fixed comparison

Use the two completed moving-scene F videos (`0238de14`, seeds 7 and 107), at
available frames 20, 32, 44 and 56. At each boundary:

1. Reconstruct the online sampler's 16 source-frame indices and select those
   images from the losslessly retained final exported video.
2. Run DA3 and VIPE on exactly those same 16 RGB images, with the same calibrated
   input intrinsics and initial anchor depth. Neither receives target poses.
3. Run VIPE on the complete causal prefix from frame 0 through that boundary.
4. Compare the boundary poses against each other, the stored full-video VIPE
   result and the recorded original online DA3 pose. Targets enter only the
   subsequent residual-direction diagnostic, never estimation or registration.

This gives eight windows and 24 estimator calls. Rejected registration and
estimator failures are retained. Both selected-image and full-prefix RGB hashes
are recorded. Position comparisons require valid registration in both pipelines.
There is no trajectory fitting, new calibration constant, command modification,
or gain tuning. Original generation and scores remain unchanged.

## Interpretation limits

The retained video has native chunk-relative Lab color correction at strength
1.0; original online DA3 saw raw decoder output. Original observer input RGB and
latents were not retained, so this diagnostic cannot recover exact original
pixels. Comparing the DA3 rerun with its old estimate mixes preprocessing and
possible numerical/decoding differences. It is not a clean color-only ablation.

The selected-image VIPE input has irregular gaps between original frame indices.
The existing worker receives their ordering at its nominal FPS; it does not
receive their original timestamps. Sparse-view failure therefore cannot be
interpreted as broad inferiority of VIPE. The full-prefix arm separately tests
its causal operation on consecutive exported frames. Neither observer is ground
truth, and two seeds of one development scene do not establish generalization.

## Verification and execution

The CPU preflight exercises the actual frozen DA3 selector using image IDs and
matches all eight reconstructed 16-view windows exactly. Both scripts compile.
This validates selection and startup syntax, not GPU efficacy. The manifest
hashes 24 source/input files, including both videos, logged observations and
existing offline trajectories. The runner uses the standard five shared GPU
locks, 4 GiB launch reserve, 2 GiB disk/host runtime reserves and a 90-minute limit.

[Frozen manifest](artifacts/dreamx_matched_observers_20261103/manifest.json) ·
[CPU selection validation](artifacts/dreamx_matched_observers_20261103/cpu_validation.json) ·
[Diagnostic source](artifacts/dreamx_matched_observers_20261103/compare.py) ·
[Queue runner](artifacts/dreamx_matched_observers_20261103/run_queue.py) ·
[Launch receipt](artifacts/dreamx_matched_observers_20261103/launch.json).

These are audit copies. The live workspace job is
`analysis/dreamx_matched_observers_20261103/run_queue.py`; its `status.json` is the
authoritative execution state.
