# DreamX command-only pilot: frozen and queued

This is a separate ablation of future-command correction. The primary
measured-memory v1 method and its original 40-pair cohort remain unchanged.
The pilot tests whether observed lag can usefully change subsequent motion
without rewriting DreamX's native KV memory. Passive execution ratios are
logged; they are not differential gains and do not enable inverse control.

The frozen design has **12 generations**: scenes `0238de14` (moving development)
and `02267acf` (near-stationary stress), seeds 7 and 107, and three arms per case.
Every generation retains the full 69 RGB frames / 18 latents. The prepared
[native camera grid](dreamx_camera_grid_20261103.md) is shared across all arms.
This changed input/FOV contract prevents pooling with historical DreamX runs.

| Arm | Observer | Future commands | Memory |
| --- | --- | --- | --- |
| A | Off | Original plan | Commanded |
| S | On | Original plan | Commanded |
| F | On | Nominal first feedback; subsequent qualified fractional recovery | Commanded |

## Causal issuance and identification limits

The callback runs after the unchanged native context-cache pass. At the first
feedback boundary, RGB is available through frame 20 and the native camera
anchor is frame 17; F leaves the next command nominal. At the next boundary,
RGB ends at 32, the native anchor is 29, and the next native camera samples are
33, 37 and 41. Thus frame **33** is the first potentially affected RGB frame.
Only the next native chunk's tensors are committed; past and later tensors
remain exact. The anchor bridge is described in the
[timeline preflight](dreamx_future_preflight_20261103.md).

The frozen recovery settings are fraction **0.5** and speed budget **2× the
planned native step**, with return motion allowed. These are explicit pilot
hyperparameters, not learned gains, universal optima, or model-specific fitted
calibrations. They supply catch-up headroom absent from the earlier non-reversal
arm. Observation rejection retains absolute commands, but the next relative
camera slice is rebuilt against the last committed anchor. Receipts distinguish
requested correction from a changed native encoding.

DA3 sees emitted RGB and the prepared initial anchor geometry only. Temporary
observer residency is released before continuation; Python, NumPy and Torch RNG
state is preserved. The execution-ratio receipt compares command and measured
motion over the same native-anchor interval. One naturally excited rollout
cannot establish a causal response curve or separate memory/history effects.

## Gates, scoring and current status

1. A and S must match all 69 RGB frames exactly before F is run.
2. S and F must match frames 0–32 and latent/cache/RNG receipts through latent 9.
3. Initial camera and noise inputs must match; the first feedback command is nominal.
4. Independent offline VIPE reports mean position error over frames 33–68,
   endpoint error, rotation, actuation counts and sampled visual evidence.
   Original targets, timestamps and anchor-depth units are retained.

Four CPU tests using the actual released encoder pass, covering nominal parity,
the explicit native anchor, next-slice isolation, rejected-observation rebasing,
and invalid timestamp/intrinsics rejection. The actual source overlay compiles
and native CLI startup passes. **GPU parity and efficacy remain unverified.**

The live queue was verified waiting for the standard GPU lock at this snapshot.
It has 36 jobs including gates, retention and scoring; required gate failures
hold the affected work. Scientific losses and abstentions remain reportable.
Manifest SHA256:
`226aab81a0601d1476be1b33444aa4cad0dcb4d3b8ee3ccb9912ef7f3cfefb33`.

[Protocol](artifacts/dreamx_future_command_20261103/protocol.json),
[CPU receipt](artifacts/dreamx_future_command_20261103/cpu_validation.json),
[queue snapshot](artifacts/dreamx_future_command_20261103/queue_snapshot.json),
and [source](artifacts/dreamx_future_command_20261103/future_hook.py) preserve
the design and provenance. These are workspace-specific audit copies; the
manifest identifies the complete frozen runtime and inputs in
`analysis/dreamx_future_command_20261103/`.
