# Astronex abstention and WorldPlay identity failure

This reviews the completed portion of `analysis/online_response_v2_resume_20261103/`:
scene `0238de14`, seed 7, 20 native chunks. It does not pool this command-response
pilot with measured-memory experiments. The queue stopped after WorldPlay S;
WorldPlay C and both seed-107 groups remain unexecuted.

## Verified Astronex result

The CPU audit decoded all three exported movies, checked every frame, compared
the recorded native commands, and checked all 20 pre-export observer-prefix
receipts. A/S/C are identical across all 637 exported RGB frames and all native
commands. The prefix receipts also match at every boundary. Those receipts hash
the decoded prefix with frame zero replaced by the fixed observer anchor; they
are not latent/RNG traces. Export equality is independently checked.

All three arms accepted 11/20 observations, activated zero axes and issued zero
corrections. The fit remained unsupported: axis-level reasons include insufficient
samples (9), unsupported positive slope (10), and insufficient command variation
(5). These are axis-event counts, not mutually exclusive boundary counts. Nine
observations were rejected for shared-frame position disagreement, at frames
348, 380, 412, 476, 508, 540, 572, 604 and 636. The requirement of eight actively
controlled future chunks was not met (actual: zero).

| Full-horizon metric | A and C, identical |
| --- | ---: |
| Mean position error, fixed anchor-depth units | 0.513330 |
| Endpoint position error, same units | 1.018151 |
| Mean rotation error | 57.7776 degrees |
| Endpoint rotation error | 89.8151 degrees |
| DINO | 0.755613 |
| LPIPS | 0.711247 |

Position registration passed the evaluator's gate. This does not validate
physical units or the entire generated geometry. The target poses are stored
batch references, potentially VIPE-derived, not asserted independent ground
truth. Identical evaluator inputs share a content-addressed estimate.

Direct visual inspection covered reference/A/C at frames 0, 106, 212, 318, 424,
530 and 636. The reference moves around the playground; generated views depart
from its composition, and late frames show distorted/duplicated railings and
smeared background structures. Both generated arms show the same artifacts.
These samples support visible degradation, not a precise onset, complete loss
of the scene, or a full-video motion/zigzag diagnosis.

The result supports safe abstention under the implemented gates. It does not
demonstrate successful online calibration, an improvement, or that active
correction would fail. Extending a rollout alone does not guarantee sufficient
command variation or separability from history. Loosening a gate merely to
obtain an active result would change the experiment.

## WorldPlay failure localization

A and S each exported 317 RGB frames with identical recorded plans and native
commands, and neither applied correction. All 20 prefix receipts are present.
They match through boundary 44 and first differ at boundary 60, then remain
different. Thus the first recorded pre-export discrepancy is in the newly added
45–60 interval; raw per-frame outputs were not retained to localize it further.

The lossy MP4s first differ at frame 7 and differ in 310 frames. That does not
move the generation discrepancy to frame 7: temporal video compression can
change earlier decoded pixels when later inputs differ. The raw-prefix receipts
are the stronger evidence for pre-export localization.

Manifest-matched source inspection shows the camera-based memory-selection
branch starts when `current_frame_idx >= context_window_length`: 16 latent
frames here, at zero-based chunk 4. The first mismatch is already present at
chunk 3 (12–15 latent indices, RGB endpoint 60), where history selection remains
the deterministic list of all prior indices. Camera-based selection therefore
cannot initiate this mismatch. This does not exclude other history-conditioned
computation, numerical nondeterminism, or hidden-state differences.

The current receipts do not contain per-chunk latent/noise/RNG hashes or
transformer/decoder intermediate parity. They do not establish the remaining
root cause. A/S both run the observer, so this pair does not isolate observer
effects against an uninstrumented upstream run.

Keep C held. The next diagnostic must preserve the original 20-chunk horizon,
seed, resolution, cameras and memory behavior, add latent/noise/RNG and native
input receipts, and repeat the unchanged-command identity pair. Inspect the
first divergent intermediate before changing generation behavior. If necessary,
separately test deterministic kernel configuration as a declared qualification
change. Do not admit C by replacing exact identity with an arbitrary tolerance.
No such new GPU diagnostic was launched in this review. A subsequent
[full-horizon traced identity pair](worldplay_identity_trace_20261103.md) is now
prepared and queued separately; it has not yet produced a GPU result.

## Evidence and reproduction

- [Read-only audit script](artifacts/online_response_review_20261103/audit.py)
- [Audit with per-frame hashes, counts and source/input SHA256s](artifacts/online_response_review_20261103/audit.json)
- [Visual review receipt](artifacts/online_response_review_20261103/visual_review.json)
- [Inspected reference/A/C sheet](artifacts/online_response_review_20261103/astronex_s7.jpg)

Run the script with `baselines/venv_vmem/bin/python`; it reads the archived pilot
and writes an audit beside the script. It performs no model inference. Three
source files used to localize WorldPlay are verified against the launch manifest.
Original run records, scores, status, movies and frozen launch sources were not
modified. The launch snapshot linked by `online_response.md` remains historical.
