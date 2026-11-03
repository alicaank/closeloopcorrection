# Matched legacy-rejoin comparison: preparation and queue

Legacy rejoin remains a leading main-method candidate following the [historical reassessment](legacy_rejoin_reassessment_20261103.md). This study compares native future commands, explicit legacy rejoin, and measured-memory correction on the original five models, four scenes and two seeds. All 40 cases remain required; no failed or near-stationary scene is replaced.

## Current execution scope

| Model | Cases | RGB frames | State at this snapshot |
| --- | ---: | ---: | --- |
| DreamX | 8 | 69 | New common driver CPU-qualified; 32 generations queued behind existing GPU work |
| LingBot | 8 | 77 | Frozen source copied and generated overlay equivalence checked; comparison queue pending |
| SANA | 8 | 121 | Transition-order resource host copied and source equivalence checked; GPU/reuse qualification pending |
| MatrixGame | 8 | 337 | Observer-lifetime resource host copied and source equivalence checked; comparison queue pending |
| Lyra | 8 | 321 | Observer-resource host copied and source equivalence checked; GPU/reuse qualification pending |

The scenes are `02267acf`, `c7418b7d`, `0238de14`, and `924228bf`; seeds are 7 and 107. The other four models' source files are byte-identical to their respective parents. Six generated driver/child overlays compile and match after normalizing only the copied source directory. This proves source equivalence, not GPU execution equivalence or availability of all paired results.

The DreamX queue is `analysis/legacy_rejoin_matched_20261103/dreamx_host/run_queue.py`, launched as PID 3021657 and confirmed alive waiting on the shared GPU lock. Its manifest SHA256 is `afbedc28cdcac69a275556c39b7f3560b61e754e485be04367cfdafb5b5f96f2`: eight cases, 32 generations and 88 total generation/retention/gate/scoring jobs. No new GPU efficacy result is claimed here. The earlier frozen queues and outputs remain unchanged.

## DreamX's common causal driver

All four arms use the same prepared initial RGB/depth grid, released BF16 camera intrinsics, original planned poses, scene, seed and horizon. The initial camera-grid conversion uses initial intrinsics and image dimensions; it does not fit a scale to a recorded future trajectory. Reprojected reference RGB enters offline evaluation only.

- **A:** observer-free native commands and native cache pass.
- **S:** causal observation with native commands and native cache pass.
- **L:** the unmodified historical rejoin curve. The measured preceding camera anchor is temporary input to native next-chunk encoding; only the next chunk's future camera tensors are committed, after the native context-noise cache pass.
- **M:** measured labels are encoded only for the current native context-noise cache pass. Future camera tensors and future commands remain unchanged.

S, L and M observe the same causal RGB window before the cache pass, querying the current native chunk's camera labels, its preceding anchor, and fixed frame zero. The first observation occurs after six latents: available RGB ends at frame 20, the preceding camera sample is frame 17, and the next native camera samples are 21, 25 and 29. All later nonterminal chunk boundaries are observed. L may correct at this first boundary; it has no nominal-first-boundary delay, speed cap, non-reversal projection, fitted gain, inverse control or probe generation.

This is a corrected legacy implementation, not an exact historical replay. Differences are the calibrated initial grid, the shared full causal observation window, observation before the context-noise pass, and explicit observation/SE3 rejection. The historical driver observed after the pass and truncated three available RGB frames. The rejoin curve itself is extracted unchanged from bundled `common.py`; CPU tests compare its translation and rotation output exactly against that source.

`issued_commands.npz` records the issued world-frame command array. Its `memory_labels` array is the buffer for the explicit M intervention; in L it is not a reconstruction of native KV conditioning. Native L conditioning must be reconstructed from issued commands, the temporary measured anchor in the boundary observation, and the released encoder. Context/future tensor hashes and boundary receipts record the actual intervention channels.

## Qualification and evaluation

Eight CPU tests use the actual released camera encoder. They verify control identity, exact historical-curve output, native measured-anchor encoding, current-context versus future-command separation, identity round trips, quality/SE3/anchor rejection, first-boundary catch-up and return after overshoot, and native timestamp/intrinsics/commit checks. The exact runtime overlay also compiles against the released pipeline, and the native interpreter accepts the driver CLI. These checks do not establish GPU parity or generator response.

Before L/M generation, A and S must have identical lossless RGB for all 69 frames and identical initial noise/conditioning and native boundary receipts. Before scoring, S/L/M must have identical RGB frames 0–20 and matching native receipts through the first pre-cache intervention point. M's cache is allowed to differ after that point; requiring equality there would reject its intended intervention. L's current cache conditioning and M's future camera tensors must remain unchanged at every boundary.

The independent offline VIPE evaluator scores all three observed arms with one contract and reports S–L, S–M and L–M comparisons. Primary mean fixed-time position uses frames 21–68; whole-rollout, endpoint, rotation, registration validity and actuation counts remain separate. Stored reference poses may themselves be VIPE-derived and are not described as independent ground truth. Contact sheets are generated for review, but automatic generation is not a completed visual review.

Raw RGB is retained through verified lossless FFV1 round trips. The queue preserves the established 4 GiB launch / 2 GiB runtime disk reserves and 2 GiB host-memory reserve. Resource failures, rejected observations, failed attribution gates and missing registrations remain visible. GPU gating or scientific failure does not permit selecting replacement scenes.

Remaining work is to freeze and queue the other 32 cases after qualifying baseline/memory reuse or arranging matched regeneration, run GPU gates and scoring, review visuals and boundary behavior, and select the policy from aggregate evidence before broader held-out evaluation.

[Full 40-case plan](artifacts/legacy_matched_preparation_20261103/plan.json) · [Four-model source preparation](artifacts/legacy_matched_preparation_20261103/host_preparation.json) · [DreamX frozen manifest](artifacts/legacy_matched_preparation_20261103/dreamx_host/manifest.json) · [CPU qualification](artifacts/legacy_matched_preparation_20261103/dreamx_host/cpu_validation.json). The archived scripts and manifests retain workspace-specific paths; the manifest records all frozen dependency hashes.
