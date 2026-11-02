# Audit corrections, 2026-11-02

Implementation: online_response.py now identifies and inverts duration-matched chunk averages, uses a maximum-step bootstrap budget, and preserves slew bounds after inverse control deactivates. Baseline/sham remain exact before any intervention.

Evaluation: evaluation_cache.py keys frozen evaluator contract plus RGB, intrinsics and reference depth. Identical inputs reuse geometry; target metrics remain separate. The bundled memory evaluator consumes it. lossless_archive.py verifies FFV1 RGB hashes before a new experiment can replace bulky frame artifacts.

Accounting: telemetry.py separates accepted feedback from scheduled calls. AdaptiveController records fresh nonzero feedback and command changes. Historical logs remain immutable; the dashboard derives corrected acceptance counts and ties from their evidence. A corrected historical SANA launcher/report source for the next ABC freeze is in analysis/method_fixes_20261102/command_gain.

The previous online-response v1 queue was stopped before any job started. Its frozen manifest is retained. Replacement response and multi-boundary memory protocols require fresh matched runs; CPU checks alone do not establish GPU efficacy.

## Disposition of the eight findings

| Finding | Correction | Verification / remaining evidence |
|---|---|---|
| Chunk coefficients iterated per latent | Fit and invert duration-matched chunk averages once, then distribute the velocity offset over the native profile. Reset a bank when the fitted duration changes; abstain on mismatched future duration. | Synthetic inverse/prediction tests at 4, 8 and 16-frame native cadences; partial future duration abstains. |
| Slew bypass after rejection | Once correction engages, limit all future transitions, including return toward baseline after rejection or lost identification. | Rejected-observation test preserves issued history and the acceleration bound. A/S remain exact before intervention. |
| Tiny first-step bootstrap inflation | Map the maximum planned step to the adapter's released navigation step. Pure rotation introduces no translation. | Tiny stationary-start and pure-rotation tests. This is an operational command budget, not physical calibration. |
| Too little active Astro horizon | New response-v2 has 20 native chunks/model; report usable samples, active future chunks and their RGB duration. | At least eight active future chunks required for the duration check; activation and efficacy remain pending. |
| Scheduled calls counted as applied feedback | Separate scheduled calls, accepted observations, fresh residual feedback, changed commands and response activation. | Regression reproduces one accepted boundary out of six scheduled calls. Historical summaries derive acceptance from observer/controller logs; unavailable fresh-feedback fields remain null. |
| Memory evidence active on one scene only | Preserve every policy result, explicitly show accepted-write coverage, and freeze four scenes with repeated native boundaries. | Original 18 complete pairs: ten one-write pairs on one scene and eight identical zero-write pairs. New 40-pair protocol is queued; qualification requires two scenes with at least two accepted writes per qualifying pair. |
| Identical inputs get different geometry | Cache exact RGB, intrinsics, reference depth and frozen evaluator contract; reuse all geometry sidecars. | Estimator called once for identical inputs. Both new queues use the cache. Published historical zero-write pairs share the commanded-arm estimate while retaining original independent scores/downloads. |
| Endpoint ties counted as losses | Compute strict wins/losses/ties separately, excluding unavailable values. | Browser regression: Matrix memory seed 7 is one loss and one tie, not two losses. |

The report's `applied_feedback` compatibility field is explicitly labelled as
accepted positive-confidence observations. It is not proof of a nonzero command
change. New events expose that distinction; historical events cannot establish
it when the necessary fields were never recorded.

## Replacement experiments and resource handling

- [Response-v2 protocol](online_response_v2_protocol.json): Astronex 637 RGB
  frames / 160 latents; WorldPlay 317 RGB frames / 80 latents. Both use 20 native
  chunks. A/S/C for seed 7 and A/C for seed 107: ten generation jobs, followed by
  four paired scores and four adaptation reports. One development scene;
  no cross-scene claim.
- [Repeated-memory protocol](method_v1_multiboundary_protocol.json): five
  models, four prespecified scenes, seeds 7 and 107; 40 pairs / 80 generation
  jobs. The two additional scenes are selected by a fixed hash rule, without
  filtering observer outcomes. Failed/rejected cases remain in policy totals.
- Both schedulers bind source/input manifests to CPU qualification receipts.
  Response-v2 waits for the existing Spatia locks. The memory dispatcher waits
  for response-v2 to terminate before acquiring those same locks.
- Memory artifacts can replace **only this new queue's** raw NPZ/PNG frames
  with FFV1 after decoding and verifying every RGB byte. RGB remains available
  losslessly. Runtime disk/RAM guards stop owned jobs; old results and protected
  checkpoints are not cleanup candidates.

Prelaunch manifest revisions are retained beside the final manifests with
receipts. No generator job had started under either draft. Historical frozen
experiments, including the original flawed response snapshot, remain unchanged.

## Validation and claims

The source suite passes **107 tests**. Frozen queue checks additionally pass
native CLI/source checks for both response models and all five memory models,
evaluator imports, response regressions, exact-input cache tests and real FFV1
round trips. Browser QA passes all **170** published comparisons, including
invalid-position handling, confounded comparisons, ties, synchronized video,
the primary measured-memory diagram, and mobile layout.

[Execution receipt](audit_fixes_20261102_execution.json) records the exact
manifests, checks and queue states at launch. GPU outcomes are pending. These
fixes do not establish improvement, global control stability, physical native
units, novelty or generalization. Historical observer pixel-correspondence
concerns remain unquantified; historical scores must not be presented as a
validation of the newer preprocessing-aware response observer.
