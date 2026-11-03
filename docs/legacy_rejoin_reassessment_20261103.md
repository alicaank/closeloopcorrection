# Reassessing legacy rejoin as a main-method candidate

The earlier decision to freeze measured-memory v1 was an experiment-control decision, not evidence that it outperforms legacy rejoin. The final paper method remains undecided. Legacy rejoin must remain a central candidate and reference comparison; measured memory, bounded command correction and online gain need demonstrated incremental value.

The audit recomputes all ten legacy cohort/seed rows in the existing summary and verifies the SHA256 of all 80 underlying source comparison records. Each row compares rejoin with its own baseline. These heterogeneous rows must not be pooled, and they do not establish a matched ranking against later memory-only experiments. Negative percentage change means lower error; DINO has the opposite preference.

| Model | Cohort / observer | Seed | Mean position change | Position lower / valid | Mean rotation change | LPIPS change |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| dreamx | re10k160 / VGGT | 7 | -32.53% | 9 / 10 | -51.61% | -3.82% |
| lingbot | re10k160 / VGGT | 107 | -29.63% | 4 / 6 | +31.71% | -3.60% |
| lingbot | re10k160 / VGGT | 7 | -46.07% | 8 / 10 | -31.72% | -12.27% |
| matrixgame | re10k160 / VGGT | 7 | -7.39% | 7 / 10 | -6.06% | -1.71% |
| astronex | minwm76 / VGGT | 7 | -10.65% | 4 / 5 | -1.65% | +0.80% |
| matrixgame | dl3dv480 / VGGT | 7 | -42.97% | 3 / 7 | +14.86% | -5.80% |
| vmem | re10k40 / VGGT | 7 | -25.06% | 8 / 10 | -40.29% | -4.32% |
| lyra | rejoin_v3 / VIPE | 7 | -54.51% | 4 / 5 | +42.03% | -10.50% |
| sana | rejoin_v3 / VIPE | 7 | -52.35% | 7 / 9 | -30.59% | -10.26% |
| spatia | rejoin_v3 / VIPE | 7 | +6.15% | 4 / 7 | +27.27% | +0.54% |

The positive evidence is not solely a VGGT effect. The later VIPE registration audit retains SANA’s 52.35% mean-position reduction (7/9 valid pairs), with lower mean rotation and LPIPS. Lyra retains a 54.51% mean-position reduction (4/5), with lower LPIPS in all five cases; mean rotation rises from 0.1723° to 0.2447°, an absolute increase of about 0.0724°. No universal observer sensitivity floor is asserted.

DreamX and LingBot seed-7 RE10K cohorts show 32.53% and 46.07% lower mean position under VGGT. These are meaningful historical leads, but the corresponding videos still require comparison under the same current scoring contract as any new competing policy. Lower LPIPS/DINO changes support appearance similarity, not a guarantee of preserved geometry or smooth camera motion.

The negative evidence remains visible: Spatia’s VIPE aggregate worsens position by 6.15% and rotation by 27.27%. MatrixGame’s DL3DV aggregate improves position by 42.97%, yet only 3/7 scenes improve and rotation worsens 14.86%; a minority of large changes drives its mean. LingBot seed 107 worsens mean rotation and DINO. Historical Astronex reference-trajectory calibration and VMem demo normalization retain their previously documented limitations.

## Next comparison and decision rule

1. Reconstruct each historical rejoin actuator from its saved launcher/arm configuration, including its memory-label policy. The label “legacy rejoin” alone does not prove a pure command-only intervention. Keep the causal unit/camera corrections already required by the adapter contract; document any change from historical execution.
2. Compare native baseline, that explicit legacy policy, and measured-memory v1 on the same original four scenes and two seeds per model, at the original native horizons. Reuse existing baseline/memory outputs only after host, inputs, native conditioning, observer and prefix equivalence are qualified. Otherwise rerun matched arms. This is a planned additional legacy arm for 40 original cases, not a replacement with selected successful scenes.
3. Use one evaluator/registration contract and fixed scoring window for all arms in each comparison. Keep unavailable registrations and failed runs in coverage accounting. Inspect full-video boundary behavior and appearance alongside mean fixed-time position, endpoint and rotation errors.
4. Choose the simplest shared policy supported by aggregate tracking and quality evidence; do not require every scene to improve. If legacy rejoin is stronger, use it as the main candidate. Keep memory-only and gain learning as ablations unless they add reproducible value. A combination requires a separately qualified comparison; it is not assumed better.
5. Freeze that selected policy before broader evaluation on prespecified held-out scenes. Existing development scenes cannot become an untouched test set by relabelling them.

The currently running frozen cohorts remain intact. This reassessment changes method-selection priority, not already issued commands, frozen manifests, stored scores or the website’s dated publication snapshot.

## What the legacy command correction actually does

Source inspection of the current bundled drivers confirms an explicit future-command intervention. This is a reconstruction of the available implementation, not yet a proof that every historical score used these exact source bytes. Historical source versions and unit conversions still need qualification before calling a new run an exact reproduction.

The shared `rejoin(measured, previous_command, future_plan)` adds a linearly decaying translation offset to the next planned chunk: for sample fraction `a = (i + 1) / n`, the desired translation is `p_i + (1 - a) * (m - u_previous)`. It similarly decays the relative rotation to identity using SLERP. The final desired pose returns to the planned chunk endpoint. An identity measurement returns the future plan unchanged.

This policy can both catch up from lag and return after overshoot. For an ideal translation-only executor, a uniformly spaced plan with step `v`, boundary error `e = m - u_previous`, and `n` future samples requires step `v - e/n`. A negative error increases speed; a sufficiently positive error reverses motion. This algebra explains the available corrective action; it does not prove that a frozen generator executes it or remains stable. The restrictive bounded, non-reversal policy removed these actions. Its losses therefore cannot establish that command correction itself is ineffective.

The desired world-frame curve and the native conditioning values are distinct. Several adapters bridge the desired curve into the frame of the stored commanded history, so their final encoded pose need not numerically equal the world-frame target.

| Model | Future-command implementation | History treatment in legacy driver |
| --- | --- | --- |
| LingBot | Rejoin against the previous issued command; encode the first future relative step from the measured pose. Reuse the plan-derived translation denominator. | Existing cached conditioning remains intact; no measured-history rewrite. The denominator support issue still needs a separately identified ablation. |
| DreamX | Rejoin the next three native camera samples. Use the measured pose as the temporary camera-encoding anchor and copy only future camera tensors after the native cache update. | No measured KV re-pass in this legacy path. Changing a temporary encoding anchor is not a persisted measured-memory write. |
| SANA | Rejoin against the planned boundary; transform the desired curve by `issued_boundary @ inverse(measured)` before encoding future ray/Plücker tensors. | Previously issued history labels remain commanded. |
| MatrixGame | Rejoin and bridge against the issued boundary before encoding the next 84 frames. | Refresh clean-context camera conditioning with the commands actually issued for those frames, not externally measured poses. This commanded-history consistency operation is part of the legacy arm. |
| Lyra | Rejoin against the planned boundary; bridge from the actual last native camera label before passing the future chunk to the native step. | Retain native history labels and normal cache updates; no external measured-history rewrite. |

The inspected paths are under `src/camera_loop/bundled/templates/analysis/plan_base_restart_20261008/`: `new_models_loop_20261028/{common,lingbot_pair,dreamx_run,sana_run}.py` and `continuous_eval_20261029/{mg35_stream_launch,lyra_plugin}.py`.

## Qualification before the matched comparison

The comparison retains all 40 original cases: LingBot, SANA, DreamX, MatrixGame and Lyra; scenes `02267acf`, `c7418b7d`, `0238de14`, `924228bf`; seeds 7 and 107. Their native RGB horizons remain 77, 121, 69, 337 and 321 respectively. Three scientific arms are native future commands with commanded history, explicit legacy rejoin, and measured-memory v1. Observer-only and identity controls qualify the host and encoding; they are not substituted for an efficacy arm.

LingBot, SANA, MatrixGame and Lyra expose an existing correction arm, but reuse of their baseline/memory outputs still requires the same input, resource host, observer configuration and native conditioning. SANA and Lyra resource retries remain qualification work; a source-level match does not establish completed-run equivalence. MatrixGame uses the same causal block export in all arms; its offline whole-history decoder audit cannot silently replace scoring RGB.

DreamX requires an explicit integration decision before launch. The current memory cohort uses a different driver from legacy rejoin. The legacy observer truncates to the last native camera-anchor frame, leaving three available RGB frames unused; the current future-command pilot also uses a newly prepared fixed-intrinsics input grid. Neither is interchangeable with the old memory cohort by changing an arm label. A common, calibrated input grid and declared observation schedule require newly matched arms unless reuse is demonstrated. Align the first intervention across arms and verify raw pre-intervention RGB, latent/RNG/cache receipts and untouched camera slices. Keep any necessary timing or calibration changes visible as a corrected legacy implementation, rather than claiming exact historical reproduction.

The source review above does not establish that legacy has beaten memory-only in a matched test. A subsequent [matched-comparison preparation](legacy_matched_preparation_20261103.md) has queued all eight DreamX cases in a corrected common host and prepared the other four hosts; its GPU gates and efficacy results remain pending.

[Machine-readable evidence and source hashes](artifacts/legacy_rejoin_reassessment_20261103/result.json) · [audit script](artifacts/legacy_rejoin_reassessment_20261103/audit.py). The script is an archival workspace-specific audit copy.
