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

[Machine-readable evidence and source hashes](artifacts/legacy_rejoin_reassessment_20261103/result.json) · [audit script](artifacts/legacy_rejoin_reassessment_20261103/audit.py). The script is an archival workspace-specific audit copy.
