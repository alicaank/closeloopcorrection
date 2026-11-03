# Primary repeated-memory cohort: original-attempt accounting

The resumed original attempt has finished with model-level resource holds.
All **16 LingBot/DreamX pairs** are scored, prefix-qualified and visually
reviewed. SANA, MatrixGame and Lyra each retain one original resource failure
and seven held pairs. The full planned denominator remains **40 pairs**.
SANA's separate 121-frame resource retry has started; its result is not yet
qualified. MatrixGame/Lyra retries and the other diagnostic queues remain separate.

| Model | Scored / planned | Active / abstained | Mean position lower / higher / unavailable | Mean rotation lower / higher / equal |
| --- | ---: | ---: | --- | --- |
| LingBot | 8 / 8 | 6 / 2 | 6 / 0 / 2 | 3 / 3 / 2 |
| DreamX | 8 / 8 | 6 / 2 | 2 / 4 / 2 | 3 / 3 / 2 |
| SANA | 0 / 8 | — | Missing | Missing |
| MatrixGame | 0 / 8 | — | Missing | Missing |
| Lyra | 0 / 8 | — | Missing | Missing |

These are strict numeric comparisons, not statistical wins or overall control
successes. The six LingBot position reductions include the two `02267acf`
collapses and moving-scene fidelity problems. DreamX's small changes lack a
matched regeneration floor. No raw-error mean is pooled across models or scenes,
no scene is excluded based on outcome, and no primary metric is selected anew.

All four abstentions are `924228bf`, both seeds, in LingBot and DreamX. Every
decoded RGB frame is identical between the arms: 77 frames for each LingBot
pair and 69 for each DreamX pair. Position registration is unavailable in both
arms of all four pairs. Identical output and identical rotation scores do not
constitute an active correction success or a zero position error.

The [complete 40-row table](artifacts/generalization_20261103/primary_cohort_summary.md)
and [machine-readable snapshot](artifacts/generalization_20261103/primary_cohort_summary.json)
retain each original case, metric window, exact-prefix qualification, accepted
writes, missing positions and qualitative review. The
[read-only summarizer](artifacts/generalization_20261103/summarize_primary_cohort.py)
binds its inputs by SHA256. The [visual audit](multiboundary_visual_audit_20261103.md)
contains the images and limitations of the unblinded qualitative assessment;
it is not a calibrated appearance metric or a blinded human study.

The archived script is a workspace-specific audit copy. Its execution location is
`analysis/memory_generalization_20261103/summarize_primary_cohort.py`; it reads the
frozen experiment and score paths recorded in that workspace.

The next required evidence is full-horizon resource qualification and paired
outcomes for the other three models, followed by the separately frozen mechanism
diagnostics. Their pending state cannot be replaced with the shorter earlier
cohort or inferred from the two models that completed. Some scene failures are
compatible with a useful general method; this incomplete five-model evaluation
does not yet establish that aggregate benefit.
