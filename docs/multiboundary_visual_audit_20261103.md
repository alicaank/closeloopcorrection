# Repeated-memory visual and reference-motion audit

This is a development-cohort audit, not new aggregate efficacy evidence. All
40 planned pairs remain in the evaluation. The source RGB frames are the
reference video; stored camera poses may be VIPE-derived and are not claimed
as independent ground truth. Translation is in each scene's reference
anchor-depth units, not established metric metres.

## Motion depends on the evaluated horizon

The following net displacements come directly from the frozen target arrays.
No generated outcome is used to select scenes or change the protocol. Native
horizons differ, so these are context values rather than comparable model scores.

| Scene | LingBot 77 frames | SANA 121 | DreamX 69 | MatrixGame 337 | Lyra 321 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 02267acf | 0.01338 | 0.05817 | 0.01171 | 0.41388 | 0.39613 |
| c7418b7d | 0.07928 | 0.11399 | 0.07086 | 0.45868 | 0.42950 |
| 0238de14 | 0.10162 | 0.17905 | 0.09005 | 0.62680 | 0.59390 |
| 924228bf | 0.00861 | 0.01330 | 0.00757 | 0.04556 | 0.04339 |

`02267acf` is nearly stationary over the short LingBot/DreamX intervals, but
moves substantially later in the long MatrixGame/Lyra intervals. A scene-level
"stationary" label would therefore misdescribe some experiments. `924228bf`
has relatively little translation throughout these windows. The full audit
includes path length, step statistics and angular path length for all 20
scene/horizon combinations. These are descriptive values, not new inclusion
thresholds; no case is removed.

## LingBot 02267acf, seed 7: confirmed visual failure

![Matched reference, commanded and measured samples](artifacts/generalization_20261103/visual_review/lingbot_02267acf_s7.jpg)

Direct inspection covers frames 0, 12, 13, 38, 57 and 76. The reference changes
little. The commanded output retains recognizable store content while drifting
strongly toward the floor. The measured output shares the sampled prefix but
shows severe overlapping and fragmented geometry already at frame 13, and the
distortion persists through the later samples. Its 74.24% reduction in mean
position error is therefore **not a successful visual-control result**. This
pair remains a policy failure in the cohort; it is not excluded from reporting.

This is sampled inspection, not a full-video review, and it cannot characterize
oscillation between the sampled frames. Thumbnails preserve native framing
rather than claiming pixel-aligned reference/output correspondence.

The [manifest](artifacts/generalization_20261103/visual_review/manifest.json)
binds score files, lossless RGB archives, reference images and sheets by SHA256.
The [inspection record](artifacts/generalization_20261103/visual_review/lingbot_02267acf_s7.review.json)
records the scope of the human-readable observations. The generator queue and
its frozen scientific sources were not changed by this audit. Additional
completed pairs still require their own visual reviews.
