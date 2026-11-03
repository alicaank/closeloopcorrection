# Method results

The main comparison is **baseline versus Method (camera rejoin)**. Generator
weights remain frozen. VMem's reported Method arm additionally writes measured
memory; its benefit must not be attributed to command correction alone.

| Model / cohort | Seed | Valid position pairs | Mean position change |
| --- | ---: | ---: | ---: |
| DreamX / RE10K160 | 7 | 10 | -32.5% |
| LingBot / RE10K160 | 7 | 10 | -46.1% |
| LingBot / RE10K160 | 107 | 6 | -29.6% |
| MatrixGame / RE10K160 | 7 | 10 | -7.4% |
| MatrixGame / DL3DV480 | 7 | 7 | -43.0% |
| Astronex / MinWM76 | 7 | 5 | -10.7% |
| VMem / RE10K40, commands + measured memory | 7 | 10 | -25.1% |
| Lyra / VIPE v3, 480 | 7 | 5 | -54.5% |
| SANA / VIPE v3, 160 | 7 | 9 | -52.3% |
| Spatia / VIPE v3, 480 | 7 | 7 | +6.1% |

Negative is lower error. Changes compare equal-scene cohort means, not pooled
cross-model scores. These 80 historical pairs contain 79 valid position pairs,
21 mean-position regressions and eight additional mean/endpoint tradeoffs.
One SANA position comparison is unavailable. Numerical validity does not certify
scene preservation; Astronex's historical calibration uses reference motion.
These results do not establish superiority over bounded B on matched inputs.

The [results website](https://camera-correction-results-20261029.alicankk.chatgpt.site)
contains all 203 comparisons, videos, averages, failure cases and source downloads
from the November 3 publication. The site's audience is unchanged.

See [failure findings](failures.md), [retained ablation conclusions](ablations.md)
and [the immutable research archive](research_archive.md). Current experiments
are not represented as completed results in this table.
