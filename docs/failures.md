# Failure findings and fixes that remain necessary

Some scene failures are compatible with aggregate benefit. Reproducible
implementation errors, invalid comparisons and feedback on unreliable geometry
require separate diagnosis and correction.

| Model / case | Established finding | Remaining question |
| --- | --- | --- |
| LingBot `d3f042a2/s107` | Large inputs precede degradation at sampled frame 48 and collapse by 64; finite feedback continues afterward | Isolate input magnitude from rotation/history/observer effects |
| LingBot measured memory, `02267acf` | Near-stationary plan normalization makes history steps about 70–106× planned maximum; visual failure occurs | Qualify a shared history/future normalization and cache rebuild |
| DreamX `e43058cb/s7` | Both arms miss the intended kitchen traversal; native-anchor replay gives zero intended reversals in all ten audited runs | Separate aggressive recovery, observer error and generator history |
| SANA `e43058cb/s7` | Both arms miss the layout/path; observers differ by 79.27 degrees at frame 144 | Establish trustworthy feedback on matched images |
| MatrixGame `0238de14` | Both arms show a seam; same-latent decoder tests confirm an export contribution | Preserve causal decode context without revising emitted RGB |
| VMem `74310829`, `d3f042a2` | Command-only exacerbates errors; all 40 command streams and latest-memory bridges replay correctly | Isolate window normalization, retrieval and reconstruction effects |
| Lyra garden / Spatia hall | Layout and tracking drift; Spatia backward estimates repeat on identical RGB with zero command reversals | Physical camera motion versus geometry/observer inconsistency |
| Astronex `minwm_001231` | Reference translation is zero; huge path ratios are denominator artifacts | Stationary-scene camera accuracy and historical calibration limits |
| WorldPlay A/S `0238de14/s7` | Equal commands, unequal decoded prefixes between boundaries 44 and 60 | Locate intermediate nondeterminism before admitting correction |

The portable controller requires `Measurement.quality['passed']`; that boolean
must come from real evidence. Historical bundled `PoseWorker` code accepts
finite poses after worker success, which is weaker and can accept collapsed RGB.
Integrating depth/image/overlap checks into those hosts requires a new qualified
version, not retrospective relabeling of old results. Quality checks also need
false-rejection testing; neither finite matrices nor one heuristic prove scene
survival.

Global translation rescaling cancels under the tested LingBot native maximum
normalization. A raw observer-scale jump is not necessary for its collapse.
There is no established universal native command threshold or angular noise
floor. A passive execution ratio is not a differential control gain.

For a proposed repair, compare baseline, original Method and one-change Method
with identical seeds, prefixes, targets and full horizons. Include successful
cases as well as diagnosed failures. Report tracking, scene preservation,
abstention and runtime failures separately; do not obtain a safer score by
preventing needed catch-up or return after overshoot.

Complete case-level replays and visual evidence remain in the
[research archive](research_archive.md) and the linked [results page](results.md).
