# Validation

## Current measured-memory repository update (2026-11-01)

The current working tree passes **67 top-level CPU tests**, both from source and
from a freshly built/installed wheel. The evaluator integration test additionally
runs its internal geometry suite in a subprocess. New coverage checks the
bundled memory contract against the portable core and compiles all seven
measured-memory drivers plus their Matrix-Game/Lyra child overlays after
relocation. No GPU is used for these package checks.

The [validation receipt](https://github.com/alicaank/closeloopcorrection/blob/b467679a28243241c3dcf5156f1c980540d879cd/docs/measured_memory_package_validation.json) records the
wheel and bundle hashes and test output. The [method reference](paper_method_reference.md)
and [development evidence snapshot](https://github.com/alicaank/closeloopcorrection/blob/b467679a28243241c3dcf5156f1c980540d879cd/docs/paper_evidence_20261101.json) distinguish
the locally GPU-tested memory runtime from the relocated source bundle. Fresh
machine GPU portability remains unqualified. Earlier version-specific counts
below describe historical releases, not the current suite.

## Historical release checks and general protocol

The source package has CPU checks for:

- Native/reference camera round trips across c2w/w2c and OpenCV/OpenGL.
- Registered depth scale, metric conversion and the Memorizon /4 mapping.
- Rejection of unresolved scales, poor shared depth coverage, malformed poses and mismatched calibration receipts.
- Crop/resize intrinsics and depth edge handling.
- Commanded-memory frame bridging and native-basis relative chains.
- Stale measurements, history writes, sham equivalence and input immutability.
- A custom model integration and a synthetic under-execution simulation.
- All 11 launcher script paths, explicit VMem measured-memory selection, historical control restrictions, hash manifests and compilation in a relocated workspace.
- Absence of model assets and machine-specific source paths in the bundle.

The initial release passed 19 CPU tests, including execution against an installed wheel. Version 0.2.0 passes 31 CPU tests against a freshly built/installed wheel. Additional dataset tests cover registered resize geometry, metadata/trajectory rejection, selected scene order, separate job outputs, fail-fast execution, changed-input rejection and child interpreter relocation. Multi-scene process execution is tested with mocked native jobs, not real GPU sampling. Every nonempty upstream patch was also checked against its pinned source revision using a temporary Git index, without modifying the research checkouts. All 11 local source revisions and interpreter paths were present during the audit.

Version 0.3.0 adds VIPE registration and relocated scoring-entrypoint checks, and verifies that preparation selects the pinned VIPE source rather than VGGT. The suite has 33 top-level tests; the evaluator test also runs four geometry cases in a separate process. A local VIPE import/configuration dry run passed with CUDA hidden. GPU qualification of VIPE scoring is pending. See [evaluation protocol](evaluation.md).

Run with `python -m unittest discover -s tests -v`. CI runs the CPU suite on Python 3.10–3.12. It does not download model weights or validate GPU behavior.

A successful CPU suite or wheel install is not an end-to-end model claim. Native environment installation, checkpoint availability, pose-estimator loading and GPU sampling must be tested on the destination machine.

For each model: first validate native-command parity and anchor projection/units, then a short baseline/sham/correction comparison with identical seeds and noise schedules. Separate a scale change from any continuation-camera or controller change. Assess position, rotation, temporal continuity and appearance together; evaluate using an estimator independent of correction/selection, and compare against seed variation. A response ratio different from 1.05 is not by itself a failed calibration.

The bundled v3 quality guards for Lyra, Spatia, SANA and WorldWarp default to enforcement. Their depth checks are heuristics, and GPU qualification remains pending. Spatia's memory optimization retains the native point partition and batches views by one; it requires mask/matching parity and peak-memory validation. WorldWarp's continuation guard supports exactly two chunks and checks the shared image anchor and depth scale; it rejects rather than silently correcting an unexplained continuation mismatch. Longer continuation is not qualified.

The historical Memorizon and WorldPlay adapters still need native/sham controls and independent projection audits. Memorizon's /4 mapping is a recorded local conditioning contract, not evidence that all observed under-execution is intrinsic to the model. WorldPlay's native unit contract remains unresolved in the new API.
