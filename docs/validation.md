# Validation

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

Run with `python -m unittest discover -s tests -v`. CI runs the CPU suite on Python 3.10–3.12. It does not download model weights or validate GPU behavior.

A successful CPU suite or wheel install is not an end-to-end model claim. Native environment installation, checkpoint availability, pose-estimator loading and GPU sampling must be tested on the destination machine.

For each model: first validate native-command parity and anchor projection/units, then a short baseline/sham/correction comparison with identical seeds and noise schedules. Separate a scale change from any continuation-camera or controller change. Assess position, rotation, temporal continuity and appearance together; evaluate using an estimator independent of correction/selection, and compare against seed variation. A response ratio different from 1.05 is not by itself a failed calibration.

The bundled v3 quality guards for Lyra, Spatia, SANA and WorldWarp default to enforcement. Their depth checks are heuristics, and GPU qualification remains pending. Spatia's memory optimization retains the native point partition and batches views by one; it requires mask/matching parity and peak-memory validation. WorldWarp's continuation guard supports exactly two chunks and checks the shared image anchor and depth scale; it rejects rather than silently correcting an unexplained continuation mismatch. Longer continuation is not qualified.

The historical Memorizon and WorldPlay adapters still need native/sham controls and independent projection audits. Memorizon's /4 mapping is a recorded local conditioning contract, not evidence that all observed under-execution is intrinsic to the model. WorldPlay's native unit contract remains unresolved in the new API.
