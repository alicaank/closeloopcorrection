# WorldPlay identity control: localized divergence, cause unresolved

The original seed-7 A/S runs on scene 0238de14 use identical saved native commands, camera intrinsics, preprocessing, seed and 20-chunk settings. Neither applies a changed command. Raw cumulative RGB hashes match at frames 12, 28 and 44, and differ at frame 60 and all subsequent checkpoints. Thus the first raw difference is somewhere in frames 45–60.

That first failed checkpoint is chunk index 3, where the native pipeline uses the fixed full prefix. Geometric retrieval selection starts at chunk index 4 (`current_frame_idx >= 16`). Retrieval selection cannot explain the first divergence. This does not identify the actual cause or certify equality of unrecorded RNG, latent, attention or cache state.

The CRF-12 MP4 exports differ substantially by the end: mean RGB MAE across 317 exported frames is 31.166/255; the maximum frame mean is 69.983/255. The first export difference is frame 7 even though raw prefixes match through 44. Lossy temporal encoding therefore cannot locate the first raw divergence or supply a raw regeneration/noise floor.

Keep the failed exact identity gate; do not reinterpret this as an adaptive-control effect or relax equality to admit the result. The next diagnostic, if prioritized, needs repeated same-arm controls plus native inputs, RNG and latent/decode hashes around the first divergent chunk. No WorldPlay controller fix or new GPU qualification is claimed by this read-only audit.

[Evidence](artifacts/worldplay_identity_audit_20261103/result.json) · [audit script](artifacts/worldplay_identity_audit_20261103/audit.py).
