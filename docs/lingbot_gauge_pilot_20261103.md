# LingBot first-boundary shared-normalization diagnostic

The [shared-normalization candidate](lingbot_shared_gauge_20261103.md) now has
a separate frozen GPU diagnostic. It is not deployed in the primary method,
does not replace any original cohort case, and does not qualify later cache
rebuilding. Its 16 generations cover `0238de14` and `02267acf`, seeds 7 and 107,
with four arms at the original **77-frame** LingBot horizon.

| Arm | First boundary | Subsequent boundaries |
| --- | --- | --- |
| A | Original commanded labels and normalization | Original commands |
| I | Added observation; exact native identity re-encoding | Original commands |
| G | Commanded history, denominator covering plan and accepted measured prefix | Shared denominator held fixed |
| M | Measured history, same denominator and future tensors as G | Shared denominator held fixed |

All first observations use emitted RGB frames 0–12 only. There is no fitted
threshold, future reference observation, probe delta, or inverse gain. A
rejected observation retains the original native conditioning and is reported
as abstention. This is validation branching, not extra probing in deployment.

At this first boundary there is no earlier generated chunk to rebuild. The
hook changes conditioning immediately before the released clean-context cache
pass; that pass replaces the current chunk's KV entries. Every subsequent
camera input is encoded under the same denominator. Later memory-write hooks
are disabled for this diagnostic, while the native read-only endpoint observer
continues. The original absolute command trajectory, generated prefix, weights,
precision and native denoising schedule remain fixed. Temporary observer
workers close before continuation, and Python/NumPy/Torch RNG state is preserved.

## Required gates and what they establish

For **every case**, I must match A across all 77 output RGB frames, the first
and final self-attention and cross-attention cache tensors, and the native
generator state. The first latent, initial noise, clean-camera and future-camera
hashes must match too. A failed identity gate holds the queue before G or M is
run for that case.

Before scoring treatment effects, A/G/M must share exact emitted RGB frames
0–12. G and M must have identical initial noise, first latent, observed RGB,
accepted measured poses, denominator and **all future native camera inputs**.
Within-arm RNG preservation is checked separately; unrelated Python process
initialization states are not asserted equal across processes.

These gates make the three comparisons interpretable:

- **G − A:** changing normalization, including smaller future conditioning.
- **M − G:** changing first-chunk memory labels under identical future inputs.
- **M − A:** the combined effect, which cannot alone isolate the memory mechanism.

Independent offline VIPE evaluates the original timestamped targets using
initial anchor-depth registration. Mean errors cover frames 13–76; endpoints,
rotation, invalid registration, abstentions and visual fidelity remain separate.
Scores use `commanded`/`measured` as generic storage keys but explicitly record
the actual source-arm mapping in each spec. These diagnostic scores must not
be pooled with the primary repeated-memory cohort or counted as independent
new scenes. Three comparisons reuse the same A/G/M generations.

## CPU validation and queued status

Three arm-selection tests and two callback tests pass. The callback tests use
the released native camera encoder on CPU, with substituted model/IPC actions;
they check accepted/rejected branches, exact no-op inputs, identical G/M future
tensors, distinct clean-history tensors, unchanged absolute commands and worker
cleanup ordering. The actual native source overlay compiles, with callbacks
immediately before and after the unchanged clean-context model call. The shared
primitive's earlier 152-test repository run remains recorded separately.

These tests **do not establish GPU cache parity, visual benefit or training
support**. The full model gates above remain pending. The queue was confirmed
alive waiting for the standard GPU lock. It contains 52 jobs, including 16
generations, lossless retention, gates and 12 paired scores. Resource guards
and frozen-source checks remain enabled.

Workspace: `analysis/lingbot_shared_gauge_20261103/pilot/`.
Manifest SHA256:
`89cb0e918ee1c3246521b023f346df7dbf9a398396a540f82ee56fc59bc175f5`.

[Protocol](artifacts/lingbot_gauge_pilot_20261103/protocol.json),
[hook](artifacts/lingbot_gauge_pilot_20261103/lingbot_gauge_hook.py),
[gates](artifacts/lingbot_gauge_pilot_20261103/check.py),
[callback validation](artifacts/lingbot_gauge_pilot_20261103/callback_validation.json)
and [queue receipt](artifacts/lingbot_gauge_pilot_20261103/queue_snapshot.json)
preserve the implementation and scope. These are workspace-specific audit
copies. The manifest references the complete frozen source and original inputs.

A successful first-boundary result would justify implementing and separately
qualifying chronological replay when normalization changes later. It would not
establish a general support boundary or excuse the existing scene failures.
