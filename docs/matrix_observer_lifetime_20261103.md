# MatrixGame full-horizon resource retry

The original 337-frame commanded arm completed, but the matched measured arm
stopped at the host-memory guard after the third block's RGB decode. Recorded
minimum available host memory was 1,533,771,776 bytes. Its process is gone; the
attempt is terminal and retained. This is missing paired evidence, not a
tracking failure or a model-level null. The main cohort holds MatrixGame and
continues with Lyra.

Source inspection identifies two auxiliary observers with overlapping lifetimes:
the endpoint observer and the measured-memory observer. Both remained resident
across generation chunks. The failed log ends during an additional VAE decode
after section 2. This supports a memory-pressure mechanism, but no per-process
RSS trace from that attempt proves that the observers alone caused the peak.

The separate retry closes the endpoint observer after its measurement, before
the memory observer starts; it closes the memory observer before native
continuation. `finally` paths also release observers when measurement raises.
Each later callback creates its own observer. Native weights, dtype, camera
policy, cache length, frame count and decoder arithmetic are unchanged. A
changed process lifetime still requires empirical parity qualification.

The bundled source hashes were updated with the two source edits. Tests execute
the actual bundled callback control flow with substituted model/IPC operations,
checking recreation, phase ordering and exception cleanup. All **145 repository
CPU tests pass**. Both Matrix native overlays compile. These checks do not
establish full-horizon GPU success.

## Preserved scope and gates

Workspace: `analysis/memory_generalization_20261103/matrix_observer_lifetime/`.
Manifest SHA256:
`8265bf0bbbf70c4813e5ee04a4e6f48fdbbda4ca835d6b8eb9edaf376509f314`.

All eight original MatrixGame pairs remain scheduled: the same four scenes,
seeds 7 and 107, and **337 RGB frames**. The first retry must:

1. Finish the original 02267acf seed-7 commanded arm at 337 frames.
2. Match every RGB byte of the previously completed 337-frame baseline.
3. Finish the matched measured arm and pass exact pre-intervention prefix parity.
4. Preserve both valid scores and scientific losses; retain unavailable
   registration as unavailable.

The retry has 16 generation jobs and 49 total jobs including retention, scoring
and gates. It holds the standard GPU locks and waits behind existing work.
Source hashes are checked before jobs; host/disk guards remain in place. A
failed required gate holds the supplement. Original failures remain in the
[40-pair coverage ledger](generalization_coverage_20261103.json).

This resource retry is separate from the same-latent whole-history decoder
diagnostic. Neither changes emitted historical RGB. Audit copies of preparation,
queue logic, gates, frozen manifest and CPU checks are in
[the artifact directory](artifacts/matrix_observer_lifetime_20261103/prepare.py).
These workspace-specific audit copies are not standalone portable launchers.
