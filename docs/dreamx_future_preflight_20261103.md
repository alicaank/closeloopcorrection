# DreamX future-only correction: native timeline preflight

The future-only arm needs two distinct timestamps: the newest emitted RGB frame
and the camera anchor used by the next native relative-pose encoding. The
released DreamX encoder uses camera indices `[0, 1, 5, 9, …]` and groups three
latents per native chunk. At the end of nine latents, RGB is available through
frame 32, while the last native camera anchor is frame 29. The next camera
samples are 33, 37 and 41.

This is a requirement for the proposed future-command integration, not evidence
of a defect in the frozen measured-memory cohort. No running source, old output,
score or method setting was changed.

## Latest observation and native reference have different roles

Let `b` be the latest emitted RGB frame and `a ≤ b` the last native camera
anchor. Given qualified observations at both frames, recovery uses the residual
at `b`. If native memory keeps commanded labels, encode the desired future pose
`D_t` using

```
U_t = U_a @ inverse(M_a) @ D_t,     t > b
```

Then the native relative transform satisfies

```
inverse(U_a) @ U_t = inverse(M_a) @ D_t.
```

Using `M_b` and `U_b` in this bridge would generally violate that identity,
particularly when the observed orientations differ between `a` and `b`.
An adapter must query the actual anchor; an unobserved anchor is rejected.
Already emitted command entries through `b` remain unchanged. The first
feedback issuance remains exactly nominal, as specified in the framing review.

`issue_fractional_future` now accepts the optional `memory_anchor_frame`.
Its default preserves the prior endpoint-anchor behavior for existing callers.
The receipt records both timestamps. This is a reusable temporal contract for
sparse camera conditioning, not a DreamX-specific fitted constant. Native stride,
frame selection and chunk size come from the adapter contract.

The host must commit only the next chunk's native tensors. Re-encoding an entire
trajectory after changing this chunk can also change a later chunk's relative
reference, even if that later absolute target was not edited. That later chunk
must be reconstructed at its own issuance boundary using the committed anchor;
it must not be silently changed during the current issuance. A rejected fresh
observation does not justify restoring an obsolete native relative reference.

## CPU qualification

The [source-backed record](dreamx_future_preflight_20261103.json) uses the actual
released `Camera`, `get_relative_pose`, `_invert_SE3` and
`cam_params_to_prope_dict` definitions, extracted without loading model weights.
It exercises all four planned scenes with synthetic lag and differing anchor
orientation. These are arithmetic tests, not generated or measured outcomes.

- Float32 native reconstruction matches the intended relative transform within
  `1.37e-7` maximum absolute matrix-entry error.
- BF16 reconstruction has maximum absolute matrix-entry error `0.001835` in
  these cases. This is observed quantization error, not a universal tolerance.
- Past tokens, the unissued tail and intrinsics remain byte-identical when only
  the next chunk is committed.
- Nominal first issuance is exact. Emitted command entries are unchanged.
- The deliberately incorrect endpoint bridge disagrees by about `0.122` in a
  matrix entry under the synthetic rotation mismatch. This is not a pose-error
  measurement or a claimed physical displacement.
- All **137 repository CPU tests pass**, including two new regressions for the
  delayed native anchor and rejection of missing/noncausal anchor observations.

The audit script is preserved in
[the artifact directory](artifacts/dreamx_future_preflight_20261103/qualify.py).
The JSON records the original script, native source, trajectory and policy
hashes. It does not assert GPU parity or corrective authority.

## Focal contract remains an independent qualification

The released native encoder ignores input camera-row intrinsics and constructs
fixed normalized focal lengths from its training configuration. At the source
864 × 480 grid these correspond to approximately `(436.36, 430.98)`, versus
source-reference focal lengths around `(388, 384)` on the four scenes. The
native/source ratios are `1.120–1.125`. This extends the prior two-scene finding
to the four planned scenes; it does not establish the generated video's true
intrinsics or explain a measured failure by itself.

Changing only the observer K would mismatch the original anchor-depth pixel
rays. Changing only native K would alter the released conditioning convention.
A separately qualified input/geometry contract is required: for example,
consistent anchor RGB/depth reprojection to the native fixed-K grid, with the
same transformation represented in evaluation. All such preprocessing must
use the initial input and declared native contract, never future reference
motion or output-score fitting. That candidate has not been implemented or
GPU-qualified in this preflight.

At the time of this preflight, the future-only GPU pilot was **not queued**. Next,
resolve that focal contract, implement the causal native next-chunk commit,
qualify observer/no-op parity, and freeze the matched moving-scene pilot with
commanded memory. Only later observed lag should perturb future commands.
Execution ratios remain observational; no inverse gain is enabled from them.

Subsequent work implemented the camera-grid contract and causal next-chunk
commit. The [separate frozen pilot](dreamx_future_command_20261103.md) is now
queued with four native-encoder CPU tests passing. GPU parity remains pending;
this does not change the historical scope of the preflight above.
