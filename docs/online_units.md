# Causal runtime camera units

The supported bundled launchers for VMem, WorldWarp, Memorizon, Astronex, and
WorldPlay now calculate unit conversions during a rollout, using the initial
input or geometry of already generated frames. They do not load a fitted scene
scale from a previous run. This change concerns unit conversion; it does not
establish improved camera tracking or qualify the command/gain ablations.

## What is implemented

| Model | Runtime source | Behavior when unavailable |
|---|---|---|
| VMem | The released `get_translation_scaling_factor` on each conditioning window | Invalid native factor stops the rollout |
| WorldWarp | TTT3R initial-anchor depth, then depth correspondence on shared emitted frames at every source window | Correspondence or depth failure stops before native geometry consumes the commands |
| Memorizon | Fresh DA3 metric depth of the initial input, registered to reference depth using the actual input preprocessing | Nonmetric, missing, or invalid depth stops; no depth=1 fallback |
| Astronex | A runtime provider must supply initial-input depth in the camera encoder's native units | No verified provider ships with this repository; launch stops explicitly |
| WorldPlay | The same runtime native-geometry provider contract | No verified provider ships with this repository; launch stops explicitly |

**Astronex and WorldPlay are guarded, not physically calibrated by this change.**
Their pose-only camera encoders do not themselves expose a depth-unit mapping.
Removing reference fits and default one prevents unsupported correction, but
does not create missing evidence. The existing performance tables describe
historical implementations and must not be relabeled as results of this patch.

## Reference and native units

Keep the desired trajectory, measured poses, and controller residuals in one
reference coordinate system. Its initial depth must describe the input image in
the same units as the supplied desired trajectory. This initial-anchor contract
is allowed input; it is not a model-specific calibration or future observation.

For corresponding valid pixels, estimate

```
k = median(native_depth[p] / reference_depth[p])
t_native = k * t_reference
t_reference = t_native / k
```

Depths must share pixels and represent the same camera z-depth convention. The
fit requires at least 256 valid pixels, at least 20% coverage, positive finite
depths, and a p90/p10 ratio no larger than 4. These are geometry rejection gates,
not confidence intervals or demonstrated optimal thresholds. Receipts contain
coverage, dispersion, and median relative residual. The implementation does not
claim to detect every geometric inconsistency.

Do not substitute executed displacement / commanded displacement for `k`.
That ratio also contains the generator's execution response. Fitting it as units
can erase real under-execution from evaluation. Response learning remains a
separate controller quantity.

## Model interfaces

**VMem.** The frozen `0.025 * 22.867` demo factor and global normalization modes
are removed from the supported fair and long launchers. They call the original
native normalizer on each current conditioning window. Its centering, native
`camera_scale`, and `+0.01` formula are preserved exactly. Those are released
encoding rules, not newly fitted controller constants. Each actual factor is
recorded as a `native_window_scale` event. This is a native normalization
contract, not proof of a metric physical scale. The window can include upcoming
commands known at issuance; it contains no future observed video.

**WorldWarp.** `OnlineDepthCalibration` retains the initial anchor and the last
calibrated source frame. For the current 81-frame/one-frame-overlap interface,
chunk 0 uses the input image; chunk 1 reconstructs emitted frames 0–80; chunk 2
reconstructs 80–160, and so on. New native depth at the shared frame is fitted to
its retained reference depth. The new window's last depth divided by its own
`k` becomes the next reference. Shared-image mean absolute RGB disagreement must
be at most 8/255, allowing codec differences; this is not feature matching.

The desired path and observer anchor depth stay in reference units. Only a copy
of the camera tensor is multiplied by the current window's factor. Native
depth and reconstructed source poses already share that window's units.
Measured memory poses are made relative to the last source camera and then
multiplied by the same current factor exactly once. Rejection invalidates the
current factor; the old factor is never silently reused. Chained depth estimates
can accumulate error, so long-rollout GPU validation is still necessary.

**Memorizon.** Use the fresh metric prediction at matching pixels, not just its
median depth. Obtain the resized/cropped pixel intrinsics from DA3's input
processor, not its predicted focal length. Divide the resulting metres per
reference unit by the native training divisor 4 exactly once. A reference depth
median different from one therefore works correctly. `is_metric=False` is an
explicit failure. The rollout log includes the fit and its initial-image hash.

**Astronex/WorldPlay.** Set `CAMERA_LOOP_UNIT_PROVIDER=module:function` only for an
audited runtime native-geometry implementation. The function is called afresh
per rollout with `model`, `anchor`, `intrinsics`, `frame_id=0`,
`available_frame=0`, and `anchor_sha256`. It receives no target path, recorded
reference video, or fitted scale. It must return:

```python
{
    "native_depth": depth_at_the_same_input_pixels,
    "source": "initial_input",
    "frame_id": 0,
    "anchor_sha256": received_anchor_sha256,
    "native_unit_evidence": "How runtime geometry matches this encoder's units",
    # optional: "mask": valid_correspondence_mask
}
```

Scalar-only answers and explicit `scale` fields are rejected. The interface
checks shapes, values, and image identity; it cannot independently prove that
an external provider's claimed native-unit semantics are true. A generic metric
depth model alone is insufficient unless the native metric conversion is known.

Astronex no longer reads batch `native_units_per_anchor_unit` or
`native_w2c_latent`; it encodes the supplied reference-unit target path after
runtime calibration. Its two older reference-calibrated entry points are
disabled with a migration error. WorldPlay no longer accepts `--unit` or receives
one from the batch metadata. Batch validation checks input geometry; successful
validation does not imply that a runtime provider is available.

## Receipts, use, and validation

The implementation is `src/camera_loop/online_units.py`, copied byte-for-byte as
`fair_common_20261025/causal_units.py` for independent native environments. Fresh
rollouts require fresh receipt paths. Receipts are output evidence, never input
calibration caches. Native camera conventions still belong to each adapter.
The older low-level `CameraAlignment`/`AlignedLoop` APIs still accept explicit
contracts and support frozen receipts; they are not an automatic runtime
calibration provider. Use the new runtime helper and updated bundled launchers
for this protocol. Existing materialized workspaces must be regenerated in a
new workspace; this patch does not mutate frozen experiment sources.

CPU tests cover changing native gauges across four windows, future-frame and
reference-video rejection, failed correspondence, stale-scale invalidation,
metric preprocessing and divisor application, fresh provider calls, and native
WorldWarp camera/memory conversion. Where the released VMem checkout is present,
its actual normalizer is tested for exact CPU parity. Bundle relocation and
measured-memory overlay compilation are also checked. Full GPU rollouts and
tracking improvements have not been established by these tests.

The follow-up [Astronex/WorldPlay source audit](astronex_worldplay_scale.md)
distinguishes physical unit calibration from an operational command-response
controller that could learn from normal rollout chunks. That route is now implemented as the opt-in [online response pilot](online_response.md);
it does not validate measured-translation memory writes. The fail-closed behavior
above describes physical-unit mode, which remains separate.
