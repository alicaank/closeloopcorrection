# LingBot shared camera-normalization candidate

The frozen repeated-memory cohort remains unchanged. This separate arithmetic
audit examines the first write, using the identical pre-intervention RGB prefix
of each seed-7 pair. It uses the actual released relative-camera and camera-ray
encoder functions, accepted emitted observations, and the known command plan.
No output score selects a denominator, threshold or scene exception.

| Scene | Measured maximum native step / plan maximum | Largest future step after shared normalization | First write |
| --- | ---: | ---: | --- |
| 02267acf | 51.2191 | 0.019524 | Accepted; later visual collapse |
| c7418b7d | 6.2146 | 0.160911 | Accepted; recognizable sampled geometry, rotation tradeoff |
| 0238de14 | 10.0175 | 0.099826 | Accepted; later playground fidelity failure |
| 924228bf | Unavailable | Unavailable | Observer rejected; stored fallback is not a measurement |

These are the **77-frame** trajectories, whose plan normalizers differ from
the shorter component tests. Each accepted first write covers RGB frames
0, 4, 8 and 12. Initial native relative rotations are below 0.5° in these three
cases. The current large translation features remain finite in BF16: this is
not demonstrated numerical overflow. Three cases cannot establish a support
threshold or prove that conditioning magnitude caused the visual failures.

## Shared arithmetic, with matched controls

Let `U_i` be the planned native framewise relative transform and `M_i` an
accepted emitted prefix of those native samples. Define

```
s = max(max_i ||translation(U_i)||, max_emitted_i ||translation(M_i)||).
```

Build a commanded-history control and a measured-history candidate using the
same `s`, dividing every translation by it. Both use the exact same planned
future transforms. Rotations remain unchanged. With zero translation everywhere,
leave the arrays unchanged and report an unidentifiable scale; do not invent a
metric-depth or native-unit value of one. This normalization does not identify
physical units or execution gain.

[conditioning_gauge.py](../src/camera_loop/conditioning_gauge.py) implements this
candidate as a pure operation. It returns immutable arrays and explicitly marks
deployment, cache-rebuild and training-support qualification as false. It is
**not connected to the primary native adapters or original cohort**. Input
alignment and observer acceptance are adapter prerequisites. Subsequent work
integrates it into the separate [first-boundary diagnostic](lingbot_gauge_pilot_20261103.md),
which is queued with mandatory GPU identity and attribution gates pending.

The shared maximum bounds each encoded translation norm by one while preserving
the ratio of history to future motion. It also makes future commands much smaller
than under the original plan-only normalization, as the table shows. Thus this
is not an innocuous unit conversion whose causal effect can be ignored. The
native function named `get_plucker_embeddings` concatenates ray origins and
directions; it does not compute cross-product Plücker moments. The audit checks
these actual channels on an 8×8 ray grid. Rotation/direction channels are exact
under translation rescaling, and future features match exactly between controls.

All **152 repository CPU tests pass**. Added tests cover unchanged inputs,
read-only outputs, identical future tensors, common positive-unit rescaling,
history/future ratio preservation, exact identity normalization, zero motion,
and malformed contracts. The initial test fixture used an Euler-angle shape
rejected by the installed SciPy; its failure log is retained, and the corrected
explicit column fixture passes. Native float32 arithmetic checks and input/source
hashes are preserved in the [audit](artifacts/lingbot_shared_gauge_20261103/audit.json).
No generated improvement, cache parity or training support is claimed.

## The next causal test must separate two effects

A same-prefix test needs three arms:

| Arm | History labels | Denominator | Future tensors |
| --- | --- | --- | --- |
| A | Commanded | Original plan maximum | Original |
| G | Commanded | Shared plan/observed-prefix maximum | Shared-normalized |
| M | Measured | The same shared maximum as G | Exactly identical to G |

`G − A` tests the consequence of common rescaling and its smaller future
conditioning. `M − G` tests the measured-history substitution under that same
conditioning. A normalization-plus-memory comparison against A alone confounds
these effects. At the first boundary, the prefix and accepted observation can
be shared causally; no outcome-dependent amplitude selection is needed.

Before interpreting a GPU test, the identity path must reproduce native camera
tensors, caches and output bytes, and all arms must preserve emitted RGB.
At later boundaries, a changed denominator requires rebuilding **every affected
retained camera-dependent state** in chronological order, including attention
sinks and the native rolling-cache indices. Multiplying existing KV tensors is
not justified: camera conditioning passes through learned nonlinear layers.
Cross-attention text state can only be reused after verifying that it is
independent of camera conditioning. Replay must use saved generated latents,
the original clean-context timestep and conditioning, with preserved RNG.

A first-boundary experiment does not qualify that later full-history replay.
Nor does a bounded feature norm prove that the model supports the resulting
history/future speed ratio. If those conditions cannot be demonstrated, keep
the previous state and report abstention rather than calling the repair deployed.
The existing 40-pair study and its failures remain the primary evidence; no
case is removed and no new threshold is fitted to these outcomes.
