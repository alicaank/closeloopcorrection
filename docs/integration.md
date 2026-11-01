# Adding a model

## Camera and timeline contract

The public controller uses anchor-relative **OpenCV camera-to-world 4×4 matrices**, with translation and z-depth in one fixed reference unit. Frame 0 normally defines the reference frame. Inputs must be finite proper SE(3) transforms. Depth must be z-depth, not inverse depth or ray distance, unless you convert it first.

Declare the native c2w/w2c direction, OpenCV/OpenGL camera basis, and scale `native_units_per_reference_unit`. For registered native/reference depths, that scale is `median(native_depth/reference_depth)`. Multiplying translation by it must produce the exact units entering the generator, after its own normalization. Use `PixelTransform` to transform pixel intrinsics and registered reference depth consistently with the model's crop/resize. It uses pixel-center resize coordinates; select the transform matching the actual upstream preprocessing.

`anchor_depth` calibrates from at least 256 valid shared pixels and 20% image coverage. A p90/p10 depth-ratio spread above 4 fails. These are initial rejection thresholds, not proof of accurate geometry. `metric` uses metres per reference unit; `metric_div4` additionally divides by four for the current Memorizon conditioning contract. `native_callback` requires the actual native normalization factor and provenance.

Changing windows may change a model's native normalization. Create a separate calibration context for each such window and convert both prior and future poses consistently into that context. Do not reuse the initial context across a changing normalization. Matched baseline/sham/correction arms use the same calibration. An existing receipt cannot be rebound to a different anchor/checkpoint/context; a fitted scale drifting more than 3% is rejected.

The observer returns poses in the fixed reference frame, after its own raw reconstruction has been registered. Model command calibration and observer gauge alignment are distinct operations. The observer adapter must gate its reconstruction quality before constructing `Measurement(..., quality={'passed': True, ...})`; the controller cannot independently certify an external estimator. `camera_loop.quality` provides depth and pose checks.

Frame IDs refer to **decoded output frames**, not latent indices or padded history. Supply an explicit mapping at the adapter boundary. All future IDs must be strictly increasing and later than the measured last available frame. Models with delayed decoding need a model-specific causal feedback boundary; never relabel an older estimate as the latest frame to bypass this guard.

The portable controller fades correction according to elapsed output-frame IDs, so sparse and dense samples of the same target give the same commands at shared timestamps. The low-level `rejoin` function accepts an explicit `progress` vector; omitting it assumes uniformly spaced samples. Neither path infers timestamps from latent indices.

`pose_quality` rejects empty/all-missing observations. Omitted context poses are allowed when actual poses exist, so this check alone does not guarantee that a particular requested endpoint was observed. `Measurement.validate()` checks the endpoint separately. Proper SE(3) algebra and a depth-scale fit are only initial checks; they do not establish shared-frame consistency or reliable current image overlap.

Sham may supply `Measurement(None, frame_id, rejected_quality_report)` and keeps planned commands. It still requires the actual boundary frame and strictly future command IDs. This controller bypass does not isolate a host's RNG, VAE cache or offloading effects; native/baseline/sham image parity must be tested in the model host.

## Minimal host integration

```python
from camera_loop import AlignedLoop, Measurement, ModelHooks, register_model

register_model('my_generator', direction='c2w', axes='opencv',
               scale_strategy='anchor_depth')

hooks = ModelHooks(
    model='my_generator',
    calibration=lambda: {
        'reference_depth': reference_anchor_depth,
        'native_depth': native_anchor_depth_on_same_pixels,
        'mask': reliable_shared_pixels,
    },
    # This includes compensation applied by previous controller calls.
    previous_command=lambda: host.last_absolute_native_camera(),
    # Encode rays/Plucker features with the model's existing implementation here.
    write_future=lambda frame_ids, native_poses: host.set_future_cameras(
        frame_ids, native_poses),
    history_labels='commanded',
    layout='absolute',
)
loop = AlignedLoop(hooks, receipt_path='outputs/scene/alignment.json',
                   checkpoint='checkpoint-content-id', anchor_id='scene-anchor')

# After generating and decoding a chunk, measure only available frames.
measurement = Measurement(measured_reference_c2w, last_output_frame,
                          observer_quality_report)
event = loop.update(
    measurement,
    planned_endpoint=original_plan[last_output_frame],
    future_reference_c2w=original_plan[next_output_ids],
    future_frame_ids=next_output_ids,
    available_last_frame=last_output_frame,
)
```

This is a host integration template: `host`, depth arrays, observer and plans come from your model. `examples/synthetic_loop.py` is the executable CPU example.

Use `history_labels='commanded'` if generated history is stored under the camera commands that produced it. The controller bridges the measured frame into that commanded-memory frame before encoding future cameras. Use `history_labels='measured'` only if the host actually stores history under the measured poses. The controller does **not** rewrite a memory bank automatically.

`stage_memory_write` validates a proposed historical pose update and
`bridge_future` expresses desired future poses against the label actually
committed by the host. The staging result is not a completed native cache write.
Seven model-specific consumers are available through the separate
[measured-memory launcher](measured_memory.md), now included in the source
bundle. See [the paper method reference](paper_method_reference.md) for the
transaction semantics, mathematical bridge, and implementation boundaries.

With `layout='relative_chain'`, the result is consecutive relative transforms in the declared native basis. The `previous_command` callback must still return an **absolute** native pose. Model-specific temporal sampling, action encodings and Plücker features remain in the host callback.

A direct `CorrectionSession` additionally exposes separate translation/rotation gains in [0,1]. The default rejoin blends the residual down across the next chunk and reaches its planned endpoint. It is a causal geometric controller, not an assertion of optimal dynamics or appearance preservation.

## Required adapter checks

Before long rollouts, check native baseline parity, exact frame-ID mapping, a synthetic translation and rotation, encode/decode round trips, anchor reprojection and units, future-only writes, and identical baseline/sham sampling. Evaluate correction with an independent pose estimator and matched seeds. Log commanded and measured poses, calibration evidence, quality decisions and native normalization context. Do not infer causality from command labels alone.
