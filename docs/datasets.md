# New datasets and more scenes

The package can import **already calibrated** scenes into the compatibility batch format, validate the selected model's input contract, and run multiple scenes serially. It does not reconstruct camera poses/depth from arbitrary videos, verify a dataset's held-out status, or install all native GPU environments. Fresh-machine GPU execution remains unverified; follow [setup](setup.md) and the [5090 guide](rtx-5090.md) before executing experiments.

## Import your own scenes

Create a JSON manifest like this, with paths relative to the manifest file:

```json
{
  "dataset": "my-dataset",
  "split": "held-out-test",
  "pose_convention": "opencv_w2c_anchor_relative",
  "depth_convention": "z_depth_same_units_as_translation",
  "scenes": [
    {
      "name": "room_001",
      "image": "source/room_001.png",
      "geometry": "source/room_001.npz",
      "caption": "A camera moves forward through a furnished room.",
      "metres_per_anchor_unit": 2.5,
      "metric_scale_evidence": "Measured calibration file or estimator/version/receipt"
    },
    {
      "name": "room_002",
      "image": "source/room_002.png",
      "geometry": "source/room_002.npz",
      "caption": "A camera turns left inside a room.",
      "metres_per_anchor_unit": 1.8,
      "metric_scale_evidence": "Measured calibration file or estimator/version/receipt"
    }
  ]
}
```

The numeric scales above are examples, **not defaults to copy**. Record your actual conversion/provenance. Do not infer metric scale from the desired camera displacement. Scene IDs must be unique letters/digits/underscores/hyphens; the package supports any number of scene entries subject to storage/runtime.

Each source NPZ must contain:

| Array | Required meaning |
|---|---|
| `w2c` | `[T,4,4]` finite proper OpenCV world-to-camera matrices, relative to frame 0, first pose identity |
| `intrinsics` | Pixel-space `[3,3]` K or `[T,3,3]`; fixed intrinsics, positive focal lengths and zero skew |
| `anchor_depth` | `[H,W]` z-depth registered to the **same source anchor image and K**, in the same units as camera translation |
| `w2c_overrun` | Optional recorded continuation, including an identical `w2c` prefix. If absent, `w2c` is reused without extending it |
| `native_w2c_latent` | Legacy archival field; the runtime Astronex adapter does not consume it |

Choose the intended frame sampling/time correspondence before import. No frames or poses are extrapolated, no latent/RGB timeline is guessed, and no raw-video pose estimation is launched. Declare camera conventions honestly; the importer cannot detect mislabeled but numerically valid matrices. Convert c2w/OpenGL inputs before using this manifest.

```bash
camera-loop import-batch --manifest my-scenes.json --out /data/my-batch
# If the calibrated source grid is not 864x480, opt into joint RGB/depth/K resizing:
camera-loop import-batch --manifest my-scenes.json --out /data/my-resized-batch --resize
camera-loop validate-batch --batch /data/my-batch --model spatia
```

Import requires a new output directory and writes images, NPZs, captions, `index.json` and an import receipt. `--resize` stretches to the legacy **864×480** grid with consistent pixel-center intrinsics/depth transformation. It does not crop away content, alter 3D translation units or add geometry. This normalization exists because several old adapters have fixed 864×480 crop/resize assumptions. The portable controller API itself has no such input-size restriction. Variable zoom/intrinsics need an adapter change and are rejected by this importer.

Optional per-scene `captions` is a dictionary such as `{"0":"Initial caption", "81":"Next caption"}`. Without it, the scene's `caption` is used throughout. Lyra uses these exact strings; rebuild/check its cache when captions change. Some other adapters consume only the scene-level caption.

## Model-specific requirements

`validate-batch --model MODEL` checks the **default schedule exposed by `camera-loop run`**, not arbitrary flags passed directly to native scripts.

| Model | Minimum main / overrun poses | Additional metadata |
|---|---:|---|
| Lyra-2 | 481 / 481 | `metres_per_anchor_unit`; exact caption embeddings prepared separately |
| Spatia | 481 / 481 | Registered anchor depth for native MapAnything scale fit |
| Matrix-Game 3.5 | 1 / 505 | `metres_per_anchor_unit` |
| VMem | 41 / 41 | Native window normalization remains inside its adapter |
| Astronex | 77 / 93 | `prompts.astronex`; runtime native-depth provider required separately (no provider ships) |
| LingBot v2 | 1 / 173 | Scene caption; native path normalization |
| DreamX | 1 / 165 | Scene caption; adapter-specific delayed feedback indexing |
| WorldWarp | 1 / 161 | Registered anchor depth; runtime shared-frame scale transfer at each window (default launch: two chunks) |
| SANA-WM | 169 / 169 | `metres_per_anchor_unit` |
| Memorizon | 161 / 161 | `prompts.memorizon`, starting with `First-person perspective — character not visible. ` |
| WorldPlay | 161 / 173 | `prompts.worldplay`, centered principal point; runtime native-depth provider required separately (no provider ships) |

The lengths differ because native chunk schedules differ. They are not a fair-evaluation recommendation: compare matched common frame intervals and retain model-specific overrun outside the scored interval. WorldPlay and Astronex require runtime native geometry; batch scalar metadata cannot qualify the conversion. Memorizon estimates metric anchor depth at rollout time and rejects nonmetric predictions. WorldWarp transfers units through shared generated frames at each geometry window. See [causal runtime units](online_units.md) for implementation, failure behavior, and validation limits. Passing input validation does not establish runtime calibration or GPU qualification.

The importer preserves explicit model-specific prompt/native metadata; it does not fabricate it from another model's prompt or a guessed scale. A dataset may pass validation for Spatia and fail for Astronex or WorldPlay. Missing fields are reported per scene before GPU execution.

## Plan or run more scenes

```bash
# Omitting --scenes selects every scene in index.json. Default: print the plan only.
camera-loop run-batch --workspace /data/camera-work \
  --model spatia --python /opt/envs/spatia/bin/python \
  --batch /data/my-batch --out /data/experiments/spatia-test \
  --arms baseline correction

# Select scenes explicitly; add --execute to launch serially on one GPU.
camera-loop run-batch --workspace /data/camera-work \
  --model spatia --python /opt/envs/spatia/bin/python \
  --batch /data/my-batch --out /data/experiments/spatia-subset \
  --scenes room_002 room_001 --arms baseline sham correction --execute
```

The runner executes scenes in the chosen order and arms in the chosen order, with the existing model-specific schedules and **seed 7**. It does not support arbitrary seed sweeps or automatic GPU scheduling yet. Use a free GPU; it does not inspect or preempt the research queue. Unsupported arms are rejected while planning.

Outputs are isolated under `EXPERIMENT/SCENE/ARM`, with one combined stdout/stderr log per job, immutable `plan.json` and an updated `status.json`. The plan records a content fingerprint of the batch and isolates calibration caches by that fingerprint. Changing input data invalidates the plan. Existing experiment roots are refused: there is no implicit resume or overwrite. To retry failures, select the failed scenes into a new experiment root.

A nonzero process exit stops the remaining jobs. `process_succeeded` means exit code zero, **not** that camera/appearance metrics passed or all native artifacts are complete; inspect native completion receipts and perform independent evaluation. This distinction matters for historical adapters that catch some errors internally.

## Lyra cache per dataset

Precompute with the relocated `lyra2_plugins/run_t5_captions.sh` into a dedicated path, as described in the 5090 guide. Then pass it explicitly:

```bash
camera-loop run-batch --workspace /data/camera-work \
  --model lyra --python /opt/envs/scoring/bin/python \
  --batch /data/my-batch --out /data/experiments/lyra-test \
  --t5-cache /data/caption-caches/my-batch.pt \
  --arms baseline correction --execute
```

Before execution, Lyra's native child interpreter checks every selected caption and the expected tensor shape with CUDA hidden. Missing captions/cache/interpreter fail before generation. Single-scene `run` accepts the same `--t5-cache` flag. Direct native launchers can use `CAMERA_LYRA_T5_CACHE`; otherwise they retain the older shared-cache default.

## What still prevents a turnkey claim

The new importer, validation and sequential planning are CPU-tested, including synthetic scenes with a different source resolution. They do not substitute for a fresh-machine model run. Native CUDA extension installation, checkpoint completeness/downloads, all-model GPU smoke tests, scientific scale audits, and automatic independent scoring are still separate work. No dataset corpus/split relationship is inferred from a model name. Keep dataset provenance and checkpoint provenance with each experiment.
