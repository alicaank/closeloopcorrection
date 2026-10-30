# VIPE evaluation

VIPE measures the generated camera path offline. DA3 remains the inference-time correction observer. The source bundle includes the isolated VIPE worker, its scale-registration layer, a combined camera/DINO/LPIPS scorer, and the previous VGGT implementations for explicit historical reproduction. Default long/pair/gate entrypoints invoke VIPE; they do not fall back to VGGT on failure.

## Environments and assets

`camera-loop prepare --clone` now checks out `nv-tlabs/vipe` at commit `95a8816947602ddc26fcb7a80bea4f9313059578` into `<workspace>/vipe`, in addition to the selected models and shared Lyra observer source. It no longer clones VGGT by default. The VGGT pin is retained for historical work.

Install VIPE using the installation instructions in that checkout and build its CUDA dependencies for your GPU. This package does not install its native environment or fetch its model weights. Our inspected local VIPE interpreter uses Python 3.10; that is an observed setup, not a guarantee for every CUDA/PyTorch combination. See the pinned source's `README.md`, `docs/` and `envs/` for its environment recipes. Upstream license notices are included under `bundled/upstream_metadata/vipe/`.

The scorer and VIPE run in separate processes:

| Process | Requirements |
| --- | --- |
| Scoring parent | NumPy, SciPy, Pillow, OpenCV, plus Torch/torchvision and LPIPS and the assets required by the bundled `evaluate_recorded.py` for image metrics |
| VIPE worker | Pinned VIPE source, its native/CUDA dependencies, default SLAM model assets and `unidepth-l` assets |

The worker uses the default VIPE pipeline with supplied fixed intrinsics, instance masking disabled, post depth alignment disabled and a 512-keyframe buffer. It runs SLAM plus dense trajectory filling and rejects missing poses or buffer overflow. It uses the optimized first-keyframe depth in the final SLAM gauge. The worker defaults to offline Hugging Face access; populate the required caches before evaluation. Check image-metric checkpoint paths in the materialized `training_runs/joint_samplerfix_20261012/evaluate_recorded.py` and configure `cache_root` or `path_map` before preparation. The extracted image scorer expects `torch/hub/facebookresearch_dinov2_main` and `torch/hub/checkpoints/dinov2_vits14_pretrain.pth` under that cache root, plus the LPIPS/AlexNet assets.

Set interpreter/source paths explicitly, or use the corresponding example `path_map` entry when materializing:

```bash
export CAMERA_VIPE_PYTHON=/opt/envs/vipe/bin/python
export CAMERA_VIPE_SOURCE=/data/camera-work/vipe
export EVAL_ROOT=/data/camera-work/analysis/plan_base_restart_20261008

# Imports/configuration only; explicitly hides CUDA and does not load models.
"$CAMERA_VIPE_PYTHON" "$EVAL_ROOT/vipe_eval_20261030/worker.py" --dry-run
```

## Score a matched pair

First prepare a registered batch using [the dataset guide](datasets.md) and generate both arms. Each source directory must contain `frames.npz` (`uint8 [N,H,W,3]`, RGB), `<scene>_frames/frame_00000.png` and following frames, or `review_full.mp4`. A video file path is also accepted. Frame zero must be the shared conditioning anchor, followed by the generated frames in order. Prefer original RGB arrays over video re-encoding when available, and keep the format identical across arms.

```bash
/opt/envs/scoring/bin/python "$EVAL_ROOT/long_domain_20261028/score_long.py" \
  --batch /data/eval-batch --model spatia --scene scene-01 \
  --horizon 480 --seed 7 \
  --runs baseline=/data/results/spatia-baseline \
         correction=/data/results/spatia-correction \
  --out /data/evaluation/vipe/spatia-scene-01.json
```

`--horizon 480` means 481 input frames including frame zero. Set the horizon to the available matched generated length for short pilots. `--seed` records generation provenance; it does not regenerate either video. The pair entrypoint at `new_models_loop_20261028/score_pair.py` defaults to 160 generated frames; the long entrypoint defaults to 480. Pass `--horizon` explicitly for comparisons. Both use dense camera evaluation but retain different historical image-sampling schedules; do not pool their appearance scores without matching that schedule.

The batch needs `index.json` plus `trajectories/<scene>.npz` with registered `w2c` (or `w2c_overrun`), pixel `intrinsics` and `anchor_depth`. Image metrics also need the recording frames indicated by each scene's `frames_dir` and `original_start`. See the dataset guide for naming and indexing. The current scorer assumes fixed intrinsics and known adapter crop/resize rules; a new camera preprocessing scheme needs its own `output_K` and reference-image transformation before use. It does not automatically support arbitrary changing intrinsics.

The 80-frame scale-gate entrypoint at `continuous_eval_20261029/scale_20261030/measure80.py` runs VIPE on all 81 frames, then computes the existing gate statistics on frames `0,4,...,80`. Its inherited execution-ratio threshold is a protocol choice, not a VIPE calibration rule or a universal property of correct camera motion. It does not load the image-quality networks.

## Outputs and interpretation

The output JSON identifies `estimator: VIPE` and `evaluation_protocol: vipe_dense_anchor_depth_v1`. Each arm includes dense position/rotation errors, measured c2w poses, source path, path-length ratio and the anchor registration diagnostics. Long/pair camera means exclude frame zero and include every generated frame. Horizon summaries include camera mean/endpoint, rotation mean/endpoint, DINO and LPIPS. The 80-frame gate reports its subsampled camera arrays explicitly.

Adjacent `<report>_trajectories/` files contain each arm's registered/raw c2w arrays, frame IDs, VIPE configuration, source commit, worker hash, timing, peak Torch allocation and worker log. A contact sheet is written for long/pair reports. Camera positions use fixed reference anchor-depth units, not verified metres.

Registration uses the median reference/native anchor-depth ratio, requires at least 256 valid shared pixels and 20% coverage, and rejects a ratio p90/p10 greater than four. These are quality heuristics, not proof of correct reconstruction. The only rigid alignment is to the first pose; there is no trajectory-wide similarity fit that could hide execution error.

Every VIPE worker exits before the parent loads DINO/LPIPS, avoiding simultaneous residency of the pose and image-metric networks. Run evaluation serially after generation on a shared GPU: the standalone scorer does not acquire your scheduler's lock. Input IPC files temporarily need RAM/disk proportional to the full RGB clip. Reports and trajectories reject overwrites; after a failed attempt use a new output stem and preserve the failed log for diagnosis.

Do not compare a previous VGGT score directly with a new VIPE score as evidence of improvement. Re-evaluate both arms with the same frames, fixed intrinsics and protocol. Reference poses may themselves have been reconstructed by VIPE, so they are not asserted to be independent ground truth. Check trajectories, images, estimator failures and seed variation before interpreting a change.

CPU tests cover registration invariance, preserving true motion error, invalid geometry rejection and model crop intrinsics. A local configuration/import dry run passed without initializing CUDA. Full GPU scoring and fresh-machine installation remain pending qualification at this release.
