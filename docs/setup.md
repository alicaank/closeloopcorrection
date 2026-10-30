# Team setup

For our single-GPU configuration, see [Spatia and Lyra-2 on the RTX 5090](rtx-5090.md).

Install the small core package in a control environment. Each video model and the pose observer retain their own CUDA/PyTorch environments; installing the core does not install them.

1. Copy `config.example.json` and select `models`. Set `conda_root`, `cache_root` and `python` interpreter overrides. Set `home_root` only when generated disk checks should use another home directory.
2. Add `path_map` entries for externally stored weights, upstream repositories, observer Python and child-process Python paths. Overrides use the template placeholders, are applied longest-first in a single pass, and must have plain path values. `python` overrides also relocate matching native child-interpreter paths; `run --python` explicitly selects the launch interpreter. Child processes must also be mapped.
3. Run `camera-loop prepare --workspace EMPTY_PATH --config CONFIG --clone`. This uses a partial clone, checks out pins, initializes required runtime submodules (including Lyra DA3/VIPE), and applies local adapter patches. Omitting `--clone` only writes the adapter sources, for environments whose upstream sources will be supplied separately. Do not materialize into an existing research workspace.
4. Install each selected upstream environment from its pinned repository instructions. The bundle keeps copies of available requirement files under `upstream_metadata/`; they are references, not a universal cross-model lockfile.
5. Obtain the model and observer assets, respecting their upstream terms. `bundled/asset_download_plan.json` and `weights_provenance.json` record known asset locations/revisions. They are inventory evidence; neither silently downloads files. Reuse teammate-provided caches or map asset paths to your own cache. Some adapters require precomputed text embeddings (the Lyra preparation scripts are included).
6. Supply ffmpeg. Map `@WORKSPACE@/analysis/plan_base_restart_20261008/model_controls_20261024/bin/ffmpeg` to your installed executable, e.g. `/usr/bin/ffmpeg`. Alternatively put an executable at that materialized path. Model-output image/video libraries are installed in native environments.
7. Run `camera-loop doctor`; it reports interpreter existence and source commit matches. It does **not** prove checkpoint completeness, CUDA compatibility or observer readiness. Review the printed run command, then start a short native-parity/correction pilot before long runs.

The dated internal directory layout is retained only inside the relocated compatibility workspace, so the existing model adapters can share helpers. It is not a requirement for new integrations using `AlignedLoop`.

## Evaluation batch

Use [import-batch / validate-batch / run-batch](datasets.md) to prepare another dataset or select more scenes before GPU work.

The adapters consume the existing batch layout:

```text
batch/
  index.json
  images/SCENE.png
  captions/SCENE.json
  trajectories/SCENE.npz
```

`index.json` contains a `scenes` list, with a unique `name`, image path, captions/prompts and model-required metric metadata. The NPZ contains `w2c` (anchor-relative OpenCV, reference-unit translation), per-frame pixel `intrinsics`, `anchor_depth` in those reference units, and `image_height`/`image_width`. The historical fair-format adapters additionally expect `w2c_overrun` for their native chunk overrun. Caption JSON uses frame-index keys. Read the selected adapter's `load_scene` / `fair_common.scene` for model-specific fields; this package does not manufacture missing metric metadata or extend short trajectories silently.

Provide enough trajectory frames for the selected launcher's native schedule (shown by `run`); the models do not all generate the same number of frames. Reference depth, image crop and intrinsics must be registered. Evaluation recordings may be used by the scorer, but future reference RGB must never enter correction inference.

Dataset/checkpoint provenance still needs to be verified per experiment. Inclusion of a model adapter does not establish its released checkpoint's training corpus or the correct held-out split.

## Controls

`baseline`, `sham` and `correction` exist where the bundled native adapter supports them. VMem's `replan_measured` is an explicit separate memory-label ablation. Generic `correction` maps to `replan_commanded`. Memorizon and WorldPlay's historical adapters expose only baseline/correction; the launcher rejects a claimed sham/native arm for them. The upstream native entrypoint is required for Lyra/Spatia/VMem native parity.

The wrapper refuses existing output directories. Use separate directories for matched arms and keep their calibration evidence identical. No queue manager, automatic cleanup, environment installer or training job is started by `prepare`.

## Camera evaluation

New scoring uses VIPE, separately from the DA3 correction observer. Preparation clones its pinned source, but you must install its native environment and cache its assets. Configure the VIPE interpreter using the example `path_map` or `CAMERA_VIPE_PYTHON`. See [VIPE setup and commands](evaluation.md). Historical VGGT dependencies are optional and are no longer cloned by default.
