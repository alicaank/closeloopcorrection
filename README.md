# Closed-loop camera correction

Model-aware camera alignment and causal command correction for video generators.
The controller measures a generated endpoint, replans **future camera commands**, and preserves the model's native camera representation and memory policy. It does not repair already-generated images or train a generator.

## Quick start

```bash
git clone https://github.com/alicaank/closeloopcorrection.git
cd closeloopcorrection
python -m pip install -e .
python examples/synthetic_loop.py
python -m unittest discover -s tests -v
camera-loop models
```

The core requires Python 3.10+, NumPy and SciPy; it does not load Torch or occupy a GPU. The demo is a simulated under-executing camera, **not a measured model result**.

## Automatic alignment

A model declares its camera direction, camera axes, native unit contract and conditioning layout. The alignment layer then:

1. Fits native/reference depth scale on registered anchor pixels, or reads the model's explicit metric/normalization conversion.
2. Checks shared depth coverage and ratio dispersion; rejects unsupported or missing unit contracts.
3. Stores the conversion against the checkpoint, anchor and normalization context; matched runs reuse it and inconsistent calibrations fail.
4. Converts commands and measured geometry consistently, including OpenCV/OpenGL and c2w/w2c conventions.
5. Rejects stale measurements, invalid poses and writes to already-generated frames.

A response ratio such as 1.05 is **not** a scale-calibration target. Calibrate geometry before interpreting under-execution. An observer's raw reconstruction gauge may change at each call: align each observation to the fixed reference independently. Do not freeze a raw observer scale or fit scale to the desired trajectory.

```bash
# NPZ arrays must already describe the same anchor pixels and depth convention.
camera-loop align --model lyra --depths registered_anchor.npz \
  --checkpoint checkpoint-sha --anchor-id scene-01 \
  --out outputs/scene-01/alignment.json
```

`registered_anchor.npz` contains `reference_depth`, `native_depth`, and optional `mask`. Other strategies accept `--metric-scale` or `--native-scale` together with `--evidence`. Unknown factors are never silently replaced by one in the core API.

## Integrate another generator

Use `ModelHooks` and `AlignedLoop`: supply native calibration, the actual previous absolute camera command, and a callback that updates future conditioning. The host model keeps its own memory, latent timeline, ray/Plücker encoding and generation loop.

See [the integration contract and example](docs/integration.md). This API supports generators that expose camera conditioning and causal generation boundaries. It cannot add camera control to a model that has no such interface.

## Run the existing model adapters

The source bundle includes local adapters for **Lyra-2, Spatia, Matrix-Game 3.5, VMem, Astronex, LingBot v2, DreamX, WorldWarp, SANA-WM, Memorizon and WorldPlay**. Their checkpoint requirements, scale contracts and validation limits differ; see [the model table](docs/models.md).

The portable API and the bundled research launchers are separate integration paths. The latter preserve the existing model-specific controller code; they are not all migrated to `AlignedLoop`. In particular, historical native-scale assumptions are not made validated merely by packaging them here.

```bash
cp config.example.json config.local.json
# Edit external interpreter/cache/weight path mappings for your machine.
camera-loop prepare --workspace /data/camera-work --config config.local.json --clone
camera-loop doctor --workspace /data/camera-work --config config.local.json

# Prints the exact command without starting GPU work.
camera-loop run --workspace /data/camera-work --model spatia \
  --python /opt/envs/spatia/bin/python --batch /data/eval-batch \
  --scene scene-01 --out /data/results/spatia-scene-01 --arm correction
# Add --execute after installing the native model/observer environments and assets.
```

`prepare` requires an empty destination, verifies source hashes, checks out pinned upstream revisions when `--clone` is set, applies the included patches, and relocates adapter paths. It does not install CUDA environments or download checkpoints/datasets. Each model runs in its own native environment. Follow [team setup](docs/setup.md) for assets, subprocess interpreters and batch format.

## Validation and scope

CPU tests cover geometry, controller semantics, calibration binding, quality/causality rejection, all 11 launcher paths and compilation after relocation. A wheel installation is also tested separately. GPU end-to-end portability has not yet been established. New Spatia visibility batching and Lyra/SANA/WorldWarp observer guards remain pending GPU qualification; WorldWarp's guarded continuation is limited to two chunks.

See [validation protocol](docs/validation.md). Source pins and available upstream license notices are retained under `src/camera_loop/bundled/`. No weights, datasets, generated videos, credentials or training checkpoints are included. See [NOTICE](NOTICE).
