# Closed-loop camera correction

**Method: causal camera rejoin for frozen video generators.** Observe generated
camera motion, express it in a consistent reference frame, and rejoin the
planned path through future native camera commands. Generator weights and
already emitted frames stay unchanged. No separate calibration probes or online
gain learning are required by the Method.

[Method and equations](docs/paper_method_reference.md) ·
[Results](docs/results.md) · [Failures](docs/failures.md) ·
[Essential ablations](docs/ablations.md) ·
[Interactive results](https://camera-correction-results-20261029.alicankk.chatgpt.site)

## Quick start

```bash
git clone https://github.com/alicaank/closeloopcorrection.git
cd closeloopcorrection
python -m pip install -e .
python examples/synthetic_loop.py
python -m unittest discover -s tests -v
camera-loop models
```

Python 3.10+, NumPy, SciPy and Pillow are required for the portable package.
The synthetic demo tests controller behavior, not measured model performance.
Native generation and evaluation require their own model environments and assets.

## How the Method works

1. Declare the adapter's camera convention, native-unit contract, timestamps and
   memory-label policy. Missing physical calibration is not silently set to one.
2. Estimate the available generated boundary pose and register it to the initial
   reference. No future reference RGB enters feedback.
3. Form a continuation that begins from the measured residual and returns to the
   planned chunk endpoint. Bridge into the coordinate frame represented by memory.
4. Encode and write only future camera inputs; keep generated history immutable.
5. Evaluate paired videos with fixed targets, identity controls, pose errors and
   visual inspection. Finite observer poses alone do not establish scene survival.

Use `CorrectionSession`, or `ModelHooks` with `AlignedLoop`, for the portable
API. Native launchers retain model-specific schedules and conditioning rules;
they are not all identical implementations of that API. See
[integration](docs/integration.md) and [known failure findings](docs/failures.md).

## Evidence and scope

Historical Method results improve mean position in 9/10 cohort/seed groups,
with scene-level failures retained. Different models, horizons and observers
are not pooled. VMem's reported arm includes measured memory; command-only
correction does not inherit that result. [Results and limitations](docs/results.md)
include the negative cases. Bounded correction and online gain learning remain
separate ablations, not additional default methods.

CPU tests check contracts and implementation behavior. They do not establish
GPU efficacy or fresh-machine portability of every adapter. Unsupported native
unit providers and failed identity controls remain explicit limitations.

## Run and integrate models

- [Installation, assets and native environments](docs/setup.md)
- [Supported adapters and qualification limits](docs/models.md)
- [Dataset preparation and serial experiments](docs/datasets.md)
- [Runtime camera units](docs/online_units.md)
- [Measured-memory adapter contract](docs/measured_memory.md)
- [VIPE evaluation](docs/evaluation.md)
- [Validation](docs/validation.md) and [32 GB GPU setup](docs/rtx-5090.md)

Source pins, upstream notices and required adapters remain in
`src/camera_loop/bundled/`. No model weights or datasets are shipped. See [NOTICE](NOTICE).

Experiment diaries, copied queue snapshots and publication media have moved out
of the tracked tree. Their immutable sources and local backup are listed in the
[research archive](docs/research_archive.md). Running experiments are unchanged.
