# Ablations retained for interpreting the Method

These comparisons answer concrete questions. They are not additional default
methods, and a CPU test does not establish generation benefit.

## Commands versus memory: VMem

Ten scenes, one seed, four historical arms:

| Intervention | Mean position error | Change from baseline |
| --- | ---: | ---: |
| Baseline | 0.67215 | — |
| Commands only | 1.30199 | +93.7% |
| Measured memory only | 0.52000 | -22.6% |
| Commands + measured memory | 0.50369 | -25.1% |

Memory materially affects the result. Adding commands to measured memory helps
5/10 cases, with an aggregate reduction of 0.01632 anchor units. Native writes
can also affect reconstruction and retrieval, so this is not a pure label-only
causal claim. Fresh sham qualification remains separate.

## Is bounded correction or gain learning better?

In the separate DL3DV A/B/C pilot, A follows planned commands, B adds bounded
future correction with response gain one, and C adds an observational gain fit.
B lowers mean position by 27.0% for LingBot and 7.0% for SANA, using two valid
position pairs per model. Mean rotation changes +0.9% and -34.2%, respectively,
over four pairs per model. C activates only one rotation component in 1/8 runs;
no translation gain activates. No additional benefit from C is established.
The Method uses a different rejoin rule; these cohorts do not rank it against B.

## Actuator and decoder checks worth retaining

- DreamX future-command pilot: two moving-scene seeds improve mean position by
  6.2% and 7.0%; near-stationary seeds change +6.0% and -0.7%. Four identity
  controls pass. Different online/offline RGB preprocessing remains a confound.
- MatrixGame: eight same-latent exports isolate a decoder-context contribution
  to seams (46.6–91.5% smaller RGB jump). This is not a tracking-error improvement.
- Astronex response pilot: no active correction; all 637 A/S/C frames match.
  Abstention is not evidence of successful gain calibration.

Long experiment diaries, plot galleries, proposed controller variants and queue
receipts are in the [archive](research_archive.md). Their optional runtime
modules remain where adapters or frozen diagnostics depend on them; they are
not promoted into the main Method API by these findings.
