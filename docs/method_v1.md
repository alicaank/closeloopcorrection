# Frozen method v1: causal measured-memory correction

Method ID: **`measured-memory-v1`**. Status: algorithm frozen for expanded
development evaluation; not a validated final paper method. This version fixes
the following decisions before inspecting the expanded cohort's outcomes.

**2026-11-02 evidence update:** the [matched mechanism audit](control_mechanism_audit_20261103.md)
finds severe measured-memory continuation collapse in both LingBot seeds on
`02267acf`, also visible in the earlier 77-frame pair. Their lower position
scores do not establish successful correction. The frozen policy is preserved;
expanded qualification is held. Input-range or visual guards would be a new
version requiring matched reruns, not retrospective edits to this protocol.

The subsequent [cross-model component study](memory_components_20261102.md)
isolates the translation component in the selected LingBot failure and finds
different responses in DreamX and SANA. It also reviews all 18 completed
expanded pairs against reference RGB. SANA can show lower pose error together
with backward estimated progress; unchanged zero-write pairs are abstentions,
not active-control successes. The amplitude-controlled diagnostic arm is not
a physically calibrated memory write and does not modify this frozen method.

## Primary algorithm

The external target trajectory remains fixed. At each supported native memory
boundary, observe already generated RGB with DA3, register poses through the
fixed anchor depth, and require the same observer and SE(3) acceptance rules.
For an accepted changed measurement, recompute the native pose-dependent
history representation and use the label actually committed by that operation.
Otherwise retain existing native memory. Where the host needs a boundary
coordinate conversion, use `C = L_boundary inverse(M_boundary) D` with the
actual committed label, including rejection fallback.

The intervention is **memory-only with the native baseline target policy**.
Future-command residual correction, adaptive B/C gains, inverse response
learning, response probes, and proposed uncertainty/authority extensions are
not part of v1. They require separately named ablations or a new method version.
Native coordinate conversion required by memory consistency is not a separate
tracking controller. Generator and observer weights remain fixed.

The same decision rules apply to every included model. Adapter-specific camera
conventions, unit contracts, preprocessing, temporal schedules, and actual
memory consumers remain explicit. This is a shared algorithm with native
writers, not one universal edit to an arbitrary KV tensor.

## Fixed observation and acceptance settings

- DA3 receives causal generated RGB and intrinsics, without target-pose labels.
- Maximum 16 views per estimator call; up to 8 query-tail frames per call.
- Register each observation to its fixed reference/adapter anchor units.
- Depth support: at least 256 valid pixels, coverage at least 0.20,
  p90/p10 depth ratio at most 4.0; finite positive registered scale.
- Require every requested measured pose to be finite and proper SE(3).
- Stage only ordered, unique, generated frame IDs; preserve the initial anchor.
- Apply an accepted changed write in full; no per-model gain or outcome-based
  blending coefficient is fitted.
- Exact identity is a no-op, distinct from observer rejection.

These are the implemented heuristic gates. They are not calibrated σ bounds,
the adaptive pilot's overlap/photometric gate, or a validated collapse detector.
Model-unit calibration is contract-based; it is not motion-response fitting.
The native writer's scope is the audited chunk/source window, not all past
hidden state. A runtime rebuild failure is a failed run, not a claimed rollback.

## What is frozen

The experiment snapshot binds the launcher, memory hooks, staging contract,
observer code, evaluator, native driver sources, selected input files, scene
order, seed order, horizons, and arm order by SHA256. It records native repository
revisions separately. Fixed model/checkpoint asset paths use the existing local
installations; an offline environment prevents implicit new downloads.

Lyra's historical driver hardcodes seed 7. The experiment overlay adds an explicit
seed argument and forwards it to both native inference and its receipt. That is
a documented experiment-interface fix, not a different memory algorithm.

No threshold, model subset, or scene selection changes in response to these
scores. A bug fix that changes execution creates a new snapshot/version and
requires matched-arm reruns. Preserve failed runs and earlier versions.

## Expanded development cohort

Five models: **LingBot, SANA-WM, DreamX, Matrix-Game 3.5, Lyra-2**. Matrix-Game
remains included despite its negative initial result.

Use the same two scenes for every model: **0238de14** and **924228bf**, from
`batch_continuous1009`. These are the first two scene names after sorting by
SHA256 of `measured-memory-v1-expanded-development:` plus the scene name.
Selection uses no outcome score. This existing development batch has prior
experimental use; it is not claimed as an untouched test split.

Seeds: **7 and 107**. Arms: **commanded and measured**, both using the native
baseline target policy. Total: **20 pairs / 40 generation jobs**, each followed
by paired offline VIPE evaluation. All pairs are generated afresh under the
same snapshot; earlier short tests remain separate evidence.

| Model | RGB frames | First potentially affected output frame |
|---|---:|---:|
| LingBot | 29 | 13 |
| SANA | 49 | 25 |
| DreamX | 33 (9 latents; first write after 6) | 21 |
| Matrix-Game | 169 (2 native blocks) | 85 |
| Lyra | 161 (2 native chunks) | 81 |

This increases scene/seed coverage, not long-horizon qualification. Horizons
follow already exercised native schedules. Multi-boundary and longer-horizon
tests follow under a separately frozen protocol rather than changing this one.

## Eligibility and failure accounting

WorldWarp and Astronex remain registered as **qualification pending**, not
discarded because of their scores. WorldWarp's previous pair differs before
the first intervention; Astronex lacks verified raw prefix equality and has
unsupported position registration in its current development pair. They need
an appropriate native-state/prefix control before joining an attribution table.

For an included model, a mismatched pre-intervention RGB prefix holds its
remaining pairs for an implementation audit. A runtime failure is logged; other
models continue. Rejected evaluator scale registration leaves position metrics
unavailable; it does not delete the pair or count as improved tracking. An
accepted zero-write run is still part of the deployed policy's outcome.

Report camera mean/endpoint errors after the first intervention at original
timestamps, whole-rollout diagnostics, accepted/rejected writes, scale validity,
prefix equality, completion, and runtime. Position and rotation stay separate.
Summarize paired results within each model and show both seeds for each scene;
do not average raw anchor-unit errors across different models/horizons as one
headline. Two scenes do not support a strong significance/generalization claim.
This queue has no complete visual-quality or collapse evaluation, so camera
improvement alone does not establish overall quality improvement.

## Follow-up ablations, after this freeze

The common-prefix mechanism study should compare unchanged memory, identical
label replay, bridge-equivalent future commands, and per-frame measured memory.
It must distinguish history repair from a changed effective future request.
Command-only and combined memory-plus-command controllers are separate arms.
Their equations, budgets, and thresholds must be fixed before comparison.

See [the full method reference](paper_method_reference.md) for derivations and
limitations, and [the frozen protocol](method_v1_protocol.json) for exact
settings and qualification status. Local execution and artifact locations are
recorded by the experiment receipt rather than modifying historical results.

The follow-up [repeated-memory protocol](method_v1_multiboundary_protocol.json)
keeps this memory policy and observer thresholds while adding two prespecified
scenes and longer native schedules. It includes exact-input evaluation reuse
and verified lossless RGB retention. Its 40 pairs are a separate development
cohort; the original 20-pair results are not silently relabelled as repeated-
write evidence. See [audit corrections](audit_fixes_20261102.md).
