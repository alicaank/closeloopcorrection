# Failure taxonomy → controller v2 (2026-11-01)

> **2026-11-03 correction:** Sections A–D below are historical proposals, not a
> validated final method. The [new legacy failure audit](../../camera-closed-loop/docs/legacy_rejoin_failures_20261103.md)
> reconstructs DreamX's temporary measured chunk anchors: all ten historical
> runs have zero adjacent intended native translation reversals, versus 20–24
> from differencing persistent absolute labels. Thus F2's DreamX reversal claim,
> the “B removed 24 reversals” interpretation, and E/H1's attribution of a DreamX
> reversal limit cycle are unsupported by that diagnostic. Large native steps
> remain (e43058cb maximum 23.43× the corresponding planned step). Passive path
> ratios do not establish the inverse gain proposed in A/B. VMem's published
> improvement combines rejoin with measured memory; command-only rejoin worsens
> mean position by 93.7% in the same ten-scene archive. The taxonomy mixes observed
> symptoms with causal hypotheses. Keep the raw evidence, but use the new audit
> for current interpretation and Sections E/F for identifiability qualifications.
> Final method selection is open; no per-model pre-calibration or requirement
> that every scene improve is adopted from the historical proposal below.

Evidence sources: per-frame VIPE v3 errors (runtime_repairs_20261031/revision2), failure_audit_20261031/{temporal,video_diagnostics}.json (path ratios, reversals, online-vs-offline disagreement, seams), adapter code (spatia_fprime.py, guarded_observer.py), adaptive.py config. Per-bin error curves are in the session log; key numbers quoted below.

## A. Six failure mechanisms (measured, not suspected)

| # | Mechanism | Where it shows | Evidence |
|---|---|---|---|
| F1 | **Systematic under/over-execution** of the command, model-specific | DreamX, LingBot, VMem, Lyra (pre-calibration) | Baseline path ratio (measured/target travel): DreamX 0.34–0.74; LingBot 2–11; VMem 1.1–8.9; Lyra 1.2–1.7; Spatia/MG ≈1.0; SANA 0.9–1.6. Rejoin with gain 1 then asks for a residual the model executes at the same wrong ratio → repeated large corrections. |
| F2 | **Delay + short return horizon → reversal every boundary** | DreamX (3-frame observation delay, 12-frame rejoin), VMem (4-frame return) | DreamX corrected arms: 20–24 direction reversals vs 0 in target, command speed 2–6× nominal (max 37×); VMem 11–18 reversals vs 1, speed max 24×. Mean position still improves because F1 is so large, but LPIPS/rotation regress on the worst scenes. |
| F3 | **Command shock through native normalization** | LingBot (relative step / max planned step) | First corrected input norms 2–15; collapse to noise at ≤ 2 chunks after a >4 input. Bounded B stops collapse but cannot catch up (see F1 + adaptive config). |
| F4 | **Observer drift / blindness** (DA3 anchor-registered window vs full-sequence VIPE) | SANA 0f12b97e, LingBot e43058cb/a4e9bdb6, Lyra 0238de14, Spatia 02267acf | Online–offline position disagreement grows with distance from anchor: SANA 0f12b97e 0.003→0.147 by frame 144 (online says 0.076, VIPE 0.204 → controller thinks it is fine); LingBot e43058cb online stays 0.4 while VIPE goes to 1.2 (observer blind to collapse); Lyra 0238de14 0.003→0.031; LingBot a4e9bdb6 0.002→0.088. |
| F5 | **Feedback on an already-accurate rollout injects observer noise** | Matrix-Game (both cohorts), Lyra 61b27b61 | MG baseline errors 0.01–0.07 with path ratio ≈1; correction = coin flip (3/10, 4/7 worse). Lyra garden: baseline 0.039 end-bin → correction 0.114, rotation 0.15→0.68°; VIPE registration itself marginal there. |
| F6 | **Memory-frame inconsistency for geometric-memory models** | Spatia (MapAnything cloud labelled with *commanded* poses), Memorizon/WorldPlay (measured relabel ↑ reprojection) | Spatia replan_commanded a126efe8: path ratio 4.06 (oscillates around a location), error 0.022→0.281 vs baseline 0.079; seams at +6 frames after every continuation in BOTH arms (native, frame change 40–45 gray levels vs median 2). WorldPlay replan_measured: reprojection +24 %. |

Cross-cutting: pose scores after visual collapse are not evidence (LingBot historical); seeds flip per-scene signs (see ASSESSMENT.md §3).

Why the adaptive B/C pilot could not fix F1/F2: `adaptive.py` clips response gains to [0.75, 1.5], uses feedback_gain 0.5, and fits only after 3 accepted, well-excited observations. DreamX needs an inverse gain ≈ 2 from the first chunk. B therefore removed reversals (24→0) but lost tracking (+27–31 % position). The failure is a configuration/prior problem, not evidence against response-aware control.

## B. Controller v2: one loop, six safeguards, each tied to a failure

| Component | Fixes | Mechanism | Cost |
|---|---|---|---|
| **C1 Calibrated feedforward gain** | F1, F2, F3 | Per-model execution-ratio prior g₀ (translation, rotation) from one calibration scene's first 2 chunks; online update per accepted chunk with regularization; clip to a *model-declared* range (e.g. DreamX [1,3], SANA [0.7,1.5]). Command = plan residual-corrected, then scaled by 1/g. Removes most of the feedback burden → smaller, non-reversing corrections. | CPU + 1 calibration run/model |
| **C2 Delay-compensated residual** | F2 | For hosts where the last observable frame precedes the boundary (DreamX 3 frames; any `available_last_frame < boundary`), propagate the measured pose through the already-issued commands × g to the boundary before computing the residual (Smith-predictor-lite). Prevents correcting motion that is already commanded. | CPU |
| **C3 Multi-chunk return with native slew bound** | F2, F3 | Return horizon H = ceil(‖residual‖ / per-chunk budget), budget = planned native envelope × (1 + margin); residual carried, never re-added after clipping; velocity continuity across the boundary. (= B's bounding, but with C1 so it can catch up.) | CPU |
| **C4 Innovation gate with observer uncertainty** | F4, F5 | σ_obs from repeated-frame disagreement between consecutive windows + registration support (valid-depth coverage, anchor overlap) + scene-loss tripwire (DINO-to-anchor / Laplacian drop). Kalman-style gain K = r²/(r²+k²σ²): sub-noise residuals are ignored (Matrix-Game, Lyra garden), confident residuals pass; freeze on scene loss (SANA/LingBot e43058cb). Report "fraction of boundaries corrected". | CPU |
| **C5 Keyframe-graph observer** | F4 | Replace "anchor + last 4 + recent 12" with anchor + K keyframes spaced by travelled distance + tail; register scale against all keyframes' stored depth (propagated), not the anchor alone; per-DOF confidence (rotation usually reliable when translation is not: SANA 0f12b97e). | observer change, same DA3 cost |
| **C6 Memory-consistent writes, gated** | F6 | For geometric-memory hosts (Spatia cloud, Memorizon bank, VMem), write generated frames at the *measured* pose only when C4 confidence is high, otherwise keep the commanded label; bridge future commands from the label actually stored. Seam-aware timing for Spatia: exclude the first 8 frames after each continuation from the observer window. | adapter flags exist (write_measured / replan_measured) |

Claim shape for the paper: *a servo loop that (i) calibrates the generator's execution ratio, (ii) corrects only residuals it can measure above its own noise, and (iii) spreads recovery within the generator's native command envelope*. Each safeguard is ablated on the failure panel below; the main 50-scene table runs the full v2.

## C. Failure panel (pre-registered, before the 50-scene runs)

12 model×scene cases that currently fail or trade off, 2 seeds, arms {baseline, sham, v1 rejoin, v2 full, v2 minus one component}:

| Model | Scenes | Targets F | Run cost |
|---|---|---|---|
| DreamX | e43058cb, a4e9bdb6 | F1, F2 (reversals 24→≤2, position ≤ v1) | 6 min |
| VMem | 74310829, d3f042a2 | F2, F6 | ~5 min |
| LingBot | e43058cb, 74310829 | F3, F4 (no collapse AND position < baseline) | 5 min |
| SANA | 0f12b97e, e43058cb | F4 (freeze on scene loss; late-half lag) | 7 min |
| Lyra | 61b27b61 | F5 (v2 ≈ baseline within noise) | 19 min |
| Matrix-Game | 0238de14 (DL3DV), e43058cb (RE10K) | F5 (v2 ≈ baseline) | ? |
| Spatia | a126efe8, 334ff324 | F6 (write_measured+replan_measured, seam-aware) | 49 min |

≈ 12 cases × 2 seeds × ~5 arms ≈ 15 GPU-h (Spatia dominates). Pass rule per case is written before running. Only then freeze v2 and launch the 50-scene cohorts.

CPU-first: C1–C4 can be replayed on the stored observations/commands of the existing runs (reversal counts, slew, fraction corrected, predicted-vs-measured residual) before any GPU time; the replay cannot predict generator response, so it is a sanity filter, not a result.

## D. What not to do

- No per-scene tuning; all thresholds from the calibration scene per model.
- No raw-scale freezing or fitting gauge to the commanded path (Lyra audit).
- No relabelling of commanded-label caches without rebuilding the memory (WorldPlay).
- No pose claims on collapsed videos; collapse rate stays in the denominator.

## E. Addendum (11-01, revised after review): what the probe calibrations show

Sources: `amplitude_calibration_20261101/results.json`, `paired_response_followup_20261101/decision.json`, `online_calibration_20261101/decision.json`, `observer_audit_20261101/{decision,results}.json`. DreamX, e43058cb (seed 107 unless noted), signed translation probes branched from a bit-identical 6-latent prefix, endpoint frame 29. "PnP" = LK tracks + frozen depth; "VIPE" = independent SLAM. All numbers below are **projections on the probe's command direction** (same statistic for command and response). The prefix-drift column is the p90 of the 3D translation change of the *shared, bit-identical* prefix when it is re-estimated inside each branch's window: it is observer re-estimation drift, not regeneration noise, and it is not a calibrated σ.

| Probe Δ | Commanded (proj.) | PnP (proj.) | VIPE (proj.) | ratio PnP / VIPE | prefix re-estimation drift p90 (3D norm) |
|---|---|---|---|---|---|
| −80 % | −0.0546 | −0.0019 (geometry gate fail, p90 3.9 px) | +0.0079 | 0.03 / −0.14 | 0.0014 |
| −40 % | −0.0272 | +0.0002 | −0.0011 | −0.01 / 0.04 | 0.0007 |
| −20 % | −0.0137 | +0.0003 | −0.0006 | −0.02 / 0.04 | 0.0006 |
| +20 % | +0.0139 | −0.0003 | +0.0002 | −0.02 / 0.01 | 0.0003 |
| +30 % | +0.0208 | −0.0003 | −0.0001 | −0.01 / −0.01 | 0.0005 |
| +40 % | +0.0273 | −0.0001 | −0.0001 | −0.00 / −0.01 | 0.0006 |
| +80 % | +0.0549 | +0.0014 | +0.0041 | 0.03 / 0.07 | 0.0019 |

Fresh +30 % holdout, translation prediction error (anchor units): PnP no-response 0.000533 → calibrated(20 %) 0.000326; VIPE 0.002287 → 0.001226. Both reductions are real in their own observer; they are not below the drift column, and the two observers' fitted gains disagree in sign at 20 % (−0.020 vs +0.028) and 40 % (cos −0.96).

Defensible statements:

1. For |Δ| ≤ 40 % the command-aligned response is ≤ 4 % of the command delta in projection and its sign is inconsistent across amplitudes and observers. At ±80 % it is 3–14 %. **Small probes have not established command-aligned translation authority**; neither useful inverse control nor literally zero response is demonstrated.
2. A 0.057° rotation refit flips the fitted gain sign while every refit stays inside the pixel-residual limit: at these amplitudes translation is not identifiable from the geometry, independent of the generator.
3. The ±80 % geometry rejection (p90 3.9 px vs fixed 3 px) coincides with the only amplitude where the response exceeds the drift column. Whether the limit is an artifact of baseline-dependent reprojection error or a real geometry failure is untested; a parallax-relative limit or a rotation-only-vs-full-fit ratio test is needed before either reading.

Hypotheses (not established), with the discriminating test for each:

- **H1 Dead zone + over-response plant** (small deltas ignored, large jumps over-executed) → relay-like limit cycle = the DreamX/VMem reversals. Test: probes that vary **duration as well as amplitude** (0.5×–2× of the planned chunk displacement, sustained over 1, 2, 3 chunks; the existing probes already scale a whole continuation chunk, so amplitude alone is not new), 2 seeds, both observers.
- **H2 Observer sign instability drives the reversals.** The audit shows sign flips within pixel tolerance; a controller fed alternating-sign residuals reverses with no plant nonlinearity. Test on stored data first: replay the existing DreamX/VMem runs' observations through the sign-stability gate and count how many issued corrections would have been abstained; correlate command reversals with flagged observations.
- **H3 History-label sensitivity.** Shift the stored labels of the last chunk, keep future commands fixed. A motion change shows label sensitivity only; it does not show that measured-label writes improve tracking or preserve geometry.

What stays decided: automatic inverse adaptation remains disabled until corrective authority is identified (Section F), on two development scenes and two seeds.

## F. Deployed method: general and on-the-fly, with calibration split in two

Deployment requirement: no probes, no fitted per-model constants. Only (i) the adapter's declared contract (camera convention, unit contract, memory-label policy *and the rebuild it implements*, observation delay), (ii) the model's own planned commands, (iii) the observer on generated frames. Thresholds are in observer-uncertainty units or the plan's own envelope and are the same for every model.

**Two estimators, kept separate.**

- **Passive motion predictor** (always on): per accepted chunk, regress measured displacement on commanded displacement *with intercept*, Δm ≈ a + g·Δu, per DOF (translation along/across the command, rotation), recursive with forgetting, prior a = 0, g = 1 with large variance. Used for delay compensation and for predicting the next endpoint. It is identifiable from the rollout because chunk displacements are large relative to drift; it says nothing by itself about corrective authority (a history-driven model with Δm ≈ 0.5 s and a responsive one with g = 0.5 give the same a + g·Δu at the planned command).
- **Corrective-authority estimate** (gates inverse adaptation): the slope g with its confidence interval, accepted only when the *naturally occurring* variation of ‖Δu‖ across accepted chunks (planned speed changes, earlier bounded corrections) is informative: Var(Δu) over the window ≥ τ² and the slope's interval excludes 0. Zero and opposite-sign responses are **kept** in this regression; they are evidence of low authority, not outliers. States: `authority_identified(g, ci)`, `authority_unidentified` (insufficient excitation), `authority_low` (interval includes 0 with sufficient excitation). Inverse scaling is applied only in the first state; in the other two the controller issues bounded plan-consistent commands without inverse scaling and reports lag.

**Observation acceptance** is about the output, not the input: a chunk pair is used only if ‖Δm‖ exceeds its own uncertainty (repeated-frame disagreement across windows, registration support) and the rotation-refit sign test passes; rejected and insufficient observations are logged separately and never silently dropped.

**Per boundary k.**

1. Observe → M_k, σ_k; scene-loss tripwire → freeze (no new innovations, prediction continues through issued commands).
2. Update the passive predictor; update the authority regression if the pair is accepted; recompute the authority state.
3. Residual with delay compensation: e_k = P_k − M̂_k, M̂_k = M_k propagated through issued-but-unobserved commands with the passive predictor.
4. Deadband: ‖e_k‖ < κ·σ_k → plan unchanged, residual kept (anti-windup: the stored residual is capped at the recoverable amount under the envelope, so clipped corrections are never re-added).
5. Bounded rate correction: speed multiplier m ∈ [1, m_max] from the plan's own envelope, spread over H chunks. **Catch-up per chunk = ŷ(m) − s**, where ŷ(m) is the passive predictor's motion at the multiplied command and s the plan's advance. With authority identified, ŷ(m) = a + g·m·s and H = ceil(‖e_k‖ / max(ŷ(m_max) − s, 0)); if ŷ(m_max) ≤ s the target is **unreachable under the envelope**: issue m_max, report retained lag, do not extend H. With authority unidentified/low, m is set from the envelope only (no inverse scaling). Bridge to the memory frame, native-encode, slew check; the clipped part is carried under the cap.
6. Memory actuation only through an adapter that actually rebuilds the representation (Spatia: MapAnything rerun with measured extrinsics = existing `write_measured`; VMem: native writer; cached-token models: relabel is a different intervention and needs a reprojection check). `history_labels="measured"` is a declaration, not an actuator.
7. Report: fraction of boundaries corrected / abstained / frozen, authority state trajectory with g and its interval, retained lag, collapse flag.

Generality: per-model inputs are only the declarations the adapter already makes; κ, τ, m_max (as a multiple of the plan's maximum native step), the output-uncertainty rule and the sign-stability test are fixed across models and reported.

**Nearest prior work to compare against explicitly**, before any "none report it" sentence: CamVerse 2512.02870 (segment-level geometric reward, training-time), PartiCam 2609.39504 (inference-time reward-guided particle selection on camera scores: the closest inference-time baseline, to be run compute-matched), WorldProbe 2610.39721 (closed-loop agent, offline ViPE). The specific claim to defend is the frozen-history differential-response protocol and the authority-gated controller, not inference-time feedback in general.

## G. Measured-memory implementation (11-01)

The new, separate runtime is `analysis/measured_memory_20261101/launch.py`.
It implements measured-pose consumers for DreamX and LingBot (native clean-context
KV forward), SANA-WM (stage-1 clean-cache replay from the pre-write cache),
Astronex (both clean KV forwards and the native PRoPE mirror), Lyra-2 (stored
camera plus Sparse3DCache world-point reconstruction), Matrix-Game (clean context
plus source-frustum registration), and WorldWarp (source poses before splat
construction). Existing experimental drivers and their historical results remain
separate. The deployed route uses generated rollout frames; no probe is issued.

The common write contract rejects noncausal IDs, unqualified observations,
invalid rigid transforms, and a changed initial anchor. Identity writes keep the
existing values exactly. Absolute-coordinate correction bridges use the boundary
label actually committed, including the commanded fallback on rejection.
`history_labels="measured"` alone still does not perform any of these operations.

Engineering evidence and current per-model GPU status are recorded in
`analysis/measured_memory_20261101/validation_summary.json`; usage and native
consumer details are in `camera-closed-loop/docs/measured_memory.md`. DreamX's
initial test passed exact commanded/identity RGB, latent, RNG and cache receipts;
its measured-cache intervention preserved the first 21 RGB frames and changed
only the continuation. LingBot similarly preserved its first 13 RGB frames.
These are actuator/implementation checks, **not** tracking improvement or a
validated collapse/reprojection gate. Section F's full uncertainty/authority
controller and the paper-level generalization claim are not established by these
checks. Automatic inverse adaptation remains disabled.
