# DreamX future-command pilot: issuance replay and observer disagreement

The two completed moving-scene F runs reproduce their saved issued commands
exactly when replayed through the frozen fractional-recovery implementation.
All six active boundaries request and geometrically serve half the online
position residual, to floating-point precision. This is a command-space result,
not a claim that the generator executes half the correction.

The nominal 2× speed budget clips no proposed RGB-frame increments in either
run. Slew limiting affects two frames in seed 7 and four in seed 107, but the
remaining increments serve the requested endpoint offset within each chunk.
Maximum proposed speed / maximum planned speed is 0.94 for seed 7 and 1.67 for
seed 107. Raising the speed budget would not remove an active speed constraint
in these cases. This replay covers float64 issuance and bridging; the separate
native-input receipts establish changed encoded chunks, not exact bf16 fidelity.

## Same-frame comparison of the observers

DA3 supplies online feedback using causal windows. VIPE supplies the independent
offline evaluation using the full 69-frame video. Both retain their existing
initial-anchor registration; this audit applies no additional trajectory or
scale fit. We compare their F-arm positions at the latest available RGB frame,
and the directions from each estimated position toward the same target.

| Seed | Available RGB frame | Position disagreement | Residual-direction cosine |
| --- | ---: | ---: | ---: |
| 7 | 32 | 0.006598 | 0.336 |
| 7 | 44 | 0.012002 | 0.179 |
| 7 | 56 | 0.014133 | 0.474 |
| 107 | 32 | 0.002042 | 0.994 |
| 107 | 44 | 0.004774 | 0.970 |
| 107 | 56 | 0.004309 | 0.999 |

Position values use the fixed anchor-depth units, not verified metres. These
are observer differences, not errors against ground truth. Causal context,
offline optimization, geometry predictions and registration can all contribute.
They cannot identify which estimate is more accurate. The first nominal
boundary at frame 20 is also retained in the machine-readable audit.

The seed-7 directions disagree substantially even though both observers return
valid geometry under their existing gates. Seed 107 has much stronger directional
agreement. Both have modest descriptive improvements in the offline position
metric, so neither disagreement nor agreement alone explains the effect size.
The six active observations are correlated within two rollouts, not six
independent trials. Do not use the offline result to change already issued
commands, retune the frozen cases or retroactively reject inconvenient scores.

## What this resolves and what remains open

- The weak benefit is not explained by a failed float64 issuance path or an
  active speed cap in these two moving-scene cases.
- A 0.5 requested recovery fraction is neither an execution gain nor a guarantee
  of a 50% reduction in observed tracking error.
- Passive execution ratios remain descriptive. The recorded F intervals range
  from 0.92–0.98 for seed 7 and 0.59–0.85 for seed 107; these are not differential
  command gains. Their changing direction agreement also prevents interpreting
  scalar length ratios alone as camera-control authority.
- Before proposing gain changes from these cases, compare independent observers
  using matched causal RGB windows and inspect scale/pose stability across
  overlapping windows. That follow-up is not executed by this CPU audit.
- Continue the frozen near-stationary cases and matched multi-model legacy
  study. Two seeds on one development scene do not settle method selection.

[Replay and comparison data](artifacts/dreamx_future_control_audit_20261103/audit.json)
and [audit source](artifacts/dreamx_future_control_audit_20261103/audit.py)
record all inputs by SHA256. The controller sources are checked against the
frozen pilot manifest before replay. Runtime files, inputs and scores are unchanged.
