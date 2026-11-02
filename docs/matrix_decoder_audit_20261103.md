# MatrixGame decoder attribution protocol

MatrixGame's current causal host exports the RGB blocks already seen by its
observer. It skips the released final whole-history VAE decode. Earlier paired
videos show a boundary contrast/saturation seam in both commanded and measured
arms, including zero-write cases. That observation cannot establish a defect
in the released model or a failure of measured-memory correction.

## Identical generated latents, different decoder context

The original four MatrixGame development pairs retain their final latent history:
`(1, 48, 43, 44, 80)` in `torch.bfloat16`, covering 169 RGB frames. This enables an
offline decoder ablation without rerunning diffusion or changing the camera
commands. The selected set is all MatrixGame cases from the original two-scene,
two-seed cohort: `0238de14` and `924228bf`, seeds 7 and 107, both arms.

Each of the eight jobs performs these steps:

1. Load only the released `WanVideoVAE38` checkpoint, with the same BF16 precision
   and native spatial tile size `(30, 52)` and stride `(15, 26)`.
2. Decode history slices `[0:22]` and `[21:43]` separately. Concatenate the first
   85 RGB frames with the last 84 RGB frames, discarding the second clean prefix.
   Require **exact equality to the saved causal RGB**. This control checks that
   the saved history and isolated decoder reproduce the actual export.
3. Decode the same full 43-latent history with the released decoder. Compare the
   first 85 previously emitted frames, per-frame RGB differences, and frame
   adjacency/brightness/contrast around the 84→85 seam.
4. Archive the joint decode losslessly for visual inspection. Preserve every
   original score, reference, command, latent and emitted RGB artifact.

The native VAE decoder explicitly clears its temporal cache at each `decode`
call. Separate block calls therefore reset decoder history. That is a concrete
mechanism to test, not proof that it explains the observed seam. Exact control
replay is required before attributing differences to decoder context.

## Interpretation and limits

If control replay succeeds and whole-history decoding reduces the seam, it
isolates a decoder/export contribution while holding generated latents fixed.
If it does not, the seam remains unresolved. If the first 85 frames change,
whole-history decoding also changes previously emitted observations and cannot
silently replace the causal loop. Even if those frames match, a state-preserving
streaming implementation would need separate equivalence and resource checks.
Pixel seams are appearance measures, not camera velocities or pose errors.

This is an offline diagnostic, not an additional camera-control method or an
updated efficacy score. It does not repair the current cohort mid-run. Native
whole-history output and the causal export must be labeled separately in any
paper or webpage comparison.

## Execution status

CPU qualification loaded the actual released VAE (704,688,668 parameters) in
BF16 with no diffusion/text models. All eight stored latent/RGB inputs and native
sources were hashed; scripts compile. **GPU replay equality and decoder results
are pending.** The queue is waiting for the shared GPU locks held by the broader
cohort, alongside the separately queued SANA resource supplement. Queue order is
not assumed from a status file alone; process existence was checked.

Workspace: `analysis/memory_generalization_20261103/matrix_decoder_audit/`.
[Protocol and audit scripts](artifacts/generalization_20261103/matrix_decoder_audit/protocol.json)
and the [dated execution snapshot](generalization_execution_20261103.json)
record the exact scope. This diagnostic adds no diffusion generations and does
not reduce the original 40-pair evaluation denominator.
