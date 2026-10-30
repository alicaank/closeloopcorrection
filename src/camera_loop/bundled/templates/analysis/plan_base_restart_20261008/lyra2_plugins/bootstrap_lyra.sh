#!/usr/bin/env bash
# User-approved 2026-10-13: delete 7 directories (SHA256 manifest + receipt), download Lyra-2 model/vae/negative_prompt/DMD LoRA,
# then start the 2-scene Lyra plug-in queue. Protected 6k / 8k+AHR / data are never touched.
set -uo pipefail
D=$(cd "$(dirname "$0")" && pwd); LOG=$D/bootstrap.log; R=@WORKSPACE@/cleanup_20261013_lyra_disk; mkdir -p "$R"
log(){ echo "$(date) $*" >> "$LOG"; }
TARGETS="analysis/eval_stack_20261008 analysis/map_consistency_20261008 analysis/memory_pose_crossover_20260919 analysis/reliability_mask_20261006 analysis/incremental_resplat_20260924 training_runs/final_h_latent_20261008 training_runs/p3_supervised_v2_20260925"
cd @WORKSPACE@
log "manifest start"
for t in $TARGETS; do
  case "$t" in *gaussian_recovery500*|*gaussian_base5000*|*ahr_corrective*|*init_wan22*|data*) log "REFUSED protected $t"; exit 2;; esac
  find "$t" -type f -print0 | xargs -0 sha256sum >> "$R/manifest.sha256"
  du -sb "$t" >> "$R/sizes.txt"
done
log "manifest done: $(wc -l < $R/manifest.sha256) files"
for t in $TARGETS; do rm -rf "$t" && log "deleted $t"; done
printf '{"approved_by_user":"2026-10-13 AskUserQuestion: Analysis dumps + 2 old lineages (57 GiB)","targets":"%s","manifest":"%s/manifest.sha256","sizes":"%s/sizes.txt","executed_utc":"%s","free_after":"%s"}\n' "$TARGETS" "$R" "$R" "$(date -u +%FT%TZ)" "$(df --output=avail -BG / | tail -1 | tr -d ' ')" > "$R/receipt.json"
log "free after delete: $(df --output=avail -BG / | tail -1)"
cd @WORKSPACE@/lyra/Lyra-2
log "download start"
env -u HF_HUB_OFFLINE @CONDA_ROOT@/envs/lyra2/bin/huggingface-cli download nvidia/Lyra-2.0 \
  --include "checkpoints/model/*" "checkpoints/vae/*" "checkpoints/text_encoder/negative_prompt.pt" "checkpoints/lora/dmd_distillation.safetensors" \
  --local-dir . >> "$LOG" 2>&1
rc=$?; log "download rc=$rc free: $(df --output=avail -BG / | tail -1)"
test -f checkpoints/model/model/.metadata || { log "model metadata missing; abort"; exit 3; }
test -L checkpoints/text_encoder/encoder.pth || log "WARNING encoder symlink missing"
log "launch queue"
nohup bash "$D/queue_lyra.sh" > "$D/runs_phase1_nohup.log" 2>&1 &
log "queue pid $!"
