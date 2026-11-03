# Resuming the November 3 disk-held experiments

**Later update:** these attempts subsequently stopped at the disk reserve again.
The [second resume](queue_disk_resume_v2_20261103.md) preserves both earlier
attempts and continues from their successful stages. This page records the first
resume's launch and results.

All ten relevant queues reached a terminal disk hold after free space fell below
their 4 GiB launch reserve. The original processes were absent and the GPU was
idle. MatrixGame had completed the seventh pair's measured generation; its
archiving, score and attribution jobs had not started. The other nine queues
had completed no jobs. This was a resource interruption, not a model result.

Eight byte-identical RGB archive allocations were replaced with hardlinks,
releasing **269,877,248 bytes (257.4 MiB)**. Every path and file byte was retained;
SHA256 was checked before and after each replacement. The changes were restricted
to completed/held November memory-study RGB archives and historical continuous
evaluation RGB previews. No dataset scene, checkpoint, configuration, initializer
or protected-model dependency was removed. The exact plan and receipt are in
[the archived evidence](artifacts/queue_disk_resume_20261103/disk_resume_dedup_receipt.json).

The new MatrixGame attempt then ran the existing verified lossless-archiving
step, recovering additional space. Its seventh score and attribution gate have
completed. The original 337-frame horizon and exact 85-frame shared prefix remain
required. See [the paired result review](matrix_full_horizon_review_20261103.md).

## Resume contracts

Each resumed runner is derived from its original frozen runner. Its only
behavior changes are separate attempt-state/log destinations and skipping jobs
already recorded as successfully completed. Runtime/source paths, environment,
job arguments, output destinations, seeds, horizons, source verification,
five shared GPU locks, 4 GiB launch reserve and 2 GiB runtime disk/host reserves
remain unchanged. The MatrixGame attempt skips 40 completed stages; it does not
regenerate the seventh pair. Other attempts skip no jobs.

Before preparation, the runner verifies that the old process is absent, the
old state is a terminal disk hold, all existing jobs succeeded, their arguments
match the frozen manifest, and all frozen source/input hashes still match.
At runtime it additionally verifies the old state and runner hashes. Old
statuses and logs are never overwritten. Per-attempt diffs, provenance records,
runner hashes and live-process launch checks are archived below.

| Resumed attempt | Original experiment | Initial resume state |
|---|---|---|
| `matrix_memory` | MatrixGame observer-lifetime retry | Seventh pair archive → score → gate; then final pair |
| `legacy_dreamx` | DreamX matched legacy/memory host | Pending original 8 cases |
| `legacy_lingbot` | LingBot matched legacy/memory host | Pending original 8 cases |
| `legacy_sana` | SANA matched legacy/memory host | Pending original 8 cases |
| `legacy_matrixgame` | MatrixGame matched legacy/memory host | Pending original 8 cases |
| `legacy_lyra` | Lyra matched legacy/memory host | Pending original 8 cases |
| `lyra_memory` | Lyra observer-resource qualification | Pending original protocol |
| `sana_transition` | SANA observer-transition qualification | Pending original protocol |
| `dreamx_future` | DreamX future-command experiment | Pending original protocol |
| `lingbot_gauge` | LingBot common-gauge encoder pilot | Pending original protocol |

All ten new queue processes were verified live after launch; MatrixGame held
the shared lock and the other nine waited. This is a dated launch observation,
not a claim that the waiting experiments have produced results. The matched
legacy study retains all 40 cases. No failure-directed replacement cohort or
smaller horizon was substituted.

Coverage accounting now joins successful original stages and new resume stages
in memory, while preserving the old held attempt and its provenance. It rejects
duplicate job execution or changed original manifests/states. The queue runtime
does not consume or modify that reporting merge.

Remaining storage is limited. The existing guards can hold a later stage again;
they were not relaxed to force progress. Terminal holds must be distinguished
from live processes waiting on the lock.

[Preparation and frozen-source verification](artifacts/queue_disk_resume_20261103/preparation.json)
· [live-process verification](artifacts/queue_disk_resume_20261103/live_verification.json)
· [preparation script](artifacts/queue_disk_resume_20261103/prepare.py)
· [artifact manifest](artifacts/queue_disk_resume_20261103/manifest.json).

Workspace attempt records live under
`analysis/queue_disk_resume_20261103/<attempt>/`. Archived status files are
snapshots; the workspace records are authoritative for subsequent progress.
