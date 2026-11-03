# Second disk-hold resume — 2026-11-03

The first resumed queues stopped at the unchanged 4 GiB launch reserve. All ten
processes were confirmed absent. MatrixGame had completed the final case's
commanded-memory generation, but had not started its scheduled lossless archival
step. This is a resource hold, not a failed model comparison.

An additional **472,322,682 bytes (450.4 MiB)** of identical archived RGB NPZ
allocations were hardlinked. Every path and SHA256 remains unchanged. Candidates
were restricted to explicitly listed historical RGB-output directories; open
files and non-private replacement inodes were rejected. Protected model trees'
file metadata remained unchanged. No dataset scene or model dependency was
deleted. [Plan](artifacts/queue_disk_resume_v2_20261103/deduplication_plan.json),
[execution receipt](artifacts/queue_disk_resume_v2_20261103/deduplication.json),
and [implementation](artifacts/queue_disk_resume_v2_20261103/deduplicate.py).

A subsequent exact-byte scan of historical output videos recovered another
**489,926,726 bytes (467.2 MiB)** across 19 groups, replacing 20 duplicate
allocations with hardlinks. Every MP4 path and SHA256 was preserved, and protected
tree metadata was unchanged. Model-repository assets were excluded. FFV1 files
whose container bytes differed were also left unchanged, even where their decoded
RGB had previously matched. Free space rose to about **5.1 GiB** while the final
MatrixGame generation remained active. [Video deduplication receipt](artifacts/storage_video_dedup_20261103/video_deduplication.json)
and [exact plan](artifacts/storage_video_dedup_20261103/video_deduplication_plan.json).

After this recovery, the existing MatrixGame archival job completed and verified
**337 × 704 × 1280 RGB frames, pixel-exact**, before replacing its bulky owned
NPZ with FFV1 storage. The remaining measured-memory generation was then launched.
The other nine queue processes were verified alive, waiting on the shared GPU
lock. At the recorded verification, free space was about **4.65 GiB**. These are
dated process observations, not completed efficacy results.

## Protocol and provenance

Each second-resume runner is derived from the original runner, preserving all
job arguments, source/runtime paths, environment, seeds, horizons, output paths,
five GPU locks, source verification, 4 GiB launch reserve, and 2 GiB runtime
disk/host-memory reserves. Only attempt status/log destinations and skipping
successful jobs change. No original or first-resume status/log is overwritten.

Preparation verified **18,341 frozen source/input entries** across ten queues,
the original manifests and CPU qualification records, both terminal prior states,
and every skipped job's arguments and completion log. MatrixGame skips 44 stages
completed across the first two attempts; five stages remain, beginning with
archiving the already-generated baseline. Other queues skip zero jobs.
At runtime the new runners verify both previous attempts' immutable state and
provenance hashes, as well as the original frozen sources.

Coverage joins job records from all three attempts in memory, preserving both
earlier holds. Validation against the real MatrixGame attempt also rejects three
deliberately corrupted in-memory inputs: a repeated completed job, an altered
skip list, and a changed previous-state hash. Runtime execution does not consume
this reporting merge. The 40-case cohort and earlier failures remain intact.

- [Preparation/source verification](artifacts/queue_disk_resume_v2_20261103/preparation.json)
- [Live process verification](artifacts/queue_disk_resume_v2_20261103/live_verification.json)
- [Coverage validation](artifacts/queue_disk_resume_v2_20261103/coverage_validation.json)
- [Preparation script](artifacts/queue_disk_resume_v2_20261103/prepare.py)
- [Coverage checks](artifacts/queue_disk_resume_v2_20261103/check_coverage.py)
- [Pixel-exact baseline retention](artifacts/queue_disk_resume_v2_20261103/matrix_baseline_retention.json)
- [Artifact manifest](artifacts/queue_disk_resume_v2_20261103/manifest.json)

Authoritative live records are under
`analysis/queue_disk_resume_v2_20261103/<attempt>/`; repository copies are dated
snapshots. Storage remains limited and the unchanged guards may hold later jobs.
The earlier [first-resume report](queue_disk_resume_20261103.md) remains historical
evidence of that attempt, not the current queue state.
