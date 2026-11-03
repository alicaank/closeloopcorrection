# Exact artifact deduplication: 3.23 GiB recovered

The broader analysis-artifact scan found 287 groups of byte-identical NPZ/MP4/MKV
files. Replacing 416 duplicate paths with hardlinks released 3,468,873,728
allocated bytes (3.23 GiB). Free space rose from approximately 4.16 to 7.39 GiB
during execution. Concurrent experiments can subsequently change free space.

Every original path remains present and every candidate's SHA256 was verified
again after execution. Only regular files under `analysis/` were eligible;
symlinks, open file handles and device/permission mismatches were excluded.
No scene, checkpoint, environment or model dependency was deleted. The protected
6k Gaussian-only and 8k+AHR checkpoint trees were outside the scan and mutation
scope. This is separate from the October 9 deletion manifest.

The duplicate copies now share an inode. Future consumers must continue treating
these archived artifacts as immutable, or replace a path atomically when writing
a new file; an in-place write would affect every hardlink. Path names and content
hashes are preserved, while independent inode identity is intentionally removed.

[Exact hash manifest](artifacts/storage_exact_duplicates_20261103/hash_candidates.json) ·
[Execution receipt](artifacts/storage_exact_duplicates_20261103/receipt.json) ·
[Execution script](artifacts/storage_exact_duplicates_20261103/apply.py).

Resource guards remain unchanged: 4 GiB before launching queued jobs and 2 GiB
during execution. No live generation was restarted or shortened for this cleanup.
