# SANA legacy-study host allocator retry

The matched SANA baseline stopped at the host-memory guard during its third
observation. It saved **73/121 RGB frames**; those 73 frames exactly match the
prefix of the preserved 97-frame baseline from the earlier memory study. This is
partial equivalence, not full-run completion or efficacy evidence.

The failed queue's recorded minimum host availability is 1,768,165,376 bytes.
Its peak aggregate process-tree RSS is 68,934,873,088 bytes; this sum can count
shared mappings more than once. Two observer placement cycles completed. The
third logged entry into observation but no restoration before termination.
Unlike the earlier transition-order failure, this run already moves stage-1
weights before decoder CPU offload. Reapplying that order is not a new fix.

## Targeted hypothesis and implementation

The stage-1 placement records identify approximately 5.3 GB of model tensors,
but the observed gain in host availability after moving them to GPU is much
smaller. This motivates testing retained allocator pages. It does not prove a
leak: live refiner/history storage, temporary allocations and other processes
also affect host memory, and the failed run did not capture an allocation stack.

The new `CAMERA_HOST_ALLOCATOR_TRIM=1` option runs Python garbage collection and
glibc `malloc_trim(0)` after stage-1 CPU-to-GPU movement and around CPU restoration.
It requests release of already freed allocations. It does not evict live tensor
storage, shorten KV history, change dtype, or modify model operations. The option
is off by default. On an unavailable allocator API it reports no trim success;
it does not substitute another memory policy. Residency records now include
process RSS and before/after-trim host availability.

The queued retry uses a separate frozen source copy; existing attempts and
currently waiting/running experiments are unchanged. The scientific cases,
generation arguments, seeds, preprocessing, horizons, observer acceptance and
legacy/memory policies are preserved. This is a resource hypothesis, not a
validated solution yet.

## Qualification and diagnostics

- All eight original SANA cases retain 121 RGB frames and their original B/S/L/M
  contracts. The retry has 59 jobs, adding one mandatory prefix gate.
- The first B arm must be generated fresh. It must complete all 121 frames and
  exactly match both the preserved 73-frame and 97-frame RGB prefixes.
- The original full-horizon B/S identity gate then applies, followed by every
  original B/L and B/L/M prefix/intrinsics gate. No score substitutes a shorter
  clip or drops a failed case.
- The 4 GiB launch and 2 GiB runtime disk/host reserves remain unchanged.
- On a resource stop, the new runner records per-process RSS/PSS/USS and asks
  the registered generator child for a Python stack. The previous runner only
  signalled a directly launched `launch.py`; this study wraps it in
  `ensure_arm.py`, so that check missed the generator child.

All **155 repository CPU tests pass**. The added checks exercise failure-safe
placement/trim ordering and a real allocator call while preserving a live CPU
tensor's storage/value and Torch RNG. The cloned native overlay, runner and
prefix gate compile; source hashes, all eight case contracts and generation
arguments are verified. CPU tests do not establish reclaimed RAM, GPU parity,
full completion or improved tracking.

The queue PID was confirmed live waiting for the existing shared GPU locks.
Workspace: `analysis/legacy_rejoin_matched_20261103/sana_host_trim/`.
Manifest SHA256:
`e083981bbe00a98147e0532260c35a1c73e5ffc58f18f4a9786f90848051c1db`.

[Failure and prefix evidence](artifacts/sana_host_allocator_retry_20261103/failure_audit.json) ·
[Protocol](artifacts/sana_host_allocator_retry_20261103/protocol.json) ·
[CPU validation](artifacts/sana_host_allocator_retry_20261103/cpu_validation.json) ·
[Repository test log](artifacts/sana_host_allocator_retry_20261103/repository_tests.log) ·
[Retry preparation](artifacts/sana_host_allocator_retry_20261103/prepare.py) ·
[Resource runner](artifacts/sana_host_allocator_retry_20261103/run_queue.py) ·
[Prefix gate](artifacts/sana_host_allocator_retry_20261103/prefix_gate.py).
