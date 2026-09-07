# Maintenance verification — 2026-09-07

## Problem and implemented behavior

The previous normalizer verified bandwidth units and numerical values, but did not tie an otherwise plausible result to the exact experiment requested. A saved result with a different reduction, dtype, warmup, missing per-iteration data, duplicate result, missing aggregate correctness or wrong executable could pass. The new `validate_contract` checks pinned JSON v4, argv/environment, logical and physical placement, requested bytes/iterations/warmup, float/count/reduction semantics, full iteration arrays and final diagnostics before a sweep accepts rows. Known numerical failures remain numerical failures; unknown evidence is rejected.

`audit.py` reconstructs a completed sweep, validates its native/process/config relationships, enforces the full unique experiment matrix, and recomputes stored rows and summary statistics. It operates on saved data, so GPU-free CI can reproduce metric derivation. Export receipts separately check bytes. This is not authentication of an untrusted producer and it does not rerun CUDA or validate physical links.

## Actual validation

- Before implementation: 11 existing Python tests passed. Current suite: 19 tests, including eight contract/replay test methods.
- A controlled CPU mutation check uses saved real GPU JSON: seven invalid variants passed the old gate at commit `3fcafdf` and are rejected by the new gate. This is a parser regression experiment, not seven hardware faults.
- Both the historical and fresh 30-process single-GPU matrices replay successfully (2 collectives × 5 sizes × 3 repeats; 60 placement records per matrix). The fresh execution used the same pinned NCCL/nccl-tests binaries, CUDA 13.2 and a real RTX 4090. Native validation reports zero incorrect elements.
- The two-GPU configuration produces `blocked_preflight`, exit 2, on this one-GPU host. No network bandwidth or performance improvement is claimed.

Commands and source hashes are in [verification.json](../evidence/maintenance-20260907/verification.json); raw data and original/exported hashes are in [the maintenance snapshot](../evidence/maintenance-20260907/). The final CPU verification records any later admission-only source changes separately from the GPU execution source snapshot.

```bash
python3 -m unittest discover -s tests -v
python3 audit.py evidence/snapshot/final-single-gpu
python3 audit.py evidence/maintenance-20260907/single-gpu
python3 scripts/export_evidence.py --verify evidence/maintenance-20260907
```

## Tradeoffs and remaining work

The adapter intentionally supports the pinned single-process, float, non-graph experiment; different upstream schema, MPI, dtype or graph modes need explicit support and tests. No parser-time performance improvement is asserted. Process output is still buffered in memory; process-group cleanup does not establish adversarial process isolation or host-crash durability.

Two-GPU placement/hypothesis experiments, faults on real communication, and physical-link interpretation remain open. Keep `qualification=false`. Single-rank bus bandwidth is zero by definition and must never become a networking achievement on a resume.
