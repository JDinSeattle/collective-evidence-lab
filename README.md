# Collective Evidence Lab

A reproducible qualification harness around **NVIDIA nccl-tests**, built to answer what a collective benchmark actually proves. It keeps hardware, placement, correctness, timings and failure evidence together.

**Status:** CPU harness tests and real RTX 4090 single-GPU checks passed. **Two-GPU qualification is pending.** No inter-GPU bandwidth or multi-node claim is made.

## What is implemented

- Pinned NCCL/nccl-tests builds; no replacement collective implementation.
- GPU/CPU topology manifest, compiler/driver versions, CPU affinity and binary hashes.
- CUDA memory validation and directional peer-copy preflight before a sweep.
- All-reduce and all-gather; in-place/out-of-place; multiple message sizes and independent process repeats.
- Default baseline, reversed GPU ordering and buffer-registration cases. Configuration rejects changes to multiple factors at once.
- Native JSON and per-iteration CUDA-event samples, with explicit missing-rank/transport fields.
- Separate launch, runtime, timeout, numerical and evidence-format outcomes; failed correctness stops the sweep.
- Whole-process-group timeout cleanup, including descendants; no unbounded collective jobs.

## Verified here

| Evidence | Result |
|---|---|
| Python regression suite | 11 tests passed, including real subprocess exit/timeout/descendant cleanup |
| Native collective smoke | 30 process runs: 2 operations × 5 sizes × 3 repeats; both placements; correctness passed |
| Single-GPU memory preflight | 128 MiB CUDA copy and full-element validation passed |
| Two-GPU gate on one-GPU host | Explicit `blocked_preflight`, exit 2 |
| Physical link/placement hypothesis | Pending a two-GPU host |

See [the experiment report](docs/REPORT.md), [methodology](docs/METHODOLOGY.md), [original scope](PROJECT_SPEC.md), and [sanitized evidence](evidence/snapshot/). Historical failures are retained alongside the final results.

## Reproduce

Python 3.10+, Git, Make, a C++ compiler and a CUDA toolkit are required. The recorded build used CUDA 13.2, NCCL 2.31.2 and `sm_89`. Supply the architecture of your actual GPU; other toolchain combinations have not been qualified here.

```bash
python3 -m unittest discover -s tests -v
python3 scripts/build.py --arch 89
```

The builder prints `bin_dir` and `library_dir`, and stores them in `evidence/local/build.json`. It uses `/tmp` because upstream Makefiles do not support spaces in build paths.

```bash
python3 lab.py run --config configs/two-gpu.json \
  --bin-dir /path/printed/by/builder/nccl-tests/build \
  --library-dir /path/printed/by/builder/nccl/build/lib \
  --preflight build/preflight --out evidence/local/two-gpu-run
```

For a deliberately limited single-GPU check, use `configs/single-gpu-smoke.json` and add `--single-gpu-smoke`. The evidence remains labelled `single_gpu_smoke`; its qualification flag is false.

```bash
python3 scripts/export_evidence.py --out evidence/new-snapshot
python3 scripts/export_evidence.py --verify evidence/new-snapshot
```

Exports redact local identity/path details and include original/exported SHA-256 receipts. Environment passed to nccl-tests is explicitly limited, because its native JSON captures the process environment.

## Why the measurement rules matter

Native `time` is in **microseconds**. `algbw = S/t`; all-reduce `busbw = algbw × 2(n−1)/n`, and all-gather uses `(n−1)/n`, with `S` the total gathered output size. These are operation-normalized metrics, not physical-link counters. On one GPU both bus factors are zero, and in-place operations may do little work.

The first local experiment caught an upstream all-gather edge: an 8-byte requested message rounded to **zero actual bytes**. The harness rejected it, and now validates the per-rank 16-byte alignment requirement before launch. See [NVIDIA's definitions](https://github.com/NVIDIA/nccl-tests/blob/b4d5beebca8a76cf01335f724d154b9b9d394d96/doc/PERFORMANCE.md) and the pinned [`all_gather.cu`](https://github.com/NVIDIA/nccl-tests/blob/b4d5beebca8a76cf01335f724d154b9b9d394d96/src/all_gather.cu).

## Role relevance

Useful evidence for communication-runtime validation and GPU performance engineering: benchmark contracts, data-volume reasoning, experimental controls, failure containment and evidence provenance. [Interview notes](docs/INTERVIEW.md) distinguish implemented work from the hardware experiments still required.
