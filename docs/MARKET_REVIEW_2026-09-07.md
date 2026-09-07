# Hiring and ecosystem review — 2026-09-07

This is a small purposive sample of official North American postings, not a survey of hiring frequency. Requirements inform engineering priorities; they do not prove candidate eligibility or sponsorship. Publication dates and observation dates are distinct.

| Official role | Date evidence | Relevant signal and fit |
|---|---|---|
| [Stripe — Software Engineer, New Grad](https://stripe.com/careers/listing/software-engineer-new-grad/8128744) | 2026-09-01T17:32:10.188Z | Programming fundamentals, learning unfamiliar systems, collaborative development. US locations including Seattle; degree by summer 2027 and at most 18 months professional experience. |
| [CLEAR — Software Engineer, Infrastructure](https://job-boards.greenhouse.io/clear/jobs/7901600) | 2026-05-06T16:05:17-04:00; updated Aug 17 ([ATS metadata](https://boards-api.greenhouse.io/v1/boards/clear/jobs/7901600)) | Reliability, Python, AWS networking, Kubernetes and observability. New York; explicitly 0–2 years. |
| [Amazon — Software Development Engineer - 2026 (US)](https://www.amazon.jobs/en/jobs/3177934/software-development-engineer-2026-us) | Absolute posting date unavailable; checked Sep 7 | Fault tolerance, CI/CD, operational troubleshooting and maintainable code. SDE-I; project experience and current degree accepted in listed qualifications; exact start window needs confirmation. |
| [NVIDIA — AI and ML Infra Software Engineer, GPU Clusters - New College Grad 2026 (JR2021591)](https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/US-CA-Santa-Clara/AI-and-ML-Infra-Software-Engineer--GPU-Clusters---New-College-Grad-2026_JR2021591) | Absolute posting date unavailable; checked Sep 7 | HPC/GPU infrastructure, high-speed networks, distributed-training optimization and Python/Go/Bash. Santa Clara/Redmond; recent MS/PhD or equivalent; specialized stretch role, not generic junior baseline. |

Amazon's separate CloudWatch requisition [10509638](https://www.amazon.jobs/en-gb/jobs/10509638/software-development-engineer-cloudwatch-application-observability) was excluded from the early-career comparison after full-page review revealed a 3+ year requirement. Search snippets alone were misleading. NVIDIA's direct ATS page was dynamic; its indexed official text supplied the reviewed requirements. Neither undated page is claimed to have been posted within six months.

Reliability and troubleshooting recur in the Amazon/CLEAR sample. Explicit AWS networking tooling belongs to CLEAR; HPC fabrics and distributed training belong to NVIDIA. Stripe supports fundamentals and system comprehension. AI wording in advertisements is not evidence that adding an agent framework improves this communication project.

Portfolio comparison found existing coverage of backend APIs, durable queues, Kubernetes/observability and general experiment supervisors. The useful addition here is communication-specific measurement admission and uncertainty-aware experiment analysis. No web service, queue, Kubernetes cluster or LLM wrapper was added.

## Upstream decisions

All four upstream repositories were unarchived and had recent pushes when queried via the official GitHub API on Sep 7. Recent push dates establish activity, not support guarantees.

| Component | Stable release observed | Existing pin and decision |
|---|---|---|
| [NCCL](https://github.com/NVIDIA/nccl/releases/tag/v2.31.2-1) | v2.31.2-1, Aug 11, 2026 | Existing `fd168324` is 30 commits ahead of the tag, not the release commit. Retain the exact experimentally built pin; no new native feature is needed for this Python contract fix. |
| [nccl-tests](https://github.com/NVIDIA/nccl-tests) | No GitHub latest release (API 404); pushed Aug 28 | Retain `b4d5beeb`; the native JSON v4 contract is pinned and will deliberately reject incompatible schemas. |
| [MSCCL++](https://github.com/microsoft/mscclpp/releases/tag/v0.10.0) | v0.10.0, Jul 22, 2026 | Existing `de2e4863` is 26 commits ahead of the release, not a stable release qualification. Retain for reproducibility; changing the transport baseline before two-GPU validation would add confounding cost. |
| [UCX](https://github.com/openucx/ucx/releases/tag/v1.22.0) | v1.22.0, Aug 3, 2026 | Leave issue-specific `bade1836` unchanged while the investigation is paused. New fault-recovery APIs do not establish that local candidate patches are qualified. |

MSCCL++ [migration #822](https://github.com/microsoft/mscclpp/pull/822), merged Jun 25, changes C++17 to C++20 and drops CUDA 11. The local extension already uses C++20/CUDA 13.2. NCCL's new CFT needs Blackwell/CUDA 13.3; it is irrelevant to this RTX 4090 host. A future migration must rebuild, repeat correctness tests and validate two-GPU behavior; a moving branch is never substituted for a pinned commit.

No dependency upgrade was necessary. The existing post-release pins remain a reproducibility tradeoff, not an assertion that development snapshots are more suitable for production. [NVIDIA's benchmark definitions](https://github.com/NVIDIA/nccl-tests/blob/b4d5beebca8a76cf01335f724d154b9b9d394d96/doc/PERFORMANCE.md) remain the source for collective bandwidth normalization.

The C build also pins [nlohmann/json v3.12.0](https://github.com/nlohmann/json/releases/tag/v3.12.0), published Apr 11, 2025 and still the GitHub latest release on this review date. No JSON-library upgrade is needed.
