# EXL3 Resident Session Experiment

## Question

Can a single-user Qwen3.8-27B session reuse resident NVFP4 KV pages and Gated DeltaNet checkpoints so that appending at most 128 tokens does not replay a 1K–256K prefix? If reuse works, does warmed-turn TTFT stop scaling with total context on the current ExLlamaV3 path?

## Configuration

- GPU: NVIDIA GeForce RTX 5090, compute capability 12.0, 32,607 MiB.
- Driver/CUDA: 610.57.04 / 13.3.
- Target: `/home/rekeyea/models/Qwen3.8-27B-EXL3-3.5bpw`.
- Draft: `/home/rekeyea/models/Qwen3.8-27B-DFlash2-EXL3-5.0bpw`.
- Donor revision: `1242187390f780dba907e30659f1639fd0b491c8`, dirty working tree.
- Target KV: NVFP4.
- Cache capacity: 262,400 tokens, providing one 256-token page of generation headroom above the 262,144-token input bucket.
- GPU split budget: 30 GiB, matching the known-good donor service.
- Sampling: greedy, one generated token.
- Workload: one seed sequence followed by three branches with the exact completed seed prefix and a distinct 128-token suffix.
- Qualification: bring-up only. EXL3 3.5 bpw cannot establish Q38X quality parity.

The direct probe bypasses the OpenAI bridge and reads `Job.cached_pages`, `Job.cached_tokens`, `Job.time_prefill`, and page-table counter deltas after each completed job. Logical usage counters are not used as cache evidence.

## Artifacts

- 1K validation: `results/20260904T192156Z-exl3-resident-probe-1k`.
- Full matrix: `results/20260904T192233Z-exl3-resident-probe-full`.

## Results

| Final context | Cached tokens | Physical prefill | Reuse minimum | Turn TTFT p50 | ExLlama prefill p50 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 768 | 255 | 75.07% | 145.69 ms | 110.63 ms |
| 32,768 | 32,512 | 255 | 99.22% | 295.82 ms | 237.05 ms |
| 131,072 | 130,816 | 255 | 99.81% | 788.04 ms | 655.63 ms |
| 262,144 | 261,888 | 255 | 99.90% | 1,455.11 ms | 1,222.95 ms |

All 12 resident turns generated one token and reused a valid recurrent checkpoint. No turn allocated KV-only pages without a matching recurrent state, and no recurrent stash became stranded. The 256K branches evicted one live tail page while retaining the 1,023-page resumable prefix.

An ordinary least-squares fit over the four turn medians gives:

```text
TTFT_ms = 134.73 + 0.005025 * context_tokens
slope = 5.146 ms per 1K context tokens
R² = 0.999919
```

The physical replay is bounded, but the measured latency remains almost perfectly linear in total context.

## SLO Verdict

- 128K p50 target `<= 200 ms`: **fail**, measured 788.04 ms.
- 256K p50 target `<= 300 ms`: **fail**, measured 1,455.11 ms.
- 256K/128K ratio target `<= 1.5`: **fail**, measured 1.846.
- Observed active device memory: approximately 26,633 MiB; no OOM occurred.

## Interpretation

Prefix reuse is working in this synthetic workload. The slope is consistent with exact long-context attention work, but this experiment does not isolate kernels, checkpoint transfers, hashing, or Generator housekeeping. Attribution of the complete latency slope requires profiling.

The recurrent checkpoints are stored in host memory and restored for new jobs; this experiment does not demonstrate a live GPU recurrent state across turns. See `2026-09-04-review-30s-latency.md` for the source audit and the revised latency proposal.

The current resume granularity also matters. ExLlamaV3 revives complete 256-token pages plus a page-aligned recurrent checkpoint. A logical 128-token branch therefore replays the incomplete tail and performs 255 physical prefill tokens. A true single-user resident session should retain the live partial page and recurrent state, cutting this to the actual delta before kernel optimization.

Exact attention still has an `O(delta * context)` component after eliminating that replay. Persistence alone cannot make asymptotic TTFT independent of context; it only removes `O(context)` transcript reconstruction from non-attention layers.

## Decision

Proceed in this order:

1. Add a live-session append experiment that retains the partial KV page and current recurrent state across turns, targeting exactly 128 physical input tokens rather than 255.
2. Profile full-attention layers separately at query lengths 1, 8, 32, and 128 for 128K and 256K contexts.
3. Profile the NVFP4 dispatch and evaluate exact split-KV sequence parallelism for the 16 full-attention layers. Online NVFP4 reads already exist. The donor also has separate split-KV kernels, but its compatible NVFP4 path selects the split-V-dimension/long-query kernels instead. The previous 300 ms aspiration requires substantial improvement; a 30-second response target needs a separately measured input/output envelope.
4. Evaluate a four-layer megakernel only for decode/verification after the attention profile; it does not remove the incremental-prefill scan and is not the first lever for this result.
5. Open the Quest-style sparse page-selection track only if the optimized exact resident path still misses the 256K gate.

The earlier HTTP baseline remains useful for end-to-end behavior, but it cannot be treated as a physical cold-prefill measurement because it did not expose cache hits. This direct probe becomes the M1 cache-reuse oracle.
