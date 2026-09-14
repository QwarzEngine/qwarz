# Qwarz

Qwarz is an independent inference-engine repository specialized for Qwen3.8-27B on one NVIDIA RTX 5090. The target runtime serves one persistent, agentic coding session with a native 262,144-token context and a model-specific 4.5–5.0 bpw artifact.

## Metrics

Live service on 2026-09-14 after the 18:07 restart (70 completed turns, production `flash` on the RTX 5090). Typical prompt in that window was ~45–50K with prefix reuse. These are measured samples, not an SLA.

| | Median | p90 |
| --- | ---: | ---: |
| End-to-end tokens/s (output / wall time, includes prefill) | **112** | 173 |
| Decode tokens/s (after the first token) | **158** | 213 |
| Prefill tokens/s (new physical tokens / host prefill) | **3,227** | 4,247 |
| TTFT (first token, includes prefill) | **441 ms** | 1.09 s |

Warm ~50K continuations: TTFT **260–550 ms**, decode **~110–175 tok/s**, append prefill **~1,800–3,700 tok/s**. Large prefills (≥8K, PRIMS path): **5,300–6,800 tok/s**, TTFT **2.4–4.2 s**. Decode tracks MTP acceptance (~0.62 this session); a 50K turn at 0.21 acceptance fell to 82 tok/s.

Phase 2 quality-gate medians (thinking medium, not warm chat):

| Context | TTFT | Decode |
| --- | ---: | ---: |
| 32K | 5.17 s | 200 tok/s |
| 64K | 11.27 s | 184 tok/s |
| 128K | 26.63 s | 164 tok/s |
| 256K | 67.27 s | 122 tok/s |

All-time store (4,598 completed turns, 5–14 Sep, many ~130K sessions): decode **138 tok/s**, end-to-end **98 tok/s**, TTFT **881 ms**. The 254K warm HTTP sample is **97.19 tok/s** and **0.593 s** to first content.

The **hybrid v1** implements a Rust HTTP/SQLite supervisor, a persistent ExLlamaV3 worker, exact generated-token history, cancellation/recovery, and Pi-compatible streaming tools. Production `flash` is the measured stack: pinned **EXL3 5 bpw + MTP6 + K8/V4**, **NVIDIA64 NVFP4 MLP** (192 matrices), Flash/8192 with **FP8 PRIMS** for Q≥8192, and **Attention64**. `baseline` is the original EXL3 Triton fallback. Benchmark launchers remain unchanged.

See [Arquitectura v1: combinación de tecnologías y verificación de pesos](docs/arquitectura-v1.md)
for the Spanish explanation of the design, measured performance, and live RTX 5090 artifact verification.

Automatic startup is installed as `qwasar.service` in the user systemd manager
with linger enabled. See [Servicio systemd](docs/servicio-systemd.md) for boot,
restart and journal commands; the existing launcher delegates to this unit.

## Use With Pi

```bash
cd /home/rekeyea/Documents/llm/qwasar
python3 scripts/configure_pi.py
python3 scripts/qwasar.py start
```

Then, from the project you want to work on:

```bash
/home/rekeyea/Documents/llm/qwasar/scripts/pi_qwasar.sh
```

Endpoint: `http://127.0.0.1:8800/v1`; model: `qwasar-qwen38-27b`. Inline
images (data URLs) are accepted in user messages; see the manual's Images
section for limits and token cost.
Use `python3 scripts/qwasar.py status`, `logs`, or `stop` to manage it.
See the [v1 manual](docs/manual-v1.md) for persistence, supported API,
security boundaries, verification, and performance limits. Cold 256K ingestion
is **not** guaranteed below 30 seconds; warm continuation is a different path.
The [real Pi/service acceptance report](docs/benchmarks/2026-09-05-v1-pi-acceptance.md)
records coding tools, durable restart, and a 254K warm HTTP sample with 0.593 s
to first content and 97.19 tokens/s. These are sample measurements, not an SLA.

## Development

```bash
uv sync --extra dev --extra benchmark
uv run pytest -q
```

`qwasar-bench` uses Python 3.12. The benchmark path depends only on the standard library plus `tokenizers`; model execution stays in the selected backend process.

## M0 Commands

```bash
./scripts/download_exllamav3_target.sh

# Immediate API/session/performance bring-up on the existing 3.5 bpw service.
./scripts/run_exllamav3_3_5_bringup.sh

export QWASAR_MODEL_PATH=/absolute/path/to/Qwen3.8-27B-EXL3-5.0bpw

uv run python -m qwasar_bench validate-manifest \
  benchmarks/manifests/qwen38-27b-rtx5090-v1.json

uv run python -m qwasar_bench doctor \
  --manifest benchmarks/manifests/qwen38-27b-rtx5090-v1.json

uv run --extra benchmark python -m qwasar_bench baseline \
  --base-url http://127.0.0.1:8888 \
  --api-model qwen38 \
  --protocol chat-completions \
  --output results/exllamav3-5bpw

uv run python -m qwasar_bench compare \
  results/exllamav3-5bpw results/qwasar-candidate \
  --output results/comparison.json
```

## M1 Resident Probe

The direct probe temporarily owns the RTX 5090, loads the pinned 3.5 bpw bring-up target, and records physical cache reuse rather than OpenAI usage counters:

```bash
QWASAR_CONTEXTS=1024,32768,131072,262144 \
QWASAR_REPETITIONS=3 \
./scripts/run_resident_probe.sh
```

The first full result is documented in `docs/benchmarks/2026-09-04-exl3-resident-session.md`. It demonstrates prefix-page and recurrent-checkpoint reuse across jobs. The observed TTFT growth has not yet been attributed to individual kernels; see `docs/benchmarks/2026-09-04-review-30s-latency.md`. New resident runs interpret context buckets as total sequence budgets and reserve output inside them.

`doctor` only approves an acceptance-grade baseline when it sees an RTX 5090 with at least 30,000 MiB, a committed Qwarz revision, an existing model path, and an artifact hash equal to the frozen manifest pin. The EXL3 5 bpw shards and tokenizer are verified, and the local artifact tree hash is now pinned. No Qwarz commit has been created.

The 3.5 bpw profile uses `--qualification bringup`. It can exercise the API, persistent turns, long-context allocation, cancellation, recovery, and optimization plumbing before 5 bpw is ready. It is structurally prevented from becoming an acceptance baseline because `doctor` requires 4.5–5.0 bpw for `acceptance`.

## Decode Screening

The same guarded launcher supports a separate source-code workload and explicit draft/cache selection:

```bash
QWASAR_PROBE_KIND=decode \
QWASAR_DRAFT_METHOD=mtp \
QWASAR_CACHE_QUANT=8,4 \
QWASAR_CONTEXTS=32768,131072,262144 \
QWASAR_MAX_NEW_TOKENS=512 \
QWASAR_REPETITIONS=2 \
./scripts/run_resident_probe.sh
```

Use `QWASAR_DRAFT_METHOD=dflash2` or `none` for controls; `QWASAR_CACHE_QUANT=nvfp4` selects the old cache path. `QWASAR_THINKING` accepts `xhigh`, `medium` (default), `low`, and `off`; `QWASAR_SAMPLER` accepts `greedy` (controlled decode screening default) and `recommended` (mode-specific sampling). These controls apply to the decode probe, not the donor HTTP bridge.

For opt-in MTP experiments, set `QWASAR_MTP_POLICY=fixed1` through `fixed7`, or `adaptive4`,
with `QWASAR_PROBE_KIND=decode`, `QWASAR_DRAFT_METHOD=mtp` and
`QWASAR_CACHE_QUANT=8,4`. Fixed policies allocate the requested draft width when
constructing the generator and disable adaptation. `fixed4` versus `adaptive4`
isolates adaptation under the same four-token ceiling. All record per-round
draft statistics. The probe verifies the
pinned EXL3 5 bpw artifact and records the effective policy in `run.json`.
Keep `QWASAR_PREFILL_SETTINGS=benchmarks/profiles/prefill-flash-k8v4.json`,
prompts, sampler and context budgets identical when comparing against v1.
Set `QWASAR_CACHE_SIZE=262144` to retain the production cache pool when testing
shorter prompts; otherwise the LRU probe allocates for its largest context bucket.
These experiment controls do not change the service's production policy.
The service now explicitly uses six fixed MTP proposals; the loader allocates the
corresponding recurrent history before constructing the generator.
The [fixed-width MTP results](docs/benchmarks/2026-09-07-mtp-fixed-width-results.md)
compare widths 1–7, long-context finalists and recommended sampling.

The [attention evaluation](docs/benchmarks/2026-09-07-attention-profile-results.md)
compares native graph decode kernels and Flash chunk sizes with MTP fixed at six.
Its measured candidate is saved in
`benchmarks/profiles/attention-5090-qwen38-mtp6.json`; this experimental profile
does not change the service. Apply `graph_attention_context(profile["decode"])`
before constructing a fresh generator and keep it active for that generator's
entire lifetime. Native head-group overrides are rejected because C++ computes
the launch grid separately.

The corpus defaults to a snapshot of installed ExLlamaV3 source files; override it with `QWASAR_CORPUS_ROOT`. It is never tiled to manufacture length. `token_safe_v2` encodes the corpus literally and splices its IDs into chat framing, so source-code delimiters cannot introduce extra chat roles. A fixed LRU coding task follows the corpus. Each bucket reserves generation plus speculative scratch within the native limit. Artifacts record actual prompt length, physical reuse, raw events, completions, host-observed first token/content, draft acceptance, and CUDA allocator peaks. First-batch tokens are excluded from the steady decode-rate numerator.

These are **screening-only warm branches**, not a continuous agent session or quality certification. The source snapshot is background for throughput testing, not a long-range retrieval exam. Prompt fitting/tokenization time is recorded separately as offline preparation; reported TTFT is not HTTP end-to-end latency. A null first-content time means no final-answer text was observed, even if reasoning streamed quickly. A 512-token cap can truncate the task; use longer outputs when evaluating thinking and correctness. CUDA allocator peaks do not include all GPU memory. Incomplete runs lack `completed.json` and must not be treated as completed measurements.

Initial measurements, the framing correction, and coding smoke checks are recorded in `docs/benchmarks/2026-09-04-decode-screen-results.md`. The old framing runs are retained only as diagnostics, not acceptance evidence.

### Append-Only Retrieval Sessions

The decode launcher can also exercise two consecutive tool-call/answer cycles over distant configuration records:

```bash
QWASAR_PROBE_KIND=decode \
QWASAR_WORKLOAD=retrieval_session \
QWASAR_MODEL_PATH=/home/rekeyea/models/Qwen3.8-27B-EXL3-5.0bpw \
QWASAR_DRAFT_METHOD=mtp \
QWASAR_CACHE_QUANT=8,4 \
QWASAR_CONTEXTS=32768,262144 \
QWASAR_REPETITIONS=1 \
QWASAR_TOOL_MAX_NEW_TOKENS=512 \
QWASAR_MAX_NEW_TOKENS=1536 \
QWASAR_SAMPLER=recommended \
QWASAR_THINKING=medium \
./scripts/run_resident_probe.sh
```

Here repetitions count independent sessions per context, not warm branches. Each session retains the exact runtime token IDs, including reasoning and message terminators; only new user/tool message tails are rendered. The snapshot contains three synthetic tables at distant positions, with 16 services and two held-out queries. A restricted `read_file` tool reads from the fixture mapping in memory, never arbitrary host files or generated code. Exact JSON fields and arithmetic are graded against a separate oracle. Actual outputs and tool results are appended without injecting the expected answer or repairing mistakes.

The initial prompt reserves both cycles' maximum output budgets, 1,024 tokens for new messages, and speculative scratch; every step checks the native limit again. Session summaries record record positions, complete user-cycle wait, transcript, and quality failures. Per-step samples retain physical prefill/cache counters and timing. `completed.json` means collection finished, **not** that quality passed. This is a synthetic agent-like session, not a multi-file coding benchmark or HTTP SLA. See `docs/benchmarks/2026-09-04-retrieval-session-results.md`.

If the runtime requeues a generation, `requeue_count` is nonzero and `cache_metrics_valid` is false. Physical cache, draft acceptance, and runtime prefill counters are null rather than misleadingly reporting only the final physical job; host-observed timing and raw events remain available.

### Cache Fidelity And Turn-Size Matrix

The selected working configuration is EXL3 **5 bpw + MTP**. To replay the archived problematic tool-result prompt both warm and cold, compare greedy tokens and capture initial full-vocabulary logits:

```bash
QWASAR_PROBE_KIND=decode QWASAR_WORKLOAD=cache_fidelity \
QWASAR_MODEL_PATH=/home/rekeyea/models/Qwen3.8-27B-EXL3-5.0bpw \
QWASAR_DRAFT_METHOD=mtp QWASAR_CACHE_QUANT=8,4 \
QWASAR_CONTEXTS=262144 QWASAR_MAX_NEW_TOKENS=512 \
QWASAR_SAMPLER=greedy QWASAR_THINKING=medium \
./scripts/run_resident_probe.sh
```

`QWASAR_SOURCE_RUN` selects the archived session (default `results/20260904-session-5.0bpw-medium-long`). The probe verifies corpus and prompt hashes, replaces both the page table and recurrent checkpoint cache for the cold pass, and requires zero physical reuse. Matching masked vocabulary entries are excluded from logit differences. A small floating-point difference alone does not prove corruption; the report includes top-token margins and the first greedy divergence. Logits diagnostics are not ordinary throughput measurements.

For controlled new-input sizes and an optional profile of the slowest valid turn:

```bash
QWASAR_PROBE_KIND=decode QWASAR_WORKLOAD=turn_matrix \
QWASAR_MODEL_PATH=/home/rekeyea/models/Qwen3.8-27B-EXL3-5.0bpw \
QWASAR_DRAFT_METHOD=mtp QWASAR_CACHE_QUANT=8,4 \
QWASAR_CONTEXTS=32768,131072,262144 \
QWASAR_APPENDED_TOKENS=128,512,2048,8192 \
QWASAR_REPETITIONS=5 QWASAR_MAX_NEW_TOKENS=1536 \
QWASAR_SAMPLER=recommended QWASAR_THINKING=medium \
QWASAR_PROFILE_MATRIX=1 ./scripts/run_resident_probe.sh
```

Both modes allocate a 262,144-position pool. Matrix nominal contexts describe base histories, not total input lengths: the largest base shrinks to reserve the largest delta and output inside the native limit. Each branch's trial ID changes before its source delta, preventing reuse of the previous branch's entire delta. Seed histories and delta token files reconstruct exact prompts. The task retrieves configuration from earlier code-context tables and applies new request values; it is not a multi-file code-editing quality benchmark. Failed responses stay in latency statistics and quality denominators. Five-sample percentiles are descriptive, not an SLA.

Warm qualification requires valid physical counters and substantial seed-prefix reuse; a correct answer with cold or requeued execution does not qualify as a warm success. Profiling resets caches and primes only the selected seed history before replaying the selected delta. It records actual prefill counters so checkpoint differences remain visible. PyTorch CPU/CUDA traces and phase ranges are saved separately; instrumented runs never enter the ordinary latency percentiles. Missing CUDA kernel activity is reported explicitly, not interpreted as zero GPU cost. Methodology and results: `docs/benchmarks/2026-09-04-fidelity-matrix-results.md`.

### Experimental Prefill Optimization

The optional `benchmarks/profiles/prefill-flash-k8v4.json` profile uses PyTorch's installed native Flash Attention with an 8,192-token prefill chunk. It preserves EXL3 5 bpw, MTP and K8/V4, reuses the donor's existing dequantization scratch, and preserves lower-right causality. It only supports the pinned single-batch Q24/KV4/D256 geometry and refuses unsupported Flash execution rather than materializing dense attention. Prefill queries below 17 tokens keep the original route; decode is unchanged.

Add this environment variable to a direct `turn_matrix`, `lru` or `retrieval_session` decode command:

```bash
QWASAR_PREFILL_SETTINGS=benchmarks/profiles/prefill-flash-k8v4.json
```

The profile is opt-in; `cache_fidelity` deliberately rejects it to keep that control unchanged. Chunk/kernel overrides are restored when a workload finishes or fails; aggregate dispatch counters are saved separately. The adapter learns the cached length from the host copy of `cache_seqlens` at upload time instead of a per-layer `.item()`; `prefill-counters.json` reports `host_length_hits` and `host_length_syncs`, and the latter must be zero on the production path (see `docs/benchmarks/2026-09-10-item-sync-results.md`). The installed donor/server and default launcher behavior remain unchanged. Passing a matrix is not BF16-model or general coding-quality parity: Flash has greater sampled numerical error than the Triton baseline against an FP32 attention oracle. See `docs/benchmarks/2026-09-04-prefill-tuning-results.md` for timings, quality controls, memory and limitations.

Isolated replay tools run with the donor Python and `PYTHONPATH=src`: `qwasar_bench.prefill_experiment` captures real attention or screens full-model candidates; `qwasar_bench.prefill_microbench` sweeps tile/staging/Flash alternatives; `qwasar_bench.prefill_oracle` checks sampled FP32 attention. Do not run GPU probes concurrently. Candidate `baseline` is the unchanged control and must have empty overrides and the default 2K chunk. All scripts preserve raw failures; collection completion is distinct from quality qualification.

### Application-Level Prefill Check

The bounded [application validation plan](docs/benchmarks/2026-09-05-prefill-application-validation-plan.md) compares the selected profile against the original prefill on LRU code generation and append-only tool cycles at 32K and near 256K. Coding is checked against both generated unittests and a frozen independent oracle; it is one task, not general coding certification.

The [completed application results](docs/benchmarks/2026-09-05-prefill-application-results.md) audit 44 generations. Flash passes 6/6 LRU responses versus 2/6 baseline; both pass 6/8 tool cycles, with different individual failures. Long warm LRU TTFT falls from 526 to 327 ms, but some complete responses get slower and exceed 30 seconds. The optimized profile remains explicit, not a universal quality or latency guarantee.

Extract LRU submissions without executing them:

```bash
PYTHONPATH=src .venv/bin/python -m qwasar_bench.lru_grade \
  --run results/EXISTING_LRU_RUN --output results/NEW_CODE_REVIEW
```

After inspecting the extracted code, repeat with a new output directory and `--execute-reviewed-code`. Execution requires Linux `bwrap`, uses isolated system Python with no host-home/network/GPU access, and enforces a timeout and resource limits. Truncated, invalid or failing submissions remain failures; the grader never repairs model output. The flag attests manual inspection, not suitability for arbitrary adversarial code. A nonzero grading exit is expected when any submission fails.

## Run Artifacts

Each completed run is published atomically and contains:

- `run.json`: immutable run identity, model, backend, revisions, hashes, and sample count.
- `environment.json`: CPU, GPU, driver, CUDA, clocks, power, and platform metadata.
- `manifest.json`: the exact workload definition used for the run.
- `samples.jsonl`: raw counters and timings for every measured repetition.
- `events/*.jsonl`: the original SSE event payloads for each measured response.
- `quality.json`: separately generated coding and tool-call quality scores required for acceptance.

Exit code `0` means success, `2` means invalid input or an environment that cannot produce an acceptance-grade run, and `3` means a valid comparison completed but failed one or more gates.

## Current Baseline Blockers

- The local ExLlamaV3 server at `../qwen38-exl3-mia` exposes `/v1/chat/completions`. Use `--protocol chat-completions`; Qwarz reconstructs `previous_response_id` history locally and derives usage from the donor's single-user `/health` counters. This bridge records TTFT but does not mislabel it as model-internal prefill time.
- The approved target source is pinned to [`thelastspark/Qwen3.8-27B-exl3`](https://huggingface.co/thelastspark/Qwen3.8-27B-exl3/tree/5.00bpw), revision `1a6fe4afb5b921fda9f93fd4b06d6c6d5c99a62c`. Its three shards total 19.9 GB and pass their expected SHA-256 checks. The local tree hash is `0b9a439ffefa45c55a2a3cb0324de9fe1b23bce69a37a399cb952d355f02d92e`.
- The host exposes an RTX 5090 (`sm_120`) and a 3090 Ti. The former 3.5 bpw HTTP service on GPU 0 was intentionally stopped for the direct resident-session experiment; launchers must still reject an unrelated inference workload on the selected GPU.
- Qwarz has no commit yet, by policy. `doctor` intentionally rejects acceptance-grade collection until an engine revision and the final local target hash are frozen.

The approved architecture is in `docs/superpowers/specs/2026-09-04-qwen38-27b-rtx5090-engine-design.md`. The active M1 experiment plan is in `docs/superpowers/plans/2026-09-04-m1-resident-session-experiment.md`.
