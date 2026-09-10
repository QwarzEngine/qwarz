# Resident Session Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether ExLlamaV3 can resume Qwen3.8-27B from resident KV pages and Gated DeltaNet checkpoints so that a turn appending at most 128 tokens avoids replaying a 1K–256K prefix.

**Architecture:** Add a direct, lazy-imported ExLlamaV3 probe to the existing Python harness. The probe loads the pinned EXL3 3.5 bpw bring-up artifact once, builds a deterministic seed sequence for each context bucket, submits branch turns that share the seed prefix, and records both wall-clock first-token latency and ExLlamaV3's physical cache counters. CPU-only unit tests use fake generators; GPU execution uses the donor virtualenv and never modifies the donor checkout.

**Tech Stack:** Python 3.12, pytest 8, ExLlamaV3 `Generator`/`Job`, PyTorch CUDA, NVML through `nvidia-smi`, JSON/JSONL artifacts.

**Spec:** `docs/superpowers/specs/2026-09-04-qwen38-27b-rtx5090-engine-design.md`

## Global Constraints

- Target exactly one NVIDIA RTX 5090 and one active generation.
- Use the pinned EXL3 3.5 bpw artifact only for bring-up; it cannot establish final Q38X quality parity.
- Keep all KV and recurrent execution state resident during an active session; do not spill KV or run model layers on CPU.
- Measure final context buckets at 1,024, 32,768, 131,072, and 262,144 tokens with no more than 128 newly appended tokens.
- Record physical reused tokens separately from logical prompt tokens; never infer reuse from an OpenAI usage field.
- Treat p50 first-delta latency `<= 200 ms` at 128K, `<= 300 ms` at 256K, and a 256K/128K ratio `<= 1.5` as the exact resident-path target.
- Preserve exact model execution. Sparse attention, KV eviction, CPU-tier cache, and approximate state reconstruction are out of scope.
- Do not edit `/home/rekeyea/Documents/llm/qwen38-exl3-mia` or transcode the EXL3 weights.
- Do not create git commits unless the user explicitly requests them.

## File Map

- `src/qwasar_bench/resident.py`: Pure configuration, observation, metric, and aggregation types shared by tests and the GPU driver.
- `src/qwasar_bench/exllamav3_probe.py`: Lazy ExLlamaV3 adapter, deterministic prompt construction, job execution, and artifact writer.
- `src/qwasar_bench/cli.py`: Adds the `resident-probe` command without importing CUDA dependencies for other commands.
- `tests/test_resident.py`: CPU-only tests for cache accounting, latency aggregation, SLO evaluation, and artifact validation.
- `tests/test_exllamav3_probe.py`: Fake-generator tests for first-token capture and exact-prefix branch construction.
- `tests/test_cli.py`: Parser coverage for the new command.
- `scripts/run_resident_probe.sh`: Validates paths/GPU ownership and launches the probe with the donor Python interpreter.
- `docs/benchmarks/2026-09-04-exl3-resident-session.md`: Records command, environment, raw artifact path, cache evidence, TTFT curve, and decision.

---

### Task 1: Resident Experiment Domain Model

**Files:**
- Create: `src/qwasar_bench/resident.py`
- Create: `tests/test_resident.py`

**Interfaces:**
- Produces: `ResidentProbeConfig(context_tokens: tuple[int, ...], appended_tokens: int, max_new_tokens: int, repetitions: int, page_size: int = 256)`.
- Produces: `JobObservation(bucket_tokens: int, phase: str, repetition: int, input_tokens: int, cached_tokens: int, prefill_tokens: int, ttft_ms: float, prefill_ms: float, generated_tokens: int, page_metrics: dict[str, int])`.
- Produces: `summarize_observations(observations: Sequence[JobObservation]) -> dict[str, object]`.
- Produces: `evaluate_resident_slo(summary: Mapping[str, object]) -> dict[str, object]`.

- [x] **Step 1: Write failing cache-accounting and SLO tests**

```python
from qwasar_bench.resident import (
    JobObservation,
    ResidentProbeConfig,
    evaluate_resident_slo,
    summarize_observations,
)


def test_observation_rejects_impossible_cache_accounting():
    with pytest.raises(ValueError, match="cached_tokens"):
        JobObservation(
            bucket_tokens=1024,
            phase="turn",
            repetition=0,
            input_tokens=100,
            cached_tokens=101,
            prefill_tokens=0,
            ttft_ms=10.0,
            prefill_ms=8.0,
            generated_tokens=1,
            page_metrics={},
        )


def test_summary_and_slo_use_turn_medians():
    def make_turn(bucket_tokens: int, ttft_ms: float) -> JobObservation:
        return JobObservation(
            bucket_tokens=bucket_tokens,
            phase="turn",
            repetition=0,
            input_tokens=bucket_tokens,
            cached_tokens=bucket_tokens - 256,
            prefill_tokens=255,
            ttft_ms=ttft_ms,
            prefill_ms=ttft_ms - 5.0,
            generated_tokens=1,
            page_metrics={"alloc_cached_pages": (bucket_tokens - 256) // 256},
        )

    observations = [
        make_turn(131072, 190.0), make_turn(131072, 210.0), make_turn(131072, 200.0),
        make_turn(262144, 280.0), make_turn(262144, 300.0), make_turn(262144, 290.0),
    ]
    summary = summarize_observations(observations)
    slo = evaluate_resident_slo(summary)
    assert summary["buckets"]["131072"]["turn_ttft_ms_p50"] == 200.0
    assert summary["buckets"]["262144"]["turn_ttft_ms_p50"] == 290.0
    assert slo == {
        "ttft_128k_pass": True,
        "ttft_256k_pass": True,
        "ratio_256k_to_128k": 1.45,
        "ratio_pass": True,
        "all_pass": True,
    }
```

- [x] **Step 2: Run the focused test and verify failure**

Run: `UV_CACHE_DIR=.uv-cache uv run --extra dev pytest tests/test_resident.py -q`

Expected: FAIL because `qwasar_bench.resident` does not exist.

- [x] **Step 3: Implement validated immutable records and medians**

```python
@dataclass(frozen=True)
class JobObservation:
    bucket_tokens: int
    phase: Literal["seed", "turn"]
    repetition: int
    input_tokens: int
    cached_tokens: int
    prefill_tokens: int
    ttft_ms: float
    prefill_ms: float
    generated_tokens: int
    page_metrics: dict[str, int]

    def __post_init__(self) -> None:
        if not 0 <= self.cached_tokens <= max(self.input_tokens - 1, 0):
            raise ValueError("cached_tokens must fit inside prefill input")
        if self.prefill_tokens != max(self.input_tokens - 1 - self.cached_tokens, 0):
            raise ValueError("prefill_tokens does not match physical cache accounting")
```

`summarize_observations` groups only `phase == "turn"` observations by decimal bucket string, reports p50 TTFT/prefill, minimum cache-reuse ratio, and maximum physical prefill tokens. `evaluate_resident_slo` reads buckets `131072` and `262144`, rounds the ratio to three decimals, and returns explicit false results when either bucket is absent.

- [x] **Step 4: Run the focused tests**

Run: `UV_CACHE_DIR=.uv-cache uv run --extra dev pytest tests/test_resident.py -q`

Expected: PASS.

- [x] **Step 5: Run the existing suite for regression**

Run: `UV_CACHE_DIR=.uv-cache uv run --extra dev pytest -q`

Expected: all tests pass.

### Task 2: Direct Generator Job Probe

**Files:**
- Create: `src/qwasar_bench/exllamav3_probe.py`
- Create: `tests/test_exllamav3_probe.py`

**Interfaces:**
- Consumes: `ResidentProbeConfig`, `JobObservation`, `summarize_observations`, and `evaluate_resident_slo` from Task 1.
- Produces: `build_branch_token_ids(seed_ids: Sequence[int], suffix_ids: Sequence[int], bucket_tokens: int) -> list[int]`.
- Produces: `run_generator_job(generator: object, job_factory: Callable[..., object], input_ids: object, *, max_new_tokens: int, seed: int, bucket_tokens: int, phase: Literal["seed", "turn"], repetition: int) -> tuple[JobObservation, object]`.
- Produces: `run_resident_matrix(generator: object, tokenizer: object, config: ResidentProbeConfig, job_factory: Callable[..., object]) -> tuple[list[JobObservation], dict[str, object]]`.

- [x] **Step 1: Write failing fake-generator tests**

```python
def test_run_generator_job_uses_first_streaming_event_for_ttft(monkeypatch):
    clock = iter([1_000_000_000, 1_011_000_000])
    monkeypatch.setattr(probe.time, "perf_counter_ns", lambda: next(clock))
    job_factory = FakeJobFactory(cached_pages=3, cached_tokens=0, time_prefill=0.008)
    generator = FakeGenerator([
        [{"stage": "started"}, {"stage": "prefill", "curr_progress": 768}],
        [{"stage": "streaming", "token_ids": FakeTokens(1), "eos": True}],
    ])
    observation, job = run_generator_job(
        generator, job_factory, FakeTokens(1025), max_new_tokens=1, seed=7,
        bucket_tokens=1024, phase="turn", repetition=0,
    )
    assert observation.ttft_ms == 11.0
    assert observation.cached_tokens == 768
    assert observation.prefill_tokens == 256


def test_build_branch_token_ids_preserves_seed_and_exact_bucket_length():
    branch = build_branch_token_ids([11, 12, 13], [21, 22, 23], bucket_tokens=5)
    assert branch == [11, 12, 13, 21, 22]
```

- [x] **Step 2: Run the focused tests and verify failure**

Run: `UV_CACHE_DIR=.uv-cache uv run --extra dev pytest tests/test_exllamav3_probe.py -q`

Expected: FAIL because the probe module does not exist.

- [x] **Step 3: Implement job timing and physical cache extraction**

`run_generator_job` must:

1. Snapshot `generator.pagetable.metrics` before enqueue.
2. Start `perf_counter_ns` immediately before `generator.enqueue(job)`.
3. Iterate until the job emits `eos` and record TTFT at the first streaming event containing at least one token.
4. Read `job.cached_pages * 256 + job.cached_tokens` only after completion.
5. Clamp cached tokens to `input_tokens - 1`, because ExLlamaV3 prefills every prompt token except the final decode input.
6. Read `job.time_prefill * 1000`, generated-token count, and per-key page-table counter deltas.
7. Raise `RuntimeError` if no token is emitted or the queue drains without EOS.

- [x] **Step 4: Implement deterministic resident branches**

Create a coding-oriented token source by encoding this fixed text once and tiling its token IDs:

```text
Repository task: inspect the parser, preserve public behavior, add a focused regression test, run the smallest relevant test target, and report exact file paths and failures. Do not change unrelated code.
```

For each bucket, make the seed input exactly `bucket_tokens - appended_tokens - max_new_tokens` tokens long. Run one greedy seed job, retain `seed_job.sequences[0].sequence_ids`, and build each branch by appending a deterministic `appended_tokens` suffix then truncating to exactly `bucket_tokens`. Use branch-specific suffix text containing the repetition number so different turn trials reuse the common seed but not another branch's tail.

- [x] **Step 5: Run probe tests and the full CPU suite**

Run: `UV_CACHE_DIR=.uv-cache uv run --extra dev pytest tests/test_exllamav3_probe.py tests/test_resident.py -q && UV_CACHE_DIR=.uv-cache uv run --extra dev pytest -q`

Expected: all tests pass without importing ExLlamaV3 or CUDA.

### Task 3: CLI and Artifact Contract

**Files:**
- Modify: `src/qwasar_bench/cli.py`
- Modify: `tests/test_cli.py`
- Modify: `src/qwasar_bench/exllamav3_probe.py`
- Modify: `tests/test_exllamav3_probe.py`

**Interfaces:**
- Consumes: `run_resident_matrix` from Task 2.
- Produces CLI: `python -m qwasar_bench resident-probe --model PATH --draft-model PATH --output PATH [--contexts 1024,32768,131072,262144] [--appended-tokens 128] [--max-new-tokens 1] [--repetitions 3] [--cache-size 262400] [--gpu-split-gb 30] [--cache-quant nvfp4]`.
- Produces artifact files: `config.json`, `environment.json`, `observations.jsonl`, and `summary.json`.

- [x] **Step 1: Write failing parser and artifact tests**

```python
def test_parser_accepts_resident_probe_defaults():
    args = cli._parser().parse_args([
        "resident-probe", "--model", "/model", "--output", "/results"
    ])
    assert args.contexts == (1024, 32768, 131072, 262144)
    assert args.appended_tokens == 128
    assert args.repetitions == 3
    assert args.gpu_split_gb == 30.0
    assert args.cache_quant == "nvfp4"


def test_write_artifacts_is_atomic(tmp_path):
    write_probe_artifacts(
        tmp_path, config=config, observations=observations, summary=summary,
        environment=environment, model_path="/model", draft_model_path="/draft",
        cache_size=262400, cache_quant="nvfp4", gpu_split_gb=30.0,
    )
    assert json.loads((tmp_path / "summary.json").read_text())["schema_version"] == 1
    assert len((tmp_path / "observations.jsonl").read_text().splitlines()) == len(observations)
    assert not list(tmp_path.glob("*.tmp"))
```

- [x] **Step 2: Run focused tests and verify failure**

Run: `UV_CACHE_DIR=.uv-cache uv run --extra dev pytest tests/test_cli.py tests/test_exllamav3_probe.py -q`

Expected: FAIL because the command and artifact writer are absent.

- [x] **Step 3: Add CLI parsing without eager CUDA imports**

Parse `--contexts` with a function returning `tuple[int, ...]`. Import `qwasar_bench.exllamav3_probe` only inside the `resident-probe` command handler so `doctor`, `baseline`, and `validate-run` remain dependency-free.

- [x] **Step 4: Add atomic artifact serialization**

Each JSON document includes `schema_version: 1`. `observations.jsonl` writes one `dataclasses.asdict` record per line. Write every file to `<name>.tmp`, call `flush()` and `os.fsync()`, then `os.replace()` it into place. `summary.json` includes the aggregate buckets, SLO result, model path, cache format, and an explicit `qualification: "bringup_only"`.

- [x] **Step 5: Run focused and full tests**

Run: `UV_CACHE_DIR=.uv-cache uv run --extra dev pytest tests/test_cli.py tests/test_exllamav3_probe.py -q && UV_CACHE_DIR=.uv-cache uv run --extra dev pytest -q`

Expected: all tests pass.

### Task 4: Safe GPU Launcher

**Files:**
- Create: `scripts/run_resident_probe.sh`

**Interfaces:**
- Consumes: `resident-probe` CLI from Task 3.
- Produces: a reproducible shell entry point using `/home/rekeyea/Documents/llm/qwen38-exl3-mia/.venv/bin/python` and `PYTHONPATH=$QWASAR_ROOT/src`.

- [x] **Step 1: Implement strict preflight checks**

The script uses `set -euo pipefail` and verifies:

```bash
MODEL_DIR=${QWASAR_MODEL_PATH:-/home/rekeyea/models/Qwen3.8-27B-EXL3-3.5bpw}
DRAFT_DIR=${QWASAR_DRAFT_PATH:-/home/rekeyea/models/Qwen3.8-27B-DFlash2-EXL3-5.0bpw}
EXLLAMA_HOME=${QWASAR_EXLLAMA_HOME:-/home/rekeyea/Documents/llm/qwen38-exl3-mia}
EXLLAMA_PYTHON=${QWASAR_EXLLAMA_PYTHON:-$EXLLAMA_HOME/.venv/bin/python}
CUDA_DEVICE=${QWASAR_CUDA_DEVICE:-0}
GPU_SPLIT_GB=${QWASAR_GPU_SPLIT_GB:-30}
```

Require the model and draft `config.json`, the donor Python binary, an output directory that does not already contain `summary.json`, and no existing compute process using at least 512 MiB on the selected GPU unless `QWASAR_ALLOW_BUSY_GPU=1` is set.

- [x] **Step 2: Launch the direct probe with pinned defaults**

```bash
exec env CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" PYTHONPATH="$ROOT_DIR/src" \
  "$EXLLAMA_PYTHON" -m qwasar_bench resident-probe \
  --model "$MODEL_DIR" \
  --draft-model "$DRAFT_DIR" \
  --output "$OUTPUT_DIR" \
  --contexts "${QWASAR_CONTEXTS:-1024,32768,131072,262144}" \
  --appended-tokens "${QWASAR_APPENDED_TOKENS:-128}" \
  --max-new-tokens "${QWASAR_MAX_NEW_TOKENS:-1}" \
  --repetitions "${QWASAR_REPETITIONS:-3}" \
  --cache-size "${QWASAR_CACHE_SIZE:-262400}" \
  --gpu-split-gb "${QWASAR_GPU_SPLIT_GB:-30}" \
  --cache-quant "${QWASAR_CACHE_QUANT:-nvfp4}"
```

- [x] **Step 3: Validate shell syntax and help path**

Run: `bash -n scripts/run_resident_probe.sh && PYTHONPATH=src /home/rekeyea/Documents/llm/qwen38-exl3-mia/.venv/bin/python -m qwasar_bench resident-probe --help`

Expected: exit 0 without loading model weights.

### Task 5: 1K GPU Validation Experiment

**Files:**
- Create: `results/<timestamp>-exl3-resident-probe-1k/` through the launcher.
- Modify: `docs/benchmarks/2026-09-04-exl3-resident-session.md`

**Interfaces:**
- Consumes: safe launcher and artifact contract from Tasks 3–4.
- Produces: first physical proof that a second job restores the seed prefix from KV pages plus a recurrent checkpoint.

- [x] **Step 1: Record and stop only the known EXL3 service**

Confirm PID 1602 still has command line `tools/serve_openai.py --model /home/rekeyea/models/Qwen3.8-27B-EXL3-3.5bpw` before sending `SIGTERM`. Do not stop download processes or unrelated GPU processes. Wait until port 8004 is closed and the process exits.

- [x] **Step 2: Run a short validation matrix**

Run:

```bash
QWASAR_CONTEXTS=1024 \
QWASAR_REPETITIONS=2 \
QWASAR_ALLOW_BUSY_GPU=1 \
QWASAR_OUTPUT_DIR=results/$(date -u +%Y%m%dT%H%M%SZ)-exl3-resident-probe-1k \
scripts/run_resident_probe.sh
```

Expected: two `phase="turn"` observations; each reports at least 768 physically cached tokens, no more than 255 physical prefill tokens beyond the 128-token branch delta plus page alignment, and one generated token.

- [x] **Step 3: Validate artifacts and cache invariants**

Run:

```bash
jq -e '.qualification == "bringup_only"' results/*-exl3-resident-probe-1k/summary.json
jq -e 'select(.phase == "turn") | (.cached_tokens >= 768 and .prefill_tokens <= 255)' \
  results/*-exl3-resident-probe-1k/observations.jsonl
```

Expected: both commands exit 0 for every turn record.

- [x] **Step 4: Diagnose before scaling if reuse is absent**

If a turn reports fewer than 768 cached tokens, record `job.cached_pages`, `generator.pagetable.metrics`, recurrent-cache size, and `generator.pagetable.audit_recurrent_sync(generator.recurrent_cache)`. Do not proceed to 256K until the missing checkpoint or page mismatch is explained.

- [x] **Step 5: Write the validation result**

Document the exact artifact path, donor git revision plus dirty state, model tree hash from `benchmarks/backends/exllamav3-3.5bpw.json`, GPU identity, cache format, and whether the physical reuse invariant passed.

### Task 6: Full Resident TTFT Curve and Decision

**Files:**
- Create: `results/<timestamp>-exl3-resident-probe-full/` through the launcher.
- Modify: `docs/benchmarks/2026-09-04-exl3-resident-session.md`

**Interfaces:**
- Consumes: validated 1K experiment from Task 5.
- Produces: measured exact-resident TTFT curve, cache-reuse evidence, SLO verdict, and the next kernel investigation decision.

- [x] **Step 1: Stabilize the measurement environment**

Record `nvidia-smi -L`, driver version, clocks, temperature, power, all compute PIDs, donor revision/dirty state, and the active 5 bpw download status. Keep one probe process on the selected RTX 5090 and avoid concurrent inference on that device.

- [x] **Step 2: Run the full matrix**

Run:

```bash
QWASAR_CONTEXTS=1024,32768,131072,262144 \
QWASAR_REPETITIONS=3 \
QWASAR_ALLOW_BUSY_GPU=0 \
QWASAR_OUTPUT_DIR=results/$(date -u +%Y%m%dT%H%M%SZ)-exl3-resident-probe-full \
scripts/run_resident_probe.sh
```

Expected: one seed plus three resident turns for every bucket, valid JSON artifacts, and no OOM at 262,144 input tokens because the cache allocation includes generation headroom.

- [x] **Step 3: Verify physical work remains bounded**

For every resident turn require:

```text
cached_tokens >= bucket_tokens - appended_tokens - max_new_tokens - 255
prefill_tokens <= appended_tokens + max_new_tokens + 255
alloc_cached_pages > 0
generated_tokens == max_new_tokens
```

Any violation invalidates the TTFT comparison and triggers recurrent/page-cache diagnosis rather than performance tuning.

- [x] **Step 4: Evaluate the TTFT curve**

Read `summary.json` and report p50 turn TTFT at every bucket, the 256K/128K ratio, p50 physical prefill time, minimum reuse ratio, peak VRAM, and page/recurrent eviction counters. Compare separately against the absolute 128K/256K gates and the ratio gate.

- [x] **Step 5: Select the next optimization track**

- If all three SLO gates pass, preserve ExLlamaV3 page/checkpoint semantics as the M1 oracle and begin the persistent Responses API session store.
- If physical prefill remains bounded but TTFT grows beyond the ratio gate, profile exact full-attention decode and implement split-KV sequence parallelism before considering sparsity.
- If physical prefill grows with context, fix page/checkpoint persistence before any kernel work.
- If exact split-KV later misses the 256K absolute gate, open the already-approved Quest-style query-aware page-selection experiment; do not enable it in normal requests.

- [x] **Step 6: Run final repository verification**

Run: `UV_CACHE_DIR=.uv-cache uv run --extra dev pytest -q && git diff --check && bash -n scripts/run_resident_probe.sh`

Expected: all tests pass, no whitespace errors, and valid shell syntax.
