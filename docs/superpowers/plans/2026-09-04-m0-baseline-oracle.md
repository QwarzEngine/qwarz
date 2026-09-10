# M0 Baseline and Oracle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible, GPU-optional benchmark and oracle harness that freezes the Qwen3.8-27B workload, records the RTX 5090 execution environment, measures OpenAI-compatible streaming backends, and evaluates future Qwasar revisions against the approved ExLlamaV3 5 bpw baseline.

**Architecture:** A small Python 3.12 package owns immutable workload manifests, strict result schemas, an OpenAI-compatible SSE client, atomic run artifacts, environment capture, and acceptance-gate comparisons. Unit and integration tests use local fake servers and injected command runners, so the complete harness remains testable when CUDA or the production model is unavailable. Real GPU collection is an explicit final step and never changes benchmark definitions.

**Tech Stack:** Python 3.12, standard library (`dataclasses`, `json`, `urllib`, `subprocess`, `hashlib`, `pathlib`), pytest, JSON/JSONL artifacts, OpenAI Responses API-compatible SSE.

**Spec:** `docs/superpowers/specs/2026-09-04-qwen38-27b-rtx5090-engine-design.md`

## Global Constraints

- Pin one production model identity and one approved ExLlamaV3 5 bpw artifact before collecting acceptance baselines.
- Keep benchmark definitions immutable during Q38X tuning; changes require a new manifest version and fresh baseline.
- Support context buckets `1K`, `32K`, `128K`, and `256K`, with separate fresh-prefill and persistent-turn cases.
- Measure one active request at a time; concurrency and batching are outside M0.
- Record raw timings and counters. Derived throughput must be reproducible from persisted fields.
- Write run artifacts atomically and never overwrite a completed run directory.
- Treat missing CUDA/NVIDIA tools as an explicit `unavailable` capability, not as a harness crash.
- Do not add PyTorch to the harness runtime. Backend processes own model-specific dependencies.
- Do not execute `git commit` until the user explicitly authorizes commits.

---

## Task 1: Scaffold the Harness and Freeze the Result Schema

**Files:**
- Create: `.gitignore`
- Create: `README.md`
- Create: `pyproject.toml`
- Create: `src/qwasar_bench/__init__.py`
- Create: `src/qwasar_bench/schema.py`
- Test: `tests/test_schema.py`

- [x] **Step 1: Write the failing schema tests**

```python
from qwasar_bench.schema import BenchmarkSample, RunMetadata


def test_benchmark_sample_round_trips() -> None:
    sample = BenchmarkSample(
        case_id="interactive-1k",
        backend="exllamav3",
        model="Qwen3.8-27B-EXL3-5.0bpw",
        context_bucket="1k",
        prompt_tokens=1024,
        reused_prefix_tokens=1000,
        output_tokens=64,
        time_to_first_delta_ms=120.0,
        elapsed_ms=2120.0,
        peak_vram_bytes=28_000_000_000,
        status="completed",
    )

    assert BenchmarkSample.from_dict(sample.to_dict()) == sample
    assert sample.prefill_tokens == 24
    assert sample.accepted_decode_tokens_per_second == 32.0


def test_invalid_token_accounting_is_rejected() -> None:
    with pytest.raises(ValueError, match="reused_prefix_tokens"):
        BenchmarkSample(
            case_id="bad",
            backend="fake",
            model="fake",
            context_bucket="1k",
            prompt_tokens=10,
            reused_prefix_tokens=11,
            output_tokens=1,
            time_to_first_delta_ms=1.0,
            elapsed_ms=2.0,
            peak_vram_bytes=0,
            status="completed",
        )
```

- [x] **Step 2: Run the test and verify it fails for the missing package**

Run: `python -m pytest tests/test_schema.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'qwasar_bench'`.

- [x] **Step 3: Add packaging metadata and the minimal schema implementation**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "qwasar-bench"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = []

[project.optional-dependencies]
dev = ["pytest>=8.3,<9"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

Implement frozen dataclasses with explicit validation and JSON-safe `to_dict`/`from_dict` methods. `BenchmarkSample` must persist raw prompt, prefix, output, TTFT, elapsed, prefill duration, decode duration, VRAM, speculative counters, status, error, artifact hash, and engine revision. Derived fields must not be persisted as authoritative values.

- [x] **Step 4: Run the focused schema tests**

Run: `python -m pytest tests/test_schema.py -q`

Expected: PASS.

- [x] **Step 5: Add repository hygiene and a concise project README**

Ignore virtual environments, Python caches, generated run results, local model paths, and editor files. Document that `qwasar` is an independent repository, M0 is the benchmark/oracle milestone, and no production inference semantics exist yet.

- [x] **Step 6: Review checkpoint**

Run: `git status --short`

Expected: only the new scaffold, tests, approved spec, and implementation plan are untracked or modified.

## Task 2: Define Versioned Workloads and Validate the Pinned Model

**Files:**
- Create: `src/qwasar_bench/workloads.py`
- Create: `benchmarks/manifests/qwen38-27b-rtx5090-v1.json`
- Create: `benchmarks/fixtures/smoke.jsonl`
- Test: `tests/test_workloads.py`

- [x] **Step 1: Write failing manifest-validation tests**

```python
def test_manifest_freezes_required_context_buckets(manifest_path: Path) -> None:
    manifest = load_manifest(manifest_path)
    assert manifest.version == 1
    assert {case.context_bucket for case in manifest.cases} == {
        "1k", "32k", "128k", "256k"
    }
    assert any(case.mode == "persistent_turn" for case in manifest.cases)
    assert any(case.mode == "fresh_prefill" for case in manifest.cases)


def test_manifest_rejects_unknown_generation_keys(tmp_path: Path) -> None:
    path = write_manifest(tmp_path, generation={"temperature": 0, "mystery": 1})
    with pytest.raises(ValueError, match="mystery"):
        load_manifest(path)
```

- [x] **Step 2: Run the tests and verify the workload module is absent**

Run: `python -m pytest tests/test_workloads.py -q`

Expected: FAIL importing `qwasar_bench.workloads`.

- [x] **Step 3: Implement strict workload types and loader**

Define `ModelPin`, `GenerationConfig`, `WorkloadCase`, and `BenchmarkManifest`. Reject duplicate case IDs, unsupported buckets, non-positive limits, unrecognized keys, and incompatible persistent-turn parent references. Hash the canonicalized manifest bytes with SHA-256.

- [x] **Step 4: Add the v1 benchmark manifest**

The manifest must pin:

```json
{
  "version": 1,
  "model": {
    "architecture": "Qwen3.8ForCausalLM",
    "repository": "Qwen/Qwen3.8-27B",
    "quantization": "EXL3",
    "target_bpw": 5.0,
    "native_context_tokens": 262144,
    "artifact_path_env": "QWASAR_MODEL_PATH",
    "artifact_sha256": null
  },
  "generation": {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_output_tokens": 256,
    "seed": 0
  }
}
```

Add deterministic coding, tool-call, long-context retrieval, multilingual technical, and persistent-turn smoke cases. Large fixtures may reference generated filler by seed and token target rather than storing hundreds of thousands of literal tokens.

- [x] **Step 5: Run workload tests**

Run: `python -m pytest tests/test_workloads.py -q`

Expected: PASS and a stable manifest SHA-256 assertion.

## Task 3: Capture Reproducible Machine and Artifact Metadata

**Files:**
- Create: `src/qwasar_bench/environment.py`
- Test: `tests/test_environment.py`

- [x] **Step 1: Write failing tests with an injected command runner**

```python
def test_environment_capture_parses_gpu_and_cpu(fake_commands) -> None:
    snapshot = capture_environment(run_command=fake_commands)
    assert snapshot.gpu.name == "NVIDIA GeForce RTX 5090"
    assert snapshot.gpu.total_memory_mib == 32607
    assert snapshot.cpu.model_name == "13th Gen Intel(R) Core(TM) i9-12900K"


def test_environment_capture_marks_nvidia_unavailable(fake_commands_without_gpu) -> None:
    snapshot = capture_environment(run_command=fake_commands_without_gpu)
    assert snapshot.gpu.available is False
    assert "NVIDIA" in snapshot.gpu.error
```

- [x] **Step 2: Run the tests and verify failure**

Run: `python -m pytest tests/test_environment.py -q`

Expected: FAIL importing `qwasar_bench.environment`.

- [x] **Step 3: Implement capability capture without shell interpolation**

Invoke commands as argument arrays. Capture `nvidia-smi` query fields, driver/CUDA version, clocks, power limit, temperature, GPU memory, `lscpu --json`, Python/platform details, current git revision, backend revision, manifest hash, and model artifact hash. Directory hashing must stream sorted relative paths and file bytes so the result is deterministic.

- [x] **Step 4: Make unavailable tools non-fatal**

Represent absent commands, permission errors, non-zero exits, and parse errors in structured capability fields. The `doctor` command will decide whether missing capabilities block a real GPU run.

- [x] **Step 5: Run environment tests**

Run: `python -m pytest tests/test_environment.py -q`

Expected: PASS.

## Task 4: Implement a Correct Responses API Streaming Client

**Files:**
- Create: `src/qwasar_bench/openai_client.py`
- Test: `tests/test_openai_client.py`

- [x] **Step 1: Write a failing fake-server integration test**

```python
def test_stream_response_collects_usage_and_first_delta(fake_responses_server) -> None:
    client = ResponsesClient(fake_responses_server.url)
    result = client.create_stream(
        model="qwen38",
        input_items=[{"role": "user", "content": "hello"}],
        max_output_tokens=16,
    )

    assert result.response_id == "resp_123"
    assert result.text == "hello world"
    assert result.input_tokens == 5
    assert result.output_tokens == 2
    assert result.first_delta_ns >= result.request_started_ns
    assert result.completed_ns >= result.first_delta_ns
```

- [x] **Step 2: Run the test and verify failure**

Run: `python -m pytest tests/test_openai_client.py -q`

Expected: FAIL importing `qwasar_bench.openai_client`.

- [x] **Step 3: Implement a bounded SSE parser and Responses client**

Use `urllib.request` with explicit connect/read timeout handling. Parse multiline `data:` fields, ignore comment heartbeats, reject oversized events, and stop only on a terminal Responses event or `[DONE]`. Record monotonic nanosecond timestamps at request start, headers, first output delta, and terminal event. Preserve raw event types for golden-fixture validation.

- [x] **Step 4: Cover failures and cancellation-shaped terminal events**

Test malformed JSON, missing terminal events, HTTP errors, `response.failed`, and `response.cancelled`. Never report a failed stream as a completed sample.

- [x] **Step 5: Run the client tests**

Run: `python -m pytest tests/test_openai_client.py -q`

Expected: PASS.

## Task 5: Execute Cases and Persist Atomic Run Artifacts

**Files:**
- Create: `src/qwasar_bench/runner.py`
- Test: `tests/test_runner.py`

- [x] **Step 1: Write failing runner tests around a fake client**

```python
def test_runner_reuses_previous_response_for_persistent_turn(tmp_path: Path) -> None:
    client = RecordingClient()
    run_manifest(load_fixture_manifest(), client, tmp_path / "run")
    assert client.calls[1]["previous_response_id"] == client.results[0].response_id


def test_run_directory_is_published_atomically(tmp_path: Path) -> None:
    output = tmp_path / "baseline-001"
    run_manifest(load_fixture_manifest(), RecordingClient(), output)
    assert (output / "run.json").is_file()
    assert (output / "samples.jsonl").is_file()
    assert not (tmp_path / ".baseline-001.tmp").exists()
```

- [x] **Step 2: Run the tests and verify failure**

Run: `python -m pytest tests/test_runner.py -q`

Expected: FAIL importing `qwasar_bench.runner`.

- [x] **Step 3: Implement sequential case execution**

Resolve fixture inputs, generate deterministic filler, maintain per-session `previous_response_id`, run warmups separately from measured repetitions, and derive TTFT, prefill throughput, accepted decode throughput, and speculative acceptance from raw backend counters when present. Mark unsupported counters as `null`, never zero.

- [x] **Step 4: Persist raw and summarized artifacts atomically**

Write into a sibling temporary directory, fsync files, create `run.json`, `environment.json`, `manifest.json`, `events/*.jsonl`, and `samples.jsonl`, then rename the directory into place. Refuse to replace an existing result directory.

- [x] **Step 5: Run runner tests**

Run: `python -m pytest tests/test_runner.py -q`

Expected: PASS.

## Task 6: Compare Baselines and Enforce Acceptance Gates

**Files:**
- Create: `src/qwasar_bench/compare.py`
- Test: `tests/test_compare.py`

- [x] **Step 1: Write failing comparison tests**

```python
def test_acceptance_requires_decode_and_large_prefill_improvements() -> None:
    report = compare_runs(baseline_run(), candidate_run())
    assert report.gates["decode_tps"].required_delta_percent == 15.0
    assert report.gates["prefill_tps_large_suffix"].required_delta_percent == 25.0
    assert report.accepted is True


def test_any_bucket_more_than_five_percent_slower_rejects_candidate() -> None:
    report = compare_runs(baseline_run(), candidate_with_slow_128k_bucket())
    assert report.gates["context_bucket_floor"].passed is False
    assert report.accepted is False
```

- [x] **Step 2: Run the tests and verify failure**

Run: `python -m pytest tests/test_compare.py -q`

Expected: FAIL importing `qwasar_bench.compare`.

- [x] **Step 3: Implement statistically explicit summaries**

Group only matching manifest hash, case ID, context bucket, generation parameters, artifact quality class, and hardware identity. Report count, median, p50, p95, arithmetic mean, standard deviation, and percent delta. Refuse comparisons with insufficient or incompatible samples.

- [x] **Step 4: Encode the approved gates**

Require at least 15% accepted decode tok/s improvement, 25% prefill improvement for appended suffixes of at least 8,192 tokens, warm persistent-turn p50 first delta at or below 200 ms for the defined 128K case, peak VRAM at or below 28.5 GiB, and no context bucket more than 5% slower. Quality gates consume a separately frozen score artifact and cannot be bypassed by missing data.

- [x] **Step 5: Run comparison tests**

Run: `python -m pytest tests/test_compare.py -q`

Expected: PASS.

## Task 7: Add the CLI and End-to-End Smoke Tests

**Files:**
- Create: `src/qwasar_bench/cli.py`
- Create: `src/qwasar_bench/__main__.py`
- Test: `tests/test_cli.py`
- Modify: `README.md`

- [x] **Step 1: Write failing CLI tests**

```python
def test_doctor_returns_two_without_required_gpu(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "capture_environment", unavailable_environment)
    assert cli.main(["doctor"]) == 2
    assert '"gpu_ready": false' in capsys.readouterr().out


def test_compare_writes_machine_readable_report(tmp_path: Path) -> None:
    output = tmp_path / "comparison.json"
    assert cli.main(["compare", "baseline", "candidate", "--output", str(output)]) == 0
    assert json.loads(output.read_text())["accepted"] is True
```

- [x] **Step 2: Run the tests and verify failure**

Run: `python -m pytest tests/test_cli.py -q`

Expected: FAIL importing `qwasar_bench.cli`.

- [x] **Step 3: Implement explicit subcommands**

Provide:

```text
python -m qwasar_bench doctor --manifest PATH
python -m qwasar_bench baseline --manifest PATH --base-url URL --output DIR
python -m qwasar_bench compare BASELINE_DIR CANDIDATE_DIR --output FILE
python -m qwasar_bench validate-manifest PATH
```

Use exit code `0` for success, `2` for invalid environment/input, and `3` for a completed comparison that fails acceptance.

- [x] **Step 4: Document exact local setup and artifact contracts**

Document editable install, test command, model path environment variable, backend endpoint expectations, result layout, exit codes, and the rule that a real baseline is valid only when `doctor` reports the RTX 5090 and all pin hashes match.

- [x] **Step 5: Run the complete CPU-safe suite**

Run: `python -m pytest -q`

Expected: PASS.

## Task 7A: Bridge the Existing ExLlamaV3 Chat Server Without Losing Session State

**Files:**
- Modify: `src/qwasar_bench/openai_client.py`
- Modify: `src/qwasar_bench/cli.py`
- Test: `tests/test_chat_client.py`
- Modify: `README.md`

- [x] **Step 1: Write failing integration tests against a local Chat Completions server**

Verify that the client expands `previous_response_id` into the complete prior message chain, preserves assistant content and reasoning, derives exact prompt/output token counts from the single-user `/health` counter deltas, and exposes the same `StreamResult` contract as the Responses client.

- [x] **Step 2: Run the focused tests and verify the client is absent**

Run: `python -m pytest tests/test_chat_client.py -q`

Expected: FAIL importing `ChatCompletionsClient`.

- [x] **Step 3: Implement the bounded stateful compatibility client**

Keep response history in process memory, reject unknown parents, serialize exactly one request at a time, parse standard Chat Completions SSE plus the donor's `reasoning_content` and tool-call deltas, and fail if `/health` counters move backwards or report an impossible delta. Do not label TTFT as model-internal prefill time.

- [x] **Step 4: Expose the protocol as an explicit baseline option**

Add `--protocol responses|chat-completions` to `baseline`. Default to `responses`; selecting `chat-completions` constructs the compatibility client and records the protocol in backend identity.

- [x] **Step 5: Run focused and complete tests**

Run: `python -m pytest tests/test_chat_client.py tests/test_cli.py tests/test_runner.py -q`

Expected: PASS.

Run: `python -m pytest -q`

Expected: PASS.

---

## Task 8: Pin the ExLlamaV3 Oracle and Collect the First RTX 5090 Baseline

**Files:**
- Create: `benchmarks/backends/exllamav3-5bpw.json`
- Create: `scripts/run_exllamav3_baseline.sh`
- Create after successful hardware run: `benchmarks/baselines/<timestamp>-exllamav3-5bpw/`
- Modify: `README.md`

- [ ] **Step 1: Record the donor and artifact identities**

Capture the exact ExLlamaV3 git/package revision, Python version, CUDA extension build metadata, target model path, target artifact SHA-256, DFlash2 draft path/hash, cache quantization, context length, graph settings, and launch command. A missing hash blocks acceptance-grade collection.

- [x] **Step 2: Add a fail-fast baseline launcher**

The script must use `set -euo pipefail`, call `doctor`, start or validate the configured Responses-compatible backend, wait on a health endpoint with a bounded timeout, run the frozen manifest, and preserve backend logs beside the run artifact. It must not download models or mutate the Python environment implicitly.

- [ ] **Step 3: Validate without a GPU**

Run: `bash -n scripts/run_exllamav3_baseline.sh`

Expected: PASS syntax validation.

Run: `python -m qwasar_bench doctor --manifest benchmarks/manifests/qwen38-27b-rtx5090-v1.json`

Expected before the source pin, local artifact hash, and engine revision are all present: exit `2` with a structured explanation. A visible non-RTX-5090 device or unavailable NVIDIA driver also blocks collection.

- [ ] **Step 4: Collect the real baseline on the RTX 5090 host**

Run: `scripts/run_exllamav3_baseline.sh`

Expected: one immutable run directory containing environment, manifest, raw SSE events, samples, backend logs, and a completed summary for every required context bucket.

- [ ] **Step 5: Verify baseline integrity**

Run: `python -m qwasar_bench validate-run benchmarks/baselines/<timestamp>-exllamav3-5bpw`

Expected: PASS with zero missing cases, zero incompatible pins, no concurrent requests, peak VRAM recorded, and all raw timing inputs present.

## Final Verification

- [ ] Run: `python -m pytest -q`

Expected: all CPU-safe unit and integration tests pass.

- [ ] Run: `python -m qwasar_bench validate-manifest benchmarks/manifests/qwen38-27b-rtx5090-v1.json`

Expected: prints the manifest SHA-256 and exits `0`.

- [ ] Run: `bash -n scripts/run_exllamav3_baseline.sh`

Expected: exits `0`.

- [ ] Run: `git diff --check`

Expected: exits `0` with no whitespace errors.

- [ ] Run: `git status --short`

Expected: only intentional M0 files are present; no generated caches, temporary directories, or local model artifacts are tracked.
