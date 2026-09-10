# Hybrid v1 with Pi

Qwasar runs a Rust HTTP/SQLite supervisor and one Python ExLlamaV3 worker.
The worker exclusively uses GPU 0 (RTX 5090). It does not execute tools:
Pi executes `read`, `write`, `edit`, and `bash` in the project you open.

## This machine

```bash
cd /home/rekeyea/Documents/llm/qwasar
python3 scripts/configure_pi.py
python3 scripts/qwasar.py start
python3 scripts/qwasar.py status
```

Standalone startup builds the release executable using the locked Cargo
dependencies; the installed systemd unit uses the already-built release binary.
The service verifies the complete frozen model artifact before
loading. Allow several minutes for compilation, hashing and GPU initialization.
The launcher refuses to take over another substantial compute process on GPU 0;
it does not stop other inference services. GPU 1 is never selected.

To use Pi, start it **from the project you want it to work on**:

```bash
cd /path/to/your/project
/home/rekeyea/Documents/llm/qwasar/scripts/pi_qwasar.sh
```

Alternatively select provider `qwasar`, model `qwasar-qwen38-27b` in Pi.
The wrapper pins the tested Pi 0.84.4 binary without invoking the mise updater;
set `PI_BIN` to override it. Additional Pi arguments are passed through,
including `--continue` and `--thinking off`. Pi owns its session files; the
server matches replayed assistant history to its durable exact token segments.

The installer adds only the Qwasar provider to `~/.pi/agent/models.json`.
It preserves other providers, makes a byte-exact protected backup, and refuses
to replace a different existing Qwasar entry. It does not touch credentials.
Use `--target /temporary/models.json` for an isolated configuration.

## Service controls

On this machine, [systemd automatic startup](servicio-systemd.md) is installed.
The launcher delegates `start`, `stop`, `status`, and `logs` to that unit. Its
persistent profile is Flash; use `systemctl --user restart qwasar.service` to
reload the worker. Changing to baseline requires an explicit unit edit and
daemon reload, not the standalone command below.

For standalone installations without the systemd unit:

```bash
python3 scripts/qwasar.py logs
python3 scripts/qwasar.py stop
python3 scripts/qwasar.py start --prefill baseline
```

`start` keeps an already-running instance; changing profile requires `stop`
then `start`. The selected Flash path uses 8192-token prefill chunks; `baseline`
is the reference fallback. Both use the same pinned EXL3 5 bpw weights, MTP
drafting and K8/V4 cache. Alternative weights intentionally fail the artifact
hash check: switching quantization is not an unvalidated path substitution.

Logs, PID identity and the SQLite WAL database live under ignored `state/`.
With systemd, process ownership belongs to the unit and new logs go to
`journalctl --user -u qwasar.service`; the old manual PID/log files are historical.
The database contains prompts, generated reasoning, tool results and token
histories in plaintext. Treat it as private project data. It is not encrypted
and v1 has no automatic history pruning. Stop the service before making a
simple file-level backup; a live SQLite backup must include WAL semantics.

HTTP is restricted to loopback, default `http://127.0.0.1:8800/v1`.
It has no authentication boundary against other local processes. The dummy
Pi API key is compatibility configuration, not a secret or access control.
Do not expose this endpoint through an unauthenticated proxy.

## Supported API

- `GET /health`, `/config`, `/v1/models`: worker readiness, effective profile,
  runtime identity and model information. HTTP health is available while loading;
  inspect `worker.status == "ready"`, not just HTTP 200.
- `POST /v1/chat/completions`: Pi-compatible text, reasoning and tool-call SSE;
  full conversation replay selects matching committed history automatically.
- `POST /v1/responses`: focused text/function subset, including
  `previous_response_id`, streaming, and fresh per-request instructions.
- `GET /v1/responses/{id}` and `POST /v1/responses/{id}/cancel`:
  retrieve persisted responses and cancel active work.
- `Idempotency-Key` replays the same request without generating it again;
  reuse with a different request returns a conflict.

Only one request generates at a time. Concurrent requests receive 409 rather
than waiting in an invisible queue. Unsupported controls are rejected rather
than silently ignored. Images, audio, structured-output constraints and general
Responses API feature parity are not part of v1.

Tool schemas support `patternProperties`, including nested argument maps used
by Pi's MCPorter extension. All matching patterns and explicit properties are
validated; `additionalProperties` applies only to unmatched names, following the
[JSON Schema object rules](https://json-schema.org/understanding-json-schema/reference/object#patternproperties).

Pi uses `openai-completions` (the Chat Completions API), Qwen template thinking,
262144 total positions and a 32768-token output allowance (the current API
maximum). This allowance includes reasoning and generated tool arguments.
Thinking `off` disables
reasoning; enabled Pi thinking modes select the worker's medium policy. Direct
API requests can select the exposed thinking controls. Generated tool arguments
are buffered until a complete, schema-valid call is available; ordinary text and
reasoning stream incrementally. Pi executes tools with your local permissions.

Pi's own default automatic compaction reserves 16384 tokens, so it normally
compacts around 245760 estimated context tokens instead of filling every native
position. Qwasar does not alter that setting. Compaction changes the prompt and
can require substantial new prefill; it is not append-only cache reuse.

With the 32768-token output allowance, Qwasar requires the rendered input to
fit within 229360 tokens (262144 minus 32768 minus the 16-token runtime reserve).
Pi's default compaction threshold does not guarantee this. Compact earlier when
approaching that limit; the engine rejects oversized input plus output budgets
rather than silently reducing the requested output. Opening `/model` in Pi
reloads `models.json`; reselect Qwasar to use an updated model configuration.

Completed turns commit exact generated token IDs, including actual model
terminators, before successful terminal events. Incomplete, cancelled or failed
responses are not eligible parents. Cancellation discards physical worker cache
state to avoid reusing uncertain recurrent state. Restart restores durable token
history but must rebuild GPU cache. Request budget checking reserves output and
speculation space inside the native context instead of silently truncating it.

## Performance expectations

Warm continuation and cold ingestion are different workloads. Prior direct
probes reached roughly 85 tokens/s near 256K and warm TTFT below one second for
small appended turns, but cold 257K prefill took about 132 seconds with Flash.
These are workload-specific screening results, not a blanket HTTP or Pi SLA.
Thinking, long tool outputs, cache eviction, restart and Pi compaction can all
increase visible response time. A 32768-token output budget is not a promise to
finish within 30 seconds. See the
[prefill application report](benchmarks/2026-09-05-prefill-application-results.md).

Flash changes numerical results and has not demonstrated BF16 parity. The
model can still select incorrect tools or arguments; strict transport parsing
does not establish model accuracy. Do not confuse a working integration smoke
test with broad coding-quality certification.

## Verification

For rejected tool-call capture and CPU-only replay, see
[Tool diagnostics](diagnostico-tools.md). Captures are private and bounded;
an already-running worker needs a restart to enable the instrumentation.

```bash
cargo test --locked --offline
cargo build --locked --offline
CUDA_VISIBLE_DEVICES='' QWASAR_TEST_MODEL=/home/rekeyea/models/Qwen3.8-27B-EXL3-5.0bpw \
  PYTHONPATH=src:.venv/lib/python3.12/site-packages \
  ../qwen38-exl3-mia/.venv/bin/python -m pytest -q
python3 scripts/verify_pi.py
python3 scripts/verify_pi_session.py
python3 scripts/verify_service.py --restart
CUDA_VISIBLE_DEVICES='' ../qwen38-exl3-mia/.venv/bin/python scripts/verify_long_context.py
```

The final four commands require the real service. The Pi check opens an isolated Pi
configuration and disposable coding project, requests all four tools, verifies
the resulting tests independently, and saves evidence under `results/pi-v1`.
The service check tests cache reuse, exact token persistence, changed headers,
cancellation, budget rejection and durable restart. The long-context check uses
the archived nonrepeated source corpus and measures HTTP cold/warm continuation;
it is a throughput/transport check, not a coding-quality evaluation. Run these
checks sequentially while Pi is idle, since only one request can run at a time.
The [v1 acceptance report](benchmarks/2026-09-05-v1-pi-acceptance.md) records the
real measurements and the limits of the completed verification.
