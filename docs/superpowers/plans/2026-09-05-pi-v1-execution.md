# Pi-ready hybrid v1 implementation

User approval: implement the previously presented v1 and make it usable in
the locally installed Pi 0.84.4. This executes the existing hybrid design;
it does not introduce Q38X, alternate weights or another kernel sweep.

## Global constraints

Rust supervisor + Python ExLlamaV3 worker, EXL3 5 bpw + MTP + K8/V4,
262144 native positions, one active generation, loopback only. GPU 0 only;
GPU 1 and donor installation untouched. No commits, branches or deletion of
existing user configuration. Tools execute in Pi, not in the engine.

## Concrete protocol and ownership

The Rust supervisor owns HTTP, SQLite WAL response durability and the child
worker lifecycle. The worker owns tokenizer/template execution and CUDA.
Version 1 newline-delimited JSON uses worker stdin/stdout; all library logs
go to stderr. Request IDs correlate every event. A dedicated input reader
can set cancellation while the only GPU owner iterates a job.

Worker startup: `python -m qwasar_runtime.worker --model PATH --prefill
baseline|flash --context-size 262144`. Stdout starts with
`{"type":"ready","protocol":1,"config":{...}}` after load.

Supervisor sends:

```json
{"op":"generate","id":"resp_...","request":{"messages":[],"tools":[],"max_tokens":4096,"temperature":1.0,"top_p":0.95,"seed":42,"thinking":"medium","tool_choice":"auto"},"parent":null}
```

`parent` is the previous terminal snapshot (opaque JSON described below),
selected by explicit response ID or longest matching canonical chat history.
The worker validates the full request and budget before enqueue. Cancellation:
`{"op":"cancel","id":"resp_..."}`; shutdown: `{"op":"shutdown"}`.

Worker events:

- `{"type":"delta","id":"...","channel":"reasoning|content","text":"..."}`.
- `{"type":"terminal","id":"...","status":"completed|incomplete|cancelled|failed","message":{"role":"assistant","content":"...","reasoning_content":"...","tool_calls":[]},"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0,"prompt_tokens_details":{"cached_tokens":0}},"metrics":{},"snapshot":{},"error":null}`.
- Terminal `error`, when present, has `code`, `message`, and integer
  `http_status`. Failed/cancelled/incomplete terminals have no usable snapshot.

A completed `snapshot` contains canonical `messages` including the generated
assistant, exact `tape` token IDs, preserved generated `segments`, and runtime
identity. The supervisor treats it as opaque except `messages` for matching.
Changed headers/schemas use a fresh prompt while retaining known generated
segments; matching append-only requests use the exact tape. No re-encoding
generated history or inventing terminators for incomplete generations.

Tool calls use OpenAI Chat form with stable generated IDs, function name and
JSON-string arguments. The worker validates the native XML against declared
schemas, including duplicate/missing parameters and call/result correspondence.
It never executes tools. Completed calls may be buffered until complete;
reasoning and ordinary final text stream incrementally without XML leakage.

## Tasks

- [x] Python runtime: literal-safe rendering, exact-history snapshots,
  incremental reasoning/tool parser, schema validation, cancel/reset, metrics,
  model loading and JSONL worker; CPU tests before implementation.
- [x] Rust service: persistent child manager, bounded streams, SQLite commits,
  one-active-request guard, Chat Completions + focused Responses adapter,
  cancellation/recovery, health/models/config, request validation and tests.
- [x] Pi integration: isolated provider config/launcher first; preserve existing
  global models and credentials, merge only the Qwasar provider after backup if
  useful. Local endpoint port 8800, model `qwasar-qwen38-27b`.
- [x] Guarded service lifecycle scripts, documentation and manual commands.
- [x] Real HTTP/Pi/GPU verification: streaming, tools that read/edit/write/test
  a disposable fixture, persistent continuation, cancel followed by success,
  restart, branch/header changes and context-limit rejection. Save evidence.

## Implementation rulings

- Ruling: tokenizer/template work stays inside the Python worker in hybrid
  v1, while durability remains in Rust. Reusing the pinned tokenizer avoids
  a second tokenizer implementation; moving it later costs an IPC migration.
- Ruling: generated token segments, rather than physical KV mirrors, provide
  durable reconstruction after restart. Recovery is correct but may require
  cold prefill; no instant-restart or cold-256K 30-second guarantee.
- Ruling: native tool-call argument deltas may be emitted as one complete
  validated JSON fragment. Pi receives valid structured calls, but does not
  see partially generated arguments in this first version.

## Progress ledger

Implementation and independent reviews complete. 275 Python tests and 13 Rust
tests pass, including the real Hugging Face wrapper and duplicate-startup
ownership regressions. Real HTTP/Pi acceptance passed: all four coding tools,
external fixture oracle, Pi session continued in a new process, exact tape
continuation, changed headers, cancellation/recovery, native budget rejection
and durable server restart. At 254098 input tokens, one warm HTTP sample returned
first content in 593ms and the complete 950-token output in 10.412s, at 97.19
steady tokens/s; cold ingestion took 129.208s. See
`docs/benchmarks/2026-09-05-v1-pi-acceptance.md` for scope and evidence.
Pi is configured with an exact backup; the managed real service is left running.
Existing benchmark results and donor remain pinned.
