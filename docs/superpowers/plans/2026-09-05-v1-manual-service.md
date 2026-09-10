# Qwasar v1 manual service: hybrid delivery plan

Status: implemented and verified with Pi 0.84.4. See the execution ledger in
`2026-09-05-pi-v1-execution.md` and acceptance report
`../../benchmarks/2026-09-05-v1-pi-acceptance.md`.
This is the M1 hybrid service in the approved design, not completion of the
future Q38X native C++/CUDA runtime. Do not wait for further kernel sweeps.

Baseline handoff: `docs/benchmarks/2026-09-05-prefill-application-results.md`.
Keep 5 bpw + MTP + K8/V4 and an explicit baseline/Flash selector. The faster
prefill is not a general quality upgrade: both routes failed two of eight
tool cycles, and Flash regressed on one previously correct long cycle.

## Fixed constraints

- Rust supervisor, independent Python/ExLlamaV3 GPU worker; one active job.
- Existing verified EXL3 5 bpw artifact, MTP, K8/V4, RTX 5090 only.
- Native 262,144-token limit includes exact prompt, maximum output and
  speculative reserve. No silent truncation, CPU offload or model switching.
- Explicit baseline/Flash profile at startup; record it with runtime identity.
  No automatic mid-request attention fallback.
- Loopback HTTP by default. Tools execute in the client, never implicitly
  in the engine. Unsupported API inputs produce errors, not silent ignores.
- No donor modifications, Git commits or new branches without user request.

## 1. Worker and protocol

Extract runtime loading and incremental Job execution from the probes into
a service worker without benchmark file writes. Version the local control
protocol: ready/config, generate, delta, terminal, cancel and shutdown.
Keep IPC separate from stdout/stderr diagnostics. Exactly one owner mutates
the generator; cancellation is checked between iterations, with job serial
tracking rather than relying on an obsolete object after a donor requeue.

Tests: fake-runtime start/stop/error/cancel state transitions, fragmented
messages, bounded output/backpressure, real GPU short text and clean shutdown.

## 2. Durable response state

SQLite WAL stores response IDs/parents, status, structured input/output,
sampling, exact generated token segments and runtime/template/model hashes.
Commit terminal state before exposing a response as a usable parent.
Incomplete, failed or cancelled outputs remain diagnostic, not valid parents.

Retain exact generated IDs; never reconstruct generated history by decoding
and re-encoding text. Reuse the live prefix for append-only turns. Branches,
instruction/schema changes and restart may require correct reconstruction.
`instructions` are not implicitly inherited from `previous_response_id`.

Tests: append continuity, instruction changes, branches, duplicate/idempotent
requests, failed-parent rejection, atomic commit and process-restart recovery.
Disk durability does not imply that GPU state survives restart or that cold
reconstruction meets the warm-turn latency target.

## 3. Streaming and tools

Implement the focused Responses surface: create, retrieve and cancel;
streaming and non-streaming; reasoning, final text and complete function
calls. Validate schemas/call IDs and accept tool results only for pending
calls of the correct parent. Parse native delimiters across chunk boundaries.
Do not leak partial XML as final text or invent terminators for capped output.

Add the approved Chat Completions adapter over the same generation/session
core, not a second engine. Document the precise supported subset. Prioritize
client-specific compatibility after the user's manual-client preference.

Tests: fragmented reasoning/tool markers, malformed arguments, unknown calls,
SSE ordering/terminal events, disconnect cancellation and oversized requests.

## 4. Launcher and manual client

Add a Qwasar-owned start command with safe GPU ownership checks and explicit
5 bpw/MTP/K8V4 defaults, plus health/model/effective-config endpoints. Provide
a terminal smoke client with session continuation/reset, streamed text,
optional reasoning, explicit tool-result submission and Ctrl-C cancellation.

Delivery ruling: the user selected Pi as the manual client. The Pi provider,
launcher and end-to-end coding/session verification replace a redundant custom
terminal UI; Pi owns tool execution and its interactive session controls.

Expose TTFT, first final content, physical cache reuse/prefill, decode rate,
completion reason and context usage. Null uncertain counters rather than
publishing fabricated values after requeue or failure.

## 5. Delivery verification

Run CPU tests, then real HTTP/GPU tests for a conversation with two tool
cycles, cancellation followed by a successful request, worker restart,
branch/instruction changes and rejection near the native limit. Exercise
the chosen client end to end and preserve a reproducible smoke transcript.

Document known model/tool failures and cold-prefill costs. Deliver the exact
start command and client setup. This makes v1 manually testable; it does not
certify BF16 parity, arbitrary agent quality or every response under 30 s.

## Deferred beyond this deliverable

Q38X conversion, native C++/CUDA worker, megakernels, alternate quants,
DFlash2 experiments, sparse attention and physical host-KV checkpoint mirrors.
Correct reconstruction from durable token history is the hybrid recovery
path; speeding that up must not block the first usable service.
