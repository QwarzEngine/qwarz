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
The service verifies the complete frozen EXL3 artifact and the NVIDIA MLP
donor shard sizes before loading. Allow several minutes for hashing, donor
substitution and GPU initialization. After this promotion, committed token
snapshots from the previous EXL3-only identity are not reused: those turns
re-render from text until new snapshots accumulate.
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
including `--continue` and `--thinking off`. The wrapper starts Qwasar at
`--thinking xhigh`. Pi owns its session files; the
server matches replayed assistant history to its durable exact token segments.

The installer adds only the Qwasar provider to `~/.pi/agent/models.json`.
It preserves other providers, makes a byte-exact protected backup, and refuses
to replace a different existing Qwasar entry. It does not touch credentials.
Use `--target /temporary/models.json` for an isolated configuration.

To use Codex with native images, from the project you want it to work on:

```bash
/home/rekeyea/Documents/llm/qwasar/scripts/codex_qwasar.sh
```

`python3 scripts/configure_codex.py` writes `~/.codex/qwasar-catalog.json` with
`input_modalities: ["text","image"]`, keeps `codex -p qwasar` pointed at
`http://127.0.0.1:8800/v1`, and leaves other Codex profiles alone. Codex rejects
attachments when the catalog is text-only; after this registration, pasted
images go through Responses `input_image` data URLs, and `view_image` arrives
as `function_call_output` whose `output` is an `input_image` part. Use
`--codex-home /temporary/codex` for an isolated configuration.

To use OpenCode against the same engine, from the project you want it to work on:

```bash
/home/rekeyea/Documents/llm/qwasar/scripts/opencode_qwasar.sh
```

`python3 scripts/configure_opencode.py` upserts provider `qwasar` in
`~/.config/opencode/opencode.json` as `@ai-sdk/openai-compatible` at
`http://127.0.0.1:8800/v1`, model `qwasar-qwen38-27b`, with `modalities.input`
`["text","image"]` so attachments are not stripped client-side. It keeps other
providers and the current default `model`, makes a byte-exact backup, and
refuses a conflicting `qwasar` entry. Use `--target /temporary/opencode.json`
for an isolated configuration. Reload OpenCode after changing the file.

To use OMP (Oh My Pi) against the same engine, from the project you want it to work on:

```bash
/home/rekeyea/Documents/llm/qwasar/scripts/omp_qwasar.sh
```

`python3 scripts/configure_omp.py` upserts provider `qwasar` in
`~/.omp/agent/models.yml` as `openai-completions` at `http://127.0.0.1:8800/v1`,
model `qwasar-qwen38-27b`, with `input: [text, image]` and Qwen thinking
(`thinkingFormat: qwen-chat-template`, `high`/`max` → `xhigh`). It keeps other
providers. If `tiny`/`smol` are missing or also point at Qwarz, they are set
to `openai-codex/gpt-5.5:off` so OMP's parallel title generator does not 409
the single worker. `modelRoles.default` is left alone. Use
`--target /temporary/models.yml` for an isolated configuration.

To use Hermes against the same engine, from the project you want it to work on:

```bash
/home/rekeyea/Documents/llm/qwasar/scripts/hermes_qwasar.sh
```

`python3 scripts/configure_hermes.py` upserts provider `qwasar` in
`~/.hermes/config.yaml` as `chat_completions` at `http://127.0.0.1:8800/v1`,
model `qwasar-qwen38-27b`, with `supports_vision` and `supports_reasoning` so
attachments are sent natively. It keeps other providers and the current
`model.default`. If `qwen38-local` is already registered and title generation
is `auto` or `qwasar`, the installer pins `auxiliary.title_generation` to
`qwen38-local`/`qwen3.8-27b` so Hermes' parallel title call does not 409 the
single worker. Use `--target /temporary/config.yaml` for an isolated
configuration. Mid-session: `/model custom:qwasar:qwasar-qwen38-27b`.

To use Qwen Code against the same engine, from the project you want it to work on:

```bash
/home/rekeyea/Documents/llm/qwasar/scripts/qwen_code_qwasar.sh
```

`python3 scripts/configure_qwen_code.py` upserts `qwasar-qwen38-27b` in
`~/.qwen/settings.json` as an OpenAI-compatible model at
`http://127.0.0.1:8800/v1`, with `modalities.image` and
`capabilities.reasoning.profile: qwen-chat-template` so thinking rides
`chat_template_kwargs.enable_thinking`. It keeps other `modelProviders.openai`
entries and the current default `model`, writes `env.QWASAR_API_KEY`, makes a
byte-exact backup, and refuses a conflicting `qwasar-qwen38-27b` entry. If
another OpenAI-compatible model exists and `fastModel` is empty or still
points at Qwarz, the installer pins it to that other model (`spark-x2.5-4b`
or `qwen3.8-27b` when present) so Qwen Code's parallel memory selector does
not 409 the single worker. Use `--target /temporary/settings.json` for an
isolated configuration. Mid-session: `/model` → `qwasar-qwen38-27b`.

To use Claude Code against the same engine, from the project you want it to work on:

```bash
/home/rekeyea/Documents/llm/qwasar/scripts/claude_qwasar.sh
```

`python3 scripts/configure_claude.py` writes `~/.claude/qwasar.env` and does
not edit `~/.claude/settings.json`. The wrapper sources that file so only that
Claude Code process uses Qwarz. It speaks Anthropic Messages
(`POST /v1/messages`) with `system`, `tool_use` / `tool_result`, base64
`image` blocks, and `thinking`. `claude-*` model ids are accepted and echoed.
`POST /v1/messages/count_tokens` estimates tokens without the worker. Use
`--target /temporary/qwasar.env` for an isolated configuration.

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
then `start`. The selected Flash path is NVIDIA64 + Flash/8192 + FP8 PRIMS
(Q≥8192) + Attention64. `baseline` is the EXL3 Triton fallback. Both keep the
pinned EXL3 5 bpw artifact, MTP drafting and K8/V4 cache; Flash additionally
substitutes the 192 pinned NVIDIA NVFP4 MLP matrices. Alternative EXL3 weights
fail the artifact hash check. NVIDIA shards are size-checked against the
recorded pin, not rehashed on every start.

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
- `POST /v1/messages` and `POST /v1/messages/count_tokens`: Anthropic Messages
  for Claude Code (`ANTHROPIC_BASE_URL=http://127.0.0.1:8800`). Thinking,
  `tool_use` / `tool_result`, and inline base64 images are translated onto the
  same worker request as Chat Completions. Errors use Anthropic's
  `{"type":"error","error":{...}}` envelope. `count_tokens` does not dispatch
  generation.
- `POST /v1/responses`: focused text/function subset, including
  `previous_response_id`, streaming, and fresh per-request instructions.
- `GET /v1/responses/{id}` and `POST /v1/responses/{id}/cancel`:
  retrieve persisted responses and cancel active work.
- `Idempotency-Key` replays the same request without generating it again;
  reuse with a different request returns a conflict.

Only one request generates at a time. Concurrent requests receive 409 rather
than waiting in an invisible queue. Unsupported controls are rejected rather
than silently ignored. Audio, video, structured-output constraints and general
Responses API feature parity are not part of v1.

## Images

Qwen3.8-27B is natively multimodal and the pinned EXL3 artifact ships its BF16
vision tower, so the worker loads it next to the text model (about 0.9 GB of
VRAM). User messages may carry inline images as Chat Completions `image_url`
parts or Responses `input_image` parts. Only base64 **data URLs** are accepted
(`data:image/png;base64,...`); remote URLs are never fetched. Supported types:
PNG, JPEG, WebP, GIF. Limits: 8 MiB per image, 16 unique images per request.
Duplicate hashes count once. If Codex keeps attaching screenshots past that,
the oldest unique images are dropped so the turn can continue. Images in
`system` or `assistant` messages are rejected. Images inside tool results —
Chat Completions `image_url` or Codex `function_call_output` `input_image`
(string or `{url}` data URL) — are hoisted onto a following user message so
the Qwen template can see the pixels.

Each image costs roughly `pixels / 1024` prompt tokens after resizing to at
most `QWASAR_MAX_IMAGE_PIXELS` (default 2,359,296; a 1920×1080 frame is not
downscaled and takes about 2,040 tokens). Those tokens count toward the native
262,144-position budget and appear in `usage.prompt_tokens`.

Image bytes are stored once in SQLite by SHA-256; durable snapshots and
`history_key` only reference the hash, so a Responses `previous_response_id`
turn does not need to resend earlier images. Chat Completions clients resend
the full conversation anyway. The worker keeps the last
`QWASAR_IMAGE_CACHE` (default 32) embeddings in memory, which is what keeps
prefix reuse working across turns that contain the same image; after a worker
restart the first turn re-embeds and re-prefills. Measured sample: a 1280×720
PNG cost 880 prompt tokens, cold first content 1.19 s, and the follow-up turn
reused 768 cached tokens with 108 ms to first content. Vision quality on the
5 bpw + NVFP4 MLP stack has not been benchmarked; see
`docs/superpowers/plans/2026-09-14-vision.md`.

Tool schemas support `patternProperties`, `propertyNames`, `prefixItems` and
`format`, including nested argument maps used by Pi's MCPorter extension and
Claude Code. All matching patterns and explicit properties are validated;
`propertyNames` is applied to each key; `prefixItems` validates tuple positions
and `items` applies to the rest (`false` forbids extra elements);
`additionalProperties` applies only to unmatched names, following the
[JSON Schema object rules](https://json-schema.org/understanding-json-schema/reference/object#patternproperties).
Known `format` values (`email`, `date-time`, `uri`, `uuid`, and the other draft
2020-12 names) are asserted; unknown format names are kept as annotations.

Pi uses `openai-completions` (the Chat Completions API), Qwen template thinking,
262144 total positions and a 32768-token output allowance (the current API
maximum). If the client omits `max_output_tokens` / `max_tokens` (Codex does this), the
service uses the native **32768** output budget so a `high`/`xhigh` reasoning
turn is not cut early. This allowance includes reasoning and generated tool arguments.
Thinking `off` disables reasoning. Enabled efforts (`low`, `medium`, `xhigh`)
keep thinking on; `xhigh` stays available. The Qwasar Pi provider sends `reasoning_effort` (`xhigh` by default in
`pi_qwasar.sh`; `high`/`max` map to `xhigh`). Hermes title generation sends
`reasoning_effort=none` and a JSON `response_format`; `none` maps to `off` and
the format is ignored (Hermes still parses free text). `ultra` maps to `xhigh`.
Native `vision_analyze` tool results that carry an `image_url` are hoisted onto
a following user message so the Qwen template can see the pixels. OMP's Qwen
template dialect sends `chat_template_kwargs.reasoning_effort` (plus
`enable_thinking`); that effort is honored. A request that only sets
`chat_template_kwargs.enable_thinking=true`
still selects **medium**. Several live
sessions used `max_tokens=8192` with thinking on; without a cap that budget was spent
almost entirely on reasoning. The worker now caps reasoning so a content reserve
remains inside `max_tokens` (1024 tokens, or half the budget when the request is
smaller). Default caps are 1024 / 2048 / 8192 for low / medium / `xhigh`.
`reasoning_budget_tokens` overrides the cap; `0` disables it. A cap stop, or a
model that emits `im_end` before `</think>`, injects the native close tokens and
continues the same request without resetting GPU cache.
After a tool result, the tools header stays in the cached prefix. The worker
appends a suffix-only hint: answer if the results are enough, otherwise call the
next tool. It does not forbid the normal tool loop. Prefer `edit`/`write` over
rewriting a file with bash. A native `<tool_call>` cut off mid-stream, or a
schema-invalid call that cannot be coerced, is `incomplete` rather than a failed
generation that wipes GPU cache. Object-shaped `ask_user_question` options
(`{label}` / `{title}`) are accepted as strings. Generated tool arguments are
buffered until a complete, schema-valid call is available; ordinary text and
reasoning stream incrementally. Pi executes tools with your local permissions.

Incomplete turns carry the reason in `qwasar_metrics.incomplete_reason`:
`max_new_tokens` and `reasoning_budget` are real output-budget stops and map to
`finish_reason: "length"` (**max_tokens** on the Anthropic wire) with
`incomplete_details.reason: "max_output_tokens"` on Responses. Rejected calls map
to `undeclared_tool`, `tool_choice_mismatch`, `missing_tool_call`,
`malformed_tool_call`, `invalid_tool_arguments` or `unclosed_tool_call`, and an
`im_end` before `</think>` maps to `unterminated_reasoning`. Those are **not**
truncation: `finish_reason` stays `stop` / `end_turn` and `incomplete_details` is
null. The reason matters because agent harnesses treat `length` as context
pressure: OMP removes the assistant turn and runs recovery compaction on every
`length` stop, so a single rejected call could loop compaction until it gives up
("Compaction freed too little context..."). `qwasar_metrics.tool_error` adds a
bounded, value-free summary (`stage`, `tool`, `call_index`, `error`); full
evidence stays under `state/tool-errors` for `diagnostico-tools.md` replay.

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
small appended turns. Cold near-256K prefill on the promoted stack was about
67 seconds (133 seconds on the previous EXL3+Flash control). See the
[phase-2 gate](benchmarks/2026-09-11-phase2-results.md). These are
workload-specific screening results, not a blanket HTTP or Pi SLA.
Thinking, long tool outputs, cache eviction, restart and Pi compaction can all
increase visible response time. A 32768-token output budget is not a promise to
finish within 30 seconds. See the
[prefill application report](benchmarks/2026-09-05-prefill-application-results.md).

Flash changes numerical results and has not demonstrated BF16 parity. The
model can still select incorrect tools or arguments; strict transport parsing
does not establish model accuracy. Do not confuse a working integration smoke
test with broad coding-quality certification.

Since 2026-09-21 the service runs the promoted F4b stack: `--prefill xqa`
(XQA decode attention over an NVFP4 one-level KV cache, per-layer decode
CUDA graphs, PRIMS prefill) plus the rendezvous fast paths (batched verify
window, GPU-chained MTP draft walk, shared embedding table on GPU, ~+2.5 GB
VRAM). The 37-cell gate passed 4/4 with the rendezvous on both arms
(coding 24/32 vs 23/32, json 4/4, acceptance +0.3 pp, peak VRAM −0.34 GiB);
matrix decode delta: median +15.8% for the candidate, +22 to +51% at 256K,
TTFT par. `QWASAR_RDZ=0` disables the rendezvous path and
`QWASAR_RDZ_EMB=0` keeps the embedding on CPU; the active state is reported
in `/health` as `worker.config` (`decode_attention`, `target_cache`,
`rendezvous`). Rollback to the previous stack: set `--prefill flash` in
`integrations/systemd/qwasar.service`, `daemon-reload`, restart. See the
[rendezvous campaign](../results/20260920-rendezvous/report.md).

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
