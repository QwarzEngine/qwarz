# Flash prefill: application validation and v1 handoff

## Decision and scope

Keep EXL3 5 bpw + MTP + K8/V4 on GPU 0 (RTX 5090). Compare the unchanged
Triton prefill with the opt-in Flash/8192 profile before freezing a usable
v1 configuration. GPU 1 and the donor installation remain untouched.
No commits or branches are created in the existing unborn repository.

The earlier matrix already demonstrates that Flash does **not** improve
every full-response latency. It accelerates prefill but changes numerical
rounding and sometimes output length. These tests must retain regressions.

## Predeclared experiment

- Same model, corpus, template, medium thinking, recommended sampler and
  seeds; both runs allocate the full native 262,144-position pool.
- LRU coding: context budgets 32,768 and 262,144; 4,096 output tokens;
  one seed and two warm branches per context, per backend (12 responses).
- Retrieval/tool sessions: the same two context budgets; two independently
  seeded sessions per bucket/backend; two append-only tool cycles each;
  512 tool-call and 1,536 answer tokens (up to 32 generations).
- Initial requests must match prompt hashes across backends. Later session
  prompts may differ because actual generated history is preserved; do
  not misrepresent these as identical-input attention microbenchmarks.
- Record cold TTFT separately from warm TTFT, first final content, full
  response and tool-cycle wait, actual output length, physical cache reuse,
  decode throughput, truncations/requeues and CUDA allocator peaks.
- Grade coding with the frozen independent `benchmarks/fixtures/lru_checks.py`
  and the generated unittest suite. Inspect generated code before execution;
  run it isolated from host files/network/GPU with a timeout. Do not repair
  responses. Count invalid/truncated/extraction failures in denominators.
- Tool calls and final JSON use the existing independent fixture oracle.
  These are allowlisted in-memory reads, not general host tools or a
  multi-file coding benchmark. LRU is one task, not coding-quality parity.
- Keep raw outputs and failure artifacts. Completion markers certify
  collection only. No performance claim may discard quality failures.

## Work sequence

1. **Complete.** Extend the guarded direct probe's opt-in profile to LRU and retrieval
   sessions with reversible chunk/kernel changes, CPU tests and review.
2. **Complete.** Run the four sequential GPU jobs, audit paired prompts, grade outputs,
   and document successes and regressions. Stop expanding kernel sweeps.
3. **Complete.** Freeze a provisional configuration for manual testing, not an acceptance
   certificate or a 30-second cold-import/full-answer SLA.
   Keep 5 bpw + MTP + K8/V4; Flash/8192 is the explicit optimized profile,
   with original Triton available as the conservative control. Both have
   observed tool-cycle failures; no universal quality promotion is made.
4. **Next.** Prioritize the v1 stateful inference service described below.

Closure: [application results](2026-09-05-prefill-application-results.md).
All 44 generations were independently audited with zero inconsistencies;
quality failures remain in the results. The service is not implemented yet.

## v1 manual-testing boundary

The next deliverable is a real Qwasar service, not another benchmark wrapper:

- One resident model and one active generation, EXL3 5 bpw + MTP + K8/V4;
  explicit baseline/Flash selector and safe fallback by restarting with
  baseline, never silently changing attention during a request.
- Local-only HTTP service, health/model/config endpoints, streaming token
  output with reasoning separate from final content, and cancellation.
- A persistent session API retaining exact token history across turns and
  preserving KV/recurrent-state reuse; explicit reset and context-budget
  rejection rather than truncating old history silently. Clarify that
  process persistence and disk/restart recovery are different guarantees.
- Structured tool-call output and tool-result input. The engine does not
  execute arbitrary model-selected host commands; the client owns tools.
- A small terminal client and reproducible start command for manual use;
  metrics for TTFT, prefill/cache reuse, decode and completion reason.
- CPU lifecycle/protocol tests plus real-GPU smoke tests for multi-turn
  continuity, tools, cancellation followed by recovery and near-limit
  rejection. Do not advertise full OpenAI protocol compatibility unless
  verified against the implemented subset.

Existing design: `docs/superpowers/specs/2026-09-04-qwen38-27b-rtx5090-engine-design.md`.
Finishing the service takes priority over additional speculative kernel work
once this bounded baseline check is complete.
