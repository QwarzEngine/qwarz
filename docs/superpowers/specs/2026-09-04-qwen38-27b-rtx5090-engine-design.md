# Qwen3.8-27B RTX 5090 Inference Engine Design

**Date:** 2026-09-04  
**Status:** Approved  
**Working name:** Q38 Engine  

## 1. Summary

Q38 Engine is a single-model, single-GPU inference runtime specialized for Qwen3.8-27B on one NVIDIA RTX 5090. It serves one active generation at a time, keeps a persistent 262,144-token session, and prioritizes agentic coding latency, output quality, and predictable operation over portability or multi-user throughput.

The project follows a staged hybrid strategy. ExLlamaV3 is the first executable backend, numerical oracle, and kernel donor. A dedicated C++20/CUDA runtime then replaces measured hot paths while a Rust supervisor provides a stateful subset of the OpenAI Responses API. The final production artifact uses one Blackwell-specific mixed-precision format, Q38X, with an effective target of 4.5–5.0 bits per model parameter. It uses native NVFP4 storage for MLP and Gated DeltaNet matrices, starts attention projections and token-space tensors in FP8, and retains recurrent state in FP32 unless validation proves a lower precision equivalent.

The production runtime initially preserves the model's exact attention structure. It also retains an experimental, reversible query-aware sparse-attention path for the 16 full-attention layers, but that path is unavailable to normal requests until the exact resident implementation has been measured at 256K and the sparse path independently passes the coding-quality gates. The engine does not introduce CPU inference fallback, runtime model switching, multi-user scheduling, or KV spill during generation.

## 2. Goals

- Serve Qwen3.8-27B on one RTX 5090 with its native 262,144-token context window fully resident.
- Deliver excellent interactive latency for a persistent agent/coding session whose context grows incrementally by turn.
- Beat the same-machine ExLlamaV3 baseline at equivalent quality by at least 15% in accepted decode tokens per second and 25% in large-suffix prefill throughput.
- Keep warmed-turn p50 time to first streamed delta at or below 200 ms for small appended turns.
- Keep warmed-turn p50 first-delta latency at or below 300 ms at 256K for at most 128 newly appended tokens, with a 256K-to-128K latency ratio no greater than 1.5.
- Bound steady-state GPU memory at 28.5 GiB or less, leaving operational headroom within a 30 GiB engine budget.
- Preserve deterministic, testable behavior across cancellation, retry, speculative rejection, context branching, and GPU-worker restart.
- Expose a focused OpenAI Responses API with streaming, reasoning text, function tools, persistent response IDs, cancellation, and retrieval.

## 3. Non-Goals

- Supporting models other than the pinned Qwen3.8-27B checkpoint family.
- Supporting GPUs other than RTX 5090 / CUDA compute capability `sm_120` in the production build.
- Continuous batching, tensor parallelism, multi-GPU execution, or more than one active response.
- Multimodal input, built-in hosted tools, background responses, distributed storage, LoRA loading, or embeddings endpoints in v1.
- Approximate attention as the mandatory or silent execution path, automatic context summarization, automatic truncation, CPU weight offload, or CPU KV spill. Query-aware sparse attention is permitted only as an explicitly measured experimental path that retains the complete KV cache and can fall back to exact attention without reconstruction.
- Preserving PyTorch as a production runtime dependency after the Q38X execution path is complete.

## 4. Fixed Workload and Model Profile

The engine compiles against the checked model manifest rather than discovering architecture at runtime. The initial checkpoint has:

- 64 language-model layers.
- A repeating group of three linear-attention Gated DeltaNet layers followed by one full-attention layer.
- 48 Gated DeltaNet layers and 16 full-attention layers in total.
- Hidden size 5,120 and MLP intermediate size 17,408.
- 24 query heads, four KV heads, and head dimension 256 in full-attention layers.
- One native MTP layer, retained as a fallback/speculation oracle.
- A native maximum position count of 262,144.
- A DFlash2 drafter with five sliding-attention layers, a 2,048-token window, block size eight, and up to seven proposed tokens per verification step.

Only text inference is loaded. Vision components and unused multimodal tensors are excluded from the Q38X artifact.

## 5. Lessons Adopted from Existing Engines

### ExLlamaV3

Use EXL3 5 bpw as the initial quality/performance baseline and numerical oracle. Reuse or port proven semantics for quantized linear layers, Gated DeltaNet, NVFP4 KV, DFlash2, MTP, speculative rewind, and CUDA graph execution. Remove its runtime architecture discovery, general model graph, Python generator loop, and unrelated model implementations from the production path.

The locally available EXL3 3.5 bpw target is permitted as a bring-up profile for API compatibility, persistent-session behavior, cache allocation, cancellation, recovery, and early performance plumbing. It is never an acceptance-quality baseline and cannot establish quality parity for Q38X. Switching between EXL3 3.5 and 5.0 uses a pinned artifact profile; model architecture and workload definitions remain unchanged.

### SGLang

Adopt explicit prefix identity and incremental prefill semantics. Do not adopt RadixAttention, a radix tree, request scheduling, or multi-request cache eviction because the target workload has one hot session and one linear cache.

### vLLM

Adopt clear separation of prefill and decode engines, chunked prefill, and measurable graph-capture modes. Do not adopt PagedAttention or continuous batching because they add indirection to solve fragmentation and concurrency that are absent here.

### TensorRT-LLM

Use its Blackwell/NVFP4 and KV-reuse implementations as performance references. CUTLASS/CuTe is the preferred implementation base for supported block-scaled Tensor Core GEMMs; a custom kernel replaces it only when profiling shows that it misses an acceptance gate. TensorRT-LLM is a benchmark target, not a runtime dependency.

### llama.cpp

Use it as a C++ architecture and correctness reference, especially for explicit backend failure handling and benchmark reproducibility. Do not inherit its cross-platform graph abstraction or generic quantization matrix.

### TabbyAPI

Use it as a behavioral reference for ExLlama-backed serving and client compatibility. Its administrative surface and general model management are outside scope.

## 6. System Architecture

Q38 Engine has four independently testable units.

### 6.1 Rust Supervisor

The supervisor owns HTTP, SSE, API validation, authentication, response IDs, request idempotency, tokenization, prompt rendering, tool-call parsing, the canonical token tape, and durable response metadata. It exposes health and metrics independently of GPU-worker health.

The supervisor starts on loopback by default. LAN binding requires an explicit address and bearer token. It never silently exposes an unauthenticated remote endpoint.

### 6.2 Session Store

SQLite in WAL mode stores response objects, input/output items, parent response IDs, status transitions, sampling parameters, tokenizer/template version, artifact hash, RNG seed/counter, and canonical token sequences. The database is not accessed per generated token. A response transaction is written before dispatch, updated at terminal state, and flushed before it becomes an eligible parent.

The store permits branches through `previous_response_id`. The GPU holds one hot branch. The supervisor retains Gated DeltaNet and DFlash2 state at the 32 most recent committed response boundaries. Selecting another stored parent computes the longest common token prefix with the hot branch, restores the nearest retained state on that common prefix, and prefills the remaining divergent suffix. If no retained state exists, it performs a correct full reconstruction from the token tape.

### 6.3 C++20/CUDA Worker

The worker loads the single Q38X artifact, owns all device memory, captures execution graphs, performs prefill, decode, DFlash2 proposal and verification, sampling, and transactional cache updates. It has no HTTP, database, or model-management logic.

The worker communicates with the supervisor over a versioned Unix-domain control protocol. Bulk cache checkpoints use supervisor-owned shared memory registered as pinned host memory. The protocol is deliberately small: load, health, restore, prefill, generate, cancel, checkpoint, and unload.

### 6.4 Offline Converter and Evaluation Tools

Python is permitted offline for checkpoint conversion, calibration, tensor sensitivity analysis, oracle execution, and benchmark orchestration. The converter emits a Q38X artifact plus a content-addressed manifest containing exact tensor shapes, layouts, precision classes, offsets, checksums, tokenizer/template hashes, kernel ABI, and required CUDA/driver versions.

## 7. Q38X Quantization Format

Q38X is one immutable, memory-mappable artifact. The runtime does not select quantization modes dynamically.

### 7.1 Precision Classes

- Large MLP and Gated DeltaNet projection matrices start in NVFP4 E2M1 with E4M3 block scales.
- Full-attention Q, K, V, and output projections start in FP8. Calibration may demote an individual projection to NVFP4 only when the complete coding and long-context suite remains inside the quality gates.
- Embedding and LM-head tensors start in FP8 to protect token-space quality while avoiding their BF16 memory cost.
- Norm weights, small coefficients, convolution parameters, and numerically sensitive reductions use BF16 or FP32 as specified in the manifest.
- Gated DeltaNet recurrent matrix state uses FP32. It is fixed-size rather than context-proportional, so reducing its precision is not part of the initial memory strategy.
- Tensor sensitivity analysis may promote any NVFP4 matrix to FP8 or BF16. It may demote an FP8 tensor only with end-to-end evidence; format choice is never based on local reconstruction error alone.
- The final artifact must remain within the 28.5 GiB runtime peak. If 5.0 effective bpw misses the memory gate, sensitivity-guided demotion reduces the artifact toward 4.7 bpw; context capacity, exact attention, and memory headroom are not reduced.

The manifest reports both body bits per weight and whole-model effective bits per parameter, including scales and promoted tensors. Acceptance uses the whole-model figure.

### 7.2 Compute Paths

Prefill dynamically quantizes activations for block-scaled W4A4 Tensor Core GEMMs on NVFP4 tensors and uses the selected FP8 or BF16 path for promoted tensors, with higher-precision accumulation where required by validation. Decode uses batch-1 W4A16 GEMV kernels that read the same packed NVFP4 weights, dequantize in registers, consume FP16/BF16 activations, and fuse adjacent norm, residual, gate, or activation work when profitable. Verification steps with two to eight target positions use native block-scaled Tensor Core GEMMs rather than the batch-1 GEMV path.

There is no second full weight representation. Kernel variants must consume the same artifact layout.

### 7.3 Quality Calibration

Calibration data is coding-focused and includes source code, diffs, tool schemas, JSON, terminal transcripts, long repository context, and multilingual technical prose. Tensor promotions are selected by a combination of layer-output error, logit KL divergence, perplexity delta, greedy token agreement, tool-call validity, and end-to-end coding evaluation.

Quantization experiments begin from the pinned BF16/FP16 source checkpoint. Q38X is never produced by transcoding EXL3 because compounded quantization error would make quality attribution invalid. The converter emits three evaluation candidates from the same source: uniform NVFP4 as a performance bound, MLP/DeltaNet-only NVFP4 as a conservative quality bound, and sensitivity-selected Q38X mixed precision as the production candidate. EXL3 5 bpw remains the deployable oracle rather than an input to conversion.

## 8. GPU Memory Contract

The production worker preallocates all steady-state memory before accepting traffic.

- Q38X target weights: expected 17–19 GiB, bounded by the final artifact manifest.
- Target NVFP4 KV: exactly 4.5 GiB of payload for 262,144 tokens, 16 layers, K and V, four KV heads, head dimension 256, and 4.5 stored bits per element. An FP8 KV candidate requires 8 GiB and is evaluated as the quality control even if it cannot satisfy the final device-memory gate.
- DFlash2 weights: expected 1.0–1.4 GiB after conversion.
- DFlash2 KV: approximately 40 MiB in FP16 because all five draft layers use a 2,048-token sliding window. It is implemented as a ring buffer and never scales with target context length.
- Gated DeltaNet recurrent state: approximately 144 MiB of primary FP32 matrix state plus convolution and history buffers.
- CUDA graphs, activations, temporary reductions, allocator metadata, and safety reserve: at most the remaining budget up to a 28.5 GiB peak.

The worker performs a startup allocation dry run, records the high-water mark, warms every supported graph shape, and then seals the allocator. Any later hot-path allocation is a test failure. Any unsupported kernel or insufficient-memory condition is a startup failure; neither condition may trigger a CPU fallback.

## 9. Cache and Session Model

### 9.1 Target KV

The target cache is a contiguous, preallocated structure-of-arrays layout specialized for the 16 full-attention layers. It has no allocation page table. A 64-bit `valid_length` determines the committed prefix. Speculative writes occupy positions after that cursor. Acceptance advances the cursor; rejection rewinds it without copying and later writes overwrite rejected entries.

The KV write path initially supports both calibrated FP8 and calibrated NVFP4. The production manifest selects one immutable format after long-context retrieval, coding, and decode benchmarks. NVFP4 is accepted only when the attention kernel consumes it directly, dequantizes online, never materializes a full FP16 cache or full-context scratch buffer, and remains inside the quality gates. The 28.5 GiB device-memory contract makes NVFP4 the expected production result, not an assumption exempt from measurement.

The physical layout has fixed logical attention pages even though allocation is contiguous. Each page can carry per-layer, per-KV-head key bounds used by an optional Quest-style selector. Exact attention ignores these bounds and scans every valid page. Sparse experiments keep the complete KV payload resident and select a union of attention sinks, a recent exact window, and query-ranked older pages; they never evict or rewrite history. Page size, recent-window size, and selected-token budget are offline-tuned constants recorded in the artifact manifest.

### 9.2 Gated DeltaNet State

Recurrent state and short-convolution state use fixed device buffers. Speculative verification maintains a small history sufficient for the maximum eight-position target step. Acceptance selects the final accepted state; rejection restores the matching checkpoint in O(1) or one bounded copy. At completed response boundaries, the supervisor retains a bounded LRU of 32 host snapshots for branch restoration; these snapshots are not part of the GPU memory budget.

### 9.3 DFlash2 State

All five DFlash2 attention caches are 2,048-token rings. Dynamic-convolution state, selector state, and RNG counters are part of the transactional generation state. Target hidden taps at layers 5, 19, 33, 47, and 61 are written directly to drafter inputs without a host round trip.

### 9.4 Prefix Reuse

Each request compiles to a token sequence and compares against the current canonical tape. An unchanged prefix incurs no model work. An appended turn prefills only its new tokens. A branch or edited earlier message restores the newest retained response-boundary state on the common prefix, rewinds target KV to that boundary, and recomputes the rest of the suffix. If the required target KV was overwritten by another branch, reconstruction begins at the nearest boundary for which both cache and recurrent state remain valid; otherwise it begins at token zero. Tokenizer, template, artifact, and prompt-renderer hashes participate in prefix identity; any mismatch invalidates reuse.

## 10. Execution Pipeline

### 10.1 Incremental Prefill

Prefill accepts only the suffix not represented by valid model state. It uses benchmark-selected chunk buckets and a separate CUDA stream for safe host/device staging. Full-attention layers use exact FlashAttention and write quantized KV directly. Gated DeltaNet layers use an exact chunk-parallel scan and emit only the final recurrent state required by the next chunk.

Chunk size is selected from an offline-autotuned table keyed by suffix length and current context bucket. The table is generated on the target 5090 and stored with the engine build; no expensive tuning occurs during normal startup.

For exact long-context attention, the worker uses sequence-parallel split-KV execution. CTAs process disjoint KV ranges, produce numerically stable partial attention states, and merge those states with an exact log-sum-exp reduction. The split count is selected from a fixed table for the 5090 so GPU utilization can rise with context length before memory bandwidth becomes the limiting term.

### 10.2 Decode and Speculation

DFlash2 proposes up to seven tokens from its block of eight positions. The target verifies one to eight positions in a single step. Sampling, candidate acceptance, RNG advancement, stop-token detection, and cache commit/rewind execute on GPU.

The worker captures a decode graph for each target verification length from one through eight. Graph inputs are stable buffers. Only input values, positions, valid lengths, stop configuration, and approved pointer fields change between replays. The desired final state is one global graph replay per target verification step. If a library primitive prevents safe global capture, the accepted fallback is a small fixed sequence of captured layer-group graphs, but it must still meet the performance gates.

Speculation is distribution-preserving: draft proposals never bypass target verification. Temperature and sampling settings are applied by the target sampler, and acceptance consumes the same counter-based RNG stream used by the numerical reference.

### 10.3 Fusions

Fusion work proceeds in measured order:

1. Persistent target KV and Gated DeltaNet state so an appended turn performs no prefix reconstruction.
2. Global or layer-group decode graphs and removal of Python launch boundaries.
3. Complete Gated DeltaNet decode blocks, including projections, recurrent update, output projection, and residual handling.
4. Full-attention decode blocks, including QKV projections, Q/K normalization, partial RoPE, quantized cache append, exact flash decoding, output gate, projection, and residual.
5. MLP paths, including fused gate/up projection, SiLU product, down projection, and residual.

After the resident runtime is profiled, the first megakernel experiment covers one repeating four-layer superblock: three Gated DeltaNet layers followed by one full-attention layer. It targets `sm_120`, batch one, and decode or verification lengths from one through eight. The experiment advances only if it preserves oracle outputs and improves end-to-end decode by at least 10% over the equivalent warmed ExLlamaV3/CUDA-graph path. A monolithic 64-layer kernel is not an initial milestone.

Query-aware sparse attention is investigated only if resident exact attention misses the 256K first-delta SLO. Its first implementation changes only the 16 full-attention kernels and reuses the same Q38X KV payload. Selector metadata and selected-page indices are inputs to a separate captured kernel path; exact and sparse execution never share mutable numerical state beyond the append-only cache. A low-confidence selector result, unsupported shape, instrumentation mismatch, or explicit exact request uses the exact split-KV path.

Each fusion is reversible behind a build-time benchmark switch until it passes correctness and performance gates. Production builds contain only the selected path.

## 11. Responses API Contract

### 11.1 Supported Surface

v1 supports:

- `POST /v1/responses`, with streaming and non-streaming output.
- `GET /v1/responses/{response_id}`.
- `POST /v1/responses/{response_id}/cancel` for the single in-progress response.
- `previous_response_id` for persistent multi-turn state and branches.
- Text input, developer/system/user/assistant messages, function definitions, function-call outputs, `max_output_tokens`, temperature, top-p, and deterministic seed metadata.
- Core streaming events for response creation/status, reasoning deltas, output-text deltas, function-call argument deltas, completed output items, completed responses, incomplete responses, cancellation, and errors.
- Usage details including input, cached-input, reasoning, visible-output, and total token counts where the model format exposes them.

`instructions` follow Responses API semantics and are not implicitly inherited merely because `previous_response_id` is present. The prompt renderer reconstructs the prior item chain and applies current instructions explicitly.

v1 includes a stateless Chat Completions adapter that translates requests into the same internal response representation. It does not own a second cache or generation implementation.

### 11.2 Explicitly Unsupported in v1

Multimodal items, hosted tools, background execution, remote file storage, multiple simultaneous responses, automatic truncation, prompt logprobs, and arbitrary structured-output grammars return a clear unsupported-feature error.

Context accounting includes the committed prefix, new input, reasoning, visible output, and tool-call tokens. If the requested maximum can exceed 262,144 total tokens, the request fails before modifying GPU state. The engine never drops earlier items silently.

## 12. Transaction and Failure Semantics

The last completed response is the durable commit point.

1. The supervisor validates the request and parent response, writes an `in_progress` record, and dispatches a canonical token suffix.
2. The worker creates a tentative state after the committed cursor and generates only into tentative cache/history positions.
3. Accepted speculative tokens may stream immediately, but the response is not eligible as a future parent until terminal metadata and the cache checkpoint complete.
4. Normal text completion or a complete function call moves the response to an internal `finalizing` state, copies the checkpoint asynchronously, commits the token cursor and SQLite transaction, and only then emits the terminal completed event.
5. Cancellation, disconnect-triggered cancellation, parser failure, CUDA error, or worker death marks the response cancelled or failed and restores the last completed commit.

The supervisor assigns the response ID before generation and supports an idempotency key. Retrying the same key returns or resumes the same stored terminal result rather than creating a duplicate response.

SSE sequence numbers are monotonic per response. The supervisor retains emitted event metadata until terminal state so retrieval can explain whether a response completed, failed, or was cancelled. Partial output from a cancelled response is diagnostic only and cannot be used as `previous_response_id`.

## 13. Recovery

After each completed response, the worker copies the dirty target-KV suffix into a 4.5 GiB supervisor-owned mirror and copies complete Gated DeltaNet state, DFlash2 ring state, cursors, and RNG counters into a 32-entry host LRU keyed by response ID. The bounded branch-state LRU consumes at most 6 GiB in addition to the target-KV mirror. The checkpoint header includes artifact hash, prompt hash, token count, generation number, byte lengths, and checksums. The supervisor exposes the response as committed only after this header is atomically published.

If the CUDA worker fails while the supervisor remains alive, the supervisor starts a fresh worker, reloads weights, verifies the checkpoint, transfers approximately 5 GiB of state back to the GPU, and restores the last completed response. The first post-recovery token must match a clean reconstruction from the same token tape and RNG state.

If the entire service or machine restarts, SQLite restores conversation metadata and the token tape. v1 rebuilds GPU state by prefill from the committed token tape. Persistent on-disk KV snapshots are excluded because their compatibility and integrity surface is not justified for the first release.

## 14. Observability

The supervisor exports structured logs and local metrics for:

- Request ID, response ID, parent ID, status, and error class.
- Input tokens, reused prefix tokens, recomputed tokens, and output tokens.
- Queue time, tokenization time, prefill time/tok/s, first-delta latency, decode accepted tok/s, and end-to-end latency.
- DFlash proposal length, accepted length, acceptance ratio, target verification length, and fallback-to-no-draft count.
- Exact KV bytes read, sparse KV bytes read, selected-page count, recent-window tokens, selector time, exact-fallback count, and shadow-attention error.
- CUDA graph replay counts by shape, uncaptured execution count, and hot-path allocation count.
- GPU memory high-water mark, checkpoint bytes/time, recovery time, and cache-valid length.

NVTX ranges cover every model group and fused kernel. The benchmark harness records GPU clocks, power, temperature, driver, CUDA version, engine commit, artifact hash, and all generation parameters.

## 15. Validation Strategy

### 15.1 Correctness

- Unit-test every quantize/dequantize primitive, fused operation, cache append, ring wrap, Gated DeltaNet update, RoPE transform, sampler, and speculative rewind against a high-precision reference.
- Compare Q38X kernel outputs with an unfused Q38X reference across boundary shapes, context buckets, and verification lengths. Packed quantization and cache writes must be byte-identical. FP32 recurrent outputs use `atol=2e-4` and `rtol=2e-3`; BF16/FP16 activation outputs use `atol=2e-2`, `rtol=2e-2`, and cosine similarity of at least 0.9999.
- Require greedy token identity between fused and unfused Q38X runtimes for the deterministic corpus.
- In sparse shadow mode, compute exact and selected-page attention for sampled decode positions and record attention-output error, final-logit error, greedy-token agreement, and the attention mass represented by selected pages.
- Verify cancellation and branch rewinds by comparing the next token/logits with a clean prefill of the same canonical token tape.
- Validate Responses API object shapes and SSE event ordering using recorded client fixtures.

### 15.2 Quality

The Q38X artifact may regress by no more than 1% relative to the approved higher-precision baseline on the aggregate coding score. The aggregate is the arithmetic mean of normalized task pass rates whose task list and weights are frozen in M0 before Q38X tuning begins. Tool-call validity may fall by no more than one absolute percentage point. The suite includes code completion, repository editing, diff generation, long-context retrieval in code, multilingual technical instructions, JSON/tool calls, and an agent loop over isolated repositories.

EXL3 5 bpw is the primary deployable quality baseline. BF16 or a sufficiently high-precision implementation is used offline for diagnostic comparisons when hardware capacity prevents fully resident execution.

### 15.3 Performance

Measure cold prefill, incremental prefill, time to first token, raw target steps/s, accepted decode tokens/s, DFlash acceptance, peak VRAM, power, and recovery time at 1K, 32K, 128K, and 256K context buckets. Prompts, outputs, sampler settings, GPU clocks, and thermal state are fixed across comparisons.

Acceptance thresholds:

- At least 15% higher accepted decode tok/s than ExLlamaV3 at the same context and equivalent quality.
- At least 25% higher prefill tok/s for appended suffixes of 8,192 tokens or more.
- Warm-turn p50 first-delta latency at or below 200 ms with 128K cached tokens, at most 128 newly appended input tokens, and greedy generation.
- Warm-turn p50 first-delta latency at or below 300 ms with 256K cached tokens and at most 128 newly appended input tokens.
- A 256K-to-128K warmed-turn p50 first-delta ratio no greater than 1.5.
- No context bucket more than 5% slower than the baseline.
- Peak device memory at or below 28.5 GiB after all graph shapes are warmed.
- Zero steady-state device allocations and zero silent fallbacks.

TensorRT-LLM, SGLang, vLLM, llama.cpp, and TabbyAPI/ExLlamaV3 are recorded as informational baselines when they can run a sufficiently comparable model and quantization. Only same-quality comparisons decide acceptance.

The sparse path has additional promotion gates: at least 99% greedy-token agreement with exact Q38X on the frozen deterministic corpus, no more than 1% aggregate coding-score regression, no more than one percentage point of tool-call-validity regression, and no failed long-context retrieval case that exact Q38X passes. Passing average perplexity or needle retrieval alone is insufficient. Until every gate passes, sparse attention remains a benchmark-only feature and exact attention remains the serving path.

### 15.4 Reliability

- Run a 24-hour agent-loop soak without memory growth, deadlock, invalid output, or OOM.
- Run 10,000 randomized cancel/retry/branch cases and compare restored state with clean reconstruction.
- Inject worker termination during prefill, decode, speculative verification, SSE delivery, and checkpointing.
- Corrupt each checkpoint section and require deterministic rejection and token-tape reconstruction.

## 16. Delivery Milestones

### M0: Baseline and Oracle

Create the reproducible benchmark harness, pin the model and ExLlamaV3 donor revision, collect RTX 5090 baselines, assemble the coding calibration/evaluation corpus, and define golden API and token fixtures. This milestone changes no inference semantics.

### M1: Stateful Hybrid Runtime

Implement the Rust Responses API supervisor, SQLite response store, canonical token tape, worker protocol, and an ExLlamaV3-backed worker. Demonstrate incremental prefix reuse, cancellation, branching, metrics, and reproducible baseline measurements.

### M2: Q38X Runtime

Implement the converter, manifest, loader, prefill kernels, decode GEMV kernels, target KV, Gated DeltaNet state, and unfused numerical reference. Replace target paths only after correctness and quality gates pass.

### M3: Fused Production Engine

Add global/layer-group CUDA graphs, full block fusion, GPU sampling and acceptance, converted DFlash2 with 2,048-token ring KV, supervisor-owned host checkpoints, worker recovery, soak testing, and final performance qualification.

If the exact M3 engine misses the 256K latency gate, add an M3 experimental track for query-aware page selection over the unchanged resident KV cache. This track cannot delay or weaken qualification of the exact engine and is promoted only after its independent performance and quality report passes every sparse-attention gate.

Each milestone produces a runnable engine and benchmark report. No milestone depends on completing all later custom kernels before it provides value.

## 17. Primary Risks and Mitigations

- **Q38X quality loss:** promote sensitive tensors using coding-specific calibration; retain EXL3 5 bpw as the deployable fallback until quality gates pass.
- **W4A4 activation sensitivity:** keep selected operations or layers on FP8/BF16 and evaluate prefill and decode separately.
- **Global CUDA graph incompatibility:** use a fixed sequence of captured layer-group graphs and accept it only if performance targets still pass.
- **NVFP4 KV attention bottleneck:** fuse dequantization into exact flash-decoding kernels and prohibit full FP16 materialization.
- **Exact attention remains context-linear:** use resident state, split-KV sequence parallelism, quantized online reads, and multi-token verification first. If those measures miss the 256K SLO, evaluate query-aware page selection while retaining complete KV and exact fallback.
- **Sparse attention misses rare coding dependencies:** keep exact mode authoritative, require shadow comparisons and task-level gates, and treat any exact-pass/sparse-fail retrieval case as a promotion blocker.
- **VRAM overrun after graph capture:** warm every graph before serving, cap the artifact, seal allocations, and preserve a 1.5 GiB margin below the 30 GiB budget.
- **DFlash2 acceptance variance on code:** record acceptance by workload and permit an automatically selected no-draft target graph only when measured latency is lower; this changes execution strategy, not output distribution.
- **Driver/CUDA instability:** pin a validated driver, CUDA toolkit, compiler, CUTLASS revision, engine binary, and artifact ABI; reject mismatches at startup.
- **Donor-code drift:** pin donor commits and copy only code covered by compatible licenses, retaining required attribution and notices.

## 18. External References

- [ExLlamaV3 repository](https://github.com/turboderp-org/exllamav3)
- [ExLlamaV3 graph-captured attention settings](https://github.com/turboderp-org/exllamav3/blob/master/doc/env_vars.md)
- [SGLang documentation](https://docs.sglang.io/)
- [vLLM V1 architecture guide](https://docs.vllm.ai/en/stable/usage/v1_guide/)
- [NVIDIA TensorRT-LLM documentation](https://docs.nvidia.com/tensorrt-llm/)
- [NVIDIA Model Optimizer LLM quantization](https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/examples/llm_ptq/README.md)
- [Luce hybrid-model megakernel](https://github.com/PixelML/luce-megakernel)
- [Flash-Decoding for long-context inference](https://princeton-nlp.github.io/flash-decoding/)
- [FlashInfer recursive and split-KV attention](https://docs.flashinfer.ai/tutorials/recursive_attention.html)
- [Quest query-aware sparsity](https://arxiv.org/abs/2406.10774)
- [SparQ bandwidth-efficient attention](https://arxiv.org/abs/2312.04985)
- [KIVI KV-cache quantization](https://arxiv.org/abs/2402.02750)
- [SnapKV cache compression](https://arxiv.org/abs/2404.14469)
- [llama.cpp repository](https://github.com/ggml-org/llama.cpp)
- [TabbyAPI FAQ](https://github.com/theroyallab/tabbyAPI/wiki/05.-FAQ)
- [OpenAI Responses streaming reference](https://platform.openai.com/docs/api-reference/responses-streaming/response/content_part)
