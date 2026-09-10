# Minima64 FP8 with isolated XQA decode evaluation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox syntax.

**Goal:** Make Minima64 plus corrected FP8 large prefill the fixed experimental base; evaluate XQA without changing K8/V4 or MTP6.
**Architecture:** Preserve validated Minima64/prefill implementation and compiled-form routing. Test already-appended K8/V4 → transient FP16 or FP8 → XQA for Q1/Q7 with all conversion costs included. Only integrate a staged route into model graphs if its numerical and latency gate justifies doing so.
**Tech Stack:** Pinned donor Torch2.14/ExLlama, FlashInfer0.6.18 and CUTLASS DSL4.7.1, RTX5090.
**Spec:** Latest user instruction: ingesta sí o sí Minima64+FP8, test XQA changing nothing else; temporarily128K allowed if needed.

## Global Constraints

GPU0 only. Keep target/draft K8/V4, MTP6, GDN/head/embed, output budgets and exact prefix semantics. Start with128K and native258K captured attention. No new cache quantization, no reduced-vocabulary MTP or new GDN quantization. Existing service remains restored between GPU windows. Repository has no initial commit, so isolate artifacts/source hashes rather than create a worktree or commits.

### Task1: Reproducible base
- [x] Provide environment.py with exact new FI/DSL and patch hash checks.
- [x] Provide base_profile.py invoking existing validated model probe with profile minima64 and mode prims; expose optional prompts and repeats but no alternative weight/cache settings.
- [x] Record fixed settings and smoke the baseline under managed GPU ownership if required by new integration.

### Task2: Staged XQA gate
- [x] Implement StagedXQA callable over already-appended decode kwargs, storage fp16/fp8; stage(), xqa_only(), diagnostics(). Never append, change length or mutate resident K8/V4.
- [x] Real Q7 capture, Q1 uses last captured query; lengths131072 and native258183, prefix-shortening disclosed as diagnostic. Compare dequantized FP32 oracle; test CUDA graph replay with changing lengths and rewind.
- [x] Time direct K8/V4, staging, XQA-only, full conversion+XQA and graph replay. Keep units and physical scratch traffic explicit.
- [x] If a validated route beats direct decode, add a guarded model graph adapter and fresh128K model pair; otherwise report measured rejection and retain the base. This conditional avoids adopting a slower route merely because its isolated kernel is fast.

### Task3: Close
- [x] Independent source and numerical/performance review, fix material issues.
- [x] Document base runner, XQA outcomes and any unsupported shape; restore original service and archive exact sources.
