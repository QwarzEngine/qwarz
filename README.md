<p align="center"><img src="docs/qwarz.jpg" width="360" alt="Qwarz"></p>

# Qwarz

Qwarz is an independent inference engine specialized for **Qwen3.8-27B on one
NVIDIA RTX 5090** (32 GB). It serves one persistent, agentic coding session
with a native **262,144-token context**, native image input, and an
OpenAI/Anthropic-compatible HTTP API. A Rust/SQLite supervisor owns sessions,
exact token history, cancellation and recovery; a persistent ExLlamaV3 worker
owns the GPU.

## The stack

Every lever is measured against a same-day control before it ships. The
production stack (`--prefill xqa`, promoted 2026-09-21 after a 37-cell gate):

| lever | effect |
|---|---|
| EXL3 5.0 bpw artifact | pinned model quantization, hash-verified at every load |
| NVIDIA64 NVFP4 MLP donor | 192 FP4/FP8 matrices + MLP CUDA graphs |
| MTP6 speculative decoding | 6 fixed draft proposals, ~0.70 acceptance on coding |
| NVFP4 one-level KV cache | quantized KV (page 256) compresses deep-context reads |
| XQA decode attention | decode kernels + per-layer decode CUDA graphs |
| PRIMS FP8 prefill | large prefills (≥8K) |
| Rendezvous | speculative loop resident on GPU: batched verify, GPU draft chain, embedding table on GPU (+2.5 GB VRAM) |
| Native vision | MM embedding tables aligned to the compute device |

Rollback to the pre-XQA stack: `--prefill flash` in
`integrations/systemd/qwasar.service`, `daemon-reload`, restart.
`QWASAR_RDZ=0` disables the rendezvous path at boot.

Pinned artifact: `thelastspark/Qwen3.8-27B-exl3` @
`1a6fe4afb5b921fda9f93fd4b06d6c6d5c99a62c`, three shards, 19.9 GB, SHA-256
verified against the frozen manifest.

## Expected numbers

Measured samples (2026-09-21, production stack, protocol cells, single user,
batch 1), not an SLA. Per-cell run noise: ±5% decode, ±2–3 pp acceptance.

| context | decode (tok/s) | TTFT |
|---|---:|---:|
| 4K | 248 | 0.67 s |
| 32K | 220 | 5.1 s |
| 64K | 199 | 11.3 s |
| 128K | 191 | 26.4 s |
| 256K | 163 | 67.0 s |

- Warm short chat end-to-end: ~156 tok/s decode, TTFT 70–140 ms warm (~1.1 s cold).
- Large prefills (PRIMS path): 5,300–6,800 tok/s.
- MTP acceptance ~0.62–0.70, workload dependent; decode tracks it.
- VRAM: ~28.7 of 32.6 GB resident (~3.4 GB free).
- Quality gate (frozen checkers): 24/32 coding cells + 4/4 json vs control 23/32.

Upgrade deltas vs the previous flash stack (same-day A/B): decode +35% @32K,
+47% @256K, TTFT −3–4%, acceptance and quality not worse. Full stats and
methodology: `results/20260920-rendezvous/community-stats.md` (local artifact).

## Use it

```bash
cd /home/rekeyea/Documents/llm/qwarz
python3 scripts/qwasar.py start    # also: status, logs, stop
```

Endpoint `http://127.0.0.1:8800/v1`, model `qwasar-qwen38-27b`. Inline image
data URLs are accepted in user messages. One GPU worker: concurrent requests
get 409; Anthropic `count_tokens` does not take the worker.

| client | install | wrapper |
|---|---|---|
| Pi | `python3 scripts/configure_pi.py` | `scripts/pi_qwasar.sh` |
| Codex | `python3 scripts/configure_codex.py` | `scripts/codex_qwasar.sh` |
| OpenCode | `python3 scripts/configure_opencode.py` | `scripts/opencode_qwasar.sh` |
| OMP | `python3 scripts/configure_omp.py` | `scripts/omp_qwasar.sh` |
| Hermes | `python3 scripts/configure_hermes.py` | `scripts/hermes_qwasar.sh` |
| Qwen Code | `python3 scripts/configure_qwen_code.py` | `scripts/qwen_code_qwasar.sh` |
| Claude Code | `python3 scripts/configure_claude.py` | `scripts/claude_qwasar.sh` |

Installers are idempotent, refuse conflicting `qwasar` entries, keep other
providers, and pin background helper roles off the single worker. Select
`qwasar-qwen38-27b` (or the provider equivalent) in each client.

## Development

The runtime tests need `torch` from the donor venv; campaign companion tests
need their local `results/<campaign>/` modules. `tests/conftest.py` skips
whatever cannot import in the current environment, so a plain run stays green
in both:

```bash
uv sync --extra dev --extra benchmark
PYTHONPATH=src /home/rekeyea/Documents/llm/qwen38-exl3-mia/.venv/bin/python -m pytest -q   # full suite
uv run pytest -q                                                                          # without torch modules
cargo test --quiet
```

`qwasar-bench` (Python 3.12) drives the GPU probes:

```bash
QWASAR_PROBE_KIND=decode QWASAR_WORKLOAD=lru \
QWASAR_MODEL_PATH=/home/rekeyea/models/Qwen3.8-27B-EXL3-5.0bpw \
QWASAR_DRAFT_METHOD=mtp QWASAR_CACHE_QUANT=8,4 \
QWASAR_CONTEXTS=32768,131072,262144 QWASAR_REPETITIONS=2 \
./scripts/run_resident_probe.sh
```

Probe families: resident sessions, decode screening, append-only retrieval
sessions, cache fidelity, turn-size matrix, prefill experiments — each with its
own env vars, documented in `docs/benchmarks/`. GPU probes must not run
concurrently with the service. Completed runs publish `run.json`,
`environment.json`, `samples.jsonl` and `quality.json` atomically; exit 2 means
invalid input or environment, exit 3 means a comparison completed but a gate
failed.

## Docs

- [v1 manual](docs/manual-v1.md) — persistence, API, security, performance, rollback
- [Servicio systemd](docs/servicio-systemd.md) — boot, restart, journal
- [Arquitectura v1](docs/arquitectura-v1.md) — Spanish design walkthrough
- [Serving evaluation](docs/benchmarks/2026-09-20-serving-evaluation.md) — the lever campaign
- `docs/qwarz.jpg` — engine icon
- `results/` (local, gitignored) — campaign data, gate decisions, community stats
