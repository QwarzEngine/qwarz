# Fase 2 — NVIDIA64 + PRIMS + Attention64 a 32K/64K/128K/256K (2026-09-11)

**Estado:** A/B completo. Puerta ±5 pp **pasa**. Detalle en [resultados](2026-09-11-phase2-results.md).

## Qué se compara

Ambos brazos: Flash/8192, MTP6, K8/V4, pool 262144, sampler recomendado, thinking medium, mismas specs congeladas.

| | Control | Candidato |
|---|---|---|
| MLP | EXL3 | NVIDIA64 (192) |
| Prefill Q≥8192 | Flash | FP8 PRIMS P×256 |
| Prefill Q<8192 | Flash | Flash |
| Decode | donor default32 | Attention64 (`block_n=64`, 4 warps, 1 stage) |

Criterios vivos (igual que Fase 1, banda ±5 pp):

1. Código: candidato ≥ control (ahora 32 celdas: 4 familias × 4 contextos × 2 semillas).
2. JSON: todas las celdas frías (4/4).
3. Aceptación MTP: mediana de `draft_acceptance` a ±5 pp.
4. Memoria: pico asignado ≤ control + 0,5 GiB.

## Contextos

| Nombre | Prompt | Nota |
|---|---:|---|
| 32K | 32768 | hash idéntico a Fase 0 |
| 64K | 65536 | nuevo |
| 128K | 131072 | hash idéntico a Fase 0 |
| 256K | 258048 nominal | clamp a `262144 − max_tokens − 16` (TTL queda en 257008) |

32K/128K se re-ejecutan: el control de Fase 0 no aplicaba Flash en el runner.

## Comandos

```bash
# Matriz (una vez)
PYTHONPATH=src /home/rekeyea/Documents/llm/qwen38-exl3-mia/.venv/bin/python \
  results/20260910-quality-gate/build_phase2_prompts.py

# Smoke del candidato (para GPU)
/home/rekeyea/Documents/llm/qwen38-exl3-mia/.venv/bin/python \
  results/20260908-hybrid-backends/managed.py qg11-p2-smoke \
  /home/rekeyea/Documents/llm/qwen38-exl3-mia/.venv/bin/python \
  results/20260910-quality-gate/phase2_runner.py \
  --profile nvidia64 \
  --output results/20260910-quality-gate/phase2-smoke \
  --prompts results/20260910-quality-gate/prompts-phase2.json \
  --labels warmup,lru-32768-0
```

No lanzar otra cosa en GPU0. El servicio se detiene y se restaura.

## Smoke (`phase2-smoke`, nvidia64)

Warmup 4K + `lru-32768-0`. FlashInfer 0.6.18 / DSL 4.7.1, kernel P×256. Contadores: 51 PRIMS (todas Q=8192), 68 Flash, saturación E4M3 libre. 192 MLP, 64 grafos. TTFT 7,77 s (incluye JIT de primera vez), decode 214 tok/s, aceptación 0,810. Servicio restaurado.