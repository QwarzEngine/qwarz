# Fase 2 — NVIDIA64 + FP8 PRIMS + Attention64 (2026-09-11)

**Puerta ±5 pp: pasa.** Código 19/32 = 19/32, JSON 4/4, aceptación −0,78 pp, pico −0,46 GiB. TTFT cae ~47–50% en los cuatro contextos. Decode sube, más en largo (+16% a 128K, +21% a 256K). El ~132→69 s a 258K queda **medido** sobre este ganador: 133,5 → 67,3 s.

Control: EXL3 + Flash/8192, decode donor default32. Candidato: NVIDIA64 + Flash/8192 + PRIMS P×256 (solo Q=8192) + Attention64. MTP6, K8/V4, pool 262144, sampler recomendado, thinking medium. Matriz `prompts-phase2.json` (32 código + 4 JSON + warmup). 32K y 131072 tienen el mismo SHA-256 que Fase 0; 64K y near-256K son nuevos. Near-256K se clampéa a `262144 − max_tokens − 16`.

## Puerta

`gate-decision-phase2-pp5.json`:

| Criterio | Control | Candidato | Puerta |
|---|---:|---:|---|
| Código (32) | 19/32 | 19/32 | pasa (≥) |
| — lru / ring / bucket / ttl | 7/8 · 5/8 · 3/8 · 4/8 | 7/8 · 6/8 · 2/8 · 4/8 | |
| JSON | 4/4 | 4/4 | pasa |
| Aceptación MTP, mediana código | 0,715 | 0,707 (−0,78 pp) | pasa (±5) |
| Pico asignado | 26,09 GiB | 25,63 GiB (−0,46) | pasa |

Los fallos no coinciden celda a celda (mismo patrón de tests generados / truncados que en Fase 0). Empate 19/19 no es equivalencia de distribución.

## Velocidad (medianas, 9 celdas por contexto, warmup fuera)

| Contexto | TTFT control | TTFT candidato | Δ | Decode control | Decode candidato | Δ |
|---|---:|---:|---:|---:|---:|---:|
| 32K | 9,74 s | 5,17 s | **−47%** | 193,6 | 200,3 | +3,5% |
| 64K | 21,45 s | 11,27 s | **−47%** | 173,4 | 183,6 | +5,9% |
| 128K | 51,41 s | 26,63 s | **−48%** | 140,8 | 164,0 | +16,5% |
| 256K | 133,52 s | 67,27 s | **−50%** | 101,5 | 122,3 | +20,6% |

PRIMS: 8.568 llamadas, todas Q=8192; Flash 1.258 en restos y warmup. Saturación E4M3 libre. FlashInfer 0.6.18, DSL 4.7.1.

La aceptación de Fase 1 (NVIDIA64 solo, −4,3 pp) no se reproduce aquí: con Flash+PRIMS+Attention64 el corrimiento queda en −0,8 pp. No se atribuye a una palanca concreta sin un A/B que aísle cada pieza.

## Límites

- Una corrida por brazo, orden control → candidato. Sin inversión.
- La puerta no certifica coding general; bucket sigue siendo la familia más débil.
- Attention64 cambia el orden de reducción; en EXL3 solo había costado calidad. Encima de NVIDIA64 el agregado empata.
- Promovido a producción el 2026-09-11: el perfil `flash` del servicio es este
  candidato. `baseline` sigue siendo EXL3+Triton. La identidad de runtime cambió
  a `5bpw-K8V4-MTP-NV64-PRIMS-ATT64`; los snapshots EXL3 anteriores no se reutilizan.

## Artefactos

- Matriz: `results/20260910-quality-gate/prompts-phase2.json`
- Control: `phase2-control/` (`qg11-p2-control`)
- Candidato: `phase2-candidate/` (`qg11-p2-cand`)
- Smoke: `phase2-smoke/` (`qg11-p2-smoke2`)
- Grados: `grades-phase2-control/`, `grades-phase2-candidate/`
- Decisión: `gate-decision-phase2-pp5.json`
