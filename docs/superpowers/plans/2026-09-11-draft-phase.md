# Plan: fase draft MTP6 sobre el perfil `flash`

> **For agentic workers:** predicción escrita antes de medir. No editar el
> donante instalado. Cada ventana GPU usa `managed.py`. No promover sin la
> matriz de fase 2.

**Goal:** recortar el coste fijo por verify de la fase draft (6 pasos MTP +
head compartido + 12 sincronizaciones host) y comprobar si el tok/s de
`flash` sube lo que predice el perfil del 2026-09-08. Es la Fase 4 de
[`2026-09-10-plan-pasos.md`](../../benchmarks/2026-09-10-plan-pasos.md).

**Baseline:** perfil `flash` de producción (NVIDIA64 + Flash/8192 + PRIMS
Q≥8192 + Attention64), MTP6, K8/V4, pool 262144. Referencia de calidad:
`results/20260910-quality-gate/phase2-candidate/` (19/32, JSON 4/4,
aceptación 0,707, pico 25,63 GiB).

**Palanca:** port process-local de ExLlamaV3 #303 ya validado en
`results/20260908-upstream-experiments/mtp/hot_head.py`: head de 65536
tokens (6 bpw, grupos Hadamard exactos), embeddings FP16 de esos tokens
en GPU, ids del draft en GPU, **una** copia host por verify. El
verificador conserva el vocabulario completo. Esto **cambia el
proponente**: la aceptación puede moverse.

## Lo que ya se sabe

| Dato | Fuente |
|---|---|
| Fase draft = 7,89 ms/verify de kernels (78 verifies Q7/258K) | `production-nsys4-report.1.analysis.json`, `groups` con `phase=qwasar::mtp_draft` |
| Head compartido 6 bpw 248320×5120 = 3,54 ms/verify (0,59 ms × 6) | ídem, `model=unmatched, module=Linear` |
| Atención del draft 3,49 (default32), MLP 0,61, fc/norm 0,22 | ídem |
| 12 D2H de 8 B + 6 H2D de 10 KiB + 18 `cudaStreamSynchronize` por verify | `memory_copies`, `cuda_api`; el embedding está en CPU (`prefer_cpu`) |
| 64K head en hybrid56+decode64: decode +7–11 % a 256K JSON, 10/10 JSON, LRU −1 % y aceptación 67,6→61,0 (una muestra) | `results/20260908-upstream-experiments/mtp/report.md` |
| Coste de memoria: +0,862 GiB (head 0,25 + embeddings FP16 0,67) | ídem |
| `flash` lru-32768-0: 214 tok/s, 5,82 tok/verify → 27,2 ms/verify | `phase2-candidate` |
| `flash` lru-258048-0: 121 tok/s, 5,22 tok/verify → 43,1 ms/verify | ídem |

## Predicción (fijada antes de medir)

Ahorro por verify con el head 64K + ids en GPU:

- Head: 3,54 → ~0,9 ms (4× menos bytes). **−2,6 ms.**
- Sincronizaciones: 12 D2H + gather CPU + 6 H2D desaparecen; el
  burbujeo GPU entre pasos baja. **−1 a −4 ms** (incierto).
- Total: **−3,5 a −6,5 ms/verify.**

Si la aceptación se mantiene (±2 pp):

| Forma | Baseline ms/verify | Predicho | tok/s predicho |
|---|---:|---:|---:|
| lru 32K | 27,2 | 20,7–23,7 | **+15 a +31 %** |
| lru 256K | 43,1 | 36,6–39,6 | **+9 a +18 %** |

Fase draft medida por eventos CUDA en el timeline: baseline 8–11 ms/verify;
candidato 4–6 ms/verify.

Riesgos declarados: aceptación en código (−6,6 pp en la única muestra
previa; la puerta es ±5 pp) y memoria (+0,86 GiB rompe el criterio 4
“≤ control + 0,5 GiB” si el control es `flash`; FP8 en embeddings baja
a ~+0,5 GiB y no está medido).

## Tasks

### Task 1: runner y A/B corto — `results/20260911-draft-phase/`

- [x] `draft_runner.py` = brazo candidato de `phase2_runner.py` +
      `--draft-head full|hot64k` + `--embedding-dtype` + medición por
      fase (`iterate_draftmodel_mtp_gen` / `iterate_gen`) con eventos
      CUDA y `perf_counter`, sin sincronizar dentro del bucle.
- [x] `prediction.json` con la tabla de arriba.
- [x] Celdas: `warmup`, `lru-32768-0`, `lru-258048-0`,
      `json-258048-0-cold`. Dos procesos frescos, full → hot64k (`ab01/`).
- [x] Comparado: fase draft −2,8 ms/verify en las tres formas (solo el
      head; sincronizaciones ≈0). ms/verify −10,8 % a 32K, −6,6 % a 256K.
      tok/s +19 % / +3,6 % / +9,7 %, dominado por ruido de aceptación.

**Stop:** si el 256K no gana ≥5 % o la aceptación cae >5 pp en lru,
cerrar con informe; no hay matriz. → Ambiguo en una celda (lru +3,6 %,
json +9,7 %, ms/verify −6,6 %); se corrió la matriz para resolverlo.

### Task 2: matriz de fase 2

- [x] 32 código + 4 JSON con `hot64k` (`matrix01-hot64k/`): 21/32,
      JSON 4/4, aceptación −3,93 pp, memoria +0,862 GiB.
- [x] Criterio 4 falla (+0,862 > +0,5 GiB). Decode pareado: +9,8 % a
      32K, +7,9 % a 64K, +3,8 % a 128K, **+2,6 % a 256K**.

### Task 3: decisión

- [x] **No promovido:** 256K queda en +2,6 % (< +5 %) y la memoria rompe
      el criterio 4. Informe: `results/20260911-draft-phase/report.md`.

## Fuera de alcance

Cambiar MTP6, DFlash, grafo del bucle draft completo, head NVFP4, cache
del draft, atención del target.
