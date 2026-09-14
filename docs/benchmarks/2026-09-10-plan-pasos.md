# Pasos en curso — donante NVIDIA y puerta de calidad (2026-09-10)

Documento de handoff pedido al interrumpir la sesión Kimi (`ses_f74c03390ffeNQf8bGW1jSDBh3`, `kimi-k3`, idle_outcome=`interrupted` a las 14:05). El último pedido en esa sesión fue dejar por escrito los pasos que se estaban siguiendo. Este archivo es esa lista, verificada contra el repo y los artefactos, no contra el transcript.

## Objetivo

Mejorar Qwasar (Qwen 3.8 27B, RTX 5090, 32 GB, MTP6, K8/V4, 256K) sin sustituir el artefacto EXL3 de producción. El donante NVIDIA NVFP4 aporta **sólo las 192 matrices MLP**. GDN, atención, embeddings, lm_head y MTP siguen en EXL3.

## Camino (fases)

| Fase | Qué | Estado |
|---|---|---|
| 0 | Puerta de calidad ampliada + baseline EXL3 | **Congelada.** Control 11/16 código, JSON 2/2, pico 24,79 GiB. |
| 1 | Donante NVIDIA MLP64: shim, oráculo, A/B vs Minima64 sobre la misma matriz | **Cerrada.** `candidate-nvidia64e`: 11/16, JSON 2/2, −0,98 GiB, TTFT −42%/−27%, decode a la par, aceptación −4,3 pp. Minima64: 8/16, −5,8 pp. **2026-09-11:** el usuario relajó el criterio 3 a ±5 pp; NVIDIA64 pasa, Minima64 no. Ver [donante](2026-09-10-nvidia-donor.md). |
| 2 | NVIDIA64 + FP8 PRIMS (P×256) + Attention64 a 32K/64K/128K/256K | **Medida. Puerta ±5 pp pasa.** 19/32 = 19/32, JSON 4/4, aceptación −0,78 pp, −0,46 GiB. TTFT −47/−47/−48/−50% a 32/64/128/256K; decode +4/+6/+16/+21%. 258K 133,5→67,3 s. Ver [resultados](2026-09-11-phase2-results.md). Producción no cambiada. |
| 3 | Decode: XQA + KV NVFP4 en target, resolviendo antes el runtime de cola FP16 2K | No empezada. Bloqueo conocido: XQA local no expone LSE; el prototipo SGLang anterior tuvo acceso ilegal. |
| 4 | Cabeza MTP 64K (la memoria la paga Fase 3). Métrica: tiempo por token aceptado | **Medida sobre `flash`, no promovida (2026-09-11).** −2,8 ms/verify constantes; aceptación −3,9 pp; decode +9,8 % a 32K, +2,6 % a 256K; 21/32, JSON 4/4; memoria +0,86 GiB rompe el criterio 4. Grafo CUDA del bucle de 6 pasos medido encima: −0,35 ms/verify a 32K, −0,76 a 256K, ids idénticos; fusión Triton de los kernels pequeños: −0,09 / −0,18 más (167 → 127 kernels). La fase draft queda limitada por bytes de pesos (sub-head 37 %). Ver [fase draft](2026-09-11-draft-phase.md). |
| 5 | Quitar los 16 `.item()` de `cache_seqlens` y auditar el hand-off MTP target→draft | **`.item()` eliminado y verificado** (eran 17 por chunk, incluida la capa MTP): 3.468/3.468 aciertos, 0 sincronizaciones, sin cambio medible de TTFT. Ver [resultados](2026-09-10-item-sync-results.md). Auditoría del hand-off MTP pendiente. |
| 6 | Vigilancia: llama.cpp #28572, vLLM #52244, FlashInfer K/V mixto | No empezada. |

Fuera de alcance explícito: lm_head NVFP4 de NVIDIA, GDN/atención FP8 en nuestro runtime, migración de artefacto completo.

## Criterio de promoción (Fase 0, fijado antes de medir)

Un candidato pasa si y solo si, contra el control EXL3 en `results/20260910-quality-gate/`:

1. Código: candidato ≥ control en entregas completas sobre 16 celdas (control = **11/16**).
2. JSON: 2/2.
3. Aceptación MTP: mediana de `draft_acceptance` a **±5 pp** del control (enmienda 2026-09-11; el candado original era ±3).
4. Memoria: pico asignado ≤ 24,79 + 0,5 GiB.

La decide `results/20260910-quality-gate/aggregate.py`. Empate agregado no demuestra equivalencia: se reportan fallos por celda.

## Dónde se cortó (Fase 1)

1. Checkpoint `nvidia/Qwen3.8-27B-NVFP4` rev `dbb8f445` descargado y SHA-256 ok (3 shards, 20,42 GiB). Ver `results/20260910-quality-gate/nvidia-download.json`.
2. Hallazgo medido: NVIDIA (ModelOpt) guarda el global **multiplicativo** (~1,5e-4); Unsloth/SGLang guarda la recíproca y **divide**. `NativeLinear` (`alpha = 1/(input*global)`) está escrito para Unsloth. El shim convierte `weight_global_scale → 1/global` al cargar (`nvidia_adapter.py`).
3. El oráculo tipo Minima (kernel vs referencia construida con los mismos operandos cuantizados) no distingue una convención errada: referencia y kernel fallan igual y el cheque pasa. Por eso se cambió a una referencia independiente.
4. Tres corridas GPU vía `managed.py` (el servicio se restauró después de cada una):

| Corrida | Fallo |
|---|---|
| `qg10-nvchk-managed` | assert de geometría: `weight_scale` [17408, 320] vs unpacked [17408, 5120]; el assert multiplicaba mal por 2 |
| `qg10-nvchk2-managed` | NaN en `gate_proj` M=1 — convención de escala aún invertida en el unpack |
| `qg10-nvchk3-managed` | `KeyError: model.language_model.layers.0.mlp.gate_proj.weight` |

El error final es de nombres, no de GPU: el artefacto EXL3 **no guarda** `.weight` BF16. Las claves reales son `trellis` / `suh` / `svh` / `mul1` bajo `model.language_model.layers.{i}.mlp.{proj}.*`. La reconstrucción BF16 de EXL3 es `LinearEXL3.get_weight_tensor()` (shape `[in, out]`).

`docs/benchmarks/2026-09-10-nvidia-donor.md` quedó diciendo «oráculo en curso»; eso era falso: el directorio `nvidia-check/` está vacío y las tres corridas salieron con returncode 1.

## Próximos pasos concretos (en este orden)

1. ~~Oráculo de dos puertas~~ **Hecho, y ampliado a cuatro** (18:00). Las puertas A/B pasaron con un adaptador que dejaba `input_scale` sin invertir; el A/B `candidate-nvidia64b` salió ruido en 19/19 celdas. Lección: un oráculo que cuantiza la referencia con los mismos globales que el kernel no detecta convenciones erradas; hace falta una referencia con activaciones sin cuantizar (Puerta C), una ida y vuelta de activaciones (Puerta D) y un brazo rojo que reproduzca el bug conocido. `nvidia-check10`: A 48/48, B 12/12, C 48/48 (máx 0,098), D 48/48 (máx 0,096), rojo 8/8 = 1,0.
2. ~~A/B NVIDIA-MLP64~~ **Hecho** (`candidate-nvidia64d` eager, `candidate-nvidia64e` con grafos MLP). El runner omitía `graph_selected_mlps()` y `64d` perdió −17% de decode a 32K por lanzamientos Python; `64e` lo recupera (−3%/+2%). El runner captura grafos por defecto.
3. ~~Calificar y decidir~~ **Hecho.** `aggregate.py` tenía dos bugs (nunca había corrido): `KeyError` en el criterio JSON y campo de aceptación nulo; corregidos. `gate-decision-nvidia64e.json`: criterios 1, 2, 4 pasan; 3 falla (−4,3 pp).
4. ~~Ganador~~ **NVIDIA64** (Minima64 en la misma matriz: 8/16, −5,8 pp, 4 truncados). La muestra LRU de 6 celdas había sobreestimado a Minima.
5. ~~Decisión del criterio 3~~ **Hecha (2026-09-11):** banda ±5 pp. `gate-decision-nvidia64e-pp5.json` pasa; `gate-decision-minima64a-pp5.json` sigue fallando (código 8/16 y −5,8 pp). NVIDIA64 es el donante de Fase 2. El corrimiento de aceptación se documenta, no se “arregla”: el drafter MTP es EXL3 y fijo.

Comando del oráculo:

```bash
/home/rekeyea/Documents/llm/qwen38-exl3-mia/.venv/bin/python \
  results/20260908-hybrid-backends/managed.py qg10-nvchk5-managed \
  /home/rekeyea/Documents/llm/qwen38-exl3-mia/.venv/bin/python \
  results/20260910-quality-gate/nvidia_linear_check.py \
  --output results/20260910-quality-gate/nvidia-check5
```

## Fase 2 — primer A/B (aún no lanzado)

Misma matriz, mismo `candidate_runner` con PRIMS y Attention64 **encendidos** sobre NVIDIA64 (en Fase 1 iban apagados a propósito). Control: `control` + `control-ttl2`. Criterio vivo: ±5 pp.

Orden de trabajo:

1. Extender `candidate_runner.py` con `--fp8-prims` y `--attention64` (parche P×256, router Q=8192, perfil `decode64`). No mezclar XQA ni cabeza MTP 64K.
2. Smoke corto (warmup 4K + una celda 32K) vía `managed.py` antes de la matriz.
3. A/B completo `candidate-nvidia64e-prims-att64` → `aggregate.py --acceptance-pp 5`.
4. No reclamar el TTFT 258K ~132→~69 s del híbrido Minima hasta medirlo otra vez. La puerta de Fase 2 no incluye 258K; esa verificación va después si la matriz pasa.

El servicio se detiene durante cada corrida GPU y `managed.py` lo restaura en MTP6. No lanzar otra cosa en GPU0 a la vez.

## Artefactos

- Puerta: `docs/benchmarks/2026-09-10-quality-gate.md`, `results/20260910-quality-gate/`
- Cabecera NVIDIA (sin pesos): `results/20260910-nvidia-header/`, `docs/benchmarks/2026-09-10-nvidia-donor.md`
- Adaptador: `results/20260910-quality-gate/nvidia_adapter.py`
- Runner A/B NVIDIA: `results/20260910-quality-gate/candidate_runner.py`
- Logs de las tres fallas: `results/20260908-hybrid-backends/qg10-nvchk{,2,3}-managed/`
