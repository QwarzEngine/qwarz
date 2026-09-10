# Pasos en curso — donante NVIDIA y puerta de calidad (2026-09-10)

Documento de handoff pedido al interrumpir la sesión Kimi (`ses_f74c03390ffeNQf8bGW1jSDBh3`, `kimi-k3`, idle_outcome=`interrupted` a las 14:05). El último pedido en esa sesión fue dejar por escrito los pasos que se estaban siguiendo. Este archivo es esa lista, verificada contra el repo y los artefactos, no contra el transcript.

## Objetivo

Mejorar Qwasar (Qwen 3.8 27B, RTX 5090, 32 GB, MTP6, K8/V4, 256K) sin sustituir el artefacto EXL3 de producción. El donante NVIDIA NVFP4 aporta **sólo las 192 matrices MLP**. GDN, atención, embeddings, lm_head y MTP siguen en EXL3.

## Camino (fases)

| Fase | Qué | Estado |
|---|---|---|
| 0 | Puerta de calidad ampliada + baseline EXL3 | **Congelada.** Control 11/16 código, JSON 2/2, pico 24,79 GiB. |
| 1 | Donante NVIDIA MLP64: shim, oráculo, A/B vs Minima64 sobre la misma matriz | **Oráculo pasado** (`nvidia-check7`). A/B pendiente. |
| 2 | Ganador de Fase 1 + FP8 PRIMS (P×256) + Attention64 + K8/V4 + MTP6 → producción | No empezada. Esperado (medido en híbrido previo): TTFT 258K ~132→~69 s no se reclama hasta medirlo otra vez sobre el ganador. |
| 3 | Decode: XQA + KV NVFP4 en target, resolviendo antes el runtime de cola FP16 2K | No empezada. Bloqueo conocido: XQA local no expone LSE; el prototipo SGLang anterior tuvo acceso ilegal. |
| 4 | Cabeza MTP 64K (la memoria la paga Fase 3). Métrica: tiempo por token aceptado | No empezada. |
| 5 | Quitar los 16 `.item()` de `cache_seqlens` y auditar el hand-off MTP target→draft | **`.item()` eliminado y verificado** (eran 17 por chunk, incluida la capa MTP): 3.468/3.468 aciertos, 0 sincronizaciones, sin cambio medible de TTFT. Ver [resultados](2026-09-10-item-sync-results.md). Auditoría del hand-off MTP pendiente. |
| 6 | Vigilancia: llama.cpp #28572, vLLM #52244, FlashInfer K/V mixto | No empezada. |

Fuera de alcance explícito: lm_head NVFP4 de NVIDIA, GDN/atención FP8 en nuestro runtime, migración de artefacto completo.

## Criterio de promoción (Fase 0, fijado antes de medir)

Un candidato pasa si y solo si, contra el control EXL3 en `results/20260910-quality-gate/`:

1. Código: candidato ≥ control en entregas completas sobre 16 celdas (control = **11/16**).
2. JSON: 2/2.
3. Aceptación MTP: mediana de `speculative_acceptance_rate` a ±3 pp del control (si el candidato la expone; el control no la registró).
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

1. **Oráculo de dos puertas** (`nvidia_linear_check.py`), vía `managed.py`:
   - **Puerta A (kernel/convención):** unpack ModelOpt multiplicativo vs `NativeLinear` con global invertido, W4A4, umbral RMS 0,003. `qg10-nvchk4` ya midió A=0,0 en `gate_proj` M=1 — la inversión es correcta. Falta la matriz completa de 12×4.
   - **Puerta B (packing):** coseno en espacio de pesos vs Minima (todas las capas) y Unsloth (sólo MLP 0–55; 56–63 no están empaquetados). Umbral > 0,95. Layer 0 `gate_proj` en CPU: NVIDIA vs Minima **0,993**, vs Unsloth **0,988**. L2 de pesos ~12–15% (calibraciones distintas, no un bug de kernel). El bound del 5% contra EXL3 BF16 era inviable: EXL3 no tiene `.weight` y el L2 entre donantes NVFP4 ya es ~12%.
2. Si ambas puertas pasan: A/B **NVIDIA-MLP64 vs Minima64** sobre `prompts.json` congelado. FP8 PRIMS y Attention64 apagados. Los MLP están muy cerca en espacio de pesos; el A/B sigue siendo la decisión de calidad e2e.
3. Calificar con `graders.py`, decidir con `aggregate.py` contra el baseline 11/16.
4. El ganador entra a Fase 2.

Comando del oráculo:

```bash
/home/rekeyea/Documents/llm/qwen38-exl3-mia/.venv/bin/python \
  results/20260908-hybrid-backends/managed.py qg10-nvchk5-managed \
  /home/rekeyea/Documents/llm/qwen38-exl3-mia/.venv/bin/python \
  results/20260910-quality-gate/nvidia_linear_check.py \
  --output results/20260910-quality-gate/nvidia-check5
```

El servicio de producción se detiene durante la corrida y `managed.py` lo vuelve a dejar en MTP6. No lanzar otra cosa en GPU0 a la vez.

## Artefactos

- Puerta: `docs/benchmarks/2026-09-10-quality-gate.md`, `results/20260910-quality-gate/`
- Cabecera NVIDIA (sin pesos): `results/20260910-nvidia-header/`, `docs/benchmarks/2026-09-10-nvidia-donor.md`
- Adaptador: `results/20260910-quality-gate/nvidia_adapter.py`
- Runner A/B NVIDIA: `results/20260910-quality-gate/candidate_runner.py`
- Logs de las tres fallas: `results/20260908-hybrid-backends/qg10-nvchk{,2,3}-managed/`
