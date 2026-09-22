# Fase 1 — Donante NVIDIA NVFP4: pasos y estado (documento de recuperación)

**Fecha:** 2026-09-14. Documento de continuidad tras el reinicio del servidor: qué se hizo, qué falta y con qué comandos se retoma cada paso. Contexto completo en [puerta Fase 0](2026-09-10-quality-gate.md) y [donante NVIDIA](2026-09-10-nvidia-donor.md).

## Objetivo de la fase

Decidir —con la puerta de calidad ampliada ya congelada— si el donante de MLP para el híbrido NVFP4 será **NVIDIA** o **Minima**, y luego promover el perfil ganador (donante + FP8 PRIMS + Attention64) como candidato a producción. Todo el trabajo es experimental y aislado; el servicio de producción (EXL3 5bpw + MTP6 + K8/V4) queda intacto y cada ventana GPU se ejecuta con `managed.py` (restauración verificada).

## Estado de los pasos

| # | Paso | Estado | Evidencia |
|---|---|---|---|
| 1 | Inspección de cabecera NVIDIA (sin pesos) | ✅ hecho (2026-09-10) | `results/20260910-nvidia-header/` (manifest, header consolidado, catálogo) |
| 2 | Fase 0: puerta ampliada + baseline EXL3 | ✅ hecho (2026-09-10) | baseline **11/16 código, JSON 2/2**, `results/20260910-quality-gate/control-baseline.json` |
| 3 | Descarga NVIDIA 3 shards + SHA-256 | ✅ hecho (2026-09-10) | `results/20260910-quality-gate/nvidia-download.json`; metadatos bajados |
| 4 | Adaptador ModelOpt→kernel (`nvidia_adapter.py`) | ✅ escrito; corrección de convención aplicada | ver hallazgo abajo |
| 5 | Oráculo numérico de 12 matrices MLP | ⏳ en corrección | fallos anteriores: (a) assert de geometría mal planteado, (b) NaN por convención inversa, (c) referencia EXL3 inexistente como BF16 plano |
| 6 | A/B NVIDIA64 vs Minima64 en la matriz congelada | ⬜ pendiente | `candidate_runner.py` listo |
| 7 | Puerta (`aggregate.py`) y decisión de donante | ⬜ pendiente | contra baseline 11/16 |
| 8 | Fase 2: perfil ganador + FP8 PRIMS + Attention64 | ⬜ pendiente | |

## Hallazgo crítico (ya corregido en el adaptador)

Las convenciones de escala global de NVIDIA (ModelOpt) y Unsloth/SGLang son **inversas entre sí**: NVIDIA guarda `w = q * block * global` (global ≈ 1,5e-4); Unsloth guarda `w = q * block / global` (global ≈ 6372). El kernel compartido `NativeLinear` espera la forma Unsloth. Corrección aplicada: el shim convierte `weight_global_scale → 1/global` al cargar. Verificado que ambas reconstrucciones coinciden (std 0,0101 vs 0,0102; absmax 0,4219 vs 0,4200).

## Por qué el oráculo anterior falló tres veces

1. **Assert de geometría:** confundí shape empaquetada con lógica; corregido (5120/16 = 320 escalas).
2. **NaN:** convención de escala inversa (punto anterior).
3. **Referencia inexistente:** el artefacto EXL3 **no contiene BF16 plano** (guarda `trellis/suh/svh/mul1` cuantizados); la referencia debe ser el **módulo EXL3 cargado por el runtime** ejecutando `forward` sobre las mismas entradas aleatorias. Esa es la corrección en curso en `nvidia_linear_check.py`.

## Comandos de retoma

```sh
# Oráculo numérico NVIDIA vs módulos EXL3 (ventana gestionada, ~10 min GPU)
cd /home/rekeyea/Documents/llm/qwasar
rm -rf results/20260910-quality-gate/nvidia-check
python3 results/20260908-hybrid-backends/managed.py qg10-nvchk4-managed \
  /home/rekeyea/Documents/llm/qwen38-exl3-mia/.venv/bin/python \
  results/20260910-quality-gate/nvidia_linear_check.py \
  --output results/20260910-quality-gate/nvidia-check

# A/B NVIDIA64 en la matriz congelada (~2,5 h GPU)
python3 results/20260908-hybrid-backends/managed.py qg10-nvidia64-managed \
  /home/rekeyea/Documents/llm/qwen38-exl3-mia/.venv/bin/python \
  results/20260910-quality-gate/candidate_runner.py \
  --profile nvidia64 --output results/20260910-quality-gate/nvidia64 \
  --prompts results/20260910-quality-gate/prompts.json

# Calificación + puerta
PYTHONPATH=src python3 results/20260910-quality-gate/graders.py nvidia64 --output grades-nvidia64
PYTHONPATH=src python3 results/20260910-quality-gate/aggregate.py \
  --control control,control-ttl2 --candidate nvidia64 \
  --grades-control grades-control,grades-control-ttl2 --grades-candidate grades-nvidia64 \
  --output gate-nvidia64.json
```

## Reglas operativas vigentes

- Nada se ejecuta en GPU sin `managed.py` (detiene el servicio solo estando libre y restaura con `finally`); el servicio debe quedar `ready` con MTP6 y config idéntica.
- Los directorios de resultados no se sobrescriben (`mkdir(exist_ok=False)`); etiquetas nuevas por intento.
- El runner de candidato mantiene PRIMS y attention64 **apagados** en este A/B: la pregunta es solo qué donante MLP es mejor; esas dos optimizaciones entran en Fase 2 sobre el donante ganador.
