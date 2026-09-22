# Fase 3 — XQA + KV NVFP4 en target: plan (2026-09-14)

**Objetivo:** llevar el decode XQA (+27–37% medido en septiembre sobre la base híbrida vieja) a la base productiva actual (`5bpw-K8V4-MTP-NV64-PRIMS-ATT64`), conservando la puerta de calidad. Es la única palanca grande pendiente del plan; Fases 0–2 y 4–5 están cerradas.

## Base de partida (verificada)

- Producción: NVIDIA64 MLP + Flash/PRIMS (P×256, solo Q=8192) + Attention64, MTP6, K8/V4, pool 262144. TTFT 256K 67,3 s; decode 256K 122,3 tok/s.
- Septiembre (base vieja MLP56+Flash): target NVFP4 + XQA dio **+26,8% decode frío / +30,0% caliente** y **−0,87 GiB**; TTFT ≈ igual (107,9 vs 107,4 s); código empatado 4/6; JSON 2/2.
- Bloqueos conocidos: XQA no expone LSE (verificado hoy en la 0.6.18 instalada); error de atención ~4,5% sin cola FP16 (~1,1–1,2% con cola de 2K, solo validado numéricamente); el prefill actual (PRIMS) consume KV derivado de K8/V4 — con caché NVFP4 hay que alimentarlo con un gather NVFP4→FP16→FP8.

## Etapas

**Etapa A — port y re-medición (sin cola FP16).**
1. Portar `results/20260908-upstream-experiments/xqa/adapter.py` a la base nueva: donante NVIDIA64 (vía `nvidia_adapter.py`), PRIMS para prefill (gather NVFP4→FP16→FP8 reutilizando la ruta septiembre), Attention64, draft K8/V4.
2. Puertas numéricas de septiembre (`adapter_check.py`: append, páginas permutadas, rewind con grafos) sobre la base nueva.
3. Smoke + matriz de Fase 2 (32 código + 4 JSON) con la misma puerta ±5 pp.
   - **Si pasa la puerta:** candidato a promoción; Etapa B queda opcional.
   - **Si falla por calidad/aceptación:** Etapa B.

**Etapa B — cola FP16 2K (solo si A falla).**
Combinación exacta NVFP4-viejo + FP16-reciente. Requiere LSE de XQA: parche JIT al módulo XQA (escribir m/l por (b,h,q)) o combinador Triton propio que recompute el viejo. Diseño detallado se fija con los números de la Etapa A en la mano; no se construye por adelantado.

## Riesgos y mitigaciones

1. **Prefill:** con caché NVFP4 el prefill pasa de K8/V4→FP16 a NVFP4→FP16→(FP8 PRIMS). El gather de septiembre costaba ~17,5 ms/capa a 258K; PRIMS ahorró ~31–35% de ingesta. Riesgo de perder parte del TTFT ganado en Fase 2 — se mide en A/3.
2. **Aceptación MTP:** la numérica de atención cambia; criterio ±5 pp. Septiembre no mostró deriva significativa (empate 4/6).
3. **Memoria:** NVFP4 KV libera ~2 GiB lógicos de pool (−0,87 GiB medido de pico); la puerta exige ≤ control + 0,5 GiB.
4. **Interacción con Attention64:** XQA reemplaza la ruta de decode; attention64 deja de aplicar en el target (sí en draft). Documentar la configuración exacta resultante.

## Fuera de alcance

Reintento de la cabeza MTP 64K (la memoria liberada podría pagarla, pero su −3,9 pp de aceptación ya se midió; solo se reabre con mapa recalibrado, decisión separada). XQA en draft. KV NVFP4 mixto K8/NVFP4 (kernel no lo soporta, verificado en septiembre).

## Artefactos previos

- Comparación final septiembre: `results/20260908-upstream-experiments/xqa/final-comparison.md`.
- Adaptador y puertas: `results/20260908-upstream-experiments/xqa/adapter.py`, `adapter_check.py`, `adapter-gate-data2/`.
- Diagnóstico cola FP16: `results/20260908-upstream-experiments/xqa/long-comparison.md` y `attention-interim.md`.
- Factibilidad K/V mixto: `results/20260908-combined-profile/mixed-kv-feasibility.md`.
