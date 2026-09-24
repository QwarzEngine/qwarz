# Grafo del bucle draft: certificación por equivalencia estadística (2026-09-23)

**Resultado: las 7 puertas congeladas pasan.** El grafo CUDA del bucle draft
diverge exactamente dentro de la banda de no-determinismo que la propia pila
eager exhibe contra sí misma a 32K+, y la ganancia de ciclo se reproduce con
4× las repeticiones de la campaña anterior. **No hay promoción automática:**
la decisión queda en el usuario.

Campaña: `results/20260923-graph-band/` · Máquina: RTX 5090 · Stack:
producción exacta (`xqa-KVNVFP4-MLPNV64-PRIMS-RDZ-MTP6-HOT64K`), único delta
process-local: el bucle draft (P producción con BC / G grafo / E eager
dispatch). Harness de `results/20260922-draft-graph/graph_runner.py` **sin
modificar**; puertas congeladas en `prediction.json` antes de medir
(SHA256 `9866d772…7382f8`).

## Cambio de criterio (decisión del usuario, no del experimento)

La campaña 09-22 demostró que la puerta bit-exacta es inexigible a 32K+:
la pila diverge contra sí misma en greedy sin grafo. El usuario eligió
**equivalencia estadística**: medir la banda eager-vs-eager y exigir que
grafo-vs-eager diverja igual o menos.

## Fase 1 — la banda (greedy, sin RNG, 6+6 corridas intercaladas por contexto)

| Contexto | Pares E-E divergentes | Mediana 1ª divergencia E-E | Pares G-E divergentes | Mediana G-E | Puerta |
|---|---|---|---|---|---|
| 32K | 15/15 | verify 9 (mín 9) | 36/36 | verify 9 (mín 8) | ✅ |
| 256K | 15/15 | verify 33 (mín 33) | 36/36 | verify 33 (mín 33) | ✅ |

A 32K+ **toda** corrida eager diverge de cualquier otra, siempre: la pila no
tiene referencia bit-estable que preservar. El grafo no la empeora: mismas
medianas de divergencia y, a 256K, 2/36 pares G-E incluso comparten
completion completa. (Las medianas difieren del verify 4-5 de ayer porque
celda y longitud difieren; la conclusión estructural es la misma.)

## Fase 2 — la ganancia (T>0, 4 reps ABBA por brazo, caché fresca)

| Contexto | Ciclo E−G (ms/verify) | Neto P→G (ms/verify) | tok/s neto mediano | Pares netos | Aceptación P→G | Δ memoria |
|---|---:|---:|---:|---|---|---:|
| 32K | **−1,125** | −0,556 | **+1,45%** | −0,3 / +9,0 / +3,2 / −9,9% | 0,779→0,773 | 0,000 GiB |
| 256K | **−0,919** | −1,149 | **+5,58%** | +7,1 / +8,1 / +4,1 / +0,5% | 0,805→0,817 | 0,000 GiB |

- El efecto-ciclo se reproduce y crece respecto a ayer (0,877→1,125 y
  0,454→0,919 ms/verify; las bandas congeladas [0,25–1,5]/[0,40–1,5] se cumplen).
- **Lectura honesta del neto:** a 32K el BC del draft ya recupera en
  producción parte del idle, así que el neto es pequeño y ruidoso (+1,45%
  mediano, 2 de 4 pares negativos). A 256K el neto es claro: los 4 pares
  positivos, +5,6% mediano, sin coste de aceptación ni memoria.
- A 4K ya se sabía: bit-exacto (60/60 ayer).

## Puertas

`BAND-32K`, `BAND-256K`, `CYCLE-32K`, `CYCLE-256K`, `NET-32K`, `NET-256K`,
`MEM`: **todas pasan** (`verdict.json`, análisis `qwasar_bench.graph_band`
fail-closed con 7 tests CPU). La validación intra-verify (mismo estado,
eager vs replay) quedó registrada pero fuera de puerta: no existe baseline
simétrica E-E mismo-estado en este harness.

## Operación

Una ventana gestionada de **37,1 minutos** (presupuesto 90), 4 procesos
secuenciales, restauración automática verificada (configuración idéntica,
`ready`, no ocupado). RTX 3090 Ti intacta. Suite completa: 507 tests pasan,
3 omitidos.

## Qué sigue

- **Decisión del usuario:** promover el grafo con el criterio estadístico
  (ganancia neta clara a 256K, marginal a 32K, sin coste medido) o dejarlo.
- Si se promueve, la **auditoría del split/combine de Attention64** sigue
  siendo la palanca para recuperar una puerta bit-exacta a 32K+ — hoy toda
  la pila (no sólo el grafo) es irreproducible bit a bit en multichunk.
