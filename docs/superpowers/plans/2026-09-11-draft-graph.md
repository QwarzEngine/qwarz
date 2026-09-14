# Grafo CUDA del bucle draft MTP6 (2026-09-11)

Seguimiento de la palanca 3 de [la fase draft](2026-09-11-draft-phase.md):
capturar los seis pasos del draft en un grafo CUDA, sobre el port hot64k
(ids y head en GPU). Process-local; el donante no se toca.

Evidencia: `results/20260911-draft-phase/{draft_graph.py, graph01/, graph02-alternate/, report.md}`.

## Tareas

- [x] Estudiar qué del paso draft no es capturable. `BC_Attention::run` y
      `BC_GatedMLP::run_bszN` terminan en `cudaGraphLaunch` (no admitido en
      captura de stream; verificado: dos capturas invalidadas). `get_for_device`
      es no-op con tensores ya en GPU; `paged_attn_triton_decode` y
      `quant_cache_paged` no sincronizan (cota desde `block_table.shape[1]`).
- [x] `draft_graph.py`: `DraftGraph` sustituye `iterate_draftmodel_mtp_gen`;
      estáticos (`block` por cubos de 32 páginas, `seqlens` en GPU, `hidden`,
      `embed0`, `out`), `_bc_bypassed()` solo durante captura/validación,
      edición del input layer para `mtp_embedding_override`, fallback eager
      (batch ≠ 1, ventana ≠ 6, sin estado MTP, dos primeros verifies).
- [x] Puerta numérica: `--validate-graph N` (eager por la misma ruta dispatch
      y luego grafo sobre las mismas entradas). 60/60 verifies con los 6 ids
      idénticos (graph01: 40, graph02: 20).
- [x] Medir: humo entre procesos (`graph01`) y A/B intercalado en el mismo
      proceso (`graph02-alternate`, `--repeat 4 --alternate-graph`).
      32K −0,35 ms/verify (−1,5 %), 256K −0,76 (−1,9 %); draft GPU en su
      suelo de replay (2,93/2,86 y 4,39/4,34); verify inalterado; pico de
      memoria idéntico.
- [x] Documentar: anexo en `report.md`, sección en
      `docs/benchmarks/2026-09-11-draft-phase.md`, fila 4 de `plan-pasos`.

## Seguimiento: perfil y fusión de kernels

- [x] `--profile-graph`: tabla CUPTI de un replay (`graph03-profile/`): 167
      kernels, suma = tiempo de replay; 85–90 % pesos (sub-head 37 %) y
      atención; kernels pequeños ≈ 227 µs/verify, techo de fusión ≈ 90 µs.
- [x] `draft_fused.py`: `_prep_kernel` (gather + 2 RMSNorm + cat) y
      `_sample_kernel` (argmax + mapa + escrituras + seqlens), autotests
      (norm a un ulp en 17/655.360; argmax exacto con empates), enganche
      `mtp_fused_input` + `_body_fused`, `--fused --alternate-fused`.
- [x] Medir (`fused01-alternate/`): 167 → 127 kernels, −48 µs de kernel por
      verify, −0,09 ms/verify a 32K y −0,18 a 256K; 40/40 ids idénticos.
      Resultado: la fusión está agotada; lo que queda es bytes de pesos.

## Decisión

No promovido por sí mismo: no cambia aceptación ni memoria de hot64k, que
sigue rompiendo el criterio 4 (+0,86 GiB). Queda como componente listo para
acompañar a hot64k si algún día pasa la puerta (p. ej. con embeddings FP8).

## Reglas respetadas

GPU0 solo, ventanas por `managed.py` (servicio parado y restaurado, config
igual antes/después), sin `.item()` ni asignaciones en el camino caliente,
buffers extra < 1 MiB más el pool del grafo, sin editar el donante.
