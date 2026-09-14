# Kernel de atención Qwasar sobre el ABI de BCAttn

Estado: diseño de la segunda generación. El usuario pidió un plan de
implementación el 2026-09-11. Este documento fija la decisión de diseño;
el plan ejecutable está en
[`../plans/2026-09-11-qwasar-attention-kernel.md`](../plans/2026-09-11-qwasar-attention-kernel.md).

## Qué ya se sabe

El 2026-09-08 se implementó un kernel CUDA que lee K8/V4 residente, es
numéricamente correcto (L2 ≤ 0,000682 vs oráculo FP32) y **perdió 63%**
frente a Attention64: 1,324 ms vs 0,813 ms por capa en Q7 / 258.183
([informe](../../../results/20260908-direct-k8v4/report.md)).

Ese prototipo no competía contra un baseline ingenuo. Attention64
(`_paged_attn_decode_split_kernel` con `block_n=64`) **ya desempaqueta
K8/V4 dentro del kernel**, usa `tl.dot`, GQA y split-KV. V3 usó WMMA
16×16×16 clásico, tiles N×256 en shared memory y combine propio 4× más
lento. Nsight: ambos leen ~430 MB de DRAM; el perdedor tuvo menos
actividad Tensor Core (16,3% vs 25,8%).

Por tanto no se vuelve a escribir “un lector + softmax + WMMA”. El
trabajo es **portar el algoritmo que ya gana** a un cubin propio, con
pipeline asíncrono y MMA SM120, sin cambiar el ABI del slot fusionado.

## Decisión

Sustituir **solo** el handle `k_split` de `BC_Attention` por un cubin
Qwasar que exporta el mismo símbolo de 15 argumentos (+ 2 punteros
nulos del wrapper `TritonKernel`). Conservar `k_combine` y `k_update`
del donante hasta que el split gane aislado.

Baseline de comparación: perfil de producción `flash`
(NVIDIA64 + Flash/8192 + FP8 PRIMS Q≥8192 + Attention64), no Minima64.

Algoritmo a clonar, no a reinventar:

- Layout de filas del donante: `BLOCK_M = next_pow2(Q)`,
  `BLOCK_H = max(16/BLOCK_M, 1)`, `BLOCK_ROWS = 16`, `programs = 12` en Q7.
- Carga vectorizada por planos de bits (`_qc_plane_kt` / `_qc_plane_v`):
  palabras coalescidas, shifts broadcast, sin gathers.
- Reconstrucción midpoint en el dominio Hadamard, misma ecuación que
  `contract.json` de 2026-09-08.
- `tl.dot` equivalente: Q `[ROWS, D]` × K `[D, N]` y P `[ROWS, N]` × V `[N, D]`,
  acumuladores y softmax online en FP32.
- Causalidad: `total = cache_seqlens[0] + Q`; fila absoluta
  `total - Q + row_q`; token lógico `<=` esa posición.
- Partials: `partial_o` / `partial_ml` con stride = `num_splits` vivo,
  no `splits_cap`. Splits inactivos no escriben.

Lo que sí puede cambiar (y es la apuesta):

1. Software pipeline de ≥2 etapas: `cp.async` / TMA de palabras y
   escalas empaquetadas solapado con MMA. Attention64 de producción
   usa `num_stages=1`; el default del donante es 2. Primero se mide
   si subir stages en Triton ya alcanza el umbral.
2. MMA nativo SM120 (CuTe / `tcgen05` o el equivalente estable en
   CUDA 13), no `nvcuda::wmma` 16×16.
3. Combine del donante reutilizado. Un combine propio solo entra si
   el split ya ganó y Nsight atribuye el resto a la reducción.

## ABI que se conserva

Split, en este orden, más dos punteros nulos que añade
`triton_kernel.cpp`:

```
q, k_cache, v_cache, block_table, cache_seqlens, out,
partial_o, partial_ml, k_scales, v_scales, h32,
split_len, num_pages_per_seq, num_splits, sinks
```

El grafo parchea índices 3, 4, 11, 12, 13. Combine:
`partial_o, partial_ml, out, h32, num_splits, sinks`; parchea índice 4.

La ruta real es `BCAttn._configure` → `BC_Attention::run`, no solo
`paged_attn_triton_decode`. Interceptar el dispatcher Python no cambia
el modelo graphed. El cubin se instala como `TritonKernel` (el wrapper
carga cubin arbitrario). Antes de atribuir diferencias al kernel
nuevo, el slot debe lanzar el **algoritmo original** a través del
mismo wrapper.

## Formas

Primera forma: Q7, batch 1, Q24/KV4/D256, páginas 256, K8/V4, MTP6.
Después Q1–Q6 y Q8. Q>16 sigue en Flash/PRIMS; este kernel no reemplaza
ingesta 8K.

Un experimento barato, **antes** del cubin, mide el prefill Triton
directo K8/V4 del donante contra Flash en Q=32/128/256. Si gana, el
turno corto es un cambio de router, no otro kernel.

## Fuera de este cambio

FP8 interno de decode, KV NVFP4, XQA runtime, espejo de contexto,
offload, sparse, cambio de paginación, GDN, MTP, donante de pesos.
Un fork de ExLlamaV3.
