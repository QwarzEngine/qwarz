# Fase draft MTP6: head de 64K sobre `flash` (2026-09-11)

**Resultado:** el head reducido de 65536 tokens con ids del draft en GPU
(port process-local de ExLlamaV3 #303) recorta la fase draft **2,8 ms por
verify** en todos los contextos, exactamente lo que predecía el perfil
para el head y nada por las sincronizaciones. La aceptación baja una
mediana de 3,9 pp en código, así que el decode neto sube +9,8 % a 32K
pero solo **+2,6 % a 256K**. Pasa código (21/32), JSON (4/4) y
aceptación (±5 pp), pero cuesta +0,862 GiB y rompe el criterio de
memoria. **No se promueve.** Producción sigue en
`5bpw-K8V4-MTP-NV64-PRIMS-ATT64`.

Plan y predicción previa: [`plans/2026-09-11-draft-phase.md`](../superpowers/plans/2026-09-11-draft-phase.md).
Informe completo: [`results/20260911-draft-phase/report.md`](../../results/20260911-draft-phase/report.md).

## Punto de partida

Perfil nsys de producción (Q7/258K, 78 verifies): fase draft 7,89
ms/verify de kernels; head compartido 248320×5120 a 6 bpw 3,54 ms
(6 pasadas), atención del draft 3,49, MLP 0,61. El embedding vive en
CPU: 12 copias D2H, 6 H2D y 18 `cudaStreamSynchronize` por verify.
Predicción: −2,4…−2,8 ms por head, −1…−4 ms por sincronizaciones.

## Medido

A/B corto (procesos frescos, `ab01/`):

| Celda | ms/verify | Fase draft GPU | tok/s | Aceptación |
|---|---:|---:|---:|---:|
| lru-32768-0 | 25,99 → 23,18 | 6,04 → 3,24 | 210 → 251 | 74,6 → 80,4 |
| lru-258048-0 | 43,14 → 40,31 | 7,91 → 5,10 | 129 → 134 | 76,4 → 73,4 |
| json-258048-0-cold | 42,66 → 40,06 | 7,89 → 5,09 | 133 → 146 | 78,6 → 80,9 |

Matriz de fase 2 (`matrix01-hot64k/` contra `phase2-candidate`,
medianas pareadas de 8 celdas de código):

| Contexto | ms/verify | tok/s | Aceptación (pp) |
|---|---:|---:|---:|
| 32K | −12,8 % | **+9,8 %** | −1,6 |
| 64K | −9,8 % | +7,9 % | −3,3 |
| 128K | −8,5 % | +3,8 % | −6,0 |
| 256K | −6,6 % | **+2,6 %** | −4,5 |

Puerta (`gate-decision-hot64k-pp5.json`): código 21/32 ≥ 19/32, JSON
4/4, aceptación 0,707 → 0,668 (−3,93 pp), memoria 25,63 → 26,49 GiB
(**falla**, límite +0,5).

## Lectura

- El coste del head era real y la reducción de bytes lo elimina
  linealmente. La palanca de sincronizaciones no existía: el host ya
  iba solapado, y el draft conserva ~1–1,4 ms/verify de huecos entre
  pasos que solo un grafo del bucle de 6 pasos quitaría.
- A 256K la fase draft residual (5,1 ms) es sobre todo atención Q1 × 6
  (~2 ms); el verify (35,4 ms) no cambia. Por eso la ganancia relativa
  cae con el contexto.
- El mapa de 64K se calibró el 2026-09-08 con dos prompts; el −4 pp de
  aceptación es la factura de esa calibración estrecha, no del mecanismo.

## Grafo CUDA del bucle de 6 pasos (medido, mismo día)

`results/20260911-draft-phase/draft_graph.py` captura los seis pasos del
draft (sobre hot64k) en un `torch.cuda.CUDAGraph`. Los caminos BC de la
atención y la MLP del draft no son capturables (terminan en
`cudaGraphLaunch`), así que dentro de la captura se desvían a sus rutas
eager; la atención sigue con Attention64 porque `graph_attention_context`
también reescribe `paged_attn_triton_decode`. `block_table` con anchura
por cubos de 8K tokens (recaptura de 6–7 ms al cambiar de cubo),
`cache_seqlens` incrementado en GPU, embedding del token 0 desde CPU a
un buffer estático, una copia D2H por verify. 60/60 verifies con las seis
propuestas idénticas al eager; pico de memoria igual.

A/B intercalado en el mismo proceso (`graph02-alternate/ab-summary.json`,
2 × grafo y 2 × eager por celda):

| Celda | ms/verify eager → grafo | Δ | draft GPU | suelo replay |
|---|---|---|---|---|
| lru 32K | 23,47 → 23,12 | −0,35 (−1,5 %) | 3,35 → 2,93 | 2,86 |
| lru 256K | 40,68 → 39,92 | −0,76 (−1,9 %) | 5,15 → 4,39 | 4,34 |

La fase draft queda en su suelo de kernels; el residuo ya no es
planificación sino la latencia de ~150 kernels pequeños por verify. El
verify no cambia. No altera la decisión sobre hot64k (aceptación y memoria
iguales). Hallazgo colateral: a 32K el verify está casi limitado por host
(19,9 ms host frente a 20,1 ms GPU).

## Perfil del replay y fusión de kernels (medido, mismo día)

Perfil CUPTI de un replay (`graph03-profile/`): 167 kernels por verify,
suma de kernels igual al tiempo de replay (sin huecos). Reparto a 32K:
sub-head 6 bpw 1.046 µs (37 %), gemv 4 bpw del bloque MTP 986 µs, k/v
94 µs, atención 507 µs (1.993 a 256K), ~107 kernels pequeños ≈ 227 µs.

`results/20260911-draft-phase/draft_fused.py` fusiona lo que queda fuera
del `TransformerBlock` del donante en dos kernels Triton: preparación de
entrada (gather + dos RMSNorm + cat → entrada del `fc`; aritmética de
`norm.cu`, 17/655.360 elementos a un ulp en el autotest) y muestreo
(argmax con empates como `torch.argmax` + mapa + escrituras +
`cache_seqlens += 1`; autotest exacto). 40/40 verifies con ids idénticos.

A/B intercalado fusión / grafo (`fused01-alternate/ab-summary.json`):

| Celda | ms/verify grafo → fusión | Δ | kernels/replay | µs kernel |
|---|---|---|---|---|
| lru 32K | 23,06 → 22,97 | −0,09 (−0,4 %) | 167 → 127 | 2.857 → 2.807 |
| lru 256K | 39,83 → 39,65 | −0,18 (−0,5 %) | 167 → 127 | 4.341 → 4.293 |

La fase draft queda agotada por planificación y fusión: lo que resta es
lectura de pesos (sub-head + bloque MTP ≈ 2,1 ms) y atención. Acumulado
hot64k + grafo + fusión frente a `flash`: −11,6 % a 32K, −4,8 % a 256K;
la decisión de no promover no cambia.

## Siguientes pasos posibles (sin ejecutar)

1. Recalibrar el mapa con las cuatro familias de la matriz y JSON, o
   subir a 96K/128K tokens (ahorro 1,8–2,3 ms en vez de 2,8, aceptación
   más cercana a la completa).
2. Embeddings FP8 (−0,31 GiB; queda +0,55 GiB; numéricamente sin medir).
3. ~~Grafo CUDA del bucle draft~~ Medido: −0,35 / −0,76 ms/verify.
   ~~Fusión de kernels~~ Medida: −0,09 / −0,18 ms/verify.
4. Menos bytes en el draft: sub-head recuantizado a 4 bpw (~−0,35
   ms/verify, exige recuantizar 64K filas y medir aceptación) o
   vocabulario caliente de 32K (~−0,5 ms, más pérdida de aceptación).
