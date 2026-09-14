# Plan: kernel de atención Qwasar sobre el ABI actual

> **For agentic workers:** implementar tarea por tarea. Los pasos usan
> checkbox (`- [ ]`). No editar el donante instalado. No promocionar a
> producción sin pasar los gates de este documento.

**Goal:** Sustituir el split de decode de `BCAttn` por un cubin Qwasar
que conserve el ABI de 15 argumentos y gane ≥15% en Q7/256K frente a
Attention64 del perfil `flash`. Si un cambio de política Triton o un
router Q=32/128 ya alcanza ese umbral, se adopta eso y el cubin queda
como trabajo posterior.

**Architecture:** Portar `_paged_attn_decode_split_kernel` (carga por
planos, `tl.dot`, softmax online FP32, layout de filas del donante) a
CuTe/CUDA SM120 con pipeline asíncrono. Reutilizar `k_combine` y
`k_update`. El wrapper `TritonKernel` carga el cubin. El algoritmo
original debe pasar primero por el mismo slot.

**Tech Stack:** Python 3.12, Torch 2.14.0+cu130, CUDA 13 / SM120,
ExLlamaV3 1.4.2 pinado, CuTe / CUTLASS del árbol local, Nsight
Systems/Compute. Reusar oráculos y lector de
`results/20260908-direct-k8v4/`.

**Spec:** [`../specs/2026-09-11-qwasar-attention-kernel.md`](../specs/2026-09-11-qwasar-attention-kernel.md).
Rechazo previo: [`2026-09-08-direct-k8v4-attention.md`](2026-09-08-direct-k8v4-attention.md)
y `results/20260908-direct-k8v4/report.md`.

## Premisa (no negociable)

No reimplementar V1–V3. Aquel kernel fue correcto y 1,63× más lento.
Attention64 ya lee K8/V4 directo. Un WMMA 16×16 que materializa tiles
N×256 en smem está prohibido como diseño inicial.

Orden de apuesta:

1. Agotar configs Triton sobre el `flash` actual.
2. Medir prefill directo vs Flash en Q=32/128/256.
3. Solo entonces escribir el cubin como **transcripción** del kernel
   Triton ganador, no como algoritmo nuevo.

## Restricciones globales

- GPU0 RTX 5090 / SM120. GPU1 no se usa.
- Batch 1, Q24/KV4/D256, página 256, pool 262144, MTP6, K8/V4.
- Perfil de referencia: `flash` = NVIDIA64 + PRIMS Q≥8192 + Attention64.
- Sin `.item()`, allocs, JIT ni autotune en el camino caliente.
- Presupuesto extra de buffers ≤ 64 MiB compartido entre capas.
  Sin espejo de contexto.
- Cada ventana GPU usa `managed.py` y restaura el servicio.
- No se modifica `qwen38-exl3-mia`. Parches solo en
  `results/20260911-attention-kernel/` hasta promoción.
- No commit salvo pedido explícito.

## Mapa de archivos

Crear en `results/20260911-attention-kernel/`:

| Archivo | Rol |
|---|---|
| `contract.json` | SHA256 del donante vigente + ABI reconfirmado |
| `policy_sweep.py` | Barrido `num_stages` / `block_n` de Attention64 |
| `midq_router.py` | Triton prefill vs Flash en Q=32/128/256 |
| `split.cu` / `split.cuh` | Cubin del split (planos + pipeline + MMA SM120) |
| `loader.cuh` | Copia verificada del lector 2026-09-08 o transcripción de `_qc_plane_*` |
| `backend.py` | Compila, carga `TritonKernel`, `prepare`/`run` |
| `slot.py` | Instala cubin en `BCAttn` sin tocar combine/update |
| `attention_gate.py` | Reusa fixtures/oráculos del rechazo |
| `bench.py` | ABBA Q7/256K vs Attention64 fresco |
| `model_probe.py` | MTP/rewind solo si el microbenchmark gana |
| `report.md` | Decisión |

Si gana el modelo completo, mover a `src/qwasar_runtime/attention/`
con selector explícito y bump de identidad. Esa promoción es una
tarea posterior con diff concreto.

Reusar, no reescribir: `results/20260908-direct-k8v4/{loader.cuh,attention_gate.py,contract.json,captures-fixed}`.
Gestor: copiar `results/20260908-hybrid-backends/managed.py` al
directorio nuevo.

---

## Task 0: Congelar el contrato contra el donante de hoy

**Files:** `results/20260911-attention-kernel/contract.json`

El `contract.json` de 2026-09-08 pinó SHA256 de `triton_paged.py`,
`bc_attn.py`, `attention.cpp` y `triton_kernel.cpp`. Volver a hashear.
Si driftó el ABI, parar y actualizar la spec antes del kernel.

- [x] Registrar SHA256 actuales de esos cuatro archivos y de
      `hybrid.DECODE_POLICY`.
- [x] Releer `attention.cpp:427-474`: args 0–14, parches 3/4/11/12/13,
      combine parche 4, grid = `splits_cap`, stride de partials =
      `num_splits` vivo.
- [x] Confirmar que `TritonKernel` sigue aceptando cubin + símbolo +
      `num_warps` + smem dinámico y añade dos punteros nulos.
- [x] Copiar la ecuación midpoint y el layout de planos
      (`_qc_plane_kt` / `_qc_load_kt`) al contrato. El lector V3 es
      válido como oráculo; el kernel nuevo debe cargar **por planos**,
      como Triton, no por grupo escalar.
- [x] Archivar `/config` del servicio `flash` (identidad
      `5bpw-K8V4-MTP-NV64-PRIMS-ATT64`) y restaurarlo.

**Entregable:** contrato vigente o stop por drift.

---

## Task 1: Agotar Attention64 sin escribir CUDA

**Files:** `policy_sweep.py`, `selection-policy.json`

`DECODE_POLICY` actual es `{block_n: 64, num_warps: 4, num_stages: 1}`.
El donante usa `num_stages=2` por defecto. Un stage extra es la
hipótesis más barata de overlap carga/MMA.

- [x] En ventana gestionada, cargar el stack `flash` (NVIDIA64 +
      Attention64 + PRIMS). No Minima.
- [x] Barrer solo valores legales de `attention_tuning.ALLOWED`:
      `block_n ∈ {32,64,128}`, `num_stages ∈ {1,2,3}`, `num_warps ∈ {4,8}`.
      Descartar configs que superen el presupuesto de smem del donante
      (`block_n = max(16, 8192/head_dim)` es la cota de tiles K+V).
- [x] Medir Q7/32K, Q7/128K, Q7/256K y Q1/256K. CUDA events, 5×20
      replays, orden ABBA contra `{64,4,1}`. Warmup fuera.
- [x] Revalidar L2 vs oráculo FP32 en la mejor config (umbral 0,003).
- [x] **Stop condicional:** si alguna config gana ≥15% en Q7/256K y
      Q1 queda dentro de 3%, promover esa política a `hybrid.py` y
      **no empezar el cubin**. Documentar y cerrar.
- [x] Si ninguna gana, congelar `{64,4,1}` como baseline del cubin y
      seguir.

**Entregable:** `selection-policy.json` con ganador o rechazo de la
pista Triton.

---

## Task 2: Router mid-Q (sin kernel nuevo)

**Files:** `midq_router.py`

Los contadores del 8 de septiembre: en Q=128 la materialización Flash
cuesta ~17 ms/capa y Flash mismo ~60 ms, con alta actividad Tensor
Core. El donante ya tiene prefill Triton que lee K8/V4 directo
(`paged_attn_triton_prefill` con `qc`).

- [x] Comparar, sobre el mismo KV ya appendeado a ~258K:
      Flash actual vs Triton prefill directo, Q ∈ {32, 64, 128, 256}.
- [x] Misma geometría, causal lower-right, sin cambiar chunk 8192 de
      la ingesta grande.
- [x] Si Triton gana ≥15% en Q=128 y pasa L2 0,003, proponer un
      router estático `17 ≤ Q < 512 → Triton directo` (umbrales
      exactos los fija la medición) y validarlo como cambio de
      `engine.tuning()`, separado del cubin Q≤8.
- [x] Si pierde, anotar: mid-Q sigue siendo Flash; el cubin no
      cubre esas formas en esta tanda.

**Entregable:** decisión de router o “Flash se queda en 17–8191”.

---

## Task 3: Transcribir el split Triton a CuTe/CUDA

**Files:** `loader.cuh`, `split.cu`, `backend.py`

Solo si Task 1 no promovió una política. El cuerpo del kernel es una
traducción literal de
`triton_paged.py:_paged_attn_decode_split_kernel` líneas 1108–1207
más `_qc_plane_kt`, `_qc_load_kt`, `_qc_load_v`, `_rot_h32`.

Prohibido en el primer cubin:

- `nvcuda::wmma` 16×16 como camino de QK/PV.
- Tile N×256 completo en smem antes del MMA (el fallo de V3).
- Combine propio.
- Autotune en caliente.
- FP8 interno.

Obligatorio:

- Carga por planos de bits, misma geometría de palabras que
  `_qc_plane_*`.
- MMA SM120 (CuTe) con operandos FP16 y acumulador FP32.
- Pipeline ≥2 etapas: prefetch del siguiente tile empaquetado
  mientras corre QK/PV del actual. Si smem no alcanza, aliasar K/V
  y documentar el recorte; no mentir overlap que no existe.
- Export C: mismos 15 parámetros + 2 punteros trailing.
- `blockDim = (32 * num_warps, 1, 1)`, grid `(programs, splits_cap, 1)`.
- Primera forma: Q7, `BLOCK_M=8, BLOCK_H=2, BLOCK_ROWS=16, programs=12`,
  `BLOCK_N=64` (el de Attention64, no el 32 del default donor).

- [x] Copiar `loader.cuh` de 2026-09-08 solo como oráculo de
      igualdad FP16. El kernel caliente usa planos.
- [x] Implementar `qwasar_split_q7` con la firma ABI. Compilar
      `-arch=sm_120`. Registrar ptxas: registros, smem, local mem.
      Spills > 0 son stop hasta reducir presión.
- [x] `backend.prepare(**kwargs)` reserva partials y carga el cubin
      **fuera** de captura. `run` no hace append ni `.item()`.
- [x] Tests CPU del empaquetado de args (orden, anchos int32 de
      11/12/13) en `tests/test_attention_kernel_abi.py` sin GPU.

**Entregable:** cubin que lanza; aún no se afirma velocidad.

---

## Task 4: Gate numérico (reusar la batería)

**Files:** `attention_gate.py`

Reusar fixtures de `results/20260908-direct-k8v4/`. No inventar otro
oráculo que comparta el lector nuevo.

- [x] 100 fixtures del lector por planos = igualdad FP16 exacta
      contra la ecuación midpoint y contra `_qc_load_*` de Triton.
- [x] Matriz chica: longitudes 7/31/32/255/256/257/4093, Q≤longitud,
      tablas permutadas, última página parcial, canarios.
- [x] Matriz larga: capturas reales del perfil **flash** (no
      reciclar Minima sin etiquetar). Q1/4/7 en capas 3, 31 y 63 a
      32K/128K/258K.
- [x] L2 relativo ≤ 0,003 vs `sampled_attention` FP32; registrar
      máximo por cabeza/fila. Comparar también vs Attention64 eager.
- [x] Replay de grafo: mutar longitud a 4093, cruzar página, cambiar
      tabla, restaurar. `replay/eager ≤ 1e-5`. KV/escalas byte a byte
      intactos.
- [ ] Compute Sanitizer: memcheck, synccheck, racecheck en fixtures
      chicos. Cero errores.  (no lanzado: Task 5 rechazó el cubin)

**Entregable:** atención correcta y capturable. Sin esto no hay bench.

---

## Task 5: Gate de velocidad del split

**Files:** `bench.py`, `selection.json`

La decisión usa **split + combine del donante**, no solo MMA.

- [x] Baseline fresco: Attention64 `{64,4,1}` sobre las mismas
      capturas flash. No reutilizar 0,813 ms de Minima.
- [x] Candidato: cubin Qwasar + combine donante. 5 rondas × 20
      replays, ABBA, clocks del equipo sin tocar.
- [ ] Nsight Compute en Q7/256K: DRAM, L2, Tensor Core, ocupación,
      registros, spills, long/short scoreboard. Misma forma en ambos
      brazos.  (omitido: 5× más lento; el rechazo no depende de ncu)
- [x] Criterio para seguir a integración:
      - Q7/256K total (split+combine) ≥15% más rápido.
      - Q1/256K dentro de 3% **o** router fijo que deja Q1 en Triton.
      - Combine no peor que +10% vs donante (si empeora, no
        sustituir combine; revisar escrituras de partials).
      - Actividad Tensor Core ≥ la del baseline (si baja, el cubin
        repite V3: cerrar).
- [x] Si no gana: `report.md` con rechazo y servicio restaurado.
      No hay Task 6–8.

**Entregable:** kernel ganador o rechazo medido. El 0,351 ms de XQA
FP8 aislado no es umbral.

---

## Task 6: Slot fusionado sin cambiar el algoritmo

**Files:** `slot.py`

- [ ] Instalar en `BCAttn._configure` un `TritonKernel` cuyo cubin
      es el **split original compilado** (o el cubin Triton
      extraído), no el nuevo. Comparar tiempo y salida contra el
      slot intacto. Cualquier delta se arregla antes de poner el
      cubin Qwasar.
- [ ] Cargar handles y buffers fuera de captura. Preservar grids,
      índices parcheados y orden de nodos (update → split → combine
      → gate).
- [ ] Insertar el cubin ganador **solo** en `k_split`. Combine y
      update siguen.
- [ ] Formas fuera de Q1–Q8: no sustituir el handle; el grafo de
      esas formas no existe en decode.

**Entregable:** el modelo graphed lanza el cubin Qwasar en verify.

---

## Task 7: Estado MTP y rewind

**Files:** `model_probe.py`

La lectura aislada no sustituye esto.

- [ ] Aceptación 0..6, rechazo total, rechazo parcial, cruce de
      página, prefijo cacheado, reset, cancelación.
- [ ] Comparar longitudes comprometidas, historial GDN/convolución,
      tokens y logits contra `flash` sin el cubin.
- [ ] Draft permanece en Attention64. Solo cambia atención target.
- [ ] Pares a 128K y 256K, mismas semillas y presupuestos.

**Entregable:** estado idéntico en las transiciones MTP.

---

## Task 8: Decisión de promoción

**Files:** `report.md`, bump condicional de identidad

- [ ] Medir tok/s aceptados, aceptación MTP, TTFT Q128, prefill 8K
      (PRIMS no debe regresionar >3%), pico de memoria.
- [ ] Matriz de código de fase 2 (32 celdas) + JSON 4/4. Candidato
      ≥ 19/32. No esconder regresiones de familia detrás del
      agregado. Aceptación ±5 pp.
- [ ] Adoptar solo si además hay ≥15% tok/s aceptados a 256K.
- [ ] Si gana solo Q7, router estático Q7 y revalidar esa
      combinación. Si falla calidad, no promover por velocidad.
- [ ] Promoción: `src/qwasar_runtime/attention/` + hook en
      `hybrid.attention_context()` + identidad
      `5bpw-K8V4-MTP-NV64-PRIMS-ATT64-QSPLIT`. Snapshots viejos no
      se reutilizan.
- [ ] Restaurar el servicio al perfil `flash` original si no se
      promueve. Archivar cubin, ptxas, ncu y hashes.

**Entregable:** decisión reproducible. FP8 interno y XQA quedan fuera.

---

## Criterios resumidos

| Gate | Umbral | Si falla |
|---|---|---|
| Task 1 política Triton | ≥15% Q7/256K | seguir al cubin |
| Task 2 mid-Q | ≥15% Q=128 | Flash se queda |
| Task 4 numérico | L2 ≤ 0,003, sanitizer limpio | no medir velocidad |
| Task 5 split | ≥15% vs Attention64 fresco; TC ≥ baseline | rechazo, no slot |
| Task 7 estado | rewind/MTP bit-compatibles en IDs | no end-to-end |
| Task 8 calidad | ≥19/32, JSON 4/4, aceptación ±5 pp | no promoción |

## Fuera de alcance

Fork de ExLlama, worker C++, megakernel, KV NVFP4, XQA runtime,
GDN, cabeza MTP 64K, cambiar MTP6, donante de pesos, commit no
pedido.
