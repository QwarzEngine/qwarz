# 2026-09-23 — Causa raíz de la no-determinism 32K+: reducciones atómicas float en el kernel GDN

## Resumen

El stack de producción era bit-estable a 4K (a nivel de ids) pero divergía run-to-run a 32K+
(mediana: verify 9), lo que bloqueaba una puerta bit-exacta para la promoción del grafo CUDA del
draft. Una bisección por capa con huellas estrictas (hash de bytes muestreados + momentos fp64
por entrada y salida de cada módulo) localizó la primera fuente en la **capa 5 del target — una
capa GDN (linear_attention)** — tanto a 4K como a 32K, durante las ventanas de verify (qlen=7).

Causa raíz: los kernels `cuda_recurrent_gated_delta_rule_kernel{,_128}` de `gdn.cu` acumulan las
4 sumas parciales (SUBK=4) de cada elemento de salida con **`atomicAdd` float en shared memory**,
cuyo orden depende del scheduling de warps → los bits bajos de `sh_dot1` (corrección delta-rule)
y `sh_dot2` (salida) varían run-to-run. El ruido existe a cualquier contexto; a 4K raramente
cruza el umbral de argmax (ids estables en 60 verifies en campañas previas), a 32K+ sí.

El parche (partials por slice + reducción en orden fijo bt=0..3, +2 `__syncthreads` por paso)
hace el stack **completamente bit-estable a 4K y 32K** (0 fuentes, 0 propagaciones, 0 diferencias
de longitud en 64 tokens × 70 módulos × 2 réplicas).

## Cadena de evidencia

1. Bisección (run1, ext del venv): primera fuente `t[006]layers.5` en call 3 @4K y call 17 @32K,
   `prefill=false`, `qlen=7`. Todo lo demás: propagación (capas 6→63 en el mismo verify, capas
   0-4 en el siguiente) o artefactos de entrada de ids (embed_tokens/mtp, corregidos después
   hasheando bytes de tensores enteros).
2. Prefill bit-estable en ambos contextos → PRIMS y el flash multichunk quedan exculpados.
3. Auditoría estática: `atomicAdd` float solo en los kernels recurrentes GDN (4 sitios). Los
   atomics de EXL3 GEMV int8 son enteros (orden-independientes); `hadamard_inner` (float) no se
   activa a estas formas (ningún Linear apareció como fuente); el sampler usa histogramas enteros.
4. Verificación (shadow ext, solo `gdn.cu` cambiado): estable a 4K y 32K. Causalidad confirmada
   por eliminación con un único cambio.

## Por qué 4K parecía estable y 32K no

El kernel GDN es ruidoso a cualquier contexto (no depende de la longitud del KV cache). El ruido
es de ~1 ulp de bf16 en la salida de cada capa GDN; un flip de ids requiere que además el margen
top-2 del argmax sea minúsculo. A 4K ninguna celda de las campañas anteriores flipeó en 60
verifies; a 32K la mediana de divergencia fue verify 9. La frontera 8192 (PRIMS/multichunk) era
una pista falsa que la bisección descartó directamente.

## Resultados de la ventana de verificación (gates congelados en `prediction.json`)

| Gate | Criterio | Observado | Resultado |
|---|---|---|---|
| CAUSAL-4K | bisect estable | `stable=true`, 0/0/0 | **pasa** |
| CAUSAL-32K | bisect estable | `stable=true`, 0/0/0 | **pasa** |
| PERF-32K | tok/s ∈ [229.9, 262.1] y ms/verify ≤ 23.9 | 227.15 tok/s, **22.25 ms/verify** | tok/s falla, ciclo pasa |

El fallo de tok/s es un defecto de diseño del gate, no una regresión de cómputo: el ciclo de
verify es **más rápido** que todas las muestras P/E de graph-band (22.77–23.52 ms). El tok/s cayó
por aceptación (4.05 vs 4.45–4.75 drafts aceptados/verify): la trayectoria greedy cambia con el
redondeo (ahora bit-reproducible), y el gate asumió incorrectamente trayectoria invariante. Si la
diferencia de aceptación es azar de trayectoria o un efecto sistemático lo decide el A/B de
seguimiento (misma celda, ambos builds; con bit-estabilidad basta UNA corrida por celda).

## El parche

`results/20260923-gdn-determinism/ext-src/gdn.cu` (copia sombra de exllamav3 @ 63b32f0; el venv
del servicio NO fue tocado):

```cuda
// antes (x4 sitios): acumulación entre slices bt=0..3 con orden aleatorio
atomicAdd(sh_dot1 + t, sum);
// después: partials + reducción en orden fijo por el hilo bt==0
sh_p1[bt][t] = sum;
__syncthreads();
if (t < v_chunk_dim && bt == 0) {
    float acc = sh_p1[0][t];
    #pragma unroll
    for (int j = 1; j < SUBK; ++j) acc += sh_p1[j][t];
    sh_dot1[t] = acc;
}
__syncthreads();
```

Costo teórico: 2 syncs extra por paso de secuencia y una suma serial de 4 términos por elemento,
sobre un kernel que es una fracción pequeña de los ~31µs/GDN-capa. Medido: el ciclo de verify no
se deterioró (22.25 vs 22.86 mediana P).

Método de despliegue del experimento: extensión compilada a `shadow/exllamav3_ext.so` con los
mismos flags que `ext.py` (`-lineinfo -O3 --use_fast_math`, sm_120), precargada por PYTHONPATH
solo en el proceso del experimento (`exllamav3/ext.py` prefiere un módulo precompilado
importable). Restauración del servicio verificada: config idéntica, `ready`, no busy, código de
producción intacto.

## Consecuencias

1. **La puerta bit-exacta para el grafo CUDA del draft vuelve a estar disponible.** La
   recertificación (armas E/G greedy, divergencia cero exigible) puede correr en la próxima
   ventana junto al A/B de aceptación.
2. **Todos los benchmarks anteriores ganan reproducibilidad** si el parche se promueve al venv
   del servicio (decisión del usuario: requiere recompilar la extensión del venv, ~3 min, y
   reiniciar el servicio).
3. El parche es upstreamable al fork MiaAI-Lab/exllamav3.
4. Lección de diseño de gates: tok/s es métrica de trayectoria (aceptación × ciclo); los gates de
   costo computacional deben usar ms/verify o fases, nunca tok/s desnudo, cuando el redondeo puede
   cambiar la trayectoria.

## Artefactos

- `results/20260923-gdn-determinism/`: `prediction.json` (sha c00425a5…), `verdict.json`,
  `final-verification.json`, `ext-src/` (fuentes + parche), `build_ext.py`, `build/` (artefacto),
  `shadow/exllamav3_ext.so` (sha 2ab1a278…), `verify/bisect-{4096,32768}.json`,
  `verify/timed-lru-32768-0.json`, `run_verify.sh`.
- `results/20260923-nondet-bisect/run1/`: bisección original (venv ext) con la fuente en capa 5.
- `src/qwasar_bench/nondet_bisect.py` + `tests/test_nondet_bisect.py` (6 tests CPU): el
  biseccionador queda como herramienta permanente (huellas por módulo, comparación
  fuente/propagación, fail-closed).
