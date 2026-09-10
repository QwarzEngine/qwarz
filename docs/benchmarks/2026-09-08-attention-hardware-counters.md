# Contadores de atención — RTX 5090 / K8/V4

**Resultado:** conviene probar acceso directo K8/V4 primero en Q=32–256, pero los contadores corrigen la hipótesis de que Flash está limitado principalmente por DRAM. La materialización sí ejerce presión sobre DRAM; Flash Q=128 y Q=8192 muestra alta actividad Tensor Core y baja presión DRAM. El reemplazo deberá conservar su eficiencia de cómputo.

## Protocolo y alcance

Nsight Compute 2026.2.1, CUDA visible sólo en GPU0, CC 12.0. Cuatro capturas completas, nueve kernels, doce pasadas por kernel. Entradas reales archivadas del primer bloque de atención del híbrido: Q=128/KV=258176, Q=7/KV≈258K y Q=8192/KV=258048. Pool de 262144; Q24/KV4/D256. No se cargan los pesos durante estos replays.

Cinco calentamientos, salida finita, limpieza previa de 256 MiB, `--replay-mode kernel --cache-control all --clock-control none`. Cada fila representa un lanzamiento aislado; las pasadas de contadores no son repeticiones independientes para calcular p50. Los relojes SM medidos varían aproximadamente 2,78–3,04 GHz. Se conservan unidades originales en CSV; GB usa base decimal.

El vaciado de cachés y la serialización de Nsight Compute alteran el contexto de ejecución. Estos datos describen los kernels aislados; no sustituyen las trazas del modelo ni validan TTFT/SLO. NVIDIA documenta estas diferencias en su [guía de profiling](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#cache-control).

## Memoria, cómputo y ocupación

Los porcentajes de DRAM y Tensor Core son las métricas normalizadas de Nsight respecto de su pico sostenido; Tensor Core indica actividad de esa ruta, no porcentaje de FLOPS útiles ni del pico NVFP4. Ocupación es warps activos respecto del máximo, no utilización de la GPU.

| Caso / kernel | Tiempo ms | DRAM leída GB | DRAM escrita GB | DRAM % | Tensor Core % | Ocupación % | L2 hit % |
|---|---:|---:|---:|---:|---:|---:|---:|
| q128-flash / materialización | 1.109 | 0.430 | 1.052 | 75.6 | 0.0 | 96.4 | 3.9 |
| q128-flash / atención | 3.690 | 1.112 | 0.048 | 17.8 | 86.9 | 8.3 | 91.1 |
| q128-flash / combine | 0.018 | 0.022 | 0.002 | 76.5 | 0.0 | 36.8 | 1.9 |
| q7-decode / atención | 1.090 | 0.431 | 0.037 | 24.3 | 18.9 | 16.1 | 75.6 |
| q7-decode / combine | 0.026 | 0.006 | 0.000 | 13.2 | 0.0 | 8.4 | 5.9 |
| q7-decode64 / atención | 0.826 | 0.431 | 0.035 | 31.9 | 25.2 | 16.2 | 70.3 |
| q7-decode64 / combine | 0.027 | 0.006 | 0.000 | 12.4 | 0.0 | 7.9 | 4.9 |
| q8192-flash / materialización | 1.102 | 0.429 | 1.052 | 76.1 | 0.0 | 96.1 | 3.8 |
| q8192-flash / atención | 240.105 | 6.135 | 0.142 | 1.5 | 85.2 | 8.3 | 99.2 |

En Q=128, la materialización lee 0,430 GB y escribe 1,052 GB por capa; Flash lee otros 1,112 GB. La secuencia completa mueve aproximadamente 2,665 GB en DRAM por capa bajo este protocolo. El kernel Flash solicita 12,737 GB a L2 y obtiene aproximadamente 91% de hits. **Hay tráfico evitable, pero las lecturas FP16 no equivalen íntegramente a DRAM ni a tiempo recuperable.**

En Q=8192, Flash solicita 799,414 GB a L2, con 99,2% de hits y sólo 1,5% del pico DRAM. Tiene 85,2% de actividad Tensor Core. Esto respalda un límite de cómputo/pipeline en esta ruta, no saturación de GDDR7.

Flash tiene 8,33% de ocupación: un CTA de cuatro warps por SM permitido por shared memory. Q=128 usa 336 CTAs, Q=8192 usa 3072; ambas formas mantienen alta actividad Tensor Core pese a esa ocupación baja. Subir ocupación no garantiza mejorar el tiempo.

## Stalls

Se muestran directamente los ratios `smsp__average_warps_issue_stalled_*_per_issue_active.ratio`; no son porcentajes de latencia ni deben sumarse a los tiempos GPU. Permiten contrastar las causas de espera dentro de cada kernel.

| Caso / kernel principal | Long scoreboard | Short scoreboard | Math pipe throttle | Wait | MIO throttle |
|---|---:|---:|---:|---:|---:|
| q128-flash / materialización | 47.007 | 1.185 | 0.122 | 1.451 | 0.549 |
| q128-flash / atención | 0.030 | 0.138 | 3.542 | 3.115 | 0.058 |
| q7-decode / atención | 0.686 | 2.336 | 0.273 | 0.882 | 1.348 |
| q7-decode64 / atención | 1.495 | 0.605 | 0.540 | 0.899 | 0.589 |
| q8192-flash / materialización | 47.537 | 1.151 | 0.112 | 1.450 | 0.468 |
| q8192-flash / atención | 0.035 | 0.097 | 3.686 | 3.174 | 0.061 |

La materialización destaca por long scoreboard; Flash por math pipe throttle y wait, con long scoreboard muy pequeño. Es evidencia adicional para separar el problema de staging del cómputo de atención.

## Verify Q=7

Con las mismas entradas, el replay standalone original suma 1.115 ms y block_n=64 suma 0.854 ms: 23.5% menos tiempo (1.31×). DRAM leída prácticamente igual (~431 MB en el split principal), mientras las solicitudes a L2 bajan 1,794→1,472 GB y se reducen short scoreboard y MIO throttle. No es una mejora por leer menos KV desde DRAM.

**Límite de equivalencia:** el original standalone declara 190 registros/hilo y el CUDA Graph de producción perfilado antes declara 204; no son binarios idénticos. La ruta 64 declara 255 en ambos. Esta comparación describe el replay aislado, no un nuevo incremento demostrado de tok/s de Qwasar. Ambas rutas ya leen K8/V4 directamente y usan 336 CTAs. El compilador especializa de manera distinta parámetros del camino graph y del standalone.

## Próximo experimento recomendado

1. Comparar el prefill directo K8/V4 existente contra Flash para Q=32/128/256 sobre el mismo KV largo y las mismas entradas, antes de escribir otro kernel.
2. Si el kernel directo pierde, usar el perfil para diseñar una fusión que preserve tiles, reutilización GQA y eficiencia Tensor Core. Descuantizar dentro del kernel introduce trabajo y presión de registros que hay que medir.
3. Mantener MTP6, híbrido MLP56, K8/V4 y Flash/8192 como referencias. En la traza del modelo híbrido, quitar los 17,44 ms de staging sin ningún costo nuevo equivaldría a ~13,3% del tiempo de kernels del target Q=128. Es un escenario ideal de ese componente, no un pronóstico ni un límite para un rediseño más amplio.
4. Validar el resultado numérico, causalidad y reutilización de prefijo; después medir latencia de servicio y calidad con el mismo presupuesto de salida y la tolerancia a pérdida de calidad ya acordada. No se hizo una nueva evaluación de coding en este perfil.

## Incidentes y trazabilidad

La ACL inicial se anulaba por `DeviceFileModify: 1`. El usuario aplicó `DeviceFileModify: 0` y la ACL efectiva; desde entonces funcionan los contadores. `counters2` conserva las primeras tres capturas completas y un Q8192 fallido: error CUDA 702 durante `SW Counters::1`. Se evitó la instrumentación por software con una lista explícita de métricas de hardware. `counters3` falló antes de ejecutar la GPU por una validación local que no distinguía métricas de lanzamiento; se corrigió. `counters4` completa las cuatro formas sin errores. No se cambiaron límites de timeout, relojes ni fuentes del runtime instalado.

Cada ventana restaura el servicio y verifica su configuración original. El resultado definitivo de esa comprobación está en [la auditoría](../../results/20260908-hot-profile/final-audit.json).

- [Resumen normalizado y hashes](../../results/20260908-hot-profile/counters4/summary.json).
- [Completitud y métricas solicitadas](../../results/20260908-hot-profile/counters4/completed.json).
- [CSV Q128](../../results/20260908-hot-profile/counters4/q128-flash.csv), [CSV Q7 original](../../results/20260908-hot-profile/counters4/q7-decode.csv), [CSV Q7/64](../../results/20260908-hot-profile/counters4/q7-decode64.csv), [CSV Q8192](../../results/20260908-hot-profile/counters4/q8192-flash.csv).
- [Captura reproducible](../../results/20260908-hot-profile/run_hardware_counters.py), [analizador](../../results/20260908-hot-profile/analyze_counters.py).
- [Perfil del modelo completo](2026-09-08-hot-context-profile.md).

Los `.ncu-rep`, órdenes exactas, logs, snapshots de fuentes y estados del servicio se conservan en `results/20260908-hot-profile/counters4*`.
