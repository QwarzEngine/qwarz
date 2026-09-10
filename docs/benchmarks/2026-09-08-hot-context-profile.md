# Perfil de contexto caliente — RTX 5090, MTP6 y K8/V4

**Resultado:** la atención sigue siendo la prioridad al usar el híbrido experimental. En Q=128, el híbrido reduce mucho el coste de los MLP y deja aproximadamente dos tercios del trabajo GPU del modelo principal en los bloques de atención. Para Q=8192, esa proporción llega al 80%. **Flash ya usa split-KV en Q=128**: el siguiente ensayo debe evaluar acceso directo a K8/V4 y sus parámetros, no asumir que falta particionar KV.

**Estado de la captura:** cuatro trazas Nsight Systems completas, con CPU/CUDA/NVTX y nodos de CUDA Graph. Nsight Compute completó después cuatro capturas aisladas con DRAM, actividad Tensor Core, ocupación y stalls. Ver [contadores de atención](2026-09-08-attention-hardware-counters.md): Flash tiene alta actividad Tensor Core y no satura DRAM; la materialización sí ejerce presión de memoria.

## Formas realmente ejecutadas

| Forma | Prefijo restaurado | KV al terminar prefill | Observaciones |
|---|---:|---:|---|
| Turno corto Q=128 | 258048 | 258176 | Una llamada target Q=128; 78 verificaciones Q=7 en cada traza corta |
| Verify Q=7 | Prefijo anterior + generación | Aproximadamente 258K | Se mide dentro del ciclo MTP6 real, no con queries sintéticas |
| Prefill Q=8192 | 249856 | 258048 | Una llamada target Q=8192 |

Pool de 262144 tokens, batch=1, sólo GPU0 RTX 5090. El límite reserva espacio para generar; no se presentan estos KV como 262144 posiciones exactas. Se mantienen seis drafts fijos, caché K8/V4, Flash/8192 y reutilización de IDs exactos.

## Coste por llamada del modelo principal

Los tiempos son sumas de duraciones de kernels atribuidas por correlación CUDA/NVTX. Q=7 muestra el promedio de 78 llamadas de cada traza corta; Q=128 y Q=8192 tienen una llamada instrumentada por perfil. Los bloques incluyen sus proyecciones, normalizaciones y operaciones internas. Los porcentajes no son utilización de GPU ni proporciones de latencia HTTP.

| Perfil | Q | Total GPU por forward | Atención | MLP | GDN |
|---|---:|---:|---:|---:|---:|
| production | 128 | 183.01 ms | 93.03 ms (50.8%) | 59.20 ms (32.3%) | 29.57 ms (16.2%) |
| production | 7 | 40.38 ms | 24.64 ms (61.0%) | 9.40 ms (23.3%) | 5.31 ms (13.1%) |
| production | 8192 | 6131.49 ms | 4156.44 ms (67.8%) | 1355.72 ms (22.1%) | 567.43 ms (9.3%) |
| hybrid64 | 128 | 130.83 ms | 87.31 ms (66.7%) | 14.57 ms (11.1%) | 27.74 ms (21.2%) |
| hybrid64 | 7 | 32.51 ms | 18.57 ms (57.1%) | 7.78 ms (23.9%) | 5.14 ms (15.8%) |
| hybrid64 | 8192 | 5319.36 ms | 4251.20 ms (79.9%) | 432.43 ms (8.1%) | 580.56 ms (10.9%) |

El resto corresponde principalmente al head, normalizaciones externas y operaciones del contenedor TransformerBlock.

## Atención: qué confirma y qué corrige el perfil

- Para Q=128, PyTorch ejecuta `flash_fwd_splitkv_kernel` y su reducción: cuadrícula 2×7×24 = 336 CTAs. El kernel de decode Q=7 también usa 12×28 = 336 CTAs. Ambas rutas ya particionan KV.
- Flash Q=128 declara 241 registros por hilo y 98304 bytes de shared memory por bloque. El decode original declara 204 registros; attention64 llega a 255. Son recursos de lanzamiento, **no ocupación lograda**.
- El split principal Q=7 promedia 1.442 ms por capa en producción y 1.069 ms en hybrid64. Es una comparación dentro de dos modelos que también difieren en sus MLP y activaciones; no sustituye el A/B aislado anterior con las mismas capturas.

| Perfil / forma | Flash: kernels GPU | Materializar KV FP16: kernels GPU |
|---|---:|---:|
| production / Q=128 | 64.35 ms | 17.51 ms |
| production / Q=8192 | 3996.16 ms | 17.72 ms |
| hybrid64 / Q=128 | 60.23 ms | 17.44 ms |
| hybrid64 / Q=8192 | 4088.23 ms | 17.70 ms |

En producción, dequant/staging ocupa aproximadamente 9,6% del target Q=128 y 0,29% de Q=8192. Por tanto, el porcentaje de la ingesta grande no describe el turno corto. Los contadores posteriores distinguen el staging (≈76% del pico DRAM) de Flash Q=128 (≈18% DRAM y ≈87% actividad Tensor Core). Fusionar puede evitar tráfico, pero también debe preservar la eficiencia de cómputo; no cabe atribuir todo el tiempo de Flash a DRAM.

## CPU, estados y especulación

- Se observan 16 copias device→host de cuatro bytes en los bloques de atención del prefill corto de cada perfil, coherentes con las 16 lecturas `cache_seqlens.item()` del adaptador. La copia escalar en sí es pequeña; el problema potencial es imponer una espera entre lanzamientos.
- El target Q=128 de producción acumula 19 `cudaStreamSynchronize` con 131,8 ms de tiempo CPU. Q=8192 acumula 5803,7 ms en esas llamadas. **Esos tiempos incluyen espera por trabajo GPU y se solapan con los kernels: no son milisegundos automáticamente recuperables al quitar `.item()`.**
- Restaurar el estado inicial del turno mueve 154927104 bytes, unos 147,8 MiB, por copias GPU; en la traza corta toma aproximadamente 8,0 ms en producción y 7,6 ms en el híbrido.
- En verify Q=7, la actualización recurrente CUDA de GDN toma aproximadamente 0,99 ms por forward en producción y 0,96 ms en el híbrido. El bloque completo toma 5,31 y 5,14 ms: gran parte del coste está en las proyecciones. ReplaySSM no puede atribuirse todo el porcentaje de GDN.
- En la traza corta de producción, el head ejecutado fuera del módulo draft consume 276 ms de los 615,5 ms de kernels de la fase de borrador: cerca del 45%. Las esperas de muestreo y las proyecciones del drafter merecen un perfil específico antes de aumentar el ancho MTP.
- No aparecen llamadas CUDA malloc/free dentro de las cuatro ventanas capturadas. Esto confirma reutilización del allocator en estas muestras, no que todo el proceso tenga un allocator sellado.

## Controles de latencia

Tres ramas por forma y perfil, con sampler greedy y presupuesto máximo de 512 tokens; las salidas pueden terminar antes. Son entradas derivadas del prompt/corpus archivado, con las mismas listas de IDs por pareja. **Estos controles se toman fuera de la ventana de captura, pero dentro de un proceso lanzado por Nsight y con anotaciones del harness.** No son una nueva validación del SLO del servicio. Se excluyen cebados, calentamientos y muestras capturadas.

| Perfil | Q | Primer token de razonamiento, mediana | Primer contenido visible, mediana | Decode, mediana |
|---|---:|---:|---:|---:|
| production | 128 | 268.8 ms | 3.700 s | 116.1 tok/s |
| production | 8192 | 6624.9 ms | 9.877 s | 119.9 tok/s |
| hybrid64 | 128 | 207.2 ms | 3.154 s | 137.1 tok/s |
| hybrid64 | 8192 | 5758.4 ms | 8.982 s | 129.0 tok/s |

No se vuelve a evaluar calidad de programación: las respuestas y las aceptaciones MTP difieren entre perfiles. Los tiempos de generación no equivalen a tiempo hasta una solución correcta. El supervisor Rust y HTTP no están instrumentados.

## Decisión que respaldan los datos

1. Mantener el híbrido experimental como referencia para optimizar: sus MLP ya reducen mucho el prefill.
2. Medir acceso directo K8/V4 frente a Flash para Q=32/128/256, afinando splits existentes. La prioridad es evitar FP16 temporal en el turno corto.
3. Quitar `.item()` como cambio aislado, pasando la longitud correcta del worker y midiendo el resultado; no usar la suma de esperas CPU como predicción del ahorro.
4. Mantener Flash/8192 para ingesta hasta que un reemplazo demuestre ganar: su atención representa el 80% del target híbrido largo.
5. Acotar ReplaySSM a su ahorro real de recurrencia/historial; mantenerlo por debajo de atención en la prioridad de rendimiento de estas cargas.

## Integridad, incidentes y restauración

Las cuatro trazas contienen 415604 kernels, todos correlacionados con un lanzamiento CUDA y ejecutados en GPU0. Las agrupaciones son exclusivas; se valida con un caso sintético donde los kernels sobreviven a los rangos CPU y se solapan entre sí. Los cuatro replays aislados (Flash Q128/Q8192 y decode Q7 original/64) completan con salidas finitas.

Los primeros cebados terminados justo en un borde de página restauraron un checkpoint anterior y reprocesaron 3840/4096 tokens adicionales. Se conservan como intentos no calificados, no como medidas Q=128. El protocolo final ceba una última página parcial, registra los checkpoints restaurables y exige los contadores físicos correctos en cada muestra medida. No se cambia la política del runtime.

Nsight se bloqueó inicialmente al ejecutar `file` desde `platform.architecture()` durante la inicialización de Triton. El harness reutiliza exactamente el mismo `platform_key` medido con el mismo intérprete antes de lanzar Nsight; el código instalado y la clave de compilación quedan intactos. Los intentos fallidos y sus restauraciones se conservan.

Cada ejecución usa el gestor que detiene el servicio sólo estando libre y comprueba la restauración de su configuración original. El perfil híbrido sigue siendo experimental; no se integra en producción. Ver `final-audit.json` para el estado final comprobado.

## Contadores completados

Cuatro capturas aisladas completadas con Nsight Compute 2026.2.1: Flash Q128, decode Q7 original/64 y Flash Q8192. El [informe de hardware](2026-09-08-attention-hardware-counters.md) contiene metodología, resultados, incidentes y límites. Artefactos definitivos en `counters4/`; las órdenes exactas están archivadas junto a cada captura. La ACL corregida funcionó.

Se usaron métricas de hardware explícitas para evitar el timeout que provocó la instrumentación por software de Q8192. Se mantuvieron relojes sin fijar y cachés vaciadas entre replays; no son contadores de todo el modelo ni una nueva medida del SLO.

Fuentes oficiales: [permisos de profiling NVIDIA](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters), [Nsight Systems y rangos de captura](https://docs.nvidia.com/nsight-systems/UserGuide/index.html).

## Artefactos

- [Resumen numérico](../../results/20260908-hot-profile/summary.json).
- [Producción, turno corto](../../results/20260908-hot-profile/production-nsys4-report.1.analysis.json) y [prefill grande](../../results/20260908-hot-profile/production-nsys4-report.2.analysis.json).
- [Híbrido, turno corto](../../results/20260908-hot-profile/hybrid64-nsys-report.1.analysis.json) y [prefill grande](../../results/20260908-hot-profile/hybrid64-nsys-report.2.analysis.json).
- [Harness](../../results/20260908-hot-profile/profile_workloads.py), [analizador](../../results/20260908-hot-profile/analyze_nsys.py), [replays](../../results/20260908-hot-profile/replay-validation/checks.json).
- [Auditoría final](../../results/20260908-hot-profile/final-audit.json), [herramientas](../../results/20260908-hot-profile/tools.json), [telemetría](../../results/20260908-hot-profile/telemetry.csv).

Los `.nsys-rep`, SQLite, prompts, eventos y capturas `.pt` están en el mismo directorio de resultados. Las órdenes exactas de captura y las fuentes de cada proceso se guardan en los directorios `*-managed`.
