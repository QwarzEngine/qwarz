# Experimentos upstream de Qwasar — 8 de septiembre de 2026

**Resultado:** se completaron los cuatro experimentos y sus combinaciones seleccionadas. Minima64 + atención FP8 ofrece el mejor equilibrio medido entre ingesta larga y calidad; KV NVFP4 + XQA destaca para decode. Son candidatos separados: todavía no se midió su combinación. El servicio quedó restaurado con su configuración original.

## Alcance y controles

Se ejecutaron las cuatro investigaciones propuestas en la [revisión upstream](2026-09-08-upstream-nvfp4-review.md): checkpoint Minima, cabeza MTP reducida, atención FP8 PRIMS y KV NVFP4 con XQA. Una RTX 5090, una sesión, pool nativo de 262144 tokens y MTP fijo en 6. La RTX 3090 Ti no se utilizó. Adaptadores, dependencias y parches están aislados en los procesos de ensayo; el donante instalado no se modificó.

El control de rendimiento habitual de esta campaña es el **híbrido experimental de 56 MLP NVFP4**, con pesos EXL3 en el resto y K8/V4. No equivale al servicio de producción EXL3 sin ese híbrido. Cada comparación identifica su control; no se suman porcentajes de ejecuciones con versiones distintas.

Los presupuestos se conservaron: 4096 tokens para código y 1536 para JSON. Las seis muestras de código son variantes de **una misma tarea LRU**, en tres contextos y dos semillas. Aprobar exige formato válido, tests generados y oráculo independiente; truncados y fallos permanecen en el resultado. No se impuso un nuevo requisito de 6/6 ni se generaliza esta muestra a la calidad completa del modelo.

La latencia caliente de referencia usa un prefijo exacto de 258048 tokens y **128 posiciones físicas nuevas**, con tres repeticiones medidas después de un calentamiento. Es distinta de los turnos JSON con 255 posiciones nuevas. Se verifican IDs, tokens reutilizados y capacidad del pool. Los máximos de memoria asignada/reservada de PyTorch se distinguen de las lecturas de memoria global de NVIDIA posteriores a cada muestra.

## 1. Pesos Minima NVFP4

Se descargó y verificó el checkpoint de [Minima](https://huggingface.co/minima-ai/mnma_qwen3.8_27b_nvfp4), revisión `16e768e7d0461b0b86e565ecedd08a24eca53e9a`. Archivo de 18788354104 bytes, SHA256 `4f046fddf0809838b7d3c7d40805c1f2bbcf2e047aac7cb0a2de2db636a569d9`. Contiene 496 matrices NVFP4: 192 de MLP, 240 de GDN y 64 de atención. No contiene tensores MTP; se conservaron embedding, head, MTP y normas del EXL3 original.

La validación de 12 formas representativas, para M=1, 7, 128 y 2048, pasó 48 comprobaciones contra un oráculo FP32 de los operandos W4A4. Error relativo RMS máximo ≈0.000245. Se repitió con las dos versiones de FlashInfer empleadas. Esto valida el adaptador numérico, no la equivalencia con el modelo antes de cuantizar.

| Perfil | Control fresco | TTFT a 32K | Decode en el screen | Resultado de código |
|---|---|---:|---:|---|
| Minima, 56 MLP | Híbrido original, 56 MLP | Similar | −1% a +8% | Truncado; control pasó |
| Minima, 64 MLP | Híbrido original, 56 MLP | −8.5% a −9.3% | Casi igual; warm 4K −4.7% | Pasó |
| + entradas GDN | Minima, 64 MLP | ≈−16% | −12% a −14% | Oráculo pasó; un test generado falló |
| + todas las proyecciones GDN | Minima, 64 MLP | ≈−23% | −32% a −41% | Pasó |
| + proyecciones de atención | Minima, 64 MLP | −6% a −8% | +2% a +5.5% | Ambos brazos rechazados por formato |
| Todas las proyecciones | Minima, 64 MLP | ≈−29% | −47% a −54% | Pasó |

Los screens Minima usaron FlashInfer 0.6.17. Las seis parejas completaron JSON 3/3 por brazo. Cuantizar más capas no aseguró mejor interacción: las variantes GDN ganaron ingesta pero perdieron decode y algunas aumentaron TTFT caliente. Cambian también las fusiones disponibles; sin un perfil adicional no se atribuye toda la pérdida a la precisión o al formato. Minima64 y sus proyecciones de atención se seleccionaron para el seguimiento largo.

Evidencia: [barrido completo](../../results/20260908-upstream-experiments/minima/screen-report.md), [resumen por pareja](../../results/20260908-upstream-experiments/minima/screen-summary.json).

## 2. Cabeza MTP de 64K

Port process-local de [ExLlamaV3 #303](https://github.com/turboderp-org/exllamav3/pull/303), commit `5705f07b39671746af336bb004ad2e324410a654`. Esta tanda usó FlashInfer 0.6.17. Sólo se restringen las propuestas; el verificador conserva el vocabulario completo y MTP6. El mapa respeta grupos Hadamard de 128 IDs.

Los 512 grupos reconstruidos fueron exactos y cinco replays de embedding con IDs mutables dieron error cero. Queda una diferencia del GEMM del proponente dependiente de la forma N: RMS relativo ≈0.4–1.3%, máximo absoluto 0.148. No se explica como simple autotuning: forzar la misma configuración no la eliminó. La cobertura de 99.9739% corresponde al corpus usado para seleccionar el mapa, no a propuestas reales ni a generalización; se comparte el mismo repositorio y la misma especificación LRU, además de plantillas, con la evaluación.

En la confirmación de contexto largo, con orden invertido y tres repeticiones:

| Métrica | Cabeza completa | Cabeza 64K | Cambio |
|---|---:|---:|---:|
| TTFT frío | 105.688 s | 104.888 s | −0.76% |
| TTFT caliente, Q255 | 279.593 ms | 271.133 ms | −3.03% |
| Decode tras ingesta fría | 131.957 tok/s | 146.449 tok/s | +10.98% |
| Decode caliente | 133.607 tok/s | 142.975 tok/s | +7.01% |

La primera pareja había mostrado una regresión fría de 104.722 a 113.457 s; se conserva y no se reprodujo al invertir el orden. Las repeticiones agrupadas no constituyen un intervalo de confianza. Coste adicional de cabeza/embeddings: 0.862 GiB. JSON 10/10 por brazo sumando ambas tandas; sólo una variante LRU larga evaluada. En esa variante, la cabeza completa truncó y la de 64K aprobó; decode pasó de 116.63 a 115.49 tok/s (−0.97%) y la aceptación de 67.61% a 61.00%. No se midió una combinación de esta cabeza con los demás ganadores.

Evidencia: [informe MTP](../../results/20260908-upstream-experiments/mtp/report.md), [agregado y trazabilidad](../../results/20260908-upstream-experiments/mtp/aggregate.json).

## 3. Atención FP8 PRIMS para ingesta grande

FlashInfer [#4714](https://github.com/flashinfer-ai/flashinfer/pull/4714), commit `af78f8fc17a9654563a619974de56e79257aaea6`, versión aislada 0.6.18 y CUTLASS DSL 4.7.1. Se compararon las capturas reales Q24/KV4/D256 con Q=128 y Q=8192, incluyendo conversiones y reordenamiento de KV.

**El kernel upstream sin cambios no pasó el control numérico largo.** El error L2 relativo frente al oráculo FP32 de sus propios operandos FP8 fue ≈23–27%. La conversión directa de las probabilidades de softmax a E4M3 perdía contribuciones pequeñas. Un parche local multiplica P por 256 antes de convertir y compensa con Vscale/256, conservando la suma de normalización en FP32. Reduce ese error a ≈0.9–1.6%; frente a la atención del K8/V4 original, el error total fue ≈2.2–2.6%. Los resultados siguientes requieren ese parche y no son exactitud de logits.

| Microbenchmark a ≈258K | Flash, con staging | FP8, conversión completa incluida | Decisión |
|---|---:|---:|---|
| Q128 | 4.835 ms | 13.354 ms | Conservar Flash |
| Q8192 | 242.333 ms | 117.509 ms | Usar PRIMS para esta forma |

El resultado Q8192 se mantuvo con caché evacuada: 250.645 frente a 121.827 ms. El router sólo activa PRIMS con Q físico de 8192. Las entradas menores siguen por Flash. **Esta ruta todavía materializa FP16**, después lo convierte a FP8: gana cómputo pero no resuelve el objetivo de eliminar el temporal.

En el modelo completo, ambos brazos usaron el mismo híbrido original de 56 MLP y las mismas versiones nuevas de GEMM:

| Métrica | Flash | PRIMS con parche |
|---|---:|---:|
| TTFT JSON a 258K | 105.479 s | 72.518 s |
| TTFT código a 258K, dos variantes | 106.439–106.484 s | 71.708–72.037 s |
| TTFT código a 131K | 36.587–37.159 s | 28.981–28.992 s |
| Mediana caliente Q128 | 210.131 ms | 209.134 ms |
| Máximo PyTorch asignado | 25.234 GiB | 25.758 GiB |
| Máximo PyTorch reservado | 26.605 GiB | 27.449 GiB |
| LRU, seis variantes | 1/6 | 4/6 |

El ahorro de ingesta larga fue 31–33%, sin mejora relevante del turno Q128, que usa Flash en ambos brazos. JSON 5/5 por brazo entre el screen válido y la pareja larga. Hubo una regresión pareada de código a 32K; las variantes largas mejoraron en esta muestra. No se interpreta el 4/6 como prueba de superioridad general de calidad. Diferencias de aceptación y de salida también afectan tok/s.

Se excluyeron del resultado el primer intento sin headers de compilación y un screen cuyo calentamiento de 8192 IDs sólo producía Q físico 8191: su primera muestra grande incluía JIT. La pareja definitiva calentó con 8193 IDs antes de medir. Se auditaron 17 capturas Q/K/V del calentamiento, correspondientes a 16 capas del target y una del MTP; esos valores fueron finitos y quedaron dentro del rango de E4M3. No es una auditoría de saturación de toda la evaluación.

Evidencia: [informe FP8](../../results/20260908-upstream-experiments/fp8/report.md), [parche exacto](../../results/20260908-upstream-experiments/fp8/pscale-patch.json), [evaluación de código](../../results/20260908-upstream-experiments/fp8/grades-quality/summary.json).

## Combinación de Minima y atención FP8

Se midieron tres procesos frescos consecutivos con FlashInfer 0.6.18, el parche P×256, el mismo K8/V4 y MTP6. El primer perfil es el híbrido original de 56 MLP; el segundo usa las 64 MLP de Minima; el tercero agrega las proyecciones q/k/v/o de las 16 capas de atención, conservando GDN en EXL3.

| Métrica | 56 MLP original + FP8 | Minima64 + FP8 | Minima64 + proyecciones de atención + FP8 |
|---|---:|---:|---:|
| TTFT JSON frío a 258K | 72.968 s | 68.820 s | 66.390 s |
| TTFT LRU a 32K, media de dos variantes | 5.798 s | 5.263 s | 4.912 s |
| TTFT LRU a 131K, media de dos variantes | 29.385 s | 27.308 s | 25.119 s |
| TTFT LRU a 258K, media de dos variantes | 73.209 s | 68.171 s | 64.346 s |
| Q128 caliente, mediana TTFT | 210.898 ms | 208.442 ms | 187.737 ms |
| Q128 caliente, mediana decode greedy | 132.940 tok/s | 138.497 tok/s | 157.355 tok/s |
| Máximo PyTorch asignado | 25.758 GiB | 25.630 GiB | 25.559 GiB |
| Máximo PyTorch reservado | 27.449 GiB | 27.195 GiB | 26.955 GiB |
| Código LRU | 3/6 | 4/6 | 2/6 |
| Respuestas de código truncadas | 0 | 0 | 2 |

**Minima64 + FP8 es el candidato más equilibrado de esta tanda.** Añade un 6–9% de ahorro de ingesta sobre la referencia FP8 en las variantes LRU y conserva todas las respuestas completas. Su 4/6 incluye una regresión pareada a 131K; no es una mejora universal. La variante de atención es más rápida, especialmente en el turno caliente, pero rechazó cuatro respuestas: dos por cantidad de bloques de código y las dos de 258K por truncado. Queda como opción con un compromiso de calidad explícito, sin imponer un umbral nuevo de aprobación.

JSON pasó 2/2 por perfil. Todos conservaron los mismos IDs, pool, MTP6, versiones y hash del kernel parcheado. El orden fue fijo y no intercalado; los porcentajes pequeños siguen sujetos a variación de ejecución. Las diferencias de decode incluyen cambios de continuación y aceptación, incluso en greedy.

Dos procesos breves de revisión inicializaron un contexto CUDA durante la petición de priming no puntuada de Minima64. No coincidieron con ninguna muestra medida ni con el calentamiento Q128 posterior; los tiempos están registrados en [la nota de integridad](../../results/20260908-upstream-experiments/fp8/incidental-context-note.json).

Evidencia: [informe combinado y revisión del adaptador](../../results/20260908-upstream-experiments/fp8/combined-report.md), [agregado reproducible](../../results/20260908-upstream-experiments/fp8/combined-summary.json), [calificación completa](../../results/20260908-upstream-experiments/fp8/grades-combined/summary.json).

## 4. KV NVFP4 y XQA

Se reconstruyó el contrato sugerido por [vLLM #53543](https://github.com/vllm-project/vllm/pull/53543) y [SGLang #36038](https://github.com/sgl-project/sglang/pull/36038): longitudes posteriores al append, máscara causal para verify, paginado, escalas y CUDA Graphs. La ruta usa FlashInfer 0.6.17 y CUTLASS DSL 4.6.2, como el híbrido original.

Los tests incluyen Q=1…8, páginas permutadas, igualdad de los bytes escritos al caché, longitudes mutables en replay y aceptar dos/rechazar cinco posiciones con rewind. Las 11 comprobaciones del adaptador pasaron; máximo error relativo frente a la ruta Triton ≈0.000445. Las primeras capturas reales fueron invalidadas por seleccionar un scratch en `cuda` distinto del de `cuda:0`; los resultados válidos posteriores comprueban además que K y V no sean cero.

En la pareja larga con el **mismo KV NVFP4**, sustituir sólo decode Triton por XQA mantuvo la ingesta JSON en ≈109 s, pero elevó decode de 26.4 a 162.3 tok/s. Para Q128 caliente: mediana de 390.822 a 205.340 ms; XQA alcanzó ≈172.7 tok/s greedy. Ambos brazos comparten prefill y su temporal FP16: esta reducción de TTFT responde al camino de primer decode, no demuestra una mejora del gather.

La variante LRU larga pasó la evaluación completa en Triton. En XQA, la clase pasó las siete comprobaciones del oráculo independiente pero falló uno de sus propios tests; el resultado global se conserva como fallo.

Ese control Triton no es el K8/V4 incumbente. Además, el constructor del donante asignaba FP16 al caché del draft al elegir NVFP4 para el target. Por ello se ejecutó una comparación adicional contra K8/V4, manteniendo el draft en K8/V4 en ambos brazos. La construcción del caché se verificó antes de capturar grafos; se comprobó la restauración de la función interceptada. Ambos procesos exigieron FlashInfer 0.6.17 y conservaron los 56 MLP originales, el head completo, MTP6 y el pool de 262144 tokens.

### Comparación final contra K8/V4

| Métrica | Target K8/V4 | Target NVFP4 + XQA |
|---|---:|---:|
| TTFT JSON frío a 258K | 107.416 s | 107.920 s |
| Decode JSON frío | 135.069 tok/s | 171.255 tok/s |
| TTFT JSON caliente, Q255 | 274.735 ms | 278.729 ms |
| Decode JSON caliente | 141.814 tok/s | 184.389 tok/s |
| TTFT código a 32K, variantes 0 / 1 | 6.021 / 6.015 s | 7.370 / 6.038 s |
| TTFT código a 131K, variantes 0 / 1 | 36.663 / 36.706 s | 37.068 / 37.877 s |
| TTFT código a 258K, variantes 0 / 1 | 106.093 / 106.457 s | 110.405 / 106.901 s |
| Decode código a 258K, variantes 0 / 1 | 85.502 / 105.368 tok/s | 141.372 / 157.618 tok/s |
| Mediana Q128 caliente | 206.740 ms | 201.397 ms |
| Mediana decode greedy Q128 | 134.875 tok/s | 184.410 tok/s |
| Máximo PyTorch asignado | 25.233 GiB | 24.367 GiB |
| Máximo PyTorch reservado | 26.605 GiB | 25.711 GiB |
| Código LRU | 4/6 | 4/6 |
| JSON largo, frío / caliente | 2/2 | 2/2 |

**XQA es el candidato más prometedor para decode:** en el control greedy Q128 ganó 36.7% de throughput, redujo TTFT 2.6% y ahorró 0.866 GiB de memoria máxima asignada. En JSON ganó 27–30% de decode, con TTFT caliente ligeramente peor. La ingesta de código a 258K aumentó entre 0.44 y 4.31 segundos según la variante; no se oculta ese coste detrás de la mejora de generación.

El resultado 4/6 está empatado sólo en agregado. K8/V4 falló un test generado a 131K/semilla 43 y truncó a 258K/semilla 42. XQA falló un test generado a 258K/semilla 42 y truncó a 258K/semilla 43. Las clases de los fallos ejecutados pasaron las siete comprobaciones del oráculo. Se conserva la regresión pareada de XQA en la segunda variante larga. El distinto texto y la aceptación influyen en los tiempos de decode: son resultados de uso del modelo, no una medida aislada del kernel.

La primera TTFT de código a 32K fue 7.370 s frente a 6.038 s en la segunda. Hubo calentamiento largo previo; no hay evidencia específica para descartar la primera como JIT. Se publican ambas. El orden de la pareja fue fijo, control seguido de XQA, con tres repeticiones calientes; no se calculan intervalos de confianza con esta muestra.

Evidencia de la pareja final: [informe](../../results/20260908-upstream-experiments/xqa/final-comparison.md), [agregado](../../results/20260908-upstream-experiments/xqa/final-comparison.json) y [calificación LRU completa](../../results/20260908-upstream-experiments/xqa/final-code-grade/summary.json).

### Conversión de KV aislada

Un gather Triton que combina K y V en un lanzamiento y procesa bloques de 4096 elementos dio 0.9865 ms frente a 1.1608 ms del original, para una capa a 258176 tokens: 1.177×, o 0.174 ms menos. Los bloques de 1024 y 2048 dieron 1.0247 y 0.9926 ms. Pasó igualdad exacta contra el decodificador independiente, con páginas permutadas y protección de la cola parcial.

El primer intento falló porque PyTorch no implementa `index_copy_` para E4M3; el segundo copió las escalas mediante vistas de bytes, sin cambiar su representación. El gather nuevo no se integró en el modelo. Es una mejora secundaria aislada, todavía escribe FP16 y no permite atribuirle una reducción de latencia de aplicación. No explica la gran diferencia entre decode Triton y XQA.

Evidencia del diagnóstico: [validación y tiempos del gather](../../results/20260908-upstream-experiments/xqa/gather-fast-data-r2/gather-fast-validation.json).

### Cola reciente FP16

Sobre una captura real Q7 de una capa, convertir K8/V4 a NVFP4 produjo ≈4.3–4.6% de error relativo de atención. Recuperar los últimos 2K tokens de esa captura en FP16 lo redujo a ≈1.1–1.2%; ampliar a 4K u 8K aportó poco en esa muestra. Es FP16 reconstruido del K8/V4, no el KV BF16 original del modelo.

Una cola de 2K en las 16 capas costaría 128 MiB extra si se conservan ambas copias. La interfaz XQA usada no devuelve LSE para combinar exactamente las dos regiones. **La mezcla sólo se evaluó numéricamente: no hay todavía un runtime mixto con latencia o calidad medidas.**

Evidencia: [pareja larga NVFP4 Triton/XQA](../../results/20260908-upstream-experiments/xqa/long-comparison.md), [calificación de código](../../results/20260908-upstream-experiments/xqa/long-ab/code-grade/summary.json), [validación del adaptador](../../results/20260908-upstream-experiments/xqa/adapter-gate-data2/adapter-validation.json), [resultados aislados e invalidaciones](../../results/20260908-upstream-experiments/attention-interim.md).

## Reproducibilidad y cierre

Los scripts, comandos, versiones, respuestas y manifiestos viven en [la carpeta de la campaña](../../results/20260908-upstream-experiments/). Cada ventana gestionada guarda la configuración previa, los procesos hijos y la restauración del servicio. El checkpoint original y el de Minima se identifican por hash; los parches son locales y no se actualizaron paquetes globales.

No se ejecutó una nueva comparación de servidor vLLM: la propuesta era condicional a poder aislar #52244 y mantener semántica equivalente de prefijo. La revisión archivada lo encontró abierto y con `mergeable: false`; no se aisló en esta campaña una ruta con semántica equivalente de prefijo. Tampoco se cambió MTP6 por DFlash ni se promovió una cuantización dispersa o un runtime nuevo.

El cierre conserva MTP6, K8/V4 y el modelo EXL3 del servicio original. Los perfiles ganadores siguen como experimentos reproducibles; no se promovieron a producción. Las revisiones independientes de las métricas y sus límites están en [review-evidence.md](../../results/20260908-upstream-experiments/mtp/review-evidence.md).

Verificación de cierre: [estado y comprobaciones](../../results/20260908-upstream-experiments/final-verification.json), [configuración restaurada](../../results/20260908-upstream-experiments/final-service-config.json) y [hashes de fuentes archivadas](../../results/20260908-upstream-experiments/final-source-sha256.json).
