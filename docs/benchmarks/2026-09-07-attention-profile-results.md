# Attention con MTP fijo de seis en RTX 5090

**Resultado:** `block_n=64`, cuatro warps y una etapa, con splits automáticos y Flash/8192, es el mejor candidato medido para acelerar attention. La matriz completa mejora aproximadamente 3%, 10% y 14% de decode a 32K, 128K y casi 256K, con 36/36 respuestas correctas por perfil. **No queda calificado como reemplazo general del servicio:** en programación las dos rondas suman 8/12 respuestas completas correctas frente a 11/12 de la referencia, y la ventaja corta cambia de signo entre rondas. Se conserva MTP6 + K8/V4 + Flash/8192 y el decode actual en producción; el candidato queda disponible como perfil experimental.

## Alcance y método

Evaluación local de Qwen3.8-27B EXL3 5 bpw en una RTX 5090, con caché K8/V4 de 262.144 tokens y seis propuestas MTP. El artefacto tiene SHA-256 `0b9a439ffefa45c55a2a3cb0324de9fe1b23bce69a37a399cb952d355f02d92e`. Cada comparación mantiene pesos, sampler y ancho MTP. Los ensayos de tamaño de prefill mantienen además fijo el nuevo perfil de decode.

El ejecutor detiene Qwasar sólo estando libre, espera a que su proceso termine y la GPU quede disponible, y restaura el servicio con la configuración completa original al terminar o fallar. Las variantes se aplican en memoria antes de cargar un modelo nuevo. No se modifican los archivos del backend donante ni se reutilizan CUDA Graphs entre perfiles.

## Dónde está el coste

Una traza instrumentada de un turno largo con 8192 tokens nuevos atribuye los kernels por contención en las anotaciones GPU de cada fase, en el mismo dispositivo y stream. Attention representa el 57,64% del tiempo sumado de kernels de verificación MTP y el 38,92% del borrador. En prefill, Flash representa el 65,92%; las multiplicaciones CUTLASS, el 27,66%.

Son proporciones de tiempo de kernels en una muestra instrumentada, no porcentajes de latencia HTTP ni medidas de utilización. La traza sirve para priorizar; los tiempos comparativos proceden de ejecuciones sin profiler.

## Barrido del kernel de decode

Se capturan Q y las páginas K8/V4 reales después de ejecutar BCAttn: borrador de una posición y verificación de siete, en contextos nominales de 32K, 128K y 256K. La última captura tiene aproximadamente 252K tokens para reservar entrada y salida dentro del límite nativo. Se repite la verificación en la capa profunda 63. Las consultas de dos a seis posiciones se obtienen de la captura de siete, conservando la máscara causal inferior derecha.

El barrido inicial prueba 52 configuraciones más el control en cuatro combinaciones de contexto y longitud de consulta. El refinamiento compara 24 más control en nueve combinaciones. La validación de dos a seis posiciones compara dos candidatos más control en quince combinaciones. En total: **2308 ensayos**, incluidos controles pareados. Hay 16 fallos de compilación en dos configuraciones que exceden la memoria compartida disponible; se conservan como fallos, no como tiempos.

Cada ensayo calienta y captura un CUDA Graph antes de medir. Se vacían 256 MiB de L2 antes de cada replay, fuera del intervalo medido. Se usan eventos CUDA, orden aleatorio y una referencia inmediatamente después de cada candidato. Se verifican finitud, cercanía al kernel original y un oráculo FP32 independiente sobre las cabezas y posiciones muestreadas. Estos tiempos excluyen pesos y el resto del modelo.

El candidato consistente usa `block_n=64`, `num_warps=4`, `num_stages=1`, con particiones automáticas. Pasa todos sus controles; el máximo error relativo L2 frente al oráculo FP32 es 0,000537. Esto valida el kernel en las capturas evaluadas, no demuestra igualdad exacta de secuencias o distribución del modelo.

| Contexto nominal | Aceleración kernel Q=1 | Aceleración kernel Q=7 | Q=7, capa 63 |
| --- | ---: | ---: | ---: |
| 32K | 1,022× | 1,206× | 1,206× |
| 128K | 1,073× | 1,285× | 1,285× |
| Cerca de 256K | 1,128× | 1,323× | 1,324× |

Modificar la agrupación de cabezas sólo desde Python sería incorrecto en el camino de CUDA Graphs: C++ calcula su propia cuadrícula. Esas variantes se exploran únicamente en el kernel aislado, con cuadrícula consistente, y no superan al candidato elegido de manera general. El adaptador experimental rechaza explícitamente ese cambio en el camino nativo.

## Comparación del modelo completo

Los perfiles reproducen los mismos historiales archivados, tareas, semillas y prompts. Hay un cebado frío por contexto y un calentamiento por tamaño de entrada, excluidos de las estadísticas calientes. Se miden tres repeticiones por combinación de contexto y entrada. El orden se baraja con una semilla fija. El sampler es el recomendado para `medium`: temperatura 1, top-p 0,95 y top-k 20.

La auditoría exige igualdad de hashes del artefacto, corpus, runtime y fuentes del harness, además de igualdad por pareja del prompt, historial, tokens reutilizados y prefill físico. Se conservan los fallos de calidad y truncamientos en los denominadores. La mediana de razones pareadas y la razón entre medianas son medidas diferentes; las tablas identifican cuál usan. La latencia completa también depende de la longitud generada.

La comparación principal suma 36 muestras medidas y 12 de calentamiento por perfil. Todas pasan calidad y reutilización. Las 36 parejas tienen exactamente el mismo prompt y prefill físico; 27 producen además el mismo texto completo. No hay requeues ni truncamientos en estas respuestas.

| Contexto nominal | Decode actual, mediana tok/s | Decode candidato, mediana tok/s | Mejora de decode, mediana pareada | Reducción de tiempo completo, mediana pareada |
| --- | ---: | ---: | ---: | ---: |
| 32K | 203,93 | 208,77 | +2,78% | 2,22% |
| 128K | 158,13 | 172,85 | +9,76% | 5,96% |
| Cerca de 256K | 107,67 | 125,69 | +14,45% | 8,71% |

La razón entre medianas de velocidad del contexto largo es 1,167×; la mediana de razones pareadas es 1,145×. La mejora de TTFT es pequeña, alrededor del 0,5–1,6%, porque se mantiene el mismo prefill. El perfil acelera la generación, no suprime el coste de ingerir una entrada grande.

El pico de PyTorch en las muestras medidas es aproximadamente 26.718 MiB asignados y 28.370 MiB reservados en ambos perfiles. Son contadores de PyTorch, no toda la memoria del dispositivo. Los registros del camino nativo confirman que el candidato se aplicó en las consultas de una a siete posiciones.

## Tamaño de prefill, manteniendo decode64

El perfil de 4K completa 18 muestras medidas y seis calentamientos, todos correctos y con caché válida. Ahorra aproximadamente 1030 MiB de memoria reservada frente a 8K: 27.340 frente a 28.370 MiB. Con 128 tokens nuevos el TTFT prácticamente no cambia en medio y largo. Con 8192 tokens nuevos es más lento:

| Contexto nominal | TTFT con chunk 8K | TTFT con chunk 4K |
| --- | ---: | ---: |
| 32K | 2882 ms | 2929 ms |
| 128K | 4529 ms | 4694 ms |
| Cerca de 256K | 6526 ms | 6742 ms |

Son medianas de tres muestras por celda. El cebado frío largo toma 127,9 s con 8K y 131,4 s con 4K, una sola observación por perfil; el primer cebado puede incluir compilación y no debe interpretarse como una comparación estadística de arranque. 4K es una alternativa para ahorrar memoria, no el ganador de velocidad.

El perfil de **16K falla por falta de VRAM durante el primer cebado de 32K**, con la caché completa y MTP6. El fallo ocurre al reservar 384 MiB para el estado de Gated DeltaNet. No produce muestras medidas y se descarta para la configuración actual, conservando log y restauración. No demuestra que 16K sea imposible con otro allocator o una caché menor; esas serían otras configuraciones.

El perfil de 12K completa sus 18 muestras y seis calentamientos, todos correctos. Reserva 29.544 MiB, **1174 MiB más que 8K**, y registra un intento de asignación fallido recuperado por el allocator durante la primera carga. No ofrece una ventaja sostenida: TTFT con entrada de 8192 tokens de 2880 / 4613 / 6887 ms para 32K / 128K / largo, frente a 2882 / 4529 / 6526 ms con 8K. El cebado frío largo toma 128,2 s frente a 127,9 s; una muestra de cada perfil. Se elige **Flash/8192** para velocidad y margen de memoria en esta configuración.

## Programación: contraste con respuestas largas

Se pide una caché LRU completa con tests, usando el mismo corpus y sampler recomendado, un presupuesto de 4096 tokens de salida y el pool completo. Por perfil se genera un cebado y dos ramas calientes en 32K y cerca de 256K. Se ejecuta primero el candidato y después la referencia. Todos los prompts y contadores físicos coinciden por pareja. El código se revisa y se ejecuta mediante bubblewrap, sin repararlo, con sus propios tests y un verificador independiente de siete comprobaciones.

| Contexto | Decode actual | Decode candidato | Cambio pareado | Respuestas calientes completas correctas, actual / candidato |
| --- | ---: | ---: | ---: | ---: |
| 32K | 171,81 tok/s | 153,95 tok/s | −10,24% | 2/2 / 1/2 |
| Cerca de 256K | 93,62 tok/s | 101,69 tok/s | +8,65% | 2/2 / 1/2 |

En el conjunto de seis salidas por perfil, la referencia pasa 6/6 respuestas completas y el candidato 4/6. Las implementaciones pasan el verificador en 6/6 y 5/6 respectivamente. El candidato presenta una expectativa LRU incorrecta en un test corto y un `NameError` en una salida larga por definir un sentinel después de usarlo como argumento predeterminado. No hay truncamientos ni requeues.

Ninguna de las cuatro parejas calientes tiene texto idéntico. En corto, la aceptación mediana de MTP pasa de 61,74% a 54,52%; en largo, de 61,15% a 58,06%. En largo se acelera el decode, pero el tiempo completo aumenta de 27,16 a 30,49 segundos porque el candidato genera más tokens. **La ganancia del kernel no garantiza mejorar la tarea completa.** Esta diferencia impide afirmar paridad de calidad o extrapolar la mejora de la matriz a toda programación.

Se amplía la comparación corta a cinco ramas medidas por perfil, invirtiendo el orden: referencia y candidato. La referencia alcanza 163,46 tok/s y el candidato 171,83 tok/s, una mejora pareada del **6,47%**, frente al −10,24% de la primera ronda. Ninguna pareja conserva el mismo texto completo. La aceptación mediana de MTP pasa de 57,90% a 59,70%. La mediana de tiempo completo aumenta de 14,33 a 16,41 segundos; la de longitud pasa de 2321 a 2744 tokens.

En esa ampliación pasan 5/6 respuestas completas de referencia y 4/6 del candidato; las 12 implementaciones pasan el verificador independiente. La referencia escribe un nombre inexistente de aserción y una expectativa de longitud incorrecta en una respuesta; el candidato falla dos expectativas en tests. Se mantienen todos los resultados.

Las dos rondas suman **24 generaciones: 11/12 respuestas completas correctas para la referencia y 8/12 para el candidato; implementaciones correctas 12/12 y 11/12**. No hay requeues ni truncamientos. Los prompts de las primeras repeticiones se repiten entre rondas: estas observaciones no son 24 tareas independientes ni una estimación general de calidad. La referencia también cambia de secuencia con la misma semilla. No se demuestra que el perfil cause la diferencia de calidad, pero tampoco que la preserve.

El control estructurado final repite las 12 muestras medidas de 32K con la configuración original. Todas pasan calidad y caché; el candidato conserva una mejora pareada de **2,83%** contra esta referencia final, frente al 2,78% contra la inicial. Esto confirma la pequeña ganancia en esa matriz, sin resolver la variabilidad del workload de programación.

## Decisión

Para las formas evaluadas, elegir **64/4/1 con splits automáticos** como candidato de attention y conservar **Flash/8192**. No hay evidencia suficiente para justificar tamaños adaptativos o cambios en la agrupación de cabezas; agregar esas variantes aumentaría complejidad sin una ganancia demostrada. 4K queda como alternativa para liberar aproximadamente 1 GiB; 12K no compensa su memoria adicional y 16K falla en la configuración actual.

Para uso general de Qwasar, **mantener el perfil actual en el servicio**. La ventaja de attention largo está respaldada por kernels, matriz y decode de programación, pero no equivale a menor tiempo de tarea ni a paridad de calidad. El resultado no obliga a reescribir el motor: identifica una sustitución pequeña y medible, cuyo perfil y adaptador quedan preparados para una eventual integración. La integración no forma parte de estas evaluaciones.

La suite local del código de evaluación termina con **301 pruebas aprobadas y 35 omitidas**. Se valida en GPU el camino nativo con CUDA Graphs, además de las pruebas de transformación y rechazo de configuraciones incompatibles. Los archivos instalados del backend permanecen intactos.

La auditoría final contabiliza **120/120 muestras estructuradas medidas correctas**, más 40/40 calentamientos correctos, y 24 generaciones de programación. Verifica los hashes de las fuentes de ejecución y que cada código revisado coincide con el código evaluado. Las 15 ventanas de experimentación restauran la configuración completa del servicio, incluyendo los dos intentos fallidos: uno por argumentos incompletos del primer capturador, antes de cargar el modelo, y otro por OOM con prefill 16K. El endpoint final devuelve `ready`, `busy=false`, MTP fijo de seis y la configuración original de attention. La parada pierde caché GPU anterior; no elimina sesiones durables.

## Artefactos

- [Desglose del profiler](../../results/20260907-attention-profile/baseline/profile/phase-breakdown.json).
- [Auditoría de kernels](../../results/20260907-attention-profile/micro-summary.json) y [script](../../results/20260907-attention-profile/analyze_micro.py).
- [Perfil candidato](../../results/20260907-attention-profile/profiles/decode64.json).
- [Ejecutor con restauración](../../results/20260907-attention-profile/managed.py), [matriz pareada](../../results/20260907-attention-profile/evaluate.py) y [auditor](../../results/20260907-attention-profile/analyze.py).
- [Comparación completa de decode](../../results/20260907-attention-profile/decode-comparison.json).
- [Comparación de prefill 4K](../../results/20260907-attention-profile/chunk4096-comparison.json) y [fallo de 16K](../../results/20260907-attention-profile/eval-chunk16384/failure.json).
- [Comparación de prefill 12K](../../results/20260907-attention-profile/chunk12288-comparison.json).
- [Perfil conservado para integración](../../benchmarks/profiles/attention-5090-qwen38-mtp6.json).
- [Comparación inicial de programación](../../results/20260907-attention-profile/coding-comparison.json), [grade de referencia](../../results/20260907-attention-profile/grade-coding-baseline/summary.json) y [grade candidato](../../results/20260907-attention-profile/grade-coding-decode64/summary.json).
- [Comparación ampliada de programación](../../results/20260907-attention-profile/coding-extended-comparison.json).
- [Control de referencia final](../../results/20260907-attention-profile/decode-vs-final-reference.json), [auditoría final](../../results/20260907-attention-profile/final-audit.json) y [estado del servicio](../../results/20260907-attention-profile/service-final.json).
