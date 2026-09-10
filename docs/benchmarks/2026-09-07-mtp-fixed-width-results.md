# MTP: barrido de cantidades fijas de propuestas

**Actualización posterior al screening:** el usuario decidió adoptar seis propuestas
fijas en el servicio. La configuración activa se actualizó; las conclusiones y la
restauración a cuatro descritas abajo corresponden al cierre original del experimento.
La evidencia de adopción se conserva en [mtp6-adoption](../../results/20260907-mtp6-adoption/after.json).

Fecha: 2026-09-07. Investigación sobre la RTX 5090 local y Qwen3.8-27B EXL3 5 bpw. **Resultado: seis propuestas es el candidato más equilibrado del screening; el servicio conserva cuatro hasta validar tareas reales.**

## Tratamiento y controles

Se varía únicamente la cantidad fija de propuestas MTP, de una a siete; la adaptación permanece desactivada. Pesos fijados por hash, K8/V4, prefill Flash con chunk 8192, sampler y prompts se mantienen iguales dentro de cada fase. El pool físico tiene 262.144 posiciones incluso al medir 32K. Todas las ventanas mantienen el máximo de query existente en ocho posiciones.

La cantidad se pasa tanto al cargador de modelo/caché como al constructor del generador. El historial recurrente necesita espacio para verificar la ventana y retroceder cuando hay rechazos; para cinco a siete propuestas crece esa reserva. Esto es una consecuencia de ampliar la ventana, no un cambio de cuantización. La ruta de producción omite el argumento experimental y conserva su valor original de cuatro.

La primera tentativa no propagaba la cantidad al cargador. Cuatro y una funcionaban, pero seis falló porque el historial seguía reservado para cuatro. El error se conservó en `screen/fixed6.log`, el servicio se restauró, se añadió una prueba de regresión y se repitió **todo** el barrido en `screen2` con el harness corregido. Las cifras de la tentativa incompleta no se mezclan con las finales. No se modificaron archivos del backend donante.

## Protocolo y alcance

Una tarea de implementación LRU con tests, precedida por un corpus de código fuente sin repetición artificial; `thinking=medium`, salida máxima 4096. Cada corrida genera una respuesta de preparación y tres ramas con caché caliente. Las preparaciones no entran en las medianas. Los contextos nominales reservan salida y scratch dentro del límite nativo: las entradas efectivas son 28.656 tokens en 32K y 258.032 cerca de 256K.

El barrido greedy de 32K sigue el orden cuatro-inicio, una, seis, dos, siete, tres, cinco, cuatro-final. Las referencias de cuatro permiten observar variabilidad entre corridas; no eliminan todos los efectos de orden. Cerca de 256K se seleccionan seis, cuatro y siete, en ese orden. Cinco queda prácticamente empatado con seis a 32K, pero no se mide en largo en este ensayo acotado. No se afirma un óptimo global.

Se auditan hashes de pesos, runtime, corpus, template, harness y configuración de prefill; también GPU/controlador, prompts exactos y contadores físicos de caché. La velocidad de decode excluye la primera tanda y cuenta razonamiento más respuesta. TTFT es el observado por el harness, no latencia HTTP. Tiempo hasta contenido y tiempo total también dependen de la longitud generada.

Las respuestas extraídas se revisan y ejecutan en bubblewrap sin red ni acceso al proyecto, con límites de recursos. Cada una debe pasar sus tests generados (al menos seis efectivos) y siete comprobaciones LRU independientes. Los fallos permanecen en los denominadores y las medidas. Es una tarea de screening, no una evaluación general de calidad ni de sesiones con herramientas.

## Resultados a 32K (greedy)

Medianas de tres respuestas calientes por corrida. La columna de cambio es la mediana de razones por prompt frente a cuatro-inicio; no es necesariamente la razón entre medianas.

| Propuestas | Decode tok/s | Cambio pareado | TTFT ms | Aceptación draft | Calidad caliente |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed1 | 99.94 | -35.96% | 163 | 92.2% | 3/3 |
| fixed2 | 127.98 | -18.46% | 164 | 85.6% | 3/3 |
| fixed3 | 147.70 | -6.04% | 165 | 77.6% | 2/3 |
| fixed4-start | 156.06 | +0.00% | 167 | 71.7% | 2/3 |
| fixed4-end | 158.64 | +1.68% | 167 | 71.8% | 2/3 |
| fixed5 | 166.90 | +5.41% | 168 | 65.7% | 3/3 |
| fixed6 | 167.67 | +5.62% | 170 | 59.3% | 2/3 |
| fixed7 | 162.59 | +4.19% | 172 | 52.0% | 2/3 |

Cinco y seis ofrecen una ventaja moderada sobre ambas referencias de cuatro. La menor aceptación porcentual de ventanas grandes no implica por sí sola menor velocidad: importa cuántos tokens finales se obtienen por unidad de tiempo. Una propuesta tiene la aceptación más alta y el decode más lento de este barrido.

Las 32 implementaciones pasan el oráculo independiente; 27/32 respuestas pasan además sus propios tests (19/24 calientes). No hay truncamientos ni requeues. Los fallos son expectativas incorrectas en tests generados. Las tres salidas greedy de cuatro-final también difieren de cuatro-inicio pese a usar el mismo ancho: la repetición evidencia variabilidad de secuencia. No se atribuyen todas las diferencias de calidad al ancho MTP ni se afirma paridad numérica.

Todas las ramas calientes usan 28.416 tokens de caché y 239 nuevos de prefill. El pico asignado por PyTorch pasa de aproximadamente 24.858 MiB con cuatro a 25.155 MiB con seis y 25.320 MiB con siete; el pico reservado pasa de 27.628 a 28.364 y 28.374 MiB. Estos contadores no incluyen toda la memoria GPU.

## Resultados cerca de 256K (greedy)

| Propuestas | Decode tok/s | Cambio pareado vs cuatro | TTFT ms | Primer contenido s | Respuesta completa s | Tokens de salida | Calidad caliente |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed4 | 84.57 | +0.00% | 326 | 5.79 | 28.15 | 2357 | 2/3 |
| fixed6 | 93.29 | +9.42% | 329 | 5.78 | 25.69 | 2348 | 1/3 |
| fixed7 | 94.19 | +12.47% | 331 | 5.77 | 24.36 | 2260 | 2/3 |

Los tres pares de seis frente a cuatro mejoran entre 9,29% y 11,40%. Siete mejora entre 6,63% y 13,92%; la mediana pareada es 12,47%. La diferencia de medianas de velocidad entre seis y siete es de sólo 0,90 tok/s: no demuestra una superioridad general de siete.

En las nueve ramas calientes coinciden los 257.792 tokens reutilizados y 239 nuevos de prefill. Cero requeues y truncamientos en las doce generaciones. Las doce implementaciones pasan el oráculo; siete respuestas completas pasan además sus tests (cinco de nueve calientes). En seis aparece una respuesta con seis expectativas incompatibles con su propio sentinel y otras con un orden de expulsión mal supuesto. Se conservan esos fallos; no se repara el código ni se descartan tiempos.

La reducción del tiempo completo también depende de las longitudes diferentes. MTP mayor no elimina el coste de cargar un contexto frío. Ninguno de los seis pares de variantes contra cuatro conserva exactamente toda la secuencia greedy.

## Comprobación con sampler recomendado a 32K

Se ejecutaron siete, cuatro y seis, en ese orden, con temperatura 1, top-p 0,95, top-k 20, min-p 0 y penalización de presencia 0 (`medium`). Todo lo demás permaneció igual. No se midió este sampler en contexto largo.

| Propuestas | Decode tok/s | Cambio pareado vs cuatro | TTFT ms | Primer contenido s | Respuesta completa s | Tokens de salida | Calidad caliente |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed4 | 162.77 | +0.00% | 169 | 6.01 | 14.14 | 2330 | 2/3 |
| fixed6 | 170.77 | +8.24% | 171 | 4.40 | 13.59 | 2252 | 2/3 |
| fixed7 | 165.19 | +3.93% | 172 | 7.70 | 17.84 | 2846 | 3/3 |

Seis mejora en los tres pares (1,66%, 8,24% y 10,24%; mediana pareada 8,24%). La razón entre medianas de velocidad es menor: **4,91%**. Son dos resúmenes distintos de las mismas tres observaciones. Siete gana 3,93% por la mediana pareada y 1,49% por razón de medianas, pero tarda más en terminar porque genera respuestas más largas. No equivale a una mejora garantizada de latencia percibida.

Las doce implementaciones pasan el oráculo y nueve respuestas pasan también sus tests; siete de nueve calientes pasan ambos. No hay truncamientos ni requeues. Tres repeticiones medidas y una sola tarea no bastan para certificar calidad general o equivalencia de distribución.

## Decisión y siguiente paso

**Mantener cuatro en producción por ahora; seis es el candidato más equilibrado para una validación posterior con tareas reales del agente.** En este screening ofrece mayor decode a 32K con ambos samplers y cerca de 256K en greedy. Siete obtiene un pequeño margen adicional en largo, pero es más lento en corto. Cinco empata prácticamente con seis en corto greedy y no se evaluó fuera de esa fase.

La mejora observada de seis frente a cuatro, usando razones entre medianas, es aproximadamente 5,7–7,4% en corto greedy según cuál referencia se use, 4,9% en corto recomendado y 10,3% en largo greedy. Las medianas pareadas correspondientes están en las tablas. Esto justifica continuar evaluando MTP sin reescribir el motor completo; no demuestra optimalidad ni una ganancia universal.

El resultado largo de seis —1/3 respuestas completas correctas frente a 2/3 con cuatro— y la variación de secuencia incluso entre referencias idénticas impiden afirmar paridad de calidad. No se ha demostrado que el ancho cause la diferencia. Antes de adoptar, corresponde contrastar tareas de Qwasar con herramientas, sesiones continuas, varios tamaños de contexto y el sampler habitual, manteniendo el resto fijo.

## Operación y verificación

Las tres fases completas suman **56 generaciones: 14 de preparación y 42 calientes**. Todas las implementaciones pasan el oráculo; **43/56** respuestas completas pasan también sus propios tests (**31/42** calientes). Las ocho respuestas completas de la tentativa inicial incompleta se conservan como diagnóstico y no integran estos totales.

La auditoría verifica invariantes también entre fases y la procedencia de cada grade mediante hashes de metadata y muestra. Los 56 resultados no tienen requeues ni truncamientos. Los comandos de grading que retornan estado 1 registran fallos del código generado; no son fallos ocultos del harness.

La suite local después de corregir la propagación al cargador da **289 pruebas aprobadas y 35 omitidas** por dependencias/entornos no disponibles. La prueba de regresión falló antes del arreglo y pasó después. El barrido real verifica los siete anchos en GPU, incluyendo los que requerían ampliar el historial recurrente. No se hicieron commits ni cambios en el backend donante.

Cada fase detuvo el servicio sólo estando libre y lo restauró automáticamente. Al finalizar, `/config` devolvió `ready`, `busy=false` y el objeto de configuración completo coincidió con el original: EXL3 5 bpw, MTP fijo de cuatro, K8/V4 y Flash. La parada pierde caché GPU previa; las sesiones durables se conservan.

## Artefactos reproducibles

- [Auditoría consolidada](../../results/20260907-mtp-fixed-width/audited-study.json).
- [Protocolo del barrido](../../results/20260907-mtp-fixed-width/screen2/protocol.json) y [resumen por muestra a 32K](../../results/20260907-mtp-fixed-width/screen2/summary.json).
- [Protocolo largo](../../results/20260907-mtp-fixed-width/long/protocol.json) y [resumen largo](../../results/20260907-mtp-fixed-width/long/summary.json).
- [Protocolo con sampler recomendado](../../results/20260907-mtp-fixed-width/validation/protocol.json) y [resumen](../../results/20260907-mtp-fixed-width/validation/summary.json).
- [Grade largo de seis](../../results/20260907-mtp-fixed-width/long/grade-fixed6/summary.json), [de cuatro](../../results/20260907-mtp-fixed-width/long/grade-fixed4/summary.json) y [de siete](../../results/20260907-mtp-fixed-width/long/grade-fixed7/summary.json). Cada fase conserva también todos los demás grades, código extraído, logs y eventos.
- [Diagnóstico del primer intento](../../results/20260907-mtp-fixed-width/screen/failure-analysis.json).
- [Ejecutor con restauración](../../results/20260907-mtp-fixed-width/execute.py), [auditor por fase](../../results/20260907-mtp-fixed-width/analyze.py) y [auditor final](../../results/20260907-mtp-fixed-width/final-audit.py).
- [Estado final del servicio](../../results/20260907-mtp-fixed-width/service-final.json).
