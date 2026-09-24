# Adaptación del MTP al target híbrido: piloto residual

## Alcance

Primera prueba de aprendizaje, no otro cambio de kernels ni entrenamiento
completo del bloque MTP. El usuario autoriza hasta 90 minutos de GPU exclusiva
con parada controlada y restauración del servicio.

Se entrena únicamente una corrección de rango bajo sobre la salida
post-normalización del MTP:

```text
z = fp16(float(h_draft) + (float(h_draft) @ A) @ B)
```

`A` tiene forma 5120×r y `B`, r×5120; r∈{16,32}. `B` empieza en cero, por lo
que el adaptador inicial conserva exactamente la salida para entradas finitas.
La salida corregida alimenta tanto el head hot64k como el siguiente paso del
rollout MTP6. No se modifica el target, sus pesos, su head completo, su sampler,
su caché ni la lógica de aceptación/rechazo.

Esto prueba si existe una desalineación aprovechable con una corrección barata.
Un resultado negativo no descarta entrenar adaptadores dentro del bloque,
entrenamiento multi-step o un drafter de mayor capacidad.

## Captura y alineación

Se usa `ExLlamaBackend(prefill="xqa")` con NVIDIA64, XQA/KV NVFP4, PRIMS,
rendezvous, hot64k y visión. La captura usa pool de 8192 posiciones para
dejar espacio a la exportación; la evaluación integrada asigna 262144.

El MTP de Qwen consume el estado del target **después de la norma final**.
Para cada paso del draft se guardan input ID, posición de caché y estado de
salida. En el verify siguiente se comprueba que cada fila del target recibió
exactamente el mismo token y prefijo especulativo; se emparejan `D[t]` y `H[t]`.
No se desplaza el target un token adicional.

Se conservan los seis niveles, incluidos prefijos que después se rechazan.
No se presentan esas filas como seis tokens independientes aceptados. Las
copias de instrumentación sincronizan y por tanto **la captura no mide velocidad**.

El smoke guardó 378 pares en 63 ventanas, con dos prompts y presupuesto 128.
La reconstrucción FP16 del head hot64k, independiente de su GEMM de decode,
dio RMS relativo **0,001063** y argmax **32/32** frente al kernel original
en entradas aleatorias. Esa prueba detecta errores de base/Hadamard, pero no
demuestra igualdad universal de logits.

## Datos y entrenamiento fijados antes de medir

- 48 documentos de entrenamiento: ocho familias, seis variantes por familia.
  CSV, INI, heap, RLE, trie, paréntesis, union-find y bitset.
- Ocho documentos de validación: percent encoding, romanos, histograma y matrices.
- Ocho documentos de test: Luhn, paginación, deduplicación y duraciones.
- División por familia, nunca por filas adyacentes del mismo documento.
- Prompts sintéticos nuevos en inglés/español; no se entrenan con completions
  de los benchmarks anteriores ni se recalibra el mapa hot64k.
- Presupuesto de captura de 1024 tokens por documento. Son prefijos generados,
  a menudo truncados y centrados en razonamiento, no una colección de programas
  completos o soluciones verificadas.
- Hasta 1024 pares por documento, distribuidos a lo largo de su secuencia.
- Head reconstruido y target congelados. Objetivo: entropía cruzada con el
  top-64 de la distribución del teacher **dentro del vocabulario hot64k**,
  más penalización del tamaño de la corrección. Se registra la masa retenida
  y cuántos argmax del teacher completo están fuera del mapa.
- AdamW, LR 1e-3, batch 128, 800 pasos por rango, gradiente limitado a 1.
  Checkpoint cada 100 pasos; selección por pérdida de validación.
- El test offline se abre sólo después de fijar el checkpoint. No se elige
  un checkpoint mirando throughput o calidad del test.

No es destilación exacta sobre todo el vocabulario: se aproxima el objetivo
con el head restringido y una cola truncada. La corrección también cambia las
entradas de los pasos siguientes; una ganancia offline puede desaparecer en
rollout cerrado. Ese contraste es parte central del piloto.

## Comparación integrada

Cuatro tareas de familias reservadas, contextos 4K/32K y cuatro ejecuciones
por tarea/contexto en orden A/B/B/A o B/A/A/B. MTP6, mapa y target constantes.
Calentamiento fuera de medida, cache reset entre muestras, mismo prompt y seed
por pareja. Se incluyen el coste del adaptador y sus copias/conversiones.

Registrar tok/s, aceptación, tokens emitidos, número de verifies y coste por
verify estimado. Este último divide tiempo de decode por número de ventanas;
no es un temporizador CUDA aislado y contiene efectos de primera/última tanda.
Las salidas acotadas no se califican como tareas correctas.

El piloto de tareas completas reutiliza el harness independiente de la
campaña NVFP4 con el perfil de proyecciones **control** en ambos brazos:
código, tools, imágenes, recuperación larga y continuaciones. Sólo el candidato
instala el adaptador. Dos seeds y orden A/B/B/A. Se mantienen las limitaciones
documentadas del grader de recuperación y del parser `string|null`; se
presentan resultados estrictos y semánticos por separado.

No se garantiza identidad de trayectorias con la misma seed: aceptación
distinta cambia el consumo del sampler y el control ya tiene variación
numérica. La inmutabilidad del target es una condición estructural, no una
certificación nueva de equivalencia del sampler ni de calidad general.

## Reproducción y aislamiento

Todo vive en `qwasar_bench`, sin imports nuevos desde producción:

```sh
# Entorno donante y exclusivamente bajo managed.py:
python -m qwasar_bench.mtp_study capture --output NUEVO/data
python -m qwasar_bench.mtp_study train --source NUEVO/data --output NUEVO/training
python -m qwasar_bench.mtp_study online --adapter CHECKPOINT --output NUEVO/online
# Orquestación con deadline interno, sin promoción:
python -m qwasar_bench.mtp_campaign campaign --output NUEVO --budget 4800
```

La ventana principal usa deadline externo de 5100 segundos, además del límite
interno de 4800, dejando margen de restauración dentro de los 90 minutos
autorizados junto al smoke. Los resultados se crean sin sobrescribir archivos.
No se usa la RTX 3090 Ti ni se exportan datos fuera de la máquina.

## Resultado de entrenamiento

Captura completa: **114.690 pares en 19.115 ventanas**, de 64 documentos y
65.088 tokens emitidos. Se retienen 49.152 filas de train, 8.192 de validación
y 8.192 de test para entrenar/evaluar.

El checkpoint seleccionado es **rango 16, paso 100**, con 163.840 parámetros
FP32 (0,625 MiB), SHA256
`0abc2785033fa898a147c378035c0d1dd87d3dcac25b23c4b9ed27068a33fbf8`.
Ni la repetición de evaluación ni los fallos de memoria cambian esta elección.

| Métrica offline | Original | Adaptador |
|---|---:|---:|
| Entropía cruzada, validación | 1,90989 | 1,87342 |
| Coincidencia argmax hot64k, validación | 67,61% | 67,00% |
| Entropía cruzada, test reservado | 1,88622 | 1,85250 |
| Coincidencia argmax hot64k, test reservado | 67,69% | 67,21% |

La corrección RMS relativa es 10,91% en test. El top-64 retiene 99,90% de la
masa **dentro de hot64k**; el argmax del head completo está dentro del mapa
en 96,77% de las filas. No son medidas sobre el vocabulario completo.

Mejora la pérdida suave pero no el argmax. La pérdida de validación empeora
con entrenamiento más largo; no hay evidencia para recomendar simplemente
más pasos. Las métricas offline tampoco incorporan la realimentación de la
salida corregida en los siguientes pasos del drafter.

## Incidencias de evaluación y trazabilidad

1. `pilot/`: captura y entrenamiento completos. Se obtienen 16 muestras
   online de 4K y se aborta al comenzar 32K por OOM, en el control.
   El servicio se restaura. No se rellena artificialmente `completed.json`.
2. El calentamiento consultaba `cache.max_seq_len`, atributo inexistente, y
   caía a 2048 tokens. Se corrige para usar `max_num_tokens`: 8200 con pool
   262144, 7168 con pool 8192. Test CPU de regresión añadido. La primera
   captura sigue siendo válida para el alineamiento, no para medir prefill.
3. `followup/`: mismo checkpoint, calentamiento corregido y allocator original.
   Se completan 32 muestras online, con reintentos de asignación OOM.
   La calidad aborta después en el control sin adaptador por otro OOM.
   El servicio se restaura. El calentamiento corregido no certifica margen
   suficiente ni demuestra por sí solo la causa de los OOM.
4. `expandable/`: repetición simétrica de online y calidad con
   `PYTORCH_ALLOC_CONF=expandable_segments:True` únicamente en los procesos
   experimentales. Se guarda el entorno. No se cambia el allocator del
   servicio, el contexto, el checkpoint ni se descarga visión.

La comparación completa con allocator original de `followup/` da:

| Contexto | Original tok/s | Adaptador tok/s | Cambio |
|---|---:|---:|---:|
| 4K | 168,13 | 156,84 | −6,72% |
| 32K | 151,16 | 152,22 | +0,70% |

Son medianas de ocho muestras por brazo/contexto. Las cuatro diferencias
pareadas por tarea van de −10,86% a +2,25% en 4K y de −9,32% a +11,64%
en 32K. No se selecciona sólo la tarea favorable.

La repetición completa `expandable/online/`, también de 32 muestras, da:

| Contexto | Original tok/s | Adaptador tok/s | Cambio | Aceptación original → adaptador |
|---|---:|---:|---:|---:|
| 4K | 167,01 | 157,31 | −5,81% | 41,19% → 38,38% |
| 32K | 150,03 | 147,49 | −1,70% | 39,07% → 38,61% |

Coste estimado por verify: 20,764→20,910 ms en 4K (+0,71%) y
22,208→22,410 ms en 32K (+0,91%). Incluye más que kernels del adaptador:
no constituye una medición aislada de su overhead. TTFT cambia menos de
0,5% en ambas longitudes. En 4K las cuatro diferencias pareadas por tarea
son negativas (−8,78%, −4,96%, −4,08%, −8,34%). En 32K hay signos mixtos
(+0,51%, +0,38%, −7,84%, +8,21%).

No se mezclan las dos políticas de allocator en una sola mediana. Tampoco
se comparan estos valores absolutos con los 230–280 tok/s de otros documentos:
prompts, sampling y trayectorias son diferentes. La evidencia permite rechazar
este candidato por falta de ganancia, no fijar un techo de rendimiento del MTP.

## Calidad de tareas (harness NVFP4, perfil control en ambos brazos)

60 respuestas, 30 por brazo, dos seeds, orden A/B/B/A. Todas terminaron en
`completed`; el código se calificó sólo en bubblewrap tras liberar la GPU.
El análisis exige cuatro brazos, hash del checkpoint igual al seleccionado,
hashes de muestras no obsoletos y configuraciones de target idénticas.

| Categoría | Original | Adaptador |
|---|---:|---:|
| Código (oracle + tests del modelo) | 3/6 | 4/6 |
| Visión (conteo de formas) | 6/6 | 6/6 |
| Tools, exacto | 0/6 | 0/6 |
| Recuperación larga, estricto | 0/6 | 0/6 |
| Continuaciones, estricto | 0/6 | 0/6 |
| **Total estricto** | **9/30** | **10/30** |
| Recuperación + continuaciones, semántico | 12/12 | 12/12 |
| **Total con recuperación semántica** | **21/30** | **22/30** |

Parejas: 2 mejoras (bimap-s193, interval-s193), 1 regresión (topology-s193).
Delta +3,3 puntos porcentuales, intervalo bootstrap descriptivo por familias
[−10, +20]: no es un certificado de calidad en ninguna dirección. Los fallos
de tools (`null` → `"null"`) y el 0/6 estricto de recuperación reproducen las
limitaciones ya documentadas del harness compartido, no del adaptador.

Largo contexto con segmentos expandibles (medianas, 2 por brazo): TTFT frío
31838 tok 4,98→5,03 s; 130140 tok 26,25→26,24 s; 257117 tok 66,67→67,06 s.
Decode de recuperación 242,6→246,6 / 213,3→211,0 / 195,2→184,4 tok/s. Pico
asignado 28,835 GiB en ambos brazos; sin OOM en esta repetición.

## Decisión

**No promover este candidato.** En la comparación simétrica completa pierde
5,8% de throughput en 4K y 1,7% en 32K; la aceptación baja; la pérdida
offline mejora pero el argmax empeora, señal de que una corrección residual
post-norma de rango 16 no captura la desalineación de forma aprovechable en
el lazo cerrado. La calidad medida no compensa porque tampoco hay ganancia
de velocidad que justifique un intercambio.

Esto **no** descarta la adaptación del MTP en general: quedan abiertos
adaptadores dentro del bloque, destilación multi-step con la corrección en
el rollout de entrenamiento, y teacher sobre el vocabulario completo.

## Estado final

Piloto terminado dentro del presupuesto: cuatro ventanas gestionadas,
~36 minutos en total de GPU exclusiva frente a los 90 autorizados, servicio
restaurado tras cada una (incluidos los dos fallos por OOM) con configuración
idéntica y `ready`, no ocupado. La RTX 3090 Ti no se tocó. Suite completa:
500 pruebas pasan, 3 omitidas. Evidencia: `results/20260923-mtp-adapter/`
(`comparison.json` con hashes de toda la evidencia). No se instala ningún
adaptador en producción.
