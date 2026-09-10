# Revisión y propuesta: respuesta útil en menos de 30 segundos

Fecha: 2026-09-04. Estado: propuesta basada en la revisión de código y resultados existentes; no se ejecutaron nuevas mediciones GPU para este informe.

## Objetivo propuesto

Tomar como objetivo principal el tiempo desde que llega un turno hasta el primer texto útil o una llamada a herramienta completa y válida. Registrar por separado el primer token, el comienzo del razonamiento, el comienzo del contenido y la finalización. Esta interpretación queda pendiente de la preferencia del usuario; si exige terminar toda la respuesta en 30 segundos, el contrato también debe acotar su longitud.

Proponer p99 <= 30 s para turnos interactivos admitidos, con sesión caliente y una envolvente explícita de entrada/salida. Registrar todos los incumplimientos. Un percentil no es un máximo absoluto: un deadline puede asegurar interrupción o error a los 30 s, pero no asegurar una respuesta correcta y completa para cualquier tarea.

La pendiente con el contexto no tiene que ser cero para cumplir este objetivo en el intervalo finito hasta 262,144 tokens. Los anteriores objetivos de 200/300 ms y ratio <= 1.5 quedan como aspiraciones de optimización, no como requisitos necesarios para el nuevo objetivo. Esta propuesta no modifica todavía la especificación aprobada.

## Evidencia disponible

Fuentes locales:

- `results/20260904T192233Z-exl3-resident-probe-full/observations.jsonl`.
- `results/20260904T172916Z-exllamav3-3.5bpw-bringup/samples.jsonl`.
- `src/qwasar_bench/exllamav3_probe.py`.
- `src/qwasar_bench/openai_client.py`.
- Donor instalado en `../qwen38-exl3-mia/.venv/lib/python3.12/site-packages/exllamav3`.

| Experimento | Resultado | Alcance |
| --- | --- | --- |
| Rama directa a 256K, delta lógico 128, salida 1 | TTFT mediano 1.455 s | Tres muestras sintéticas; incluye overhead de Generator |
| Rama directa a 128K, mismo delta/salida | TTFT mediano 0.788 s | Tres muestras sintéticas |
| Turno HTTP persistente próximo a 256K | TTFT mediano 2.590 s; decode 15.65 tok/s; elapsed 18.44 s | Reasoning incluido; salida corta; cache hits físicos desconocidos |
| Ampliación de seed hacia 256K | 259.682 s, 131,198 tokens de prefill; 130,816 tokens cacheados | Ampliación masiva con reutilización parcial, no carga fría completa de 256K |

Con una tasa orientativa de 15 tok/s, 256 tokens requieren unos 17 s, 512 unos 34 s y 1,024 unos 68 s, además de ingesta y overhead. Estos cálculos son ilustrativos: no extrapolan un rendimiento garantizado desde el texto sintético a coding real.

## Correcciones a las conclusiones anteriores

1. **Reutilización comprobada no equivale a sesión viva GPU.** El donor libera el estado recurrente al finalizar un job y restaura un checkpoint al abrir otro. `RecurrentCache` mantiene stashes en memoria de sistema. Los datos comprueban reutilización de KV y checkpoints, no continuidad del estado recurrente en GPU entre turnos.
2. **La atribución completa a atención todavía no está medida.** La pendiente observada es compatible con atención exacta, pero faltan tiempos de kernels, transferencias, hashing, restauración y housekeeping. `run_generator_job` arranca después de construir el Job y toma el tiempo cuando `iterate()` devuelve eventos; en una salida de un token puede incluir tareas de cierre. `job.time_prefill` es un reloj del donor, no un evento CUDA aislado.
3. **Hay split-KV en el donor, pero NVFP4 sigue otra ruta.** `modules/attention_fn/dispatch.py` selecciona `_fns_fp8` para NVFP4 compatible. Esta lista apunta a `fn_triton_paged_attn` y `fn_triton_paged_attn_longq`. La primera usa `_paged_attn_splitdv_kernel`: divide la dimensión V, no la secuencia KV. La ruta larga agrupa consultas/cabezas sin eje de splits KV. Los kernels `paged_attn_triton_decode/prefill` con `num_splits` existen aparte. Debemos confirmar este dispatch con una traza antes de portarlo o cambiarlo; no basta con activar un flag genérico.
4. **El benchmark no demuestra calidad ni respuesta útil.** Repite un texto, genera un token y crea ramas desde un seed común. No incluye conversaciones largas continuas, resultados de herramientas grandes, invalidaciones, razonamiento prolongado ni calidad de código. El cliente HTTP cuenta `reasoning_content` o fragmentos de herramientas como primer delta, aunque todavía no haya contenido visible o JSON ejecutable.
5. **Hay sólo tres muestras por bucket.** El ajuste casi lineal describe esos cuatro puntos; no establece p95/p99 ni un máximo. Tampoco prueba el coste de una entrada nueva grande.
6. **256K es el presupuesto de posiciones de la secuencia.** El config local tiene `max_position_embeddings=262144`. Para uso soportado, contexto previo + entrada nueva + salida reservada deben caber dentro de ese límite. Asignar 262,400 slots KV no amplía por sí solo el contexto del modelo. La prueba de 262,144 tokens de entrada y un token de salida no es un contrato válido para una respuesta larga.
7. **La arquitectura se debe identificar por el artefacto.** La carpeta comercial se llama Qwen3.8, pero el config declara `Qwen3_5ForConditionalGeneration`, 64 capas, 16 de atención completa y 48 lineales. La etiqueta de arquitectura del manifest de bring-up difiere. Congelar hashes de config, template, pesos y código realmente importado antes de comparar kernels o 5 bpw.

## Ruta recomendada

### 1. Medir el tiempo que espera el usuario

Añadir relojes monotónicos para llegada, tokenización, cola, restauración, fin del prefill, primer token, primer reasoning, primer contenido, herramienta completa y fin de respuesta. Medir con eventos CUDA la atención, matrices, DeltaNet y transferencias en corridas de profiling separadas de las de latencia. Guardar los nombres de kernels ejecutados, tokens físicos procesados, aceptación del draft y memoria máxima.

No convertir el primer reasoning ni un heartbeat en cumplimiento del objetivo de respuesta útil. Para tools, exigir JSON completo válido. Un límite de salida alcanzado no es una respuesta completa satisfactoria.

### 2. Conservar la sesión caliente y el prefijo estable

Retener KV, página parcial, estado recurrente FP32 y posición exacta en GPU mientras la sesión esté abierta. Mantener también el estado necesario del draft. Añadir sólo tokens nuevos. Evitar cambios retroactivos de system prompt, herramientas, chat template o serialización que invaliden el prefijo. Separar el camino de append de ramas/rewind/cancelación y comprobar equivalencia con una reconstrucción exacta.

El objetivo inmediato es eliminar la repetición de la cola parcial y la restauración desde RAM en el turno habitual. No atribuir una mejora de 2x a pasar de 255 a 128 tokens antes de medir: el resto de costes permanece, y el token generado pendiente debe contarse exactamente una vez.

### 3. Controlar la ingesta de herramientas

Medir deltas de 128, 512, 2,048, 8,192 y 32,768 tokens sobre contexto casi lleno. La envolvente inicial propuesta para turnos interactivos es delta <= 2,048; es un límite a validar, no un SLA ya logrado.

Para coding, preferir diffs, rangos de archivo, búsquedas acotadas y resultados paginados. Conservar los resultados completos en artefactos accesibles por herramienta y marcar cualquier recorte explícitamente. No truncar silenciosamente información que pueda cambiar la respuesta. Cuando el resultado serializado sea definitivo, adelantar su prefill durante períodos ociosos si el flujo de la herramienta lo permite; no ingerir parciales que después se vayan a reescribir.

Chunked prefill acota unidades de trabajo y facilita cancelación/solapamiento, pero no elimina el coste total de leer una entrada nueva. Las importaciones grandes y recuperación de sesión deben tener métricas propias; actualmente no podemos prometer 30 s para ellas.

### 4. Optimizar atención NVFP4 con evidencia

Medir la ruta NVFP4 actual frente a split-KV exacto especializado para batch 1, 24 cabezas Q, 4 KV y dimensión 256. Barrer particiones 1/2/4/8/16/32/64 según longitud de consulta y contexto, con límites de scratch. Agrupar las seis cabezas Q que comparten KV cuando resulte rentable, leer/decuantizar bloques online y reducir softmax de manera estable. Comparar numéricamente contra atención densa sobre el mismo KV cuantizado, además de greedy/logits y tareas de coding.

Una referencia pertinente es [FlashInfer append/decode y Split-K](https://flashinfer.ai/2024/02/02/introduce-flashinfer.html). Su análisis justifica el experimento, no un multiplicador de velocidad transferible a esta 5090 y este formato.

### 5. Optimizar generación y razonamiento

Comparar target sin draft, DFlash2 y MTP con salida suficiente para medir aceptación y tokens útiles/s. DFlash2 ya estaba activo en los experimentos anteriores; habilitarlo otra vez no constituye una mejora. No asumir que reducir el bloque DFlash a cualquier tamaño es compatible con el checkpoint.

El bridge fuerza `enable_thinking=True`. Probar una política explícita de razonamiento para tareas simples frente a complejas usando únicamente controles que el modelo y backend soporten. Evaluar calidad al reducirlo: cortar la secuencia o forzar un cierre no garantiza que el modelo haya terminado de razonar. Reservar presupuesto tanto para reasoning como para contenido/tool JSON.

Para respuesta completa, un ejemplo de presupuesto sería 5 s de ingesta/overhead + 25 s de generación. Completar 1,024 tokens dentro de ese tiempo exigiría unos 41 tok/s sostenidos, frente a los aproximadamente 15 observados en las pruebas cortas. Es una meta por validar.

### 6. Escalar complejidad sólo si falla la envolvente

Si sesión viva, entradas acotadas y atención exacta optimizada cumplen 30 s, priorizar fiabilidad y calidad. El megakernel queda después del perfil: fusionar lanzamientos no elimina por sí solo la lectura de contexto o el coste de producir una respuesta larga.

Si el decode exacto sigue impidiendo la respuesta útil dentro del presupuesto, evaluar [Quest](https://arxiv.org/abs/2406.10774): selección de páginas dependiente de la consulta, manteniendo el KV completo. Medir coste del selector y calidad de referencias distantes/coding; no asumir complejidad constante, equivalencia exacta o que reintentar exacto después de un fallo sparse conserve el deadline. Una recuperación exacta puede requerir replay desde un estado limpio porque pasos aproximados anteriores ya afectaron capas posteriores y estado recurrente.

La distinción entre ahorro de prefill y coste de decode también está documentada por [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/).

## Presupuesto inicial a validar

Ejemplo para la primera respuesta útil o herramienta completa:

| Componente | Presupuesto propuesto |
| --- | ---: |
| Recepción, tokenización, cola y coordinación | 1 s |
| Append y primer paso | 6 s |
| Reasoning + contenido necesario para la primera acción | 20 s |
| Margen de variabilidad | 3 s |

Con 20 tok/s, 20 s permiten aproximadamente 400 tokens combinados hasta esa acción. Esto no demuestra que todas las tareas produzcan una respuesta útil con ese presupuesto. El envelope debe fijarse a partir de calidad y latencia medidas; las tareas que excedan su razonamiento/salida requieren otro contrato o una mejora de velocidad.

## Próximos experimentos y aceptación

1. Corregir instrumentación y la reserva de salida. Distinguir cold, warm resume y live append; no mezclar fases para los percentiles.
2. Barrer contextos 32K/128K/cerca de 256K, deltas 128/512/2K/8K/32K y salidas 1/256/1K. Respetar `prefijo + delta + salida <= 262144`. Usar fixtures reales de código y herramientas, además de controles sintéticos.
3. Probar conversaciones append-only durante al menos 20 turnos, incluido cruce de páginas, cancelación y reanudación. Registrar cuánto se invalida y cuánto se reejecuta.
4. Hacer primero screening corto de sesión viva y kernels; aumentar muestras sólo en configuraciones finalistas. Para estimar p99 de manera útil hacen falta muchas más muestras que las tres actuales; registrar intervalo de confianza, máximo y número de violaciones, no sólo p50.
5. Comparar cada optimización cambiando una variable y revalidar con EXL3 5 bpw cuando esté disponible. No inferir su memoria, velocidad o calidad desde 3.5 bpw.

Criterio de promoción: mejoras observadas en el tiempo útil extremo a extremo, sin regresión de calidad acordada, con memoria suficiente para completar la salida y recuperar/cancelar. La implementación del runtime y la certificación de un máximo de 30 s quedan pendientes de estos experimentos.
