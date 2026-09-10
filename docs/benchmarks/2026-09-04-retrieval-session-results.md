# Recuperación distante y sesiones append-only

Fecha: 2026-09-04. Una RTX 5090; SGLang en la 3090 Ti no participa. Continuación de `2026-09-04-decode-screen-results.md`.

## Pregunta y protocolo

La batería LRU medía velocidad de código, pero no obligaba al modelo a usar el contexto distante. Esta extensión comprueba si puede recuperar datos distribuidos, llamar una herramienta, consumir su resultado y continuar con un segundo pedido sin reconstruir el historial.

Cada sesión inserta tres tablas sintéticas entre fragmentos únicos del snapshot de ExLlamaV3: contratos cerca del inicio, factores en el medio y rutas cerca del final. Hay 16 servicios con identificadores aleatorios reproducibles. Dos consultas distintas requieren recuperar límites y tags del contrato, un factor y tag de la tabla intermedia y una ruta del registro final. La herramienta devuelve otro tag y un offset, ausentes del snapshot. El resultado exigido es `min(max(valor, mínimo), máximo) * factor + offset` junto con los cuatro identificadores correspondientes.

El harness valida la llamada XML de `read_file` y despacha únicamente rutas incluidas en un mapa de fixtures en memoria. No lee rutas arbitrarias del host ni ejecuta código generado. Esta herramienta y la recuperación son sintéticas: no confundirlas con un agente que modifica un repositorio real.

Son cuatro generaciones por sesión: llamada, respuesta, nueva llamada, nueva respuesta. Cada prompt conserva exactamente los tokens anteriores que devuelve el runtime, incluyendo reasoning y terminadores. No se retokeniza ni se rerenderiza el historial generado. Se agrega únicamente la cola de usuario/herramienta usando el template nativo; cuerpo y respuestas de herramienta se tokenizan literalmente. Una comprobación CPU con el tokenizer real confirma que estas colas equivalen al texto del render nativo en medium/off y no convierten delimitadores literales en roles nuevos.

Configuración: EXL3, MTP, K8/V4, sampler recomendado por modo. `medium` usa temperatura 1, top-p 0.95 y presencia 0; `off` usa temperatura 0.7, top-p 0.8 y presencia 1.5. Comparar políticas completas, no atribuir todo el efecto al flag de thinking.

Presupuestos: hasta 512 tokens por llamada y 1,536 por respuesta. El prompt inicial reserva ambas parejas completas, 1,024 tokens de mensajes adicionales y 16 de margen. Entrada inicial: 27,632 en 32K y 257,008 en 262,144 posiciones. Se verifica de nuevo el límite antes de cada generación. Una truncación o llamada inválida hace fallar la sesión; no se rescata aumentando el presupuesto selectivamente.

La espera por ciclo abarca preparación del prompt, generación de la llamada, despacho de herramienta y generación de la respuesta final. Los tiempos son del proceso directo, no HTTP: excluyen carga del modelo y tokenización inicial del corpus. La duración del segundo ciclo es la medida interactiva caliente; el primer ciclo incluye el prefill inicial. Los contadores físicos permiten distinguirlo. Los tokens/s incluyen reasoning y excluyen el primer batch, igual que en el probe anterior; salidas breves no son un benchmark sostenido de código.

## Piloto a 32K, EXL3 5 bpw

Los dos pilotos asignan el pool sólo a 32,768 posiciones. No mezclar sus latencias con la matriz que asigna el pool completo.

- `off`: recupera ambas rutas correctamente, pero el primer paso de respuesta repite la llamada en lugar de entregar JSON. En el segundo entrega los tags correctos pero calcula 38 en vez de 66, ignorando el mínimo de 15. Ningún ciclo aprueba.
- `medium`: ambas llamadas y ambos JSON aprueban, incluyendo resultados 408 y 66. Esto muestra que un fallo de off no equivale a incapacidad de recuperar todos los datos o a un problema de framing.
- Los resultados no prueban superioridad general de medium: sólo hay dos consultas de configuración y una semilla. Tampoco permiten atribuir los fallos de off a la cuantización sin controles adicionales.

Artefactos: `results/20260904-session-5bpw-off-32k/` y `results/20260904-session-5bpw-medium-32k/`. Los errores del modelo se conservan sin modificaciones.

## Matriz MTP con pool completo

Cada proceso asigna 262,144 posiciones y evalúa 32K y luego 256K con `medium`, una sesión por tamaño. Los registros largos comienzan en las posiciones 13,025, 128,349 y 243,547 del prompt. Cada nueva sesión es independiente; la sesión larga de 5 bpw registra cero tokens reutilizados en su primer job.

| Target / contexto | Espera ciclo inicial | Espera segundo ciclo completo | Calidad por ciclo |
| --- | ---: | ---: | --- |
| EXL3 3.5 bpw / 32K | 11.503 s | 1.860 s | Pasa / pasa |
| EXL3 5 bpw / 32K | 12.907 s | 2.817 s | Pasa / pasa |
| EXL3 3.5 bpw / 256K | 205.572 s | 5.531 s | Falla / falla |
| EXL3 5 bpw / 256K | 210.174 s | 5.337 s | Falla / pasa |

En 5 bpw a 256K, el primer job tarda 204.943 s hasta el primer token y procesa físicamente 257,007 tokens. Después de recibir la herramienta, el modelo recupera los tags correctos y calcula 408 en su reasoning, pero su contenido final repite `read_file` en vez de entregar el JSON solicitado. No se cuenta el reasoning correcto como una respuesta correcta, ni se reintenta o se inyecta el resultado esperado. El segundo pedido se agrega al historial tal como quedó.

Ese segundo ciclo pasa ambos pasos y completa en 5.337 s. Sus prompts tienen 257,616 y 257,869 tokens; reutilizan 257,024 y 257,536, con sólo 591 y 332 tokens de prefill físico. TTFT de esos jobs: 1.031 y 0.727 s; decode: 104.09 y 106.10 tokens/s. Termina con 258,031 tokens en el historial. Esta velocidad corresponde a respuestas cortas de alta aceptación especulativa, no a código largo arbitrario.

Artefacto: `results/20260904-session-5.0bpw-medium-long/`. El fallo de entrega demuestra por qué no basta con medir retrieval, tok/s o el comienzo del reasoning por separado.

En 3.5 bpw a 256K, ambos pasos de respuesta recuperan y calculan correctamente en el reasoning (408 y 66), pero vuelven a emitir `read_file` en lugar del JSON. El segundo ciclo tarda 5.531 s y sus jobs alcanzan 108.15/109.36 tokens/s, pero ambos datos describen una entrega fallida, no una respuesta útil más rápida. Reutilizan 257,280/257,536 tokens y procesan 366/376. El primer job tiene TTFT de 199.488 s sin reutilización; el historial termina con 258,126 tokens. Artefacto: `results/20260904-session-3.5bpw-medium-long/`.

Los prompts iniciales de cada contexto coinciden por SHA-256 entre ambas cuantizaciones; los siguientes difieren porque preservamos lo que cada modelo realmente generó. No hay comparación pareada de idénticos prompts en los cuatro jobs ni suficiente muestra para ordenar calidad general. La evidencia útil es que el prefijo no se reproduce completo en los pasos calientes, mientras aparecen fallos de entrega que un benchmark de velocidad ocultaría.

## Control sin MTP: mismo resultado, menos velocidad

Se repitió 5 bpw, K8/V4, medium y 262,144 posiciones con `QWASAR_DRAFT_METHOD=none`, sin cambiar presupuestos ni semillas. El primer prompt coincide con la matriz MTP y ambos registran cero reutilización inicial. En este control coinciden los prompts de los cuatro jobs, las cuatro completions completas y sus secuencias de tokens emitidos; el auditor comprueba cada igualdad.

| Paso largo | MTP tok/s | Sin draft tok/s |
| --- | ---: | ---: |
| Primera llamada | 100.25 | 43.79 |
| Primera respuesta, fallida | 98.88 | 43.86 |
| Segunda llamada | 104.09 | 43.86 |
| Segunda respuesta, correcta | 106.10 | 43.85 |

Sin draft, el primer ciclo falla de la misma manera: reasoning correcto y repetición de la herramienta. El segundo pasa y tarda 10.426 s frente a 5.337 s con MTP. El decode de este caso mejora aproximadamente 2.25–2.42 veces con MTP y el ciclo caliente completo casi dos veces. Esto es evidencia pareada para estos outputs, no garantía universal de igualdad de distribución o rendimiento.

Por tanto, **desactivar MTP no corrige este fallo y pierde el objetivo de 50 tokens/s en esta carga**. No se atribuye la repetición a MTP: el control la reproduce sin él. Ambos caminos todavía usan K8/V4 y reutilización de estado; no separa los posibles efectos de la cuantización KV, del estado cacheado, del contexto largo o de la política de formato del modelo.

Artefacto: `results/20260904-session-5bpw-no-draft-long/`. Auditoría externa de siete sesiones y 28 generaciones: `results/20260904-session-audit.json`, con hashes de samples y completions, recalificación estricta y comprobación de ausencia de reencolados. No modifica los JSON originales.

## Decisión provisional

1. Por decisión del usuario, mantener EXL3 5 bpw + MTP como configuración de trabajo. K8/V4 continúa como caché experimental actual. Esta elección no certifica todavía calidad frente a BF16 ni un SLA; 3.5 bpw queda como control, no como foco de optimización.
2. Mantener medium como política a evaluar para agente; los dos éxitos anteriores de off en LRU no se trasladaron a esta tarea de herramientas. No convertir un piloto pequeño en una política universal.
3. Antes de optimizar kernels, comparar continuación cacheada contra replay frío con el mismo prompt y semilla, y después una caché de mayor precisión. Así se puede separar un problema de estado/KV de uno de comportamiento o formato.
4. Medir entrega válida, no sólo tokens/s: la repetición de herramientas necesita un control explícito de la sesión y una política de salida evaluada. No ocultarla con reintentos selectivos ni contar reasoning correcto como respuesta final.

## Reproducción y límites

La sección "Append-Only Retrieval Sessions" del README contiene el comando completo. Cada directorio es nuevo y conserva `run.json`, corpus, fixture/oráculo, eventos, samples y resúmenes. Un marcador de colección completada no significa aprobación de calidad. El oráculo no se entrega al modelo; sólo se incluyen las tablas y las respuestas reales de la herramienta.

Persistencia aquí significa continuidad exacta entre jobs en un mismo proceso. No certifica persistencia tras reiniciar, ausencia de checkpoints recurrentes en RAM, ejecución end-to-end de un cliente de agente ni un SLA de 30 segundos. Sigue pendiente medir edición multiarchivo, recuperación más difícil, varias semillas y calidad frente a BF16.

La revisión del harness añadió rechazo de campos JSON duplicados y señalización de métricas de caché/draft no disponibles cuando el runtime reencola un job. Las métricas de la última porción física no deben confundirse con las de toda la generación. Esto no cambia prompts ni sampling. Los hashes del harness distinguen los runs anteriores y posteriores a esos cambios. Las respuestas de los pilotos y de la matriz MTP fueron recalificadas con el evaluador final: todos los dictámenes coinciden; sus eventos no contienen reencolados. Los artefactos originales permanecen intactos.

Al cierre, 76 tests de pytest pasan, el launcher pasa `bash -n` y ambos manifests siguen válidos. La revisión independiente confirmó las dos correcciones del evaluador/contadores. Los cinco procesos de benchmarking terminaron y liberaron la 5090; el servicio SGLang de la 3090 Ti permaneció activo. No se modificó el checkout donante ni se creó un commit.
