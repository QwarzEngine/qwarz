# V1 híbrida: verificación real con Pi

Fecha: 2026-09-05. EXL3 5 bpw + MTP (4 drafts) + K8/V4,
Flash/8192, RTX 5090 GPU 0, pool de 262.144 posiciones. El proceso
SGLang PID 3547 en GPU 1 y los cambios preexistentes del donante permanecen
intactos. Servicio local en `127.0.0.1:8800`, modelo `qwasar-qwen38-27b`.

## Entrega

- Supervisor Rust/Axum, SQLite WAL durable y worker Python ExLlamaV3.
- Chat Completions para Pi, SSE de texto/reasoning/tools y adaptador Responses
  acotado con recuperación por `previous_response_id`.
- Historial generado conservado como IDs exactos, incluidos los terminadores
  nativos reales; continuidad append-only y reconstrucción después de reinicio.
- Un trabajo activo; rechazo 409 de concurrencia, cancelación, recuperación
  y backpressure que nunca reporta una respuesta truncada como exitosa.
- Hash completo del artefacto validado antes de cargar CUDA; guardas de GPU,
  propiedad exclusiva de base de datos, scripts de arranque/parada y configuración
  Pi con respaldo exacto de proveedores anteriores.

La revisión independiente encontró y cerró problemas de herramientas con el
wrapper HF real, EOF/cancelación, nombres históricos de parámetros, restauración
de reasoning repetido, identidad de pesos, timeouts, pérdida de deltas y arranque
duplicado sobre SQLite. Las regresiones permanecen en los tests.

## Pruebas automáticas

**275 tests Python y 13 tests Rust pasan**, incluyendo 10 pruebas HTTP con
workers fake explícitos, 34 pruebas de runtime con tokenizer/template real en
CPU y pruebas de ciclo de vida con reloj simulado. Build release y formato Rust
verificados. Fake nunca se selecciona en el servicio real.

El smoke inicial del worker real asignó la ventana completa y respondió READY.
Detectó una métrica de decode engañosa para una sola tanda seguida de EOS:
se corrigió el numerador para contar tokens efectivamente emitidos después de
la primera tanda, no contabilidad adicional del terminador. Una sola tanda
ahora reporta velocidad `null`, no miles de tokens/s artificiales.

## Pi 0.84.4 real

Proyecto desechable, configuración aislada, herramientas reales de Pi:
leer código/tests, corregir suma usando `edit`, escribir un test negativo usando
`write` y ejecutar `python3 -m unittest -v` usando `bash`. El verificador comprueba
por separado que el test original no cambió y ejecuta un oráculo externo fijo.

| Política | Pasos de herramientas | Ciclo completo | Resultado |
| --- | ---: | ---: | --- |
| medium, smoke inicial | 6 | 5,42 s | 2 tests pasan, sin errores de tools |
| medium, verificador final | 6 | 4,22 s | 2 tests + oráculo pasan |
| off, verificador final | 5 | 2,77 s | 2 tests + oráculo pasan |

También se cerró Pi y se abrió un proceso nuevo con `--continue`: recuperó la
misma sesión, recordó HAZEL en el segundo turno y registró 256 tokens de cache
leídos. El proveedor global está registrado; no se modificaron credenciales ni
otros proveedores. Las pruebas de coding desactivaron extensiones para aislar
el contrato básico de Pi; no certifican todos los plugins personales.

## Servicio real y recuperación

`scripts/verify_service.py --restart` pasó:

- Primer prompt de 6.173 tokens y continuación de 6.202: prefijo de IDs
  exactamente igual al snapshot anterior, 6.144 tokens físicos reutilizados.
- Cambio de system header: conserva los segmentos generados, pero correctamente
  hace prefill nuevo en lugar de declarar reutilización falsa.
- Trabajo activo rechazó concurrencia; cancelación dejó estado `cancelled`,
  sin snapshot elegible, y una generación posterior respondió RECOVERED.
- Prompt de 270.013 tokens rechazado antes del enqueue con HTTP 400 y código
  `context_length_exceeded`; se reserva salida y scratch dentro de 262.144.
- GET Responses devuelve el objeto guardado; tras parar/iniciar, el snapshot
  permanece idéntico y `previous_response_id` recuerda AMBER.

## HTTP cerca de 256K

Una secuencia fría y su continuación caliente sobre 254.000 tokens de corpus
fuente **no repetido**. Thinking off, greedy, sin tools; tarea caliente:
generar `merge_sorted` y sus tests. Esto mide transporte/latencia, **no** calidad
del código generado ni el coste del thinking medium. La salida no se ejecutó.

| Medición | Frío | Continuación caliente |
| --- | ---: | ---: |
| Input efectivo | 254.044 tokens | 254.098 tokens |
| Tokens cacheados físicamente | 0 | 253.952 |
| Prefill físico nuevo | 254.043 tokens | 145 tokens |
| Primer contenido HTTP | 129,208 s | **0,593 s** |
| Respuesta HTTP completa | 129,255 s | **10,412 s** |
| Output contabilizado | 3 tokens | 950 tokens |
| Decode posterior a primera tanda | no aplicable | **97,19 tokens/s** |

Ambas respuestas terminaron realmente; no hubo requeue. La cinta caliente
comienza con todos los IDs exactos de la cinta fría. Aceptación MTP caliente:
85,35%. La medición HTTP incluye preparación/transporte y el commit terminal;
no es sólo el temporizador del generador.

Es una muestra, no un percentil/SLA ni garantía de 30 s para cualquier tarea.
Thinking, ingesta nueva grande, eviction, reinicio y compaction cambian el coste.
Pi reserva por defecto 16.384 tokens para compaction; su ventana efectiva de
historial puede ser menor que la capacidad nativa. Flash sigue sin certificación
de paridad BF16 o de calidad general del modelo.

## Evidencia y reproducción

- `results/v1-runtime-smoke/`: carga y primer worker real.
- `results/pi-v1/`, `results/pi-v1-final-medium/`, `results/pi-v1-final-off/`:
  eventos JSONL, fixture, tests, oráculo y resumen de Pi.
- `results/pi-v1-session/`: sesión Pi reabierta en proceso nuevo.
- `results/v1-service/evidence.json`: controles de servicio y reinicio real.
- `results/v1-long-context/`: eventos SSE, respuesta y tiempos frío/caliente.

Los resultados y `state/` permanecen ignorados por Git. Los comandos
reproducibles están en el [manual v1](../manual-v1.md). No se creó ningún commit
ni se cambió de rama durante esta implementación.

## Corrección posterior: esquemas de extensiones de Pi

El primer «Hola» del usuario reveló una omisión de compatibilidad: su petición
incluía 16 herramientas, y `mcporter_call` declaraba `args` como un objeto con
`patternProperties: {"^.*$": {}}`. Las pruebas iniciales de Pi sin extensiones
no cubrían ese esquema; el worker lo rechazaba antes de generar.

Se implementó la validación de `patternProperties`, no su descarte: patrones
superpuestos, interacción con `properties`/`additionalProperties`, esquemas
booleanos, expresiones inválidas y parámetros string nativos. Hay 11 regresiones
nuevas, incluida una HTTP; la suite Python pasa con 286 tests. Después de
reiniciar el servicio se reprodujo exactamente la petición fallida con sus 16
esquemas, y terminó correctamente. Evidencia local:
`results/pi-pattern-properties/original-replay.json`.

También pasó Pi real con `compound-engineering-compat.ts` cargada explícitamente:
8 herramientas declaradas, incluida `mcporter_call`, y respuesta HOLA sin ejecutar
tools. El verificador aísla stdin con DEVNULL para no convertir el script de
diagnóstico en parte del prompt. Resultado:
`results/pi-pattern-properties/pi-extension-summary.json`.
