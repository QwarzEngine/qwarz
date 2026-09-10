# Fidelidad de caché, tamaño del turno y perfil

Fecha: 2026-09-04. Configuración seleccionada: EXL3 5 bpw + MTP, K8/V4, RTX 5090. La 3090 Ti mantiene su servicio SGLang y no participa.

## Controles

La primera etapa reconstruye el historial archivado de la primera llamada de herramienta y su resultado. Verifica por hash el prompt inicial de 257,008 tokens y el prompt de respuesta de 257,271. Una pasada prima el historial original; luego compara ese prompt de respuesta en caliente y tras reemplazar tabla de páginas y caché de checkpoints recurrentes. Los contadores físicos deben confirmar reutilización en caliente y cero reutilización en frío.

Ambas respuestas usan greedy, mismo límite y semilla. Se guardan los tokens generados y los logits del primer token sobre el vocabulario completo. Las entradas de padding enmascaradas con `-inf` coincidentes se excluyen de las diferencias; NaN, infinito positivo o máscaras distintas son errores. Las diferencias numéricas se interpretan junto con margen top-1/top-2 y divergencia de la secuencia, no con una regla arbitraria de igualdad bit a bit. Capturar logits altera el coste del diagnóstico: no se presenta como benchmark normal de velocidad.

### Resultado del control de fidelidad

Artefactos: `results/20260904-cache-fidelity-5bpw-mtp/`. Ambos caminos reciben exactamente el prompt `ab51961e8911f2dd948a6df1365606faf655920e97cddec3c10ac2b5ae4ff04f` y generan los mismos **184 tokens**, incluidos terminadores. Ambos calculan 408 correctamente en el razonamiento y luego repiten incorrectamente `read_file`: calidad final 0/2.

| Camino | Tokens reutilizados | Prefill físico | TTFT | Respuesta completa |
| --- | ---: | ---: | ---: | ---: |
| Caliente | 257,024 | 246 | 0.542 s | 2.455 s |
| Replay frío | 0 | 257,270 | 205.266 s | 207.163 s |

No hubo requeues. Sobre 248,077 logits finitos, diferencia absoluta máxima 0.143555 y media 0.022431; 243 posiciones coinciden en `-inf`. Top-1 idéntico, margen frente al segundo 4.640625 en ambos. El doble de la perturbación máxima representa sólo 6.19% de ese margen. No hay divergencia greedy: este caso no respalda la hipótesis de caché corrupta. Junto con el control anterior sin MTP, orienta la investigación del fallo hacia comportamiento del modelo y contrato/template de herramientas; no distingue todavía entre esos dos factores ni certifica toda la caché.

Incidencia del diagnóstico: el proceso original terminó con código 1 al comparar logits porque su versión ya cargada rechazaba también el padding legítimo `-inf`. Las dos generaciones y vectores ya estaban guardados. Se corrigió el comparador, se agregaron regresiones y se recalculó **sin regenerar ni modificar los artefactos originales**. `fidelity-postprocessed-summary.json` documenta la recuperación y valida hashes, contadores y tokens emitidos/retenidos. No se creó un `completed.json` ficticio.

## Matriz de nuevos tokens

Doce celdas: tres tamaños nominales de historial por deltas de 128, 512, 2,048 y 8,192 tokens, cinco ramas distintas por celda. Cada rama conserva el historial semilla real, pero no incorpora las respuestas de otras ramas. Es un control reproducible del coste de turno, no 60 sesiones independientes de agente.

El pool siempre tiene 262,144 posiciones. Los prefijos nominales 32K/128K dejan crecer el prompt con el delta; el prefijo mayor se reduce a un máximo de 252,400 tokens antes de descontar el margen para la respuesta de primado. Se reserva salida de 1,536 tokens y scratch de 16. Cada sample registra las longitudes efectivas; ninguna llamada puede exceder el límite nativo.

El contexto combina un snapshot de código sin repetición artificial y contratos/factores de configuración a distintas distancias. El delta añade código fuente no incluido en el prefijo y un pedido corto con nuevos valores. Se exige JSON exacto con tags y resultado calculado. Los bytes de código son carga de prefill; este examen no certifica comprensión de todas las líneas ni edición multiarchivo. No se declaran disponibles herramientas que la tarea no necesita.

El identificador de la rama cambia al comienzo del delta, no sólo al final, para impedir que la segunda muestra reutilice el cuerpo nuevo de la primera. La espera total medida incluye preparación local y entrega completa, pero excluye carga inicial, tokenización global del corpus y transporte HTTP. El primado se registra aparte. Las respuestas inválidas permanecen tanto en los percentiles de espera general como en el denominador de calidad; se informa por separado el p95 entre respuestas válidas y físicamente calientes. Para calificar caliente se exige contador válido y al menos historial menos intervalo de checkpoint menos una página (256 tokens) reutilizados; una respuesta correcta reprocesada en frío no cuenta como éxito caliente. Cinco muestras no bastan para certificar un SLA.

### Resultados de los 60 turnos

Artefactos: `results/20260904-turn-matrix-5bpw-mtp/`, resumen `matrix-summary.json` y auditoría independiente de métricas `matrix-audit.json`. **60/60 respuestas exactas, 60/60 reutilizaciones calientes verificadas, cero requeues o truncamientos.** Todas las celdas quedan por debajo de 30 s en sus cinco observaciones. La respuesta más lenta tarda **14.583 s**.

Tiempos en segundos. TTFT incluye el primer delta de razonamiento; primer contenido mide el comienzo de la respuesta final después de thinking. Completa incluye preparación local.

| Historial nominal | Tokens nuevos | TTFT p50 | Primer contenido p50 | Completa p50 | Completa p95 | Calidad |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32K | 128 | 0.232 | 1.625 | 2.034 | 2.073 | 5/5 |
| 32K | 512 | 0.456 | 1.749 | 2.108 | 2.216 | 5/5 |
| 32K | 2,048 | 1.022 | 2.317 | 2.780 | 2.843 | 5/5 |
| 32K | 8,192 | 3.575 | 5.058 | 5.454 | 5.626 | 5/5 |
| 128K | 128 | 0.344 | 2.497 | 2.991 | 3.233 | 5/5 |
| 128K | 512 | 0.780 | 2.604 | 3.176 | 3.518 | 5/5 |
| 128K | 2,048 | 1.935 | 3.923 | 4.388 | 4.477 | 5/5 |
| 128K | 8,192 | 6.869 | 8.958 | 9.436 | 9.467 | 5/5 |
| Casi 256K | 128 | 0.518 | 3.014 | 3.682 | 3.790 | 5/5 |
| Casi 256K | 512 | 1.051 | 3.639 | 4.252 | 4.548 | 5/5 |
| Casi 256K | 2,048 | 2.967 | 5.584 | 6.237 | 6.605 | 5/5 |
| Casi 256K | 8,192 | 10.872 | 13.458 | 14.131 | 14.511 | 5/5 |

Los historiales efectivos son 32,642, 130,946 y 252,274 tokens. El grupo mayor recibe de 252,402 a **260,466 tokens de entrada**, no 262,144 más salida: se mantiene la reserva nativa. Reutiliza siempre 252,160 posiciones y procesa físicamente 241/625/2,161/8,305 tokens según el delta; la pequeña diferencia corresponde al borde de página/checkpoint.

En los 20 turnos del grupo mayor, decode observado **96.65–103.97 tokens/s**, mediana **99.92**. Son tokens de razonamiento y respuesta combinados, excluido el primer batch y prefill; no tokens visibles por segundo extremo a extremo. Las salidas tienen 250–382 tokens en toda la matriz. El pico PyTorch asignado es 24.511 GiB y reservado 25.670 GiB; no incluye toda la memoria del proceso/driver.

La auditoría reconstruye los 60 prompts a partir de historial y delta, verifica hashes únicos, largo exacto, reserva de contexto, tokens emitidos/retenidos, contadores físicos, hashes del harness/corpus y JSON contra el resultado esperado sin reparar ninguna salida. El resultado favorable **no** convierte esta tarea corta en garantía para ediciones largas ni resuelve el fallo del ciclo real con herramientas.

## Perfil

La selección es la respuesta válida no truncada más lenta de la matriz. Antes de instrumentarla se descartan las cachés y se prima únicamente su historial semilla: conservar el delta seleccionado en caché invalidaría el perfil de su prefill. Se registra de nuevo el prefill físico para comprobar la comparabilidad.

No hay Nsight Systems/Compute instalados; se usa PyTorch Profiler CPU/CUDA. Rangos: asignación/restauración, prefill, checkpoints, draft MTP, verificación del target y mantenimiento al vaciar la cola. Las trazas incluyen kernels reales y operadores; los tiempos inclusivos solapados no se suman como un desglose de pared. Se conservan cuenta de lanzamientos, duración acumulada y unión temporal de kernels. El perfil queda fuera de todos los percentiles normales y declara explícitamente si no se capturó actividad CUDA.

### Resultado del perfil

Caso seleccionado: `turn-262144-8192-0`, 260,466 tokens de entrada. El primado de control confirma cero reutilización; la repetición instrumentada reutiliza 252,160 tokens y procesa **los mismos 8,305 tokens físicos** que el original. Coinciden hash de prompt, los 360 tokens generados y resultado exacto. El conteo de propuestas aceptadas cambia ligeramente; no cambia la respuesta. Captura CUDA válida, seis métodos anotados, sin rangos faltantes.

Respuesta instrumentada: 14.700 s frente a 14.560 s del sample original sin preparación; prefill reportado por runtime 10.801 s. Los kernels suman **14.372 s**, con unión temporal de 14.372 s, 103,516 ejecuciones y 71 nombres distintos. No interpretar esa unión como utilización de SM ni sumar rangos CPU/GPU inclusivos. Archivo de traza: `profile/chrome_trace.json` (223.5 MB); auditoría: `profile-audit.json`.

| Familia | Duración acumulada de kernels | Fracción del total de kernels |
| --- | ---: | ---: |
| `_paged_attn_prefill_kernel` | 8.420 s | 58.59% |
| `_paged_attn_decode_split_kernel` | 2.012 s | 14.00% |
| `_paged_attn_decode_combine_kernel` | 0.032 s | 0.22% |
| Toda la atención paginada, incluidas variantes menores | 10.469 s | **72.84%** |
| Dos variantes principales de GEMM CUTLASS FP16 | 1.768 s | 12.30% |

La fila de toda la atención contiene las anteriores, no se suma a ellas. La asignación/restauración aparece en un rango CPU inclusivo de 15.2 ms; checkpoint, 143.9 ms; draft MTP, 562.3 ms; verificación, 3,312.1 ms. Son rangos con trabajo asíncrono y esperas, no un desglose aditivo de tiempo real.

Advertencia importante del perfil: `cudaMemcpyAsync` acumula 8.15 s de tiempo CPU y aparecen 5.38 s de eventos `Command Buffer Full`, pero las copias/memset reales en GPU suman sólo **58.6 ms**. No son 8 s de transferencia PCIe: no confundir espera/backpressure en la API con tiempo del motor de copia. La traza no permite afirmar por sí sola si la atención está limitada por ancho de banda, cómputo u ocupación; faltan contadores de hardware.

## Qué cambia en la decisión

1. **Mantener EXL3 5 bpw + MTP y K8/V4.** Esta matriz supera 50 tokens/s cerca del límite sin cambiar cuantización ni atención exacta. Cambiar a NVFP4 de pesos no ataca directamente el 72.84% identificado en atención y abriría otra evaluación de calidad.
2. **Primer objetivo de optimización: atención de prefill con prefijo largo.** Aislar `paged_attn_triton_prefill` del runtime instalado en `modules/attention_fn/triton_paged.py`, con las formas reales observadas y caché K8/V4. Medir bloques M/N, warps, stages y partición KV que ya admite la implementación; luego comparar un kernel especializado. Mantener la máscara/softmax exactos y validar salidas/logits y la matriz completa antes de adoptar un candidato. No prometer un porcentaje de mejora antes de medir.
3. **Después, atención de verificación especulativa.** `_paged_attn_decode_split_kernel` cuesta otros 2.012 s. Comparar particionado y reparto entre cabezas para los tamaños reales de MTP. Un megakernel global o quitar Python no es la primera intervención respaldada por esta traza: fusionar lanzamientos no elimina el trabajo de atención dominante.
4. **Separar corrección de herramientas de rendimiento.** Hacer A/B del mismo ciclo `read_file` con template nativo, serialización de resultados y política de herramientas controlados. La repetición incorrecta persiste con replay frío y en el control anterior sin MTP; no justifica tocar la caché. El control sin tools de esta matriz pasa, pero cambia también la tarea/prompt y no demuestra por sí solo un defecto del template.

### Límite honesto de los 30 segundos

La meta se cumple **en estos 60 turnos calientes cortos**, incluidos 8K nuevos sobre casi 256K. No equivale a cumplirla con un prompt frío de 256K (el control tarda 207 s), salida de miles de tokens, cadenas de varias herramientas, reinicios o edición multiarchivo. Para producción hace falta medir ciclos completos de coding y presupuesto de thinking/salida sin dar por válida una respuesta truncada.

La caché evita reprocesar todo el historial, pero la atención exacta sigue consultándolo: con 128 nuevos, TTFT mediana sube de 0.232 a 0.344 y 0.518 s; con 8K, de 3.575 a 6.869 y 10.872 s. No hemos logrado TTFT constante ni el objetivo secundario de 300 ms cerca del límite. La sesión persistente es necesaria; optimizar atención es el siguiente paso para reducir esa dependencia.

## Verificación final

- 104 tests CPU pasan; revisión independiente del comparador, perfil y clasificación física de caché completada.
- `bash -n scripts/run_resident_probe.sh` pasa. Auditorías reconstruyen prompts, calidad, contadores, tokens y traza; el perfil no entra en los 60 samples.
- La matriz termina con código 0 y `completed.json`; el control de fidelidad conserva su recuperación explícita, no un marcador falso de éxito.
- GPU 0 liberada al terminar. El proceso SGLang 3547 de la 3090 Ti sigue activo. No se modificaron el donante ni sus dependencias, ni se crearon commits o ramas.
