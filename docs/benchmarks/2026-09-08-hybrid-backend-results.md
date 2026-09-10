# NVFP4 híbrido, contexto largo y componentes de SGLang — 2026-09-08

El candidato que conserva la capacidad completa combina el motor actual de Qwasar con NVFP4 sólo en los MLP de las primeras 56 capas. Mantiene MTP6, atención decode original, prefill Flash y caché K8/V4 de 262.144 tokens. Se probaron además dos rutas SGLang para caché NVFP4, caché FP8 y ReplaySSM. El prototipo está aislado en `results/20260908-hybrid-backends/`; el servicio de producción termina con su configuración original.

**Decisión:** conservar el perfil actual como predeterminado. El híbrido mejora mucho el prefill y preserva 256K/MTP6/reutilización, pero todavía registra una entrega de código truncada que la referencia resolvió. Es un candidato experimental para contexto largo; no está validado como reemplazo general.

## Comparación completa contra Qwasar actual

Qwen3.8-27B, entradas idénticas por SHA-256, una RTX 5090 de 32 GB y concurrencia uno. Temperatura 1, top-p 0,95, top-k 20, thinking medium y semilla 42 + repetición; salida máxima de 1.536 tokens para JSON y 4.096 para código. SGLang usa rechazo especulativo estándar en estos perfiles. Tres repeticiones por combinación de longitud y estado de caché; una ejecución inicial de calentamiento excluida. Los tiempos incluyen entrega del primer token de razonamiento. “Frío” vacía el contexto, pero conserva el modelo cargado y las compilaciones.

| Tokens de entrada | Primer token actual | Primer token híbrido | Reducción | Decode actual | Decode híbrido |
|---:|---:|---:|---:|---:|---:|
| 4,096 | 1.156 s | 0.681 s | 41.1% | 245.7 tok/s | 257.8 tok/s |
| 32,768 | 9.527 s | 5.878 s | 38.3% | 218.3 tok/s | 218.6 tok/s |
| 131,072 | 50.775 s | 36.050 s | 29.0% | 164.1 tok/s | 159.4 tok/s |
| 258,048 | 132.388 s | 103.591 s | 21.8% | 121.0 tok/s | 118.3 tok/s |

| Tokens de entrada | Primer token con prefijo actual | Híbrido | Tokens reutilizados en ambos |
|---:|---:|---:|---:|
| 4,096 | 153.5 ms | 112.6 ms | 3,840 |
| 32,768 | 177.6 ms | 125.7 ms | 32,512 |
| 131,072 | 241.9 ms | 188.1 ms | 130,816 |
| 258,048 | 330.6 ms | 277.2 ms | 257,792 |

Estas consultas reutilizadas cambian un sufijo corto. No son una evaluación de múltiples sesiones concurrentes ni de recuperación después de reiniciar el proceso. La capacidad de 256K se verifica ejecutando entradas de 258.048 tokens, generación posterior y otra consulta que reutiliza 257.792 tokens, no sólo leyendo un parámetro de configuración.

## Código, calidad y tiempo total

Consultas estructuradas: actual **24/24**, híbrido **24/24**. Código: actual **6/6**, híbrido **5/6** entregas completas que pasan los siete controles independientes y al menos seis tests generados. No se corrige código ni se cambian los tests producidos por el modelo.

| Contexto de código | Tiempo total actual | Tiempo total híbrido | Tokens generados actual / híbrido | Decode actual / híbrido |
|---:|---:|---:|---:|---:|
| 32,768 | 28.27 s | 27.49 s † | 2980 / 3998 | 159.3 / 185.5 tok/s |
| 131,072 | 71.74 s | 62.33 s | 2544 / 2932 | 121.4 / 112.8 tok/s |
| 258,048 | 162.20 s | 136.60 s | 2706 / 3117 | 91.4 / 94.3 tok/s |

Las celdas † incluyen una entrega fallida o truncada: indican duración observada, no tiempo hasta una solución correcta. Son medianas de dos tareas por longitud. Las respuestas y su cantidad de razonamiento cambian con la cuantización: un menor tiempo de prefill no garantiza la misma reducción en el tiempo de respuesta completa. El límite de salida es de 4.096 tokens para ambos. La muestra detecta fallos concretos; no demuestra equivalencia general de calidad.

- Híbrido: `coding-32768-1` — `rejected`, truncated.

Pico de memoria CUDA asignada en la matriz principal: **26.09 → 25.25 GiB**. Son mediciones del allocator, no toda la memoria de la GPU. El híbrido gana margen, pero el ahorro observado es de unos 0,84 GiB.

## Combinaciones SGLang y caché

Los ensayos SGLang usan el checkpoint mixto NVFP4/FP8 completo, MTP6, estado SSM FP32, FlashInfer 0.6.17 y fuentes copiadas de SGLang 0.5.18. Los perfiles de 144K permiten entradas de 128K más salida; no se presentan como sustitutos de la capacidad de 256K.

| Perfil | Capacidad configurada | Primer token a 128K | Con prefijo a 128K | Decode a 128K | JSON |
|---|---:|---:|---:|---:|---:|
| SGLang FP8 KV + ReplaySSM | 147.456 | 34.845 s | 938 ms | 143.0 tok/s | 6/6 |
| SGLang NVFP4 KV + workspace + ReplaySSM | 147.456 | 36.072 s | 974 ms | 118.9 tok/s | 6/6 |
| SGLang NVFP4 KV + XQA directo + ReplaySSM | 147.456 | Ensayo no completado | — | — | — |
| Híbrido K8/V8, Flash/4096 | 262.144 | 36.766 s | 191 ms | 149.6 tok/s | 8/8 |

| Perfil adicional | Código completo y tests correctos |
|---|---:|
| SGLang FP8 KV | 1/1 |
| SGLang NVFP4 KV + workspace | 1/1 |
| Híbrido K8/V8 | 3/6 |

En K8/V8, los fallos detectados en `coding-32768-0` y `coding-131072-0` son expectativas incorrectas en tests generados. `coding-258048-0` anida sus tests bajo `__main__`: no se descubren al importar, pero una ejecución suplementaria del script sin modificar corrió sus 15 tests y confirmó un error por expectativas contradictorias para claves ausentes. No se penaliza únicamente su organización. El detalle queda en `grade-kv88/coding-258048-0-main.json`.

Estos perfiles son ensayos de selección con una repetición por celda; la comparación principal usa tres. Revisar sus resultados de código y los logs de fallos junto a esta tabla. SGLang reutilizó 129.024 tokens a 128K, frente a 130.816 en Qwasar: sus distintas granularidades de caché son parte del coste observado, no una comparación aislada de kernels. K8/V8 se ensayó junto con chunks de 4.096 para ajustar memoria; no se puede atribuir toda la diferencia exclusivamente a los bits de caché.

La ruta SGLang de workspace solicitando 262.144 tokens redujo el pool a **224.768** y sufrió OOM en la primera generación. A 147.456 tokens completó las muestras, pero el allocator registró fallos recuperados al asignar buffers durante prefill largo. FP8 evita esa conversión adicional en cada verificación; el perfil XQA directo intenta leer los bloques NVFP4 sin convertir todo el prefijo. **La adaptación local de XQA falló:** 0/2 consultas JSON cortas válidas, texto repetitivo y un acceso ilegal de memoria CUDA al comenzar 32K. No llegó a validar 128K. Los tests aislados de máscara causal y replay habían pasado, pero no garantizan una integración correcta con el motor. La causa raíz no está aislada; esto no demuestra que el PR original o XQA en general fallen.

## Qué se reutilizó y qué queda como candidato

- **GEMM NVFP4:** se reutilizó la convención de escalas y disposición de bloques de SGLang y los kernels compartidos de FlashInfer, manteniendo el runtime de Qwasar. B12X se usa para M ≤ 128; CUTLASS para lotes de prefill mayores.
- **Memoria de MTP en SGLang:** adelantar la operación existente que comparte embeddings y cabeza de salida liberó **3,554 GiB** antes de presupuestar la caché. No cambia los cálculos del modelo.
- **ReplaySSM:** se probó dentro de SGLang para reducir estados intermedios de verificación. Portarlo a Qwasar sigue siendo trabajo separado: debe conservar rollback de tokens rechazados, historial de convolución y snapshots de prefijos.
- **XQA con KV NVFP4:** el prototipo SGLang adapta la ruta nativa para batch uno y cadena MTP de seis drafts. Falló la integración local con el motor y queda descartado de los perfiles utilizables. El parche original depende de APIs más nuevas que las de SGLang instalado; la adaptación parcial necesita depuración antes de otra evaluación.

La capa de salida, atención/GDN, últimas ocho capas y MTP del híbrido proceden del artefacto EXL3; sólo se reemplazan 168 matrices MLP por los tensores originales del checkpoint NVFP4 verificado. La proyección nativa entrega FP16 y se promueve a FP32 donde lo pide el runtime: es un cambio numérico real, incluido en la evaluación de calidad.

## Próximas evaluaciones justificadas

- Priorizar una implementación aislada de ReplaySSM en Qwasar, con pruebas de aceptación/rechazo MTP, rewind y prefijos antes de medir velocidad. El resultado en SGLang no establece por sí solo cuánto ganaría Qwasar.
- Para el híbrido, medir sensibilidad por grupo de capas y ampliar tareas de código antes de seleccionar cuántos MLP pasar a NVFP4. No basta aumentar el límite de salida para ocultar la entrega truncada.
- Reintentar XQA sobre una revisión completa y compatible de SGLang que contenga la integración upstream, después de aislar el fallo local; el prototipo parcial actual no es una base válida de comparación.

## Verificación y reproducción

Los 16 casos de matrices reales pasaron contra una multiplicación FP32 de los operandos NVFP4 desempaquetados. Las 112 comprobaciones de MLP con entradas cambiantes, anchos uno y siete, fueron idénticas con y sin CUDA Graph. Esto valida el adaptador y la captura; no mide por sí solo la pérdida de cuantización frente al modelo original. El binario CUTLASS compilado contiene instrucciones `OMMA.SF…E2M1.E2M1` de SM120.

Cada ejecución gestionada detiene únicamente el servicio de Qwasar libre, usa GPU0 y restaura la configuración completa en `finally`. GPU1 no participa. La restauración final se comprueba en `final-audit.json`. No se modifican los pesos ni los módulos instalados del runtime; las modificaciones SGLang están en overlays locales.

```sh
python3 results/20260908-hybrid-backends/managed.py NUEVO-managed \
  /home/rekeyea/Documents/llm/qwen38-exl3-mia/.venv/bin/python \
  results/20260908-hybrid-backends/hybrid_probe.py \
  --output results/20260908-hybrid-backends/NUEVO \
  --prompts results/20260908-hybrid-backends/prompts.json \
  --backend adaptive --graph-mlp --validate-graphs
```

Artefactos: [comparación](../../results/20260908-hybrid-backends/comparison.json), [entradas](../../results/20260908-hybrid-backends/prompt-manifest.json), [código híbrido](../../results/20260908-hybrid-backends/grade-hybrid/summary.json), [código de referencia](../../results/20260908-hybrid-backends/grade-exl3/summary.json), [código K8/V8](../../results/20260908-hybrid-backends/grade-kv88/summary.json), [auditoría final](../../results/20260908-hybrid-backends/final-audit.json), [entorno](../../results/20260908-hybrid-backends/environment.json), [instrucciones nativas](../../results/20260908-hybrid-backends/hybrid-native-instructions.json), [detalle técnico y fuentes](../../results/20260908-hybrid-backends/research-notes.md).

Fuentes primarias: [FlashInfer mm_fp4](https://docs.flashinfer.ai/generated/flashinfer.gemm.mm_fp4.html), [SGLang ReplaySSM](https://github.com/sgl-project/sglang/issues/28511), [PR de workspace especulativo NVFP4](https://github.com/sgl-project/sglang/pull/36045), [PR de XQA nativo para MTP](https://github.com/sgl-project/sglang/pull/36038). Ambos PR seguían abiertos al revisarlos. Todas las métricas de esta página proceden de ejecuciones locales.
