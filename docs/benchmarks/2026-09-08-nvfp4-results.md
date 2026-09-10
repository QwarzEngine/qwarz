# NVFP4 nativo en RTX 5090: comparación con EXL3

**Resultado:** NVFP4 acelera mucho el procesamiento de entradas nuevas en esta prueba. El tiempo hasta el primer token baja de 8,87 a 3,90 segundos con 30.720 tokens. La configuración vLLM evaluada reutiliza peor los prefijos, necesita más memoria y no ofrece una mejora uniforme de generación. Se conserva el servicio actual: EXL3 5 bpw, MTP6, K8/V4, Flash/8192 y contexto de 262.144 tokens.

## Qué se comparó

Experimento local, exclusivamente en la RTX 5090 de 32 GB, con un solo pedido activo. La RTX 3090 Ti no se utiliza. Se comparan configuraciones completas: cambian motor, cuantización de pesos, formato de caché y kernels de atención. Los números no permiten atribuir toda la diferencia únicamente a NVFP4.

| Parámetro | Qwasar actual | Candidato |
| --- | --- | --- |
| Motor | ExLlamaV3 del entorno donante | vLLM 0.27.1 instalado localmente |
| Pesos | Qwen3.8-27B EXL3 5 bpw | Unsloth Qwen3.8-27B NVFP4/FP8 mixto |
| MTP | Seis propuestas fijas | Seis propuestas fijas, cabeza BF16 y embedding/lm_head compartidos |
| Caché de atención | K8/V4 | FP8 E4M3 |
| Contexto configurado | 262.144 | 49.152 |
| Reserva de caché | Pool actual completo | 3 GiB explícitos; capacidad informada de 50.045 tokens |
| Prefill | Torch Flash, chunks de 8192 | FlashInfer, máximo 8192 tokens por paso |
| Grafos | Configuración actual | CUDA Graphs PIECEWISE; modo completo incompatible con esta combinación de MTP y attention |

El checkpoint local utiliza NVFP4 en los MLP de las capas 0–55; atención, proyecciones principales de GDN, últimos ocho MLP y lm_head utilizan FP8. Sus shards ocupan 21,81 GiB frente a 18,53 GiB del artefacto EXL3. vLLM informa 21,26 GiB al cargar los modelos. Por tanto, el nombre «4 bits» no implica que este checkpoint mixto ocupe menos que EXL3 5 bpw.

La revisión de Unsloth es `7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108`. Se verificaron los SHA-256 completos de los dos shards contra Hugging Face. [Procedencia](../../results/20260908-nvfp4/provenance.json).

El runtime resuelve `mamba_cache_mode=align`, páginas de atención de 1632 tokens y estado SSM FP32. Aunque se solicitó calcular escalas KV, la versión instalada lo desactiva para modelos híbridos por calibración recurrente no fiable y utiliza escalas predeterminadas de 1.0. Se conserva la [configuración resuelta](../../results/20260908-nvfp4/eval-nvfp4-3g/resolved-config.json), además de los argumentos solicitados.

## Evidencia de ejecución nativa

vLLM seleccionó `FlashInferCutlassNvFp4LinearKernel`. Una generación separada, instrumentada después de las mediciones, registra 1008 invocaciones de kernels CUTLASS SM120 con operandos `float_e2m1_t`. El binario utilizado contiene instrucciones `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`. Esto confirma cómputo FP4 nativo, además del almacenamiento de pesos en cuatro bits.

[Resumen de la traza](../../results/20260908-nvfp4/native-trace-summary.json), [instrucciones y hash del binario](../../results/20260908-nvfp4/native-instructions.json). La implementación de referencia de NVIDIA documenta las instrucciones Tensor Core con escalas por bloque para [NVFP4 sobre SM120](https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu).

## Protocolo

Se congelaron los IDs de 29 entradas: un calentamiento, 24 consultas estructuradas y cuatro tareas de programación por motor. Son 56 respuestas medidas en total. Las consultas recuperan etiquetas de tres tablas intercaladas en un corpus de código sin repetir y calculan un resultado verificable. Hay tres repeticiones por combinación de contexto y caché. Cada turno con caché sigue a su pareja fría y cambia únicamente los últimos 162 tokens del prompt.

Las tareas de programación piden una caché LRU completa con tests, con dos semillas en contextos de 2048 y 30.720 tokens. Se mantiene thinking medium, temperatura 1, top_p 0,95, top_k 20 y semillas 42–44. Los límites son 1536 tokens para JSON y 4096 para código. Ninguna respuesta de las rondas completas se truncó.

Los archivos de tokenizador difieren en su representación, pero tienen el mismo vocabulario. Se verificaron los mismos IDs, texto decodificado y recodificación en las 29 entradas. Cada pareja medida tiene el mismo SHA-256 de prompt. La semilla común no garantiza iguales muestras entre motores con samplers distintos.

TTFT y tiempo completo usan reloj de pared en el cliente del motor, sin HTTP. Decode se calcula como tokens generados después del primer lote divididos por el tiempo restante. Se excluyen carga, compilación, calentamiento y profiler. Los valores de las tablas son medianas; las variaciones pareadas de decode usan media geométrica de razones. Tres repeticiones describen esta prueba, no una estimación general de producción.

## Tiempo hasta el primer token

| Tokens de entrada | Caché | EXL3 | NVFP4 | Resultado |
| ---: | --- | ---: | ---: | --- |
| 4096 | Fría | 1,155 s | 0,416 s | NVFP4 2,78× más rápido |
| 8192 | Fría | 2,211 s | 0,830 s | NVFP4 2,66× más rápido |
| 30.720 | Fría | 8,866 s | 3,897 s | NVFP4 2,28× más rápido |
| 45.056 | Fría | 13,669 s | 10,421 s | NVFP4 1,31× más rápido |
| 4096 | Reutilizada | 0,153 s | 0,262 s | EXL3 más rápido |
| 8192 | Reutilizada | 0,157 s | 0,852 s | EXL3 más rápido |
| 30.720 | Reutilizada | 0,171 s | 1,025 s | EXL3 más rápido |
| 45.056 | Reutilizada | 0,185 s | 0,551 s | EXL3 más rápido |

En los turnos reutilizados, EXL3 recupera respectivamente 3840, 7936, 30.464 y 44.800 tokens. vLLM recupera 1632, 0, 24.480 y 42.432. La configuración híbrida de caché y sus checkpoints cambian sustancialmente el trabajo efectivo de prefill. En particular, a 8192 tokens vLLM informa cero aciertos de caché pese al prefijo compartido. Estas cifras describen esta configuración; no prueban que toda configuración vLLM tenga el mismo comportamiento.

## Velocidad de generación

| Tokens de entrada | EXL3 frío, tok/s | NVFP4 frío, tok/s | EXL3 reutilizado, tok/s | NVFP4 reutilizado, tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 4096 | 244,3 | 241,1 | 253,4 | 237,1 |
| 8192 | 238,6 | 230,7 | 245,6 | 233,3 |
| 30.720 | 215,6 | 227,5 | 223,7 | 221,1 |
| 45.056 | 206,3 | 217,1 | 210,4 | 219,4 |

Las diferencias pareadas por celda oscilan entre −6,48% y +5,72%. No aparece una mejora uniforme de decode.

En programación, NVFP4 registra 167–177 tok/s con contexto corto frente a 178–192 de EXL3: −7,0% pareado. Con 30.720 tokens, obtiene 176–185 frente a 160–165: +11,0%. El tiempo completo de las dos tareas largas baja de 27,02 / 26,61 segundos a 22,29 / 25,18 segundos. En las cortas sube de 12,66 / 16,21 a 17,05 / 19,55 segundos; también cambian las longitudes de las respuestas.

## Calidad

Ambos motores aciertan **24/24** consultas estructuradas. En programación, ambos obtienen **4/4 implementaciones que pasan los siete controles independientes de LRU**, pero **3/4 entregas completas con todos sus tests propios correctos**. NVFP4 genera un test que intenta borrar una clave ya expulsada. EXL3 genera dos expectativas de recencia/evicción incorrectas dentro de una misma entrega.

El extractor estricto previo rechazó además una respuesta EXL3 por incluir un bloque bash de uso junto al bloque Python. Se preserva ese resultado y se añade una evaluación simétrica de los ocho bloques Python únicos, intactos, permitiendo texto y bloques no Python adicionales. El código revisado se ejecuta dentro de bubblewrap con límites de recursos; no se corrigen respuestas ni tests generados. [Resultados complementarios y código exacto](../../results/20260908-nvfp4/grade-python-blocks/summary.json).

La muestra permite detectar fallos concretos, pero no demuestra equivalencia de calidad entre cuantizaciones.

## Memoria y ensayos fallidos

Dos configuraciones con reserva automática terminaron por falta de memoria durante inferencia: utilización 0,93 y 0,89. La segunda reservó incluso más caché, porque el arranque con compilaciones reutilizadas estimó menos memoria de activaciones. Por eso se fijaron 3 GiB explícitos para la ronda completa.

La ronda de 3 GiB terminó todas las muestras, aunque el allocator registró un intento de asignación de 272 MiB fallido y recuperado durante la primera entrada de 30K. El máximo de las lecturas de memoria del dispositivo después de cada muestra fue 31.239 MiB para NVFP4 y 30.779 MiB para EXL3. Incluyen escritorio y memoria reservada; no son un muestreo continuo del pico. Se necesita más margen para un despliegue estable.

Un primer smoke test completó carga y compilación, pero no generó por un error del script al pasar el resultado de `apply_chat_template` como IDs. Se corrigió codificando el texto renderizado explícitamente. Se conservan todos los logs y las restauraciones, incluidos los intentos fallidos.

El candidato medido queda limitado a 49.152 tokens; no se validó NVFP4 a 262.144. Esto es una limitación de la configuración probada y del checkpoint mixto, no del formato NVFP4 en general.

## Decisión y reproducción

NVFP4 merece continuar como optimización de prefill. Para Qwasar, el siguiente objetivo sería combinar esos kernels con una reutilización de contexto y un presupuesto de memoria comparables a los actuales, y volver a medir antes de migrar. Los resultados no justifican reemplazar hoy el servicio por esta configuración vLLM.

No se modificaron el runtime de producción ni los pesos. Cada experimento se ejecutó con el servicio libre, detención controlada y restauración en `finally`; las configuraciones completas anterior y posterior coinciden. El servicio terminó `ready` con MTP6.

Artefactos: [comparación completa](../../results/20260908-nvfp4/comparison.json), [entradas y protocolo](../../results/20260908-nvfp4/prompt-manifest.json), [entorno](../../results/20260908-nvfp4/environment.json), [ejecutor NVFP4](../../results/20260908-nvfp4/vllm_probe.py), [referencia EXL3](../../results/20260908-nvfp4/exl3_probe.py), [control de servicio](../../results/20260908-nvfp4/managed.py), [análisis](../../results/20260908-nvfp4/analyze.py).

Comando de la ronda NVFP4 completa, utilizando un nombre nuevo para cada directorio de salida:

```sh
python3 results/20260908-nvfp4/managed.py NUEVO-managed \
  /home/rekeyea/.venvs/qwen38/bin/python results/20260908-nvfp4/vllm_probe.py \
  --output results/20260908-nvfp4/NUEVO \
  --prompts results/20260908-nvfp4/prompts.json \
  --max-len 49152 --gpu-util .89 --kv-gib 3 --profile
```

Fuentes externas: [checkpoint de Unsloth](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4), [API FlashInfer mm_fp4](https://docs.flashinfer.ai/generated/flashinfer.gemm.mm_fp4.html), [ejemplo NVFP4 SM120 de NVIDIA](https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu). Las métricas proceden de las ejecuciones locales, no de cifras publicadas por terceros.
