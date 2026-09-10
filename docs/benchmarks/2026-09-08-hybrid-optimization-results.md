# Optimización del híbrido NVFP4 — 2026-09-08

Se evalúan fusiones de MLP, FP8 nativo en las últimas ocho capas, prefill de 16K y el perfil experimental de atención de bloques de 64. El usuario acepta el resultado anterior de 5/6 en código y permite una reducción adicional de calidad a cambio de rendimiento. Se mantienen MTP fijo de seis drafts, caché K8/V4 de 262.144 tokens y reutilización de prefijos.

**Selección:** híbrido NVFP4 + attention64, con las últimas ocho capas MLP en EXL3 y prefill Flash de 8K. Con el margen de calidad autorizado es el candidato para generación larga: conserva capacidad y MTP, obtiene 4/6 entregas de código completas y correctas frente a 5/6 del híbrido anterior y consume menos memoria que la variante con cola FP8. Esta última termina 3/6, con dos truncamientos y mayor tiempo de código largo; su pequeño ahorro de prefill no justifica seleccionarla. No hay un ganador universal para todas las consultas: el prefill frío no muestra una mejora concluyente y varía entre corridas; la reducción de tiempo total depende de cuánto texto se genere.

## Comparación completa

La referencia procede de la ronda anterior y sus archivos se copiaron sin modificar; la procedencia y hashes están en `reference-provenance.json`. Los dos candidatos usan las mismas 31 entradas congeladas: calentamiento, 24 consultas estructuradas y seis entregas de código. GPU0 RTX 5090, concurrencia uno, temperatura 1, top-p 0,95, top-k 20, thinking medium, semilla 42 + repetición. Tres repeticiones JSON por longitud/estado y dos tareas de código por longitud. La muestra de código repite una especificación de caché LRU; no es una tasa general de corrección de programación.

| Perfil | JSON | Código completo y tests correctos | Pico CUDA asignado |
|---|---:|---:|---:|
| Híbrido anterior | 24/24 | 5/6 | 25.25 GiB |
| Híbrido + attention64 | 24/24 | 4/6 | 25.23 GiB |
| Híbrido + attention64 + FP8 final | 24/24 | 3/6 | 26.77 GiB |

### Primer token con contexto frío

| Entrada | Híbrido anterior | + attention64 | + attention64 / FP8 final |
|---:|---:|---:|---:|
| 4,096 | 0.681 s | 0.693 s | 0.682 s |
| 32,768 | 5.878 s | 6.010 s | 5.741 s |
| 131,072 | 36.050 s | 36.664 s | 35.662 s |
| 258,048 | 103.591 s | 104.919 s | 103.295 s |

### Primer token con prefijo reutilizado

| Entrada | Híbrido anterior | + attention64 | + attention64 / FP8 final |
|---:|---:|---:|---:|
| 4,096 | 0.113 s | 0.112 s | 0.108 s |
| 32,768 | 0.126 s | 0.122 s | 0.117 s |
| 131,072 | 0.188 s | 0.188 s | 0.181 s |
| 258,048 | 0.277 s | 0.273 s | 0.268 s |

### Generación después del primer token

| Entrada | Híbrido anterior | + attention64 | + attention64 / FP8 final |
|---:|---:|---:|---:|
| 4,096 | 257.8 tok/s | 254.4 tok/s | 244.5 tok/s |
| 32,768 | 218.6 tok/s | 233.6 tok/s | 221.2 tok/s |
| 131,072 | 159.4 tok/s | 176.1 tok/s | 175.6 tok/s |
| 258,048 | 118.3 tok/s | 133.9 tok/s | 134.2 tok/s |

TTFT incluye el primer token de razonamiento. La velocidad de generación usa `(tokens generados − primera entrega)/(tiempo total − TTFT)`. Frío significa caché de contexto vacía con modelo cargado; no incluye arranque. Los prefijos reutilizados son de 3.840, 32.512, 130.816 y 257.792 tokens, y se comprueba que no haya requeues.

### Tiempo total de código

| Entrada | Híbrido anterior | + attention64 | + attention64 / FP8 final |
|---:|---:|---:|---:|
| 32,768 | 27.49 s † | 25.67 s | 21.74 s |
| 131,072 | 62.33 s | 60.81 s † | 69.74 s † |
| 258,048 | 136.60 s | 129.48 s † | 147.26 s † |

† Incluye al menos una entrega fallida o truncada: es duración observada, no tiempo hasta una solución correcta. Los límites de salida se mantienen en 1.536 tokens para JSON y 4.096 para código. Respuestas, razonamiento y aceptación MTP pueden variar; una diferencia de tiempo total no aísla la velocidad del kernel.

- Híbrido anterior: `coding-32768-1`, estado `rejected`; truncated.
- Híbrido + attention64: `coding-131072-0`, estado `failed`; detalle en el informe de ejecución de tests.
- Híbrido + attention64: `coding-258048-0`, estado `failed`; detalle en el informe de ejecución de tests.
- Híbrido + attention64 + FP8 final: `coding-131072-0`, estado `rejected`; truncated.
- Híbrido + attention64 + FP8 final: `coding-131072-1`, estado `failed`; detalle en el informe de ejecución de tests.
- Híbrido + attention64 + FP8 final: `coding-258048-0`, estado `rejected`; truncated.

En el perfil híbrido + attention64, los fallos `coding-131072-0` y `coding-258048-0` son expectativas erróneas de los tests generados; ambas implementaciones pasan el oráculo funcional independiente. Se mantienen como entregas fallidas porque la tarea también exige tests correctos.

## Ensayos de selección

| Variante | JSON | Primer token a 128K | Primer token a ~256K | Decode a ~256K |
|---|---:|---:|---:|---:|
| NVFP4 con MLP fusionado | 8/8 | 36.788 s | 106.572 s | 124.1 tok/s |
| NVFP4 + últimas ocho capas FP8 | 8/8 | 35.317 s | 102.670 s | 119.3 tok/s |
| Híbrido con prefill de 16K | 8/8 | 35.991 s | 109.641 s | 111.6 tok/s |

Estos screens tienen una repetición por celda y no incluyen código. No se extrapola un 8/8 JSON a calidad de programación. La primera consulta 32K del perfil 16K incluyó compilación de una forma nueva y se excluye de comparaciones de velocidad estable.

## Control contemporáneo

La referencia completa es histórica. Se ejecuta además una repetición nueva del híbrido sin los cambios, con el mismo entorno y lanzador, para comprobar la deriva de rendimiento. No reemplaza las tres repeticiones de la referencia ni permite atribuir diferencias pequeñas sólo al kernel.

| Entrada | Referencia histórica: TTFT / decode | Referencia nueva: TTFT / decode |
|---:|---:|---:|
| 4,096 | 0.681 s / 257.8 tok/s | 0.729 s / 239.0 tok/s |
| 32,768 | 5.878 s / 218.6 tok/s | 6.112 s / 221.4 tok/s |
| 131,072 | 36.050 s / 159.4 tok/s | 37.418 s / 159.2 tok/s |
| 258,048 | 103.591 s / 118.3 tok/s | 106.483 s / 118.2 tok/s |

## Implementación y límites numéricos

- La pareja gate/up comparte escalas exactamente en los 56 MLP NVFP4. Se concatenan pesos y bloques completos de 128 filas sin recuantizarlos. Los 15 casos de esta fusión fueron idénticos al MLP anterior, pero copiar las mitades contiguas para la activación empeoró el prefill grande.
- Fusionar también SiLU, multiplicación y cuantización con FlashInfer 0.6.17 mejoró el microbenchmark del MLP aproximadamente 10–12% en prefill y 4% en pasos pequeños. El error RMS relativo máximo frente al híbrido previo fue 1,264%; la mejora aislada no se trasladó de forma consistente al modelo completo.
- La cola FP8 reemplaza las 24 matrices MLP de las capas 56–63 con tensores FP8 originales y escalas por canal del checkpoint. Las activaciones se cuantizan por token; se usa `torch._scaled_mm`, FP16 de salida y promoción a FP32 donde corresponde. La cuantización, el checkpoint y el redondeo difieren de EXL3.
- Attention64 reutiliza el perfil existente: block_n=64, cuatro warps, una etapa, splits automáticos. Cambia el orden de reducción numérica y se aplica a las rutas capturadas y al fallback durante toda la vida del modelo. No cambia el número de drafts ni la capacidad de caché.

Selección de acumulación FP8: `{"passed": true, "selected_fast_accum": false, "median_speedup": 0.9954153883818797}`. El ensayo numérico previo usa multiplicación FP32 de los operandos cuantizados y pesos originales, no equivalencia con el modelo BF16.

## Verificación y reproducción

Las fusiones se comprueban con pesos reales de las capas 0, 27 y 55, M=1/7/128/2048/8192 y replay de CUDA Graph con entradas cambiantes. La integración del modelo comprueba también los MLP capturados a anchos uno y siete. Las pruebas FP8 cubren gate/down en las cinco formas. Los tiempos micro repiten pesos en caliente; las mediciones completas son las que deciden.

Cada proceso experimental usa GPU0 y el gestor restaura el servicio original al terminar o fallar. Los pesos y los paquetes instalados no se modifican. Las fuentes, flags, resultados, tests y configuraciones previas/posteriores quedan bajo `results/20260908-hybrid-optimization`. La restauración final se audita en `final-audit.json`.

```sh
python3 results/20260908-hybrid-optimization/managed.py NUEVO-managed \
  /home/rekeyea/Documents/llm/qwen38-exl3-mia/.venv/bin/python \
  results/20260908-hybrid-optimization/hybrid_probe.py \
  --output results/20260908-hybrid-optimization/NUEVO \
  --prompts results/20260908-hybrid-optimization/prompts.json \
  --backend adaptive --graph-mlp --decode64
```

El comando reproduce el candidato attention64; agregar `--fp8-tail` para el segundo finalista. Usar nombres de salida nuevos porque los resultados existentes no se sobrescriben. `hybrid_probe.py` admite `--decode64`, `--fp8-tail`, `--fp8-fast-accum`, `--fusion quantized` y `--chunk`; se ejecuta mediante `managed.py` para restaurar el servicio.

Artefactos: [perfil seleccionado](../../results/20260908-hybrid-optimization/selected-profile.json), [comparación](../../results/20260908-hybrid-optimization/comparison.json), [pruebas de fusión](../../results/20260908-hybrid-optimization/fusion-micro2/samples.json), [pruebas FP8](../../results/20260908-hybrid-optimization/fp8-micro/samples.json), [código attention64](../../results/20260908-hybrid-optimization/grade-eval-decode64/summary.json), [código attention64/FP8](../../results/20260908-hybrid-optimization/grade-eval-decode64-fp8tail/summary.json), [procedencia de referencias](../../results/20260908-hybrid-optimization/reference-provenance.json), [notas técnicas](../../results/20260908-hybrid-optimization/experiment-notes.md), [auditoría](../../results/20260908-hybrid-optimization/final-audit.json).

Fuentes: [FlashInfer SiLU/NVFP4](https://docs.flashinfer.ai/generated/flashinfer.quantization.silu_and_mul_nvfp4_quantize.html), [FlashInfer mm_fp4](https://docs.flashinfer.ai/generated/flashinfer.gemm.mm_fp4.html). Se revisó el código instalado de 0.6.17; la documentación web consultada corresponde a 0.6.18. Todas las cifras de este informe proceden de la 5090 local.
