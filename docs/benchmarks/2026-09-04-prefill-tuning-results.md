# Optimización de prefill exacto: resultados

Configuración: Qwen3.8-27B EXL3 5 bpw + MTP, K8/V4, RTX 5090. Sin cambios del donante, cuantización, atención causal completa ni GPU 1.

## Captura y controles

`results/20260904-prefill-capture/attention.pt` captura una atención real del prompt archivado `d6600c0384cbe67d1aca0c2791245b57af2ce1e1ab028c189c8701ec7e570424`. Q tiene forma `[1, 2048, 24, 256]`; cuatro cabezas KV, caché paginada K8/V4. El modelo reutiliza 252,160 tokens y realiza 8,305 de prefill físico para el turno completo. Respuesta exacta, sin requeue. La captura no es una medida limpia de latencia: incluye copias a CPU y escritura.

El microbenchmark reproduce esos tensores sin cargar los pesos. No vuelve a añadir K/V ni modifica las páginas. Descarta el `out` capturado y clona la referencia para impedir alias. Separa warmup/compilación de CUDA Events; intercala baseline después de cada candidato, aleatoriza orden y conserva errores de compilación/memoria y fallos numéricos. Las métricas incluyen staging, no sólo el kernel final. La tolerancia previa al barrido es `atol=0.01, rtol=0.01`; no implica paridad de calidad del modelo completo.

## Barridos de Triton

Coarse: 67 configuraciones; refine: 44 configuraciones adicionales de barrido, con algunas superpuestas a coarse. Se exploran bloques M/N, 4/8 warps, stages, splits y staging frente a lectura cuantizada directa. Cada candidato válido tiene dos rondas de tres mediciones; el baseline se intercala durante ambas. Se conservan las configuraciones que exceden recursos como fallidas, no como medidas de tiempo cero. En modo directo el backend impone N=64 para K8/V4 y dimensión 256; por eso no se acepta un override que sería ignorado.

El mejor tiempo del refinamiento es 74.037 ms con M64/N32, cuatro warps, dos stages y 32 splits, frente a 120.080 ms del baseline. Ocho splits tarda 74.516 ms, pero reduce el pico aislado de 3.012 a 1.878 GiB: una diferencia temporal pequeña no justifica automáticamente más de 1 GiB adicional. Se mantienen ambos como finalistas.

## Alternativa Flash ya instalada

Se añade un adaptador opt-in que descomprime las mismas páginas K8/V4 al mismo scratch reutilizable del donante, pero usa Flash Attention nativa de PyTorch. Conserva escala, agrupación GQA y causalidad **lower-right** para consultas cortas sobre un prefijo largo. No se instala `flash-attn`, no se usa `is_causal=True` indiscriminadamente en SDPA pública, ni se permite fallback a atención densa/math. Las geometrías no soportadas se rechazan.

Verificación formal: tres rondas de cinco mediciones por candidato y baseline intercalado. Para consultas menores se usa el último subconjunto del Q capturado, conservando el KV y su longitud: es un control de forma/aritmética, no otra sesión independiente.

| Consultas Q | Baseline | Triton 8 splits | Triton 32 splits | Flash nativa |
| ---: | ---: | ---: | ---: | ---: |
| 2,048 | 119.702 ms | 74.396 ms | 73.769 ms | 62.542 ms |
| 512 | 29.083 ms | 22.207 ms | 19.720 ms | 16.119 ms |
| 256 | 15.509 ms | 11.817 ms | 10.593 ms | 8.265 ms |
| 241 | 19.649 ms | 20.249 ms | 19.112 ms | 8.265 ms |
| 113 | 10.957 ms | 12.288 ms | 10.032 ms | 4.710 ms |

Son medianas aisladas sobre KV largo; no se trasladan directamente a velocidad de todo el modelo. Todos estos finalistas pasan el control numérico. En el piloto Flash Q2048, diferencia absoluta máxima 0.005859, RMSE 0.000349 y L2 relativa 0.001344 frente a Triton. Diferencia de reducción numérica, no una modificación de la cuantización.

Artefactos: `results/20260904-prefill-coarse`, `results/20260904-prefill-refine` y `results/20260904-prefill-finalists-q{2048,512,256,241,113}`. Cada ejecución guarda candidatos, hashes de captura y código, referencia de entorno, rondas y fallos. La recomendación final exige además el screen de modelo completo y su matriz de calidad.

## Validación de modelo completo

El screen compara baseline, Triton de ocho splits y Flash con chunks 2K/4K/8K sobre el mismo historial archivado. Cada delta cambia su identificador temprano, para no reutilizar todo el delta anterior. Un warmup por candidato y delta queda fuera de los percentiles; tres repeticiones posteriores se ordenan aleatoriamente. Se comprueban calidad, reutilización física y presupuesto nativo, y se conserva por separado la finalización de colección y la calificación.

La implementación Flash sincroniza para leer la longitud efectiva de KV (`cache_seqlens.item()`). Ese coste está incluido en el screen; la mejora no se calcula suponiendo que esa sincronización es gratis.

### Resultado del screen

**80/80 respuestas exactas y calientes**: 20 warmups excluidos y 60 turnos medidos. Medianas de TTFT en segundos, tres repeticiones por celda:

| Ruta | +128 | +512 | +2K | +8K |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 0.518 | 1.050 | 2.968 | 10.872 |
| Triton ajustado, 8 splits | 0.518 | 0.932 | 2.222 | 7.859 |
| Flash, chunk 2K | 0.324 | 0.730 | 1.908 | 6.892 |
| Flash, chunk 4K | 0.323 | 0.731 | 1.906 | 6.776 |
| Flash, chunk 8K | 0.323 | 0.732 | 1.908 | **6.532** |

Se selecciona Flash con chunk 8K por prefill, no por cuál respuesta casualmente generó menos tokens. En +8K, la respuesta completa mediana pasa de 14.225 a 9.882 s. El pico asignado del candidato es 25.799 GiB, frente a 24.505 GiB del baseline en ese screen. La memoria reservada comparte el historial del allocator entre variantes y no se interpreta como memoria exclusiva de cada candidato.

El perfil opt-in queda en `benchmarks/profiles/prefill-flash-k8v4.json`. Usa Flash para Q >= 17 y conserva la ruta previa para consultas menores, sin modificar decode/MTP. No se cambia el backend instalado ni el comportamiento por defecto del launcher.

## Comprobación independiente FP32

`results/20260904-prefill-fp32-oracle.json` calcula softmax y productos FP32 con TF32 desactivado sobre el **mismo KV descomprimido K8/V4**. Muestrea 120 vectores: cinco posiciones Q por 24 cabezas, con agrupación GQA y frontera causal verificadas en tests CPU contra doble precisión.

| Implementación | Error absoluto máximo | RMSE | Error L2 relativo |
| --- | ---: | ---: | ---: |
| Triton baseline | 0.001693 | 0.000115 | 0.0449% |
| Flash nativa | 0.005113 | 0.000400 | 0.1561% |

**Flash no es bit a bit equivalente y su error en este muestreo es mayor.** No se presenta como una mejora de precisión. Ambos usan atención causal completa y el mismo formato; el screen de recuperación pasa, pero no prueba paridad con el modelo BF16 ni garantiza todo el coding. Se mantiene Triton disponible como control y se exige la matriz completa antes de recomendar esta ruta experimental.

## Matriz final y decisión

La validación continúa el 2026-09-05 en `results/20260905-prefill-flash-matrix/`. **60/60 respuestas exactas, 60/60 ejecuciones calientes, cero requeues y truncamientos.** Los 60 prompts coinciden por hash con la matriz original, y todos tienen exactamente el mismo prefill físico y reutilización. Hay 34/60 secuencias generadas idénticas; las otras cambian su razonamiento o redacción, aunque el JSON final sigue siendo correcto.

Tiempos en segundos. TTFT es mediana; respuesta completa es p95 descriptivo de cinco muestras e incluye preparación local.

| Contexto nominal | Nuevos | TTFT baseline → Flash | Reducción TTFT | Completa p95 baseline → Flash |
| --- | ---: | ---: | ---: | ---: |
| 32K | 128 | 0.232 → 0.224 | 3.4% | 2.073 → 2.098 |
| 32K | 512 | 0.456 → 0.414 | 9.2% | 2.216 → 2.276 |
| 32K | 2,048 | 1.022 → 0.886 | 13.3% | 2.843 → 2.730 |
| 32K | 8,192 | 3.575 → 2.890 | 19.2% | 5.626 → 4.870 |
| 128K | 128 | 0.344 → 0.294 | 14.5% | 3.233 → 3.180 |
| 128K | 512 | 0.780 → 0.574 | 26.4% | 3.518 → 3.432 |
| 128K | 2,048 | 1.935 → 1.367 | 29.3% | 4.477 → 3.911 |
| 128K | 8,192 | 6.869 → 4.555 | 33.7% | 9.467 → 7.150 |
| Casi 256K | 128 | 0.518 → 0.322 | 37.7% | 3.790 → 4.011 |
| Casi 256K | 512 | 1.051 → 0.732 | 30.4% | 4.548 → 4.211 |
| Casi 256K | 2,048 | 2.967 → 1.910 | 35.6% | 6.605 → 5.621 |
| Casi 256K | 8,192 | 10.872 → 6.540 | 39.9% | 14.511 → 10.161 |

El caso mayor conserva 252,274 tokens de historial y llega a 260,466 de entrada, reservando salida y scratch dentro de 262,144. La espera máxima observada es **10.252 s**. Decode permanece entre **96.34 y 103.18 tokens/s** en el grupo largo. Pico asignado: **25.805 GiB**; reservado: **26.986 GiB**, sin contar todo el driver/proceso.

No todas las respuestas completas mejoran: en +128 cerca del límite el p95 pasa de 3.790 a 4.011 s porque también cambia la generación. Esto no invalida la mejora de prefill, pero impide prometer una reducción uniforme de espera total o paridad bit a bit. El objetivo secundario de TTFT <300 ms cerca del límite todavía no se cumple: mediana 322 ms.

### Perfil de confirmación

Se repite el peor caso válido tras reset y primado frío. Misma entrada, mismo prefill físico de 8,305, calidad correcta y captura CUDA verificada; fuera de todos los percentiles. Prefill del runtime: **6.469 s**, frente a 10.801 s del perfil anterior. La traza contiene 94,482 kernels y suma 9.959 s de ejecución de kernels, frente a 103,516 y 14.372 s anteriormente. Auditorías: `matrix-audit.json` y `profile-audit.json`.

### Recomendación y límites

**Mejor opción medida: staging K8/V4 + Flash nativa de PyTorch, chunk 8K, consultas Q >=17.** Mantener EXL3 5 bpw y MTP. Se entrega como opción experimental reversible, no como cambio silencioso del servidor ni como garantía de óptimo global. Se comprobaron 111 entradas de configuraciones en los dos barridos, finalistas con cinco longitudes de Q, 80 pruebas comparativas de modelo, 60 turnos finales y un oráculo FP32 muestreado.

Triton M64/N32, cuatro warps, dos stages y ocho splits queda como alternativa si se prioriza seguir más cerca de la aritmética del baseline: TTFT +8K 7.859 s en el screen, frente a 6.532 s de Flash. Antes de convertir Flash en predeterminado del motor, falta ampliar calidad a edición multiarchivo y ciclos reales con tools; la prueba corta no resuelve el fallo de herramientas previo ni certifica fidelidad contra pesos BF16. No se añaden atención dispersa, compresión de historial, offload ni otra cuantización.

Verificación final: 175 tests CPU pasan con Torch disponible y CUDA oculta; launcher válido con `bash -n`. Revisión independiente de causalidad/GQA/scratch y auditoría independiente de los 80 samples sin excepciones. Donante y GPU 1 intactos; sin commits ni ramas.
