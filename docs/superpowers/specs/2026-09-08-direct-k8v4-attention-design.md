# Atención directa K8/V4 para Qwasar

Estado: propuesta de diseño; no implementada. El usuario solicita planificar cuidadosamente la variante después del experimento de XQA con conversión temporal.

## Decisión propuesta

Mantener el KV activo K8/V4 en VRAM. Construir un lector especializado que expanda bloques dentro del kernel de atención, sin escribir un espejo completo en memoria global. Conservar Minima64 + FP8 PRIMS como base experimental y MTP6. Empezar con aritmética FP16 y acumuladores/softmax FP32; FP8 interno será una variante posterior y separada.

El decode actual ya combina lectura cuantizada directa, Hadamard, GQA y split-KV. Es el competidor a superar, no una implementación ingenua. XQA FP8 aislado es una señal de oportunidad, no un objetivo de tiempo garantizado para otro lector y otra aritmética.

## Por qué el espejo fuera de VRAM no resuelve la latencia

Un espejo FP8 de K y V para 16 capas, 4 cabezas KV y dimensión 256 ocupa:

`bytes = 16 * 2 * tokens * 4 * 256`

Son 8 GiB a 262144 tokens y 4 GiB a 131072, sin metadatos ni capa draft. La atención exacta vuelve a recorrer el contexto para cada ronda de verificación. Compartir una lectura entre siete consultas no elimina la siguiente ronda.

| Ubicación / supuesto | Lectura de 8 GiB | Lectura de 4 GiB |
|---|---:|---:|
| RAM a través de PCIe 5.0 x16, 63.015 GB/s teóricos unidireccionales | 136.3 ms | 68.2 ms |
| RAM a través de PCIe 4.0 x16, 31.508 GB/s teóricos | 272.6 ms | 136.3 ms |
| SSD hipotético a 12 GB/s efectivos | 715.8 ms | 357.9 ms |

Son cálculos de bytes/ancho de banda, no mediciones del equipo ni tiempos que se deban sumar ciegamente al cálculo: puede existir solapamiento. Aun con solapamiento perfecto, el flujo sostenido de rondas queda limitado por esos bytes. El ejemplo de SSD no identifica ni mide el SSD instalado. DMA directo tampoco elimina el límite de transferencia.

La consulta local de nvidia-smi reportó soporte máximo Gen5 x16 y Gen2 x16 en el instante de reposo; no se infiere de ello el enlace negociado bajo carga ni el rendimiento H2D. No es necesario ejecutar un benchmark de transferencia para descartar este límite optimista frente al decode residente.

RAM/SSD sí son ubicaciones posibles para snapshots y sesiones inactivas. Reanudar implica una transferencia amortizable una vez. Un snapshot completo de este modelo híbrido debe incluir también estado GDN, historial de convolución, tokens y metadatos de prefijo; guardar sólo KV no permite restaurar la sesión correctamente. Esa persistencia queda fuera de este cambio.

Fuentes: [ancho de banda PCIe de Intel](https://edc.intel.com/content/www/us/en/design/products/platforms/details/raptor-lake-s/13th-generation-core-processors-datasheet-volume-1-of-2/003/pci-express-support/), [memoria host mapeada y transferencias CUDA](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html), [oversubscription de NVIDIA](https://developer.nvidia.com/blog/?p=37205).

## Alternativas

1. Espejo FP8 en RAM/SSD: ahorra VRAM, introduce lectura por PCIe por ronda; descartado para acelerar esta sesión activa.
2. Conversión directa K8/V4 a scratch FP8: elimina el paso FP16 del experimento previo, pero todavía escribe y relee todo el espejo. No es el objetivo principal.
3. Lector K8/V4 fusionado con atención: mantiene memoria y semántica del caché; requiere un kernel especializado y trabajo real de integración. Es la propuesta.

Reutilizar XQA significa estudiar/reusar piezas de pipeline, cargas y MMA con sus licencias y revisión del target SM120. No significa que su API Python ni su lector homogéneo acepten el caché actual. Antes de portar su kernel completo se debe demostrar que el lector propuesto encaja en su organización. Si la adaptación arrastra interfaces de paginación, escalas o tipos incompatibles, usar un kernel CUDA/CuTe pequeño de Qwasar que conserve el contrato de split/combine existente; registrar esa elección antes de optimizar.

## Contrato que se preserva

- GPU0 RTX5090 / SM120 exclusivamente; GPU1 no se usa.
- Batch 1, Q24/KV4/D256, páginas de 256 tokens, contexto nativo 262144.
- MTP fijo 6; formas Q1..7 y Q8 sólo como comprobación de borde/futuro, sin cambiar width.
- Cachés target y draft K8/V4; Minima64 MLP; FP8 PRIMS únicamente para la ingesta grande según router existente.
- Proyecciones GDN/atención, embeddings y cabezas según la base experimental existente.
- Conservar tabla de páginas, escrituras y cursor actuales. No introducir KV contiguo ni un formato nuevo en este experimento.
- Atención causal completa. Una forma no soportada elige el backend actual antes de capturar el grafo; no se cambia de backend dentro de un grafo ni se ocultan fallos numéricos con fallback.
- El lector opera después de append y no escribe KV ni escalas. Lee `cache_seqlens` en device y aplica el contrato exacto de longitud pre-append más Q.
- No sincronización `.item()`, asignaciones, compilaciones ni búsqueda de configuración en el camino caliente.

## Matemática y layout

El caché no es INT8/INT4 convencional: empaqueta códigos por grupos de 32 valores en el dominio rotado. Para cada código sin signo `c`, escala `s` y ancho `b`, el lector existente reconstruye:

`x_rot = FP16((FP32(c) - (2**(b-1) - 0.5)) * (FP32(s) / 2**(b-1)))`

Se deben preservar el orden de operaciones y redondeo inicial al comparar lectores. Las orientaciones K transpuesta y V normal tienen direccionamiento distinto; copiar la misma rutina para ambas no basta. Las primitivas `_qc_plane_kt`, `_qc_plane_v`, `_qc_load_kt` y `_qc_load_v` son el contrato a verificar.

Con `R = H32 / sqrt(32)`, operar con Q rotada y K/V en el dominio rotado; restaurar la salida con R. Comparar tanto con el decode directo como con un oráculo FP32 sobre K/V reconstruidos. No confundir equivalencia matemática con identidad bit a bit después de redondeos.

Cada bloque procesa cabezas Q hermanas y posiciones de verify para reutilizar K/V; usar split-KV para poblar la GPU. Estas técnicas ya existen: las variables a evaluar son distribución de filas, tamaño de bloque, presión de registros, disposición en memoria compartida, solapamiento de carga/cálculo y combinación de parciales. No atribuir una ganancia a una característica ya presente.

Mantener softmax online y acumuladores FP32. El único temporal global permitido para atención es el estado parcial de reducción `(m, l, acc)`, cuyo tamaño depende del número de splits y Q, no de materializar todos los valores K/V del contexto. Fijar un presupuesto adicional inicial de 64 MiB compartido entre capas secuenciales para buffers del backend; medir memoria real, sin reservar una copia por capa. Registros derramados a memoria local también cuentan como tráfico y deben perfilarse.

## Integración en Qwasar

La ruta normal pasa por `BCAttn._configure` y `BC_Attention::run`, no sólo por el dispatcher Python. El bloque fusionado ejecuta proyecciones, normalización/RoPE, append cuantizado, split/combine de atención y salida. Interceptar sólo `paged_attn_triton_decode` no cambia necesariamente el modelo real.

El ABI actual del split tiene 15 argumentos en este orden:

`q, k_cache, v_cache, block_table, cache_seqlens, out, partial_o, partial_ml, k_scales, v_scales, h32, split_len, num_pages_per_seq, num_splits, sinks`

El grafo parchea tabla en índice3, longitudes en4 e ints runtime en11/12/13. Combine recibe `partial_o, partial_ml, out, h32, num_splits, sinks`; parchea splits en4. Grid capturado usa el máximo de splits; los splits inactivos no deben contaminar la reducción.

Preferir compatibilidad con este ABI y layout de parciales. Si se necesitan otros grids o tipos de handles, crear una extensión experimental explícita del slot y sus parámetros de grafo. No asumir que un handle CUDA arbitrario funciona como `TritonKernel`. Probar la integración con el algoritmo original antes de atribuir diferencias al kernel nuevo.

## Evidencia mínima y decisión

La evidencia previa está en `results/20260908-combined-profile/report.md`: Q7/258183, decode directo0.7905 ms, XQA FP8 preparado0.3511 ms, conversión+XQA2.4427 ms. La captura128K fue sintética y una sola capa no representa todas las capas.

Antes de promover: lectores correctos; atención correcta en múltiples capas y formas; grafo con longitudes y tablas mutables; append/reject/prefijo correctos en el modelo; ausencia de scratch completo; mejora repetible del bloque integrado y del modelo completo.

La meta de continuación a integración es al menos15% menos tiempo de atención Q7 a256K frente al baseline fresco, con Q1 sin regresión superior a3% o router fijo que conserve Q1 original. Son umbrales de investigación, no predicciones. La meta de adopción es +15% tokens aceptados/s a256K, TTFT Q128 p50<=300 ms, sin empeorar prefill grande más de3% y sin empeorar las variantes de calidad emparejadas. Conservar presupuesto de salida; el usuario acepta menos de6/6, pero no ocultar regresiones individuales detrás del agregado.

## Fuera del primer cambio

FP8 dentro de decode, NVFP4 KV, espejos persistentes, offload, sparse attention, cambio de paginación, proyecciones adicionales y cambios de MTP/GDN. Se podrán estudiar por separado si la primera variante ofrece evidencia útil.
