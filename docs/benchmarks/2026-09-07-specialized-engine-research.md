# Investigación: siguiente motor de Qwasar para RTX 5090

Fecha: 2026-09-07. Alcance: Qwen3.8-27B, una RTX 5090, una generación activa y contexto nativo de 262.144 posiciones. Investigación de código, resultados archivados y documentación primaria; no se ejecutaron nuevas generaciones ni se modificó el servicio.

**Recomendación: avanzar hacia un backend propio mediante sustituciones medidas.** La especialización tiene fundamento, pero el beneficio depende de los kernels, la precisión y la especulación. Reescribir la coordinación en C++ no garantiza acelerar el trabajo GPU dominante. Conservar el supervisor Rust y el backend EXL3 como referencia permite obtener mejoras antes de completar un runtime nativo.

## Punto de partida verificado

La [arquitectura v1](../arquitectura-v1.md) usa EXL3 5 bpw, cabeza a 6 bits, MTP a 4 bits, cuatro tokens draft, caché K8/V4 y prefill Flash/8192. El [diseño inicial](../superpowers/specs/2026-09-04-qwen38-27b-rtx5090-engine-design.md) ya contemplaba C++/CUDA y un formato mixto Q38X. Sus elecciones futuras de NVFP4 y DFlash2 son hipótesis de diseño, no resultados demostrados.

Se inspeccionó el `config.json` del artefacto local fijado por el manifiesto. Declara 64 capas: 48 Gated DeltaNet y 16 de atención completa; hidden size 5120, MLP 17408, 24 cabezas Q, cuatro KV y dimensión 256. Tiene una capa MTP. Esto permite compilar formas y asignaciones específicas; no es un MoE que pueda acelerar omitiendo expertos inactivos. La publicación oficial identifica el checkpoint como [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).

Una consulta de solo lectura con `nvidia-smi` confirmó GPU 0 RTX 5090, compute capability 12.0, 32.607 MiB expuestos y 29.223 MiB ocupados en ese instante. Es memoria total del dispositivo, no una atribución exclusiva a Qwasar. NVIDIA especifica 32 GB GDDR7 y 1792 GB/s teóricos para esta GPU: [especificaciones del lanzamiento](https://www.nvidia.com/es-la/geforce/news/rtx-50-series-graphics-cards-gpu-laptop-announcements/).

## Qué dicen las mediciones existentes

La [matriz posterior a Flash](2026-09-04-prefill-tuning-results.md) mantiene cerca de 256K unos 96–103 tokens/s de decode en tareas cortas. Con 128 tokens nuevos, TTFT mediana 322 ms; con 8192 nuevos, 6,540 s. Los 60 casos pasan el control de respuesta y reutilización. Son muestras de un workload acotado, no percentiles de producción ni certificación general de coding.

Recalculé el desglose sobre **todos los eventos kernel** de la traza posterior a Flash, evitando reutilizar el porcentaje antiguo de 72,84%. Caso: 260.466 tokens de entrada y 8305 de prefill físico, con instrumentación. Resultado derivado y hash de la traza en [profile-breakdown.json](../../results/20260907-engine-research/profile-breakdown.json).

| Familia | Tiempo acumulado | Porcentaje del tiempo de kernels |
| --- | ---: | ---: |
| Atención Flash | 4,216 s | 42,33% |
| Atención de decode/verificación, split y combine | 1,996 s | 20,04% |
| GEMM CUTLASS FP16 | 1,783 s | 17,91% |
| Operaciones EXL3 identificadas por nombre | 1,468 s | 14,74% |
| Descompresión explícita de KV | 0,038 s | 0,38% |
| Resto | 0,459 s | 4,61% |

Total: 94.482 kernels, 9,959 s acumulados. Estos porcentajes **no son utilización del hardware ni un desglose completo de latencia HTTP**. El caso incluye un sufijo grande y no representa el perfil de decode puro o de un turno de 128 tokens. Faltan contadores de ancho de banda, Tensor Cores, ocupación y stalls para identificar el límite físico de cada kernel.

Como cálculo ilustrativo de Amdahl, reducir a la mitad ese 62,37% de atención bajaría el tiempo acumulado de kernels alrededor del 31,2%: aproximadamente 1,45× de aceleración si todo lo demás permaneciera igual. No es una predicción del motor ni de la respuesta completa. Duplicar solamente las GEMM CUTLASS afectaría aproximadamente al 9% de ese total.

## Prioridades propuestas

### 1. Especializar atención exacta para las formas reales

Es la inversión con mayor respaldo en el perfil largo. Separar prefill de verificación especulativa, conservando inicialmente pesos EXL3 y K8/V4.

- Medir Q=1,2,3,4,5,8,16 y sufijos de 128/512/2048/8192 sobre 32K, 128K y casi 256K.
- Ajustar splits, bloques, warps y agrupación de las seis cabezas Q por cabeza KV. El donante ya tiene split-KV y agrupación GQA: el trabajo es mejorar su ejecución para estas formas, no introducir mecanismos ausentes.
- Comparar un kernel CUDA/CuTe especializado con los mejores Triton y Flash actuales. Mantener la causalidad lower-right, softmax estable, escalas y transformaciones de K8/V4.
- Evaluar lectura cuantizada directa frente a staging con la operación completa. El coste explícito de descomprimir es pequeño en la traza; un beneficio mayor tendría que venir de reducir tráfico posterior, scratch o mejorar cómputo/ocupación. Debe medirse.
- Explorar chunks determinados por longitud del sufijo y contexto, en lugar de usar siempre 8192.

Hay bibliotecas útiles como referencia, pero **soportar SM120 no implica soportar Qwasar**. La entrada [FlashInfer `fmha_v2_prefill_sm120`](https://docs.flashinfer.ai/generated/flashinfer.prefill.fmha_v2_prefill_sm120.html) documenta FP8, MHA con igual cantidad de cabezas y dimensiones 64/128; excluye GQA. No sustituye directamente Q24/KV4/D256. La API [NVFP4 attention SM120](https://docs.flashinfer.ai/generated/flashinfer.nvfp4_attention_sm120.nvfp4_attention_sm120_fwd.html) utiliza Q/K/V precuantizados, escalas, V transpuesta y corrección QK. Su compatibilidad exacta y calidad necesitan validación; no consume sin adaptación la caché EXL3 actual.

### 2. Mejorar especulación antes de fijar un drafter definitivo

El [loader de Qwasar](../../src/qwasar_bench/exllamav3_probe.py) construye `Generator` sin fijar `num_draft_tokens` ni `dynamic_draft_tokens`. En el donante instalado, MTP usa cuatro por defecto y el ajuste dinámico está desactivado. Este ya ofrece longitud adaptativa y omisión temporal del draft cuando la aceptación es baja.

Primero compararía MTP fijo 1/2/4/7, modo adaptativo y control sin especulación, con pesos, caché, prompts y sampler idénticos. La métrica decisiva es tiempo por token aceptado: `(draft + verificación + commit) / tokens emitidos`, junto con tiempo hasta herramienta ejecutable y respuesta completa. Más propuestas o mayor aceptación porcentual no garantizan mejor latencia.

Después reabriría DFlash2 en un A/B limpio. El [screen antiguo](2026-09-04-decode-screen-results.md) cambió simultáneamente drafter y caché, usó 3,5 bpw y otro protocolo; no permite concluir que DFlash2 sea intrínsecamente peor. Tampoco permite trasladar sus cifras a v1.

La [publicación de DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) muestra ventajas frente a MTP en H200, incluyendo una petición concurrente. Es evidencia para probarlo, no una previsión para la 5090 cuantizada a 256K. Su [configuración](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2/blob/main/config.json) tiene cinco capas de ventana 2048 y bloque ocho. Un caché circular puede explotar esa ventana, pero los pesos adicionales, los taps del target y el rollback deben entrar en el presupuesto. Si no cabe con K8/V4, documentar esa restricción en vez de cambiar dos variables y atribuirle toda la diferencia al drafter.

### 3. GEMM y precisión diseñadas para SM120

En el perfil aparecen kernels `cutlass_80_tensorop...`: son una razón para comparar implementaciones más ajustadas, no prueba de que falte soporte de la GPU ni de que se estén usando CUDA cores exclusivamente. Ya usan Tensor Cores FP16.

Haría dos experimentos separados. Primero, mejorar multiplicaciones con las mismas representaciones numéricas y dimensiones fijas. Después, crear desde el checkpoint original BF16 una variante NVFP4/mixed precision, priorizando matrices grandes y protegiendo tensores sensibles. No convertir EXL3 a NVFP4: encadenaría pérdidas de cuantización y dificultaría evaluar calidad.

NVIDIA proporciona [un ejemplo CUTLASS de GEMM NVFP4 para SM120](https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu), basado en instrucciones block-scaled y orientado a GeForce RTX 50. Hay que seleccionar código para esa arquitectura; un resultado en B200/SM100 no basta. La disponibilidad de la instrucción no determina el rendimiento con M=1–8, el coste de cuantizar activaciones ni la calidad del modelo.

Mantener rutas distintas para GEMV/decode pequeño, verificación de varios tokens y GEMM/prefill grande. Un kernel que gana con M=8192 puede perder con M=1. Un artefacto de 4,5–5 bpw efectivos requiere contar escalas, padding y tensores conservados a mayor precisión; no basta el número nominal del formato.

### 4. Runtime nativo y memoria estática, después de medir su contribución

Propuesta de destino: supervisor Rust existente → worker C++/CUDA específico → conjunto pequeño de kernels seleccionados para SM120. ExLlamaV3 queda como control ejecutable y fuente de semántica durante la migración.

El donante instalado ya tiene rutas C++ con CUDA Graphs por bloque de atención, MLP y Gated DeltaNet. Hay que medir cuáles se activan realmente y qué queda fuera: lanzar todo el paso mediante un grafo estable, evitar sincronizaciones host/device y fijar buffers podría ayudar, pero no es una ganancia desde cero. El adaptador Flash actual ejecuta `cache_seqlens.item()`; conviene eliminar esa dependencia del host cuando el estado de longitud pueda mantenerse de forma correcta.

Para una sesión, evaluar buffers contiguos y un cursor de longitud, con checkpoints recurrentes explícitos. Eliminar administración multiusuario simplifica el runtime, aunque el ahorro de indireccionamiento paginado no está medido. Además, el loader conserva defaults genéricos como `max_batch_size=256`; fijar uno merece un ensayo de memoria y rendimiento antes de una reescritura.

El estado recurrente es crucial: rebobinar KV no restaura Gated DeltaNet. Aceptación parcial MTP, rechazo, cancelación y ramas deben restaurar también estado recurrente y convolucional, RNG y longitud comprometida. Preservar los IDs exactos y el protocolo durable existente es parte del criterio de éxito.

### 5. Caché más compacta y atención aproximada como investigaciones separadas

Con 16 capas, 262.144 tokens, cuatro cabezas KV y dimensión 256, K8/V4 necesita **6 GiB de valores empaquetados**. Las formas de escalas FP16 del adaptador actual añaden **0,5 GiB**, para 6,5 GiB lógicos de KV completo, sin buffers adicionales. KV NVFP4 con 4 bits y una escala de 8 bits cada 16 elementos requiere 4,5 GiB bajo ese layout: ahorro teórico de 2 GiB. Son cálculos de formatos, no nuevas mediciones de VRAM.

Esto podría dar espacio a otro drafter o a mejores buffers; no implica más velocidad automáticamente ni igual fidelidad. Validar recuperación distante, atención numérica y coding, especialmente si se reducen las claves de 8 a 4 bits. Conservar inicialmente el estado recurrente FP32.

Atención dispersa/selectiva podría reducir el trabajo que crece con el contexto, pero altera el cálculo y exige evaluar omisiones de información lejana. Dejarla como experimento explícito posterior. Un megakernel persistente completo también requiere pruebas propias de ocupación, sincronización y rendimiento por fase; la cantidad de lanzamientos por sí sola no justifica hacerlo primero.

## Secuencia de trabajo y criterios de decisión

| Etapa | Trabajo concreto | Condición para avanzar |
| --- | --- | --- |
| A | Nueva línea base con v1 Flash: decode puro, turnos chicos, +8K, ingesta fría y ciclos de tools | Prompts reproducibles, precisión/configuración fijas y perfiles por fase |
| B | MTP adaptativo/fijo, batch uno y tuning de atención existente | Ganancia estable de extremo a extremo con misma calidad y contexto |
| C | Un kernel de atención propio y una variante GEMM | Ganancia dentro de Qwasar, incluyendo staging, lanzamiento y memoria |
| D | NVFP4/mixed precision y DFlash2, con controles separados | Calidad y memoria aceptables en 32K/128K/256K |
| E | Migrar el bucle a C++/CUDA e integrar kernels ganadores | Justificación por perfil y pruebas de estado, tools y recuperación |

Usaría como metas de aceptación iniciales las del diseño existente: al menos +15% de decode aceptado y +25% de throughput de prefill grande frente a v1, a calidad comparable. Se convierten en gates del nuevo ensayo, no en mejoras prometidas. Las ganancias pequeñas y baratas pueden adoptarse sin esperar a la migración completa.

En cada A/B: mismo artefacto salvo el ensayo explícito de precisión; mismo prompt/sampler; warmup fuera de medición; orden intercalado; longitudes reales y reutilización física verificadas. Separar TTFT, primer contenido, herramienta ejecutable y finalización. Incluir respuestas incorrectas/truncadas en los resultados. Ampliar calidad a edición multiarchivo y herramientas, dado que la validación existente es limitada. Si cambia la aritmética, comparar logits y tareas; igualdad greedy aislada no prueba equivalencia de distribución.

Para diagnosticar el techo, medir con Nsight Systems/Compute cuando estén disponibles: DRAM efectiva, ocupación, uso de Tensor Cores, stalls y esperas CPU. La cota de ancho de banda por paso depende de bytes realmente leídos de pesos, KV y temporales; la especulación amortiza parte de ese tráfico entre tokens aceptados. Dividir el tamaño del archivo por el ancho de banda no produce por sí solo un techo válido de tokens/s especulativos.

La investigación alcanza para priorizar experimentos, no para fijar una aceleración final. El siguiente entregable útil es un A/B pequeño de especulación y atención contra **la v1 con Flash**, seguido por un prototipo de kernel donde ese A/B muestre margen.
