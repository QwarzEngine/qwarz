# Revisión upstream y checkpoints NVFP4 — 2026-09-08

**Conclusión:** hay componentes nuevos o recién integrados que conviene evaluar antes de reescribir atención. Sobresalen FlashInfer FP8 GQA D256, XQA NVFP4 con máscara especulativa y el head reducido MTP de ExLlamaV3. También existe un checkpoint con las 496 proyecciones del backbone en NVFP4; su archivo no contiene MTP. Descargar pesos evita volver a cuantizar, pero no reemplaza los contratos de kernels, caché y especulación.

Revisión realizada 2026-09-08T16:04:50.200986+00:00. Búsquedas GitHub por repositorio, PR actualizados desde 2026-08-20, NVFP4/SM120/GDN/attention y revisión de PR previos citados. Se consultó la API para distinguir abierto, cerrado e integrado; se inspeccionaron diffs de ocho PR principales y del backend FlashInfer. No es una auditoría exhaustiva de cada PR de los repositorios. Los números publicados por terceros no son nuevas mediciones de Qwasar.

## Selección de cambios

| Proyecto / PR | Estado consultado | Aplicación y límites |
|---|---|---|
| [vllm #55170](https://github.com/vllm-project/vllm/pull/55170) | Integrado 2026-09-08 | Prioriza W4A4 sobre W4A16 en SM120. Útil al repetir benchmarks con otro runtime; el híbrido Qwasar ya elige GEMM nativo. |
| [vllm #53543](https://github.com/vllm-project/vllm/pull/53543) | Abierto | XQA NVFP4 con máscara para especulación. Test GPU 5090 Q24/KV4/D256/Q8; resultados de servidor proceden de un backport, DFlash2 y otra configuración. Candidato para entender/reintentar la integración que falló. |
| [vllm #54772](https://github.com/vllm-project/vllm/pull/54772) | Abierto | Otra habilitación NVFP4 KV SM120: XQA decode y FA2 prefill. Descripción sin resultados de prueba; no tomar como solución validada. |
| [vllm #55065](https://github.com/vllm-project/vllm/pull/55065) | Abierto | Integra FP8 PRIMS prefill SM120. API admite D256 y GQA; depende de FlashInfer #4714 y CUTLASS DSL reciente. Validación publicada usa otro Qwen, 32 requests concurrentes y salida de un token. |
| [vllm #52244](https://github.com/vllm-project/vllm/pull/52244) | Abierto | Corrige publicación de estados GDN y colas parciales de caché bajo MTP/EAGLE con hashing fino. MR V1; no equivale a reutilización exacta token a token ni valida Qwasar a 256K. |
| [vllm #55688](https://github.com/vllm-project/vllm/pull/55688) | Abierto | ReplaySSM con ciclo de vida de prefijos y anillos recurrentes. Depende de #52928 y FlashInfer #4815; evidencia H100/GB200, otro modelo y batch 64. |
| [vllm #54614](https://github.com/vllm-project/vllm/pull/54614) | Abierto (draft) | Propone dispatch W4A16/W4A4 por M con layout compartido. Ningún kernel del árbol declara todavía ese contrato: infraestructura experimental, no aceleración disponible. |
| [vllm #55643](https://github.com/vllm-project/vllm/pull/55643) | Integrado 2026-09-08 | Corrige padding de escalas y dirección de escala global en SiLU+NVFP4. Revisar al reutilizar esa fusión; no atribuir este fallo al adaptador local de FlashInfer sin prueba. |
| [sglang #38170](https://github.com/sgl-project/sglang/pull/38170) | Abierto | Selecciona b12x por defecto en SM120 y conserva autotuning. Qwasar ya usa b12x para M≤128: confirma nuestra decisión, no agrega ese ahorro otra vez. |
| [sglang #36043](https://github.com/sgl-project/sglang/pull/36043) | Cerrado sin merge | Prototipo skinny GEMM, sustituido por #36865. No tratar el cierre como merge ni importar su política original de forma indiscriminada. |
| [sglang #36865](https://github.com/sgl-project/sglang/pull/36865) | Integrado 2026-09-02 | Kernels KDA integrados. En Qwen3.8, dispatch validado sólo down M=9/K17408/N5120, para DSpark. Nuestro verify MTP6 usa Q7; no aplica automáticamente. Expone una regresión por persistir escalas en L2. |
| [sglang #36038](https://github.com/sgl-project/sglang/pull/36038) | Abierto | Ruta NVFP4 nativa para verify/draft extend; sigue abierta. Máscaras, ragged queries, escalas y metadatos son parte del contrato. El port local previo falló; no se prueba aquí otra revisión. |
| [sglang #36045](https://github.com/sgl-project/sglang/pull/36045) | Abierto | Alternativa con workspace FP8 GPU y CUDA Graph safe; sigue abierta. Conserva conversión del prefijo, no elimina el temporal. Ya fue evaluada parcialmente en el experimento local. |
| [sglang #35824](https://github.com/sgl-project/sglang/pull/35824) | Cerrado sin merge | Recuantización FP8→NVFP4 de proyecciones de entrada, cerrada sin merge. No es equivalente a usar un checkpoint calibrado desde BF16; el autor reporta degradación severa al extenderla a salidas. |
| [llama.cpp #28572](https://github.com/ggml-org/llama.cpp/pull/28572) | Abierto | Pipeline cp.async/TMA para MMQ NVFP4 y menos spills. Autor reporta +14% prefill Qwen3.8-27B en 5090; es GEMM de pesos, no un kernel de KV. Backend/layout diferentes del nuestro. |
| [llama.cpp #28549](https://github.com/ggml-org/llama.cpp/pull/28549) | Abierto | Separa cachés de CUDA Graph para draft con salida y catch-up sin salida. Evita recaptura repetida; la mejora publicada usa otro Qwen y MTP3. |
| [llama.cpp #27161](https://github.com/ggml-org/llama.cpp/pull/27161) | Abierto | Conversor de checkpoints mixed FP8/NVFP4 compressed-tensors, incluido Unsloth Qwen3.8 con MTP. Sigue abierto; relevante para una comparación GGUF, no necesario para importar tensores a Qwasar. |
| [llama.cpp #23572](https://github.com/ggml-org/llama.cpp/pull/23572) | Abierto | Cuantizador NVFP4 simplificado: su autor declara pérdida respecto a otros formatos y no equivalencia con una receta NVFP4 calibrada. No elegirlo como referencia de calidad del formato. |
| [llama.cpp #26556](https://github.com/ggml-org/llama.cpp/pull/26556) | Cerrado sin merge | Otro cuantizador NVFP4, cerrado sin merge. Los resultados indexados por buscador que aún lo muestran abierto están desactualizados. |
| [exllamav3 #303](https://github.com/turboderp-org/exllamav3/pull/303) | Abierto | Head de vocabulario reducido sólo para borradores, verificación con vocabulario completo. Ataca directamente el head externo que ocupa ~45% de la fase draft en nuestra traza de producción. Costo publicado ~0,55 GiB; datos de 4060 Ti/MTP4, no ganancia medida en 5090/MTP6. |
| [exllamav3 #334](https://github.com/turboderp-org/exllamav3/pull/334) | Abierto | DFlash2 greedy sin cambios de extensión nativa, validado con GLM-5.3; requiere siete drafts en ese checkpoint. No es una mejora conservando MTP6, sino un experimento posterior de drafter. |
| [exllamav3 #247](https://github.com/turboderp-org/exllamav3/pull/247) | Cerrado sin merge | Carga directa de bytes Q8, cerrado sin merge y anterior al período principal. Idea para el lector K8; las cifras son microbench Q8/Q8, no K8/V4 a 256K. |
| [flashinfer #4714](https://github.com/flashinfer-ai/flashinfer/pull/4714) | Integrado 2026-09-02 | Backend CuTe DSL PRIMS FP8 con paged/ragged, causal, LSE y grafos. El validador admite D32/64/128/256 y Hq divisible por Hkv, incluida 24/4; requiere probar la forma completa. Páginas 16/32/64/128 y HND compacto difieren del layout local. |

## Qué cambia respecto a la investigación anterior

La exclusión de FlashInfer por D64/D128 y ausencia de GQA era específica de `fmha_v2_prefill_sm120`. No describe el backend `cute-dsl-prims` integrado el 2 de septiembre. Su API admite nuestra relación 24/4 y D256. La diferencia de formato continúa: pide FP8, no K8/V4, y otro layout de páginas. La primera comparación útil sería aislada con las capturas existentes, incluyendo conversión y memoria, antes de tocar el pool residente.

XQA es otra ruta: vLLM #53543 incluye un test explícito de 24 Q/4 KV/D256/Q8 en una 5090. Además, reporta una diferencia grande entre replay aislado y ejecución integrada, mitigada con un stream separado. Esto es una pista para depurar, no evidencia de que nuestro acceso ilegal anterior tenga esa misma causa. KV NVFP4 no demuestra por sí mismo que las multiplicaciones QK/PV sean FP4×FP4.

El head MTP reducido de ExLlamaV3 #303 encaja con una medición propia: 276 ms de head externo sobre 615,5 ms de kernels de la fase draft. Restringe propuestas, no el vocabulario usado para verificar. La selección debe respetar grupos Hadamard de 128 tokens; se debe medir aceptación y tiempo por token aceptado con MTP6 y corpus propio. El +21,9% del autor no se extrapola a 256K.

B12X ya se usa en el híbrido para M≤128. SGLang #38170 y vLLM #55170 evitan selecciones genéricas malas; no son una mejora nueva de nuestros GEMM actuales. SGLang #36865 enseña por qué el microbench de pesos calientes puede engañar: persistir escalas desplazaba otros datos y empeoraba el modelo entero.

La copia local del donante ya contiene `cache/nvfp4.py` y atención Triton para páginas NVFP4. Eso es almacenamiento/lectura de KV, no un cargador general de pesos NVFP4 ni prueba de aritmética FP4 nativa en atención. No se modifica ni se presenta como novedad upstream.

## Checkpoint descargable

[Minima Qwen3.8-27B NVFP4](https://huggingface.co/minima-ai/mnma_qwen3.8_27b_nvfp4) es el candidato para ampliar el híbrido. Revisión fijada: `16e768e7d0461b0b86e565ecedd08a24eca53e9a`. El archivo pesa 18,788,354,104 bytes (17.50 GiB). Se leyó sólo su cabecera safetensors de 277616 bytes, además de configuración y catálogo; no se descargaron los pesos.

Comprobación directa de la cabecera: 496 matrices `weight_packed`, con escalas por bloques FP8 y escalas globales FP32; embeddings y lm_head BF16. No hay tensores MTP/nextn. `config.json` declara `mtp_num_hidden_layers=1`, lo que por sí solo no certifica presencia de esos pesos. Incluye calibración de KV FP8, que no es KV NVFP4.

El esquema de nombres `weight_packed`, `weight_scale`, `weight_global_scale` e `input_global_scale` coincide con el adaptador experimental de Qwasar. Se pueden importar las matrices calibradas por grupos, conservando inicialmente el MTP, head y embeddings actuales. Es una mezcla numérica que exige validar las proyecciones, el estado recurrente y la aceptación, no un reemplazo de archivo transparente.

La ficha publica evaluación con vLLM/RTX PRO 6000 de 96 GB; el estudio llega a 32K de perplexity y 64K de recuperación. No valida por sí sola 256K en 32 GB ni MTP6. Pesos de ~17,5 GiB más K8/V4 de 6,5 GiB suman ~24 GiB teóricos antes de MTP, estados, grafos, scratch y allocator: hay que medir capacidad real.

El [Unsloth actual](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) es mixto, no 496 proyecciones NVFP4. Su catálogo sí incluye `model_mtp.safetensors` y su índice contiene claves MTP. Se archivó su revisión actual para no confundirla con revisiones usadas en experimentos anteriores.

## Orden propuesto

1. Usar el checkpoint Minima como fuente de pesos originales ya calibrados. Barrer primero MLP56→MLP64 y luego proyecciones GDN/atención por grupos, conservando MTP6 y K8/V4. No recuantizar EXL3.
2. Probar el head MTP reducido en un A/B aparte. Evaluar mapas derivados del corpus propio y el costo de memoria; mantener siempre el verificador completo.
3. Comparar Flash actual, K8/V4 directo y FP8 PRIMS con las capturas reales Q128/Q8192. Las conversiones deben entrar en el tiempo medido; no adoptar FP8 KV global sin medir el pool.
4. Reintentar XQA NVFP4 con un contrato completo de máscara/longitudes/grafos y pruebas de rewind. Después, evaluar cola reciente FP16. No se encontró en esta selección un PR validado que implemente exactamente esa mezcla para Q24/KV4/D256/MTP6 a 256K.
5. Repetir una referencia vLLM cuando se integre o pueda aislarse #52244, manteniendo el costo de prefijo como criterio central. Una corrección de reutilización parcial merece reabrir la comparación, no dar por resuelta la equivalencia con Qwasar.

## Artefactos y verificación

- [Estados, fechas y commits de PR](../../results/20260908-upstream-review/pr-summary.json).
- [Cabecera real del checkpoint Minima](../../results/20260908-upstream-review/minima-ai-safetensors-header.json).
- [Configuración Minima](../../results/20260908-upstream-review/minima-ai-config.json).
- [Catálogo de archivos y hashes LFS Minima](../../results/20260908-upstream-review/minima-ai-model.json).
- [Manifest del análisis](../../results/20260908-upstream-review/manifest.json).

La carpeta de resultados conserva respuestas JSON de GitHub/Hugging Face y diffs. Los PR de búsqueda inicial pueden quedar fuera de la selección por no aportar a este hardware/modelo; por ejemplo llama.cpp #28552 no es el PR NVFP4 #28572, aunque un resultado de búsqueda asoció mal la numeración. Se priorizó la API exacta.
