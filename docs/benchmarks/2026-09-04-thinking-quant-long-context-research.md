# Thinking, cuantización y decode cerca de 256K

Fecha: 2026-09-04. Investigación documental y revisión local; sin nuevas ejecuciones GPU. Complementa `2026-09-04-review-30s-latency.md`. No cambia todavía el runtime ni la especificación aprobada.

## Conclusión

Más de 50 tokens/s es un objetivo razonable de ingeniería, no una capacidad que nuestra implementación haya demostrado. Los aproximadamente 15.65 tokens/s locales no constituyen un límite del modelo o de EXL3. Antes de desarrollar otro kernel, comparar configuraciones completas y verificar calidad, ocupación real del contexto y aceptación especulativa.

No hay evidencia suficiente para declarar un ganador universal entre EXL3 5 bpw, NVFP4 y GGUF. Propuesta: mantener EXL3 como referencia inmediata, probar NVFP4 calibrado como candidato de rendimiento y considerar FP6 especializado como alternativa experimental.

## Thinking

La [documentación oficial](https://huggingface.co/Qwen/Qwen3.8-27B) admite `xhigh`, `medium`, `low` y thinking desactivado. `xhigh` y conservar el thinking histórico son los valores por defecto. Advierte que reducir esfuerzo puede aumentar reintentos y tiempo total del agente. Los parámetros recomendados de sampling difieren entre thinking y no-thinking.

[Simon Willison](https://simonwillison.net/2026/Aug/16/qwen-38-27b/) documenta razonamiento excesivo en tareas pequeñas: una generación SVG consumió 22,276 tokens de razonamiento. También muestra un caso de código que perdió corrección al apagarlo. Es evidencia cualitativa con trazas, no un benchmark de agentes a 256K.

Hallazgo local: `../qwen38-exl3-mia/tools/serve_openai.py`, función `generate_full`, fuerza `enable_thinking=True` y no pasa `reasoning_effort`. El template del artefacto 3.5 bpw resuelve ese caso como `xhigh` e inserta instrucciones en el prefijo. `medium` no añade esas instrucciones especiales; no impone un límite duro de tokens.

Propuesta experimental: comparar `medium` como candidato interactivo, `xhigh` como control de calidad y `low`/off para tareas sencillas. Mantener la política fija por sesión durante la comparación: modificar instrucciones iniciales o quitar razonamiento histórico retroactivamente invalida el prefijo. No confundir cambiar el sampler con desactivar thinking. Medir tareas resueltas y tiempo total, no sólo cuánto tarda una respuesta aislada.

## Qué aporta MiaAI-Lab

El [kit oficial](https://github.com/MiaAI-Lab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw) distingue target EXL3 **3.5 bpw** y draft DFlash2 **5 bpw**: el nombre del repositorio no implica un target de 5 bpw. Publica resultados GB10 y capacidad nativa 262K; su README consultado todavía aclara que no son benchmarks RTX.

Localicé el [reporte de Melvin Vivas, reproducido en un espejo](https://zamantika.com/hi/melvindvivas/status/2094687609910128781): aproximadamente 64.5 tokens/s en RTX 3090, `DRAFT=mtp`, `CONTEXT_SIZE=262144`, `CACHE_QUANT=8,4`, `GPU_MEM_GB=22`. El [original de X](https://x.com/melvindvivas/status/2094687609910128781) devolvió 403. Es una pista, no una medición validada aquí: el texto disponible no documenta tokens de entrada efectivos, fixture, longitud de salida ni aceptación. Reservar 262K no demuestra tenerlos ocupados.

Nuestra prueba usó DFlash2 y KV NVFP4, no esa combinación. El parser local interpreta `8,4` como bits K/V. Debemos comparar MTP/DFlash2/sin draft y caché `8,4`/NVFP4 por separado. Una tasa de aceptación diferente puede cambiar mucho el resultado.

## Evidencia de cuantización y velocidad

| Fuente primaria | Resultado publicado | Límite de interpretación |
| --- | --- | --- |
| [Quesma](https://quesma.com/blog/qwen38-27b-quantizations-benchmarked/) | Q4_K_M próximo a BF16 en GPQA, IFBench y Terminal-Bench 2.1 | KV F16; coding con 98K reservados y xhigh; algunas revisiones de pesos ya no están disponibles. No prueba fidelidad de KV de 4 bits a 256K |
| [Minima, preprint](https://arxiv.org/html/2609.04098v1) | NVFP4 W4A4: 17.53 GiB de pesos; promedio de cinco tareas 0.52 puntos por debajo de BF16 | RTX PRO 6000; recuperación evaluada hasta 64K. Decode publicado con concurrencia 32, no velocidad de un usuario en 5090 |
| [qwentin](https://github.com/kacper-daftcode/qwentin/blob/main/README.md) | FP6 + K4/V FP8 + MTP: 111.8 tokens/s inicial y aproximadamente 105.6 en follow-up a 243K; prefill frío 294 s | Autorreporte de motor de investigación. Calidad principal por coincidencia top-1, no éxito de coding; no es todavía certificación a 262,144 |

El paper Minima incluye calibración de escalas KV FP8 y correcciones de escalas en GEMMs fusionadas. Sus resultados justifican probar la receta exacta, no trasladar el nombre NVFP4 a cualquier checkpoint. Tampoco justifican cuantizar directamente el estado recurrente FP32.

qwentin es interesante por atención especializada SM120, verificación especulativa batched y prefill ancho. Sus benchmarks usan MTP y ajustes de draft; hay que auditar el verificador y el sampler antes de asumir equivalencia con nuestro régimen estocástico. La coincidencia top-1 con BF16 no es porcentaje de inteligencia conservada. Sus funcionalidades opcionales que alteran el comportamiento del modelo quedan fuera de esta comparación.

## Artefactos y decisión provisional

- EXL3 3.5 bpw: conservar el baseline existente para aislar mejoras de runtime.
- EXL3 5 bpw: los tres shards ya existen localmente, con tamaños esperados que suman 19,901,680,029 bytes, unos 18.54 GiB. No se verificaron hashes en esta revisión; el perfil sigue marcado como no descargado. No confundir archivo presente con artefacto validado.
- NVFP4 calibrado: candidato prioritario a comparar por rendimiento, empezando por recetas publicadas y sin mezclar revisiones. Evaluar pesos y caché independientemente.
- GGUF Q4_K_M/Q5: control externo de calidad y rendimiento, no ganador elegido por bpw.
- FP6 especializado: candidato posterior si reproducción y calidad justifican su coste de integración.

Cambiar de EXL3 3.5 a 5 bpw requiere volver a medir memoria máxima, espacio de KV/draft/scratch y aceptación; no garantiza velocidad ni calidad de tarea. Comparar bpw nominal entre formatos omite escalas, capas protegidas, activaciones y buffers.

## Experimentos propuestos, en orden

1. Congelar hashes de pesos, template, backend y código importado; verificar el target 5 bpw. Corregir la medición para separar primer token, reasoning, primer contenido y herramienta completa válida.
2. Reproducir EXL3 3.5 con MTP y K8/V4 frente al baseline DFlash2/NVFP4. Primero screening corto; después barrido factorial para no atribuir al draft un cambio causado por caché.
3. Probar thinking xhigh/medium/low/off con tareas pareadas de edición, diagnóstico y herramientas. Evaluar sampling recomendado y cualquier variante greedy como regímenes distintos.
4. Comparar finalistas EXL3 5 y NVFP4 sobre las mismas tareas y configuraciones de esfuerzo. Separar fidelidad numérica contra BF16 de rendimiento del sistema completo. BF16 necesita un entorno de referencia con suficiente memoria; no cabe plenamente residente en una sola 5090.
5. Contextos efectivos 32K, 128K, 240K y cerca del límite: `prefijo + delta + salida_reservada <= 262144`. Probar deltas 128/512/2K/8K, salidas suficientemente largas y al menos 20 turnos append-only. Separar cold, warm resume y sesión viva.
6. Registrar tokens aceptados por segundo, tokens de reasoning, aceptación del draft, latencia por ronda, máximo/p50/p95, VRAM pico, replays, reintentos, herramientas inválidas y pruebas de código. Usar código real y tareas con dependencias distantes; una aguja sintética no basta.

Objetivo propuesto: decode sostenido superior a 50 tokens/s en tareas representativas cerca de 256K, sin offload de pesos/KV, junto con respuesta útil en menos de 30 s dentro de una envolvente explícita. No se ha logrado ni certificado aquí. Exigir calidad no inferior dentro de un margen acordado, con incertidumbre reportada; ausencia de significación estadística no demuestra equivalencia.

A 50 tokens/s, 1,000 tokens combinados de razonamiento y respuesta consumen 20 s; 4,000 consumen 80 s, sin contar prefill. Por tanto, cumplir decode no basta para cumplir espera útil. El presupuesto de thinking y mantener el prefijo caliente siguen siendo esenciales.
