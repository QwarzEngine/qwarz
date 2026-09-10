# Fidelidad, tamaño del turno y perfil: ejecución aprobada

Configuración de trabajo: EXL3 5 bpw + MTP, K8/V4, una RTX 5090. No modificar el donante, la 3090 Ti, ramas ni commits.

- [x] Reconstruir por hash el prompt problemático, comparar continuación caliente/replay frío con generación greedy y logits iniciales. No llamar corrupción a cualquier diferencia de coma flotante.
- [x] Medir 128/512/2,048/8,192 tokens nuevos sobre prefijos nominales 32K/128K/casi 256K. Cinco ramas por celda, con semilla compartida pero deltas distintos desde el comienzo para evitar reutilización del delta anterior. Guardar tamaños reales, prefill físico, TTFT, contenido final, respuesta completa, calidad y percentiles descriptivos.
- [x] Perfilar una repetición del caso válido más lento con rangos de restauración/asignación, prefill, checkpoints, draft y verificación. Separar completamente la ejecución instrumentada de los percentiles normales.
- [x] Revisión independiente, tests CPU completos, auditoría de artefactos e informe con límites y siguiente optimización respaldada por el perfil.

Decisiones de alcance: las ramas controlan tamaño y mantienen constante el prefijo; no equivalen a conversaciones independientes de agentes multiarchivo. Se usa recuperación de configuración sobre código y nuevos datos de herramienta, con JSON/calculo verificables. Los percentiles de cinco muestras no certifican un SLA. Los contextos nominales 32K/128K son prefijos; el mayor se reduce lo necesario para reservar 8K nuevos y salida, sin exceder nunca 262,144 posiciones.

Instrumentación: no hay `nsys` ni `ncu` instalados; usar PyTorch Profiler CPU/CUDA, conservando trazas y reportando explícitamente si falta actividad CUDA. No instalar herramientas del sistema ni sumar tiempos inclusivos solapados como si fueran un desglose aditivo.
