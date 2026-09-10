# Prefill exacto: investigación aprobada

Configuración fija: EXL3 5 bpw + MTP, K8/V4, GPU 0 RTX 5090. GPU 1 y donante intactos; sin commits ni ramas. Se usa ejecución con subtareas independientes y tests antes de implementación.

- [x] Capturar Q/K/V y metadatos de una atención real sobre el historial largo; conservar prompt y hashes.
- [x] Comparar baseline, staging/directo, bloques, warps, stages y splits con compilación fuera del tiempo, repeticiones y controles numéricos.
- [x] Refinar candidatos y comprobar longitudes de consulta/contexto; no afirmar optimalidad global de un barrido finito.
- [x] Validar candidatos ganadores con el modelo completo, prefill físico comparable, calidad y memoria; evaluar tamaño de chunk si es seguro.
- [x] Repetir matriz 32K/128K/casi 256K con la opción seleccionada, auditar, revisar y documentar opciones rechazadas y mejoras reales.

Ruling: se trabaja en el repo qwasar existente y sin worktree/commits porque es un repo sin primer commit y el usuario pidió preservar ese estado. La propuesta aprobada en chat es la especificación. No se instala tooling del sistema ni se modifica el paquete donante: variantes mediante controles reversibles dentro del proceso de prueba.

Interfaces: captura produce diccionario torch con kwargs CPU de paged_attn_triton_prefill; microbenchmark lo consume sin modelo. Integración consume el mismo formato de candidatos y conserva el baseline al salir de cada contexto. El benchmark aislado sólo selecciona candidatos; calidad final depende de pruebas de modelo completo.

Revisión: corregidos rechazo de chunks no alineados, selección que antes podía nombrar ganador a una regresión, warmups separados del screen, bloqueo de block_n ignorado en modo directo y requisito físico de captura caliente. La captura original ya realizada verifica 252,160 reutilizados y 8,305 de prefill, sin requeue; el endurecimiento posterior no cambia sus artefactos.

Ruling: incluir la implementación Flash nativa ya instalada en PyTorch, además del tuning Triton. Usa el mismo staging K8/V4 y causalidad lower-right; se prohíbe fallback a atención densa/math. No instala flash-attn ni altera precisión de pesos. Permite comprobar si un kernel existente supera la mejor configuración del donante.

Resultado: Flash + chunk 8192 es la mejor opción medida; 80/80 respuestas del screen y 60/60 de la matriz final pasan. Reducción de TTFT cercana a 40% en +8K/casi 256K. Oráculo muestreado FP32 detecta mayor error numérico de Flash (L2 relativa 0.1561% frente a 0.0449% del baseline); queda documentado como ruta experimental opt-in y no como paridad general de coding. 175 tests CPU pasan. Informe: `docs/benchmarks/2026-09-04-prefill-tuning-results.md`.
