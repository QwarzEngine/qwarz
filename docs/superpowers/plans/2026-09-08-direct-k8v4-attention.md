# Direct K8/V4 Attention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Execution status (2026-09-08):** investigación cerrada con rechazo medido: mejor Q7 1,324 ms frente a 0,813 ms. Tareas 1–3 completadas; tarea 4 produjo un rechazo. Tareas 5 y métricas integradas de 6 no aplican porque no pasó el gate de velocidad. Informe: `results/20260908-direct-k8v4/report.md`. Los casilleros pendientes conservan explícitamente trabajo no ejecutado; no implican una promoción pendiente.

**Seguimiento (2026-09-11):** no reabrir este plan. La segunda generación (portar Attention64 al ABI, no otro WMMA) está en [`2026-09-11-qwasar-attention-kernel.md`](2026-09-11-qwasar-attention-kernel.md).

**Goal:** Evaluar e integrar únicamente si gana una atención especializada que lea K8/V4 sin espejo completo.

**Architecture:** Lector por bloques en el dominio Hadamard, cálculo FP16 y acumulación/softmax FP32. Mantener append, paginación y MTP6; validar primero el lector, después el kernel y finalmente el slot del bloque fusionado. El backend actual es referencia y fallback por forma.

**Tech Stack:** Python3.12, Torch2.14.0+cu130, ExLlamaV3 local, CUDA13 / SM120, FlashInfer0.6.18 fijado y Triton de la base. CUDA C++/CuTe sólo donde el estudio del pipeline justifique su uso.

**Spec:** `docs/superpowers/specs/2026-09-08-direct-k8v4-attention-design.md`.

## Global Constraints

GPU0 RTX5090 / SM120 exclusivamente; GPU1 no se usa.

Batch 1, Q24/KV4/D256, páginas de 256 tokens, contexto nativo 262144.

MTP fijo 6; cachés target y draft K8/V4; Minima64 MLP y FP8 PRIMS de ingesta según router existente.

No sincronización `.item()`, asignaciones, compilaciones ni búsqueda de configuración en el camino caliente.

Presupuesto adicional inicial de buffers del backend: 64 MiB compartido entre capas secuenciales. Sin espejo completo.

Este documento planifica investigación e integración condicional; no declara un kernel ya escrito ni resultados futuros. No se cambia producción durante microbenchmarks. Usar el gestor existente para stop/restauración del servicio. El repositorio no tiene commit inicial: preservar fuentes, hashes y parches en un directorio aislado; no crear un commit global de todos los archivos sin relación.

## Mapa de archivos previstos

Crear durante ejecución en `results/20260908-direct-k8v4/`:

- `contract.json`: formas, versiones, hashes y ABI verificables.
- `loader.cuh`, `loader_gate.py`: lector y comprobación aislada contra reconstrucción independiente.
- `attention.cu`, `backend.py`: kernel especializado, preparación de buffers y llamada sin append.
- `attention_gate.py`: oráculos, máscaras, páginas y mutaciones de grafos.
- `bench.py`, `selection.json`: medición y perfil seleccionado por forma.
- `bc_adapter.py`, `donor-slot.patch`: conexión explícita del slot, si requiere un cambio del donante.
- `model_probe.py`, `report.md`: comparación integrada y decisión final.

Reusar sin sobrescribir `results/20260908-combined-profile/environment.py`, su gestor, base runner y fuentes de referencia. Copiar el gestor al nuevo directorio preservando su raíz relativa; guardar snapshot del gestor original. No instalar un fork experimental sobre el donante usado por producción.

Si gana el modelo completo, trasladar el código validado a `src/qwasar_runtime/attention/` con un selector explícito y manifiesto; hasta entonces todo permanece experimental. Esta promoción es una tarea posterior con revisión del diff concreto.

### Task 1: Congelar contrato y elegir punto de adaptación

**Files:** Crear `contract.json` y una sección de decisión en `report.md`. Leer `triton_paged.py`, `bc_attn.py`, `exllamav3_ext/libtorch/attention.cpp` y XQA `mha.cu` de las versiones locales fijadas.

- [x] Registrar SHA256 del donante y de XQA, licencia/atribución de cualquier fragmento reusado, versiones y arquitectura de compilación. No importar ExLlama fuera de una ventana GPU propia.
- [x] Transcribir el ABI y layout de parciales de la spec, y comprobar índices3/4/11/12/13 del split e índice4 de combine contra C++.
- [x] Mapear las cargas de K/V de XQA, tipos, etapas y consumo por MMA. Registrar si se pueden sustituir sin modificar scheduler, máscaras y reducción. Elegir adaptación de XQA sólo si ese mapa es concreto; en caso contrario implementar un kernel CUDA/CuTe reducido con el ABI split/combine. No portar todo el runtime XQA por defecto.
- [x] Registrar formato de código y escala de K/V, orden de redondeos y Hadamard. Usar el código existente como referencia, no como oráculo único.
- [x] Ejecutar el baseline vigente en una ventana gestionada antes de tocar el algoritmo; archivar configuración y restauración exacta.

**Entregable:** contrato revisable y decisión fundada sobre el punto de adaptación. Si el ABI requiere modificación, describir el diff del slot antes de escribir el kernel.

### Task 2: Lector correcto de K8/V4

**Files:** `loader.cuh`, `loader_gate.py`.

**Interface:** el lector recibe punteros a palabras y escalas, índice físico del token, cabeza KV y bloque de dimensiones; produce un tile FP16 en dominio rotado. No tiene acceso de escritura al caché.

- [x] Crear fixtures con todos los códigos K8 (0..255) y V4 (0..15), escalas0/pequeña/1/grande finita y posiciones alrededor de límites de grupos32 y páginas256.
- [x] Comparar el desempaquetado contra el layout empaquetado del donante y la ecuación independiente siguiente, verificando K transpuesta y V normal por separado:

```python
midpoint = 2 ** (bits - 1) - 0.5
expected_rotated = ((codes.float() - midpoint) *
                    (scales.float() / 2 ** (bits - 1))).half()
assert torch.equal(actual_rotated, expected_rotated)
```

- [x] Incluir tablas no contiguas y permutadas, última página parcial y memoria canaria detrás del rango válido. Mantener iguales datos lógicos al permutar físicamente el almacenamiento.
- [x] Implementar sólo las cargas y conversión necesarias para8 y4 bits; exigir resultados finitos donde la referencia sea finita y coincidencia exacta del lector antes de combinarlo con MMA.
- [x] Correr Compute Sanitizer sobre fixtures pequeños dentro de la ventana GPU. Conservar el error exacto si falla; no pasar a atención con un lector ambiguo.

**Entregable:** lector sin errores de dirección, escala o redondeo; sin scratch de contexto completo.

### Task 3: Atención especializada y oráculo

**Files:** `attention.cu`, `backend.py`, `attention_gate.py`.

**Interface:** `prepare(**decode_kwargs)` prepara buffers/configuración fuera de captura; `run(**decode_kwargs)` consume el caché ya actualizado, respeta `out` y no hace append. Los kwargs se corresponden con la llamada de decode actual; sólo Q1..8, causal completo, K8/V4 y geometría fija se aceptan.

- [x] Implementar inicialmente Q7 con bloques KV64, aritmética FP16, Q rotada y combinación final inversa; conservar softmax y parciales FP32. No introducir FP8.
- [x] Hacer explícito el reparto de filas entre cabezas GQA y consultas. Manejar fila causal completamente enmascarada y split inactivo sin NaN ni lectura de parciales de otra llamada.
- [x] Extender Q1..6 y Q8 después de pasar Q7; formas fuera del contrato usan el backend actual antes de captura.
- [x] Comparar las mismas consultas con `sampled_attention` FP32 y con el decode directo. No comparar sólo contra un oráculo que comparte el nuevo lector.
- [x] Matriz pequeña completa: longitudes7/31/32/255/256/257/4093, Q<=longitud, tablas permutadas y restauración de punteros. Matriz larga: Q1/4/7 a32K/128K/258K con capturas de capas temprana, media y tardía. Capturas recortadas se etiquetan sintéticas; obtener capturas reales Minima64 para integración.
- [x] Exigir error relativo L2<=0.003 frente a FP32 en cada caso largo; registrar máximo por cabeza/fila y error absoluto, sin esconder outliers en un promedio global. Investigar cualquier caso que empeore claramente frente al baseline aun si pasa el umbral.
- [x] Repetir con el mismo grafo tras mutar longitud a4093 y cruzar páginas, sustituir tabla por otra válida, restaurar y comparar con eager. Exigir replay/eager<=1e-5 y una salida de control vieja distinguible. Comprobar KV/escalas byte a byte.

**Entregable:** atención correcta y capturable. La comparación usa el KV ya cuantizado; no afirma identidad del modelo BF16.

### Task 4: Medir y optimizar sólo cuellos observados

**Files:** `bench.py`, `selection.json`, `report.md`.

- [x] Calentar las formas antes de medir. Usar CUDA events con al menos5 rondas de20 replays e intercalar baseline/candidato en orden ABBA; excluir oráculos y compilación. Registrar temperatura, clocks, memoria y variación, sin cambiar clocks del equipo automáticamente.
- [ ] Medir split, combine y ruta total; contrastar eager/grafo y memoria máxima. La decisión usa total, no sólo MMA ni loader.
- [x] Perfilar Q7/256K con Nsight: DRAM, L2, instrucciones de desempaquetado, Tensor Core, ocupación, registros, spills y barreras. El baseline también tiene perfil con la misma forma.
- [ ] Si domina latencia de carga y queda margen de recursos, probar doble buffer compartido. Si dominan registros o padding, probar repartos de filas GQA/Q. Barrer bloques KV32/64/128 y splits16/32/64/128 únicamente donde sean legales; descartar por presupuesto de memoria antes de ejecutar.
- [x] Congelar la mejor configuración por Q y cubo de contexto; no autotuning en caliente. Revalidar numéricamente cada ganador con la Task3.
- [x] Continuar a integración sólo con mejora repetible>=15% en tiempo total Q7/256K. Q1 debe quedar dentro de3% del baseline o conservar explícitamente su ruta original. Si ninguna variante gana, cerrar con evidencia y no modificar el runtime.

**Entregable:** kernel ganador o rechazo medido. El objetivo0.351ms de XQA FP8 aislado no se usa como garantía ni umbral de calidad.


**Notas de ejecución:** barrido completo de bloques/splits indicado, oráculo y replay/eager numéricos, tiempos ABBA de grafos y split/combine Nsight realizados. No se midió latencia eager por separado ni pico global de VRAM: se registraron buffers del backend y telemetría. Se optimizaron lector, softmax y acumuladores; no se implementó doble buffer asíncrono. Ninguna configuración ganó; no se activó router.

### Task 5: Conservar el bloque fusionado y el estado MTP

**Files:** `bc_adapter.py`, `donor-slot.patch`, `model_probe.py`.

- [ ] Conectar primero el slot experimental al mismo algoritmo original; comparar baseline original contra baseline por el slot nuevo. Cualquier cambio de tiempo o salida de esa conexión debe resolverse antes de activar el kernel.
- [ ] Cargar handles y buffers fuera de captura. Preservar grids, ABI, argumentos parcheados y orden de nodos. Si el kernel exige otro layout de parciales, cambiar split/combine y reserva juntos en el slot experimental; nunca reinterpretar silenciosamente los buffers viejos.
- [ ] Insertar el kernel únicamente entre append cuantizado y salida de atención. Mantener proyecciones/RoPE, append único, gate y proyección de salida en el grafo.
- [ ] Probar aceptación MTP0..6, rechazo completo, rechazo parcial, transición entre widths soportados, cruce de página, prefijo cacheado y reset. Comparar longitudes comprometidas, historial GDN/convolución, tokens y logits con la ruta original; la prueba de lectura aislada no sustituye estos tests de estado.
- [ ] Correr pares completos a128K y256K con las mismas entradas, semillas y presupuestos. Baseline y candidato usan Minima64 + FP8 PRIMS y el mismo draft. Empezar sólo por target; el draft continúa original para aislar la variable.

**Entregable:** comparación del modelo completo que cambia exclusivamente la atención target seleccionada.


**No ejecutada:** integración condicionada a una mejora que no se obtuvo. No existen bc_adapter.py, donor-slot.patch ni pruebas integradas MTP de este candidato.

### Task 6: Decisión y cierre

**Files:** `report.md`, manifiesto de fuentes y configuración final.

- [ ] Medir tokens realmente aceptados/s, aceptación MTP, TTFT Q128 mediana/p95, prefill grande y memoria. Mantener warmups fuera de la muestra.
- [ ] Correr las6 variantes LRU previas y JSON exacto con el presupuesto original; añadir fixtures de recuperación larga y operaciones de prefijo/rewind. Registrar cada mejora/regresión; no aumentar max_tokens para evitar truncado.
- [ ] Adoptar sólo si logra+15% tokens aceptados/s a256K, TTFT Q128 p50<=300ms, prefill grande sin regresión>3% y calidad emparejada sin empeorar. El usuario permite menos de6/6; el criterio no impone6/6 universal, pero tampoco acepta esconder una regresión individual con un total igual.
- [ ] Si gana sólo Q7 o un cubo de contexto, proponer un router estático para ese dominio y volver a validar esa combinación exacta. Si falla calidad, no promover por velocidad.
- [x] Revisar diff, licencias, fuentes compiladas y rutas activadas; archivar resultados y restaurar el servicio original. La promoción a `src/qwasar_runtime/attention/` se presenta como cambio concreto posterior, sin mezclarla con nuevos formatos o cambios de MTP.

**Entregable:** decisión sustentada y reproducible. FP8 interno es un experimento posterior; no se mezcla con esta primera comparación FP16.

**Cierre por rechazo:** no se corrieron métricas/calidad end-to-end; se preservaron fuentes, resultados y servicio original. El runtime original permanece activo.
