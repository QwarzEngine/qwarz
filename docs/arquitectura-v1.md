# Qwasar v1: la combinación que usamos y por qué funciona

Actualizado: 2026-09-11.

## Resumen

Qwasar v1 es un **motor híbrido especializado** en Qwen3.8-27B para una
RTX 5090 y un único trabajo de generación concurrente. No es una mezcla de
servidores SGLang, vLLM, llama.cpp y ExLlama: el backend elegido es
**ExLlamaV3**, con supervisión, persistencia e integración propias.

| Capa | Elección | Aporte |
| --- | --- | --- |
| Pesos | EXL3 **5 bpw** + **NVIDIA64 NVFP4** en 192 MLP | El artefacto EXL3 sigue pinado; gate/up/down de las 64 capas salen del donante NVIDIA. |
| Decode | **ExLlamaV3 + MTP6** y **Attention64** | Seis propuestas fijas; decode Triton con `block_n=64`. |
| Caché | **K8/V4** | Reduce el tamaño de claves y valores para alojar la ventana larga. |
| Prefill | **Flash/8192** + **FP8 PRIMS** (P×256, Q≥8192) | PRIMS cubre los chunks grandes; el resto sigue en Flash. `baseline` es Triton. |
| Continuidad | Historial con **IDs exactos** y reutilización del prefijo | Evita volver a procesar el historial completo en cada turno compatible. |
| Visión | Torre **BF16 del propio artefacto** + `MMEmbedding` | Imágenes inline en mensajes de usuario; ~0,9 GB extra; tokens ≈ píxeles/1024. |
| Servicio | **Rust + Axum + SQLite WAL** | HTTP, admisión, persistencia (incluidas imágenes por hash), streaming y supervisión del worker. |
| GPU | Worker **Python/ExLlamaV3** en GPU 0 | Un solo propietario del generador y del estado de inferencia. |
| Cliente | **Pi**, mediante Chat Completions | Interacción, sesiones y ejecución local de herramientas. |

## Cómo se conectan las piezas

```text
Pi: conversación y ejecución de herramientas
                  |
        HTTP /v1/chat/completions
                  |
       Rust + Axum <----> SQLite WAL
                  |
           IPC JSONL local
                  |
        Python + ExLlamaV3
                  |
      RTX 5090: modelo + estado GPU
```

Rust recibe la petición y busca el historial durable correspondiente. El worker
valida herramientas, renderiza el prompt con el tokenizer/template del artefacto,
comprueba el presupuesto de contexto y ejecuta la generación. Los eventos de
reasoning y texto vuelven a Pi por streaming. Las llamadas a herramientas se
entregan cuando sus argumentos están completos y validados.

**Pi ejecuta las herramientas, no Qwasar.** Los resultados vuelven como nuevos
mensajes para continuar la inferencia. También existe un adaptador Responses
acotado sobre el mismo núcleo, no un segundo motor o una segunda caché.

## Son tres decisiones distintas de cuantización y generación

### 1. Pesos EXL3 de 5 bpw

El modelo principal utiliza el artefacto seleccionado como **EXL3 5 bpw**.
No significa que absolutamente todos sus componentes estén configurados con
cinco bits. El `config.json` del artefacto declara:

```json
{
  "quant_method": "exl3",
  "bits": 5.0,
  "head_bits": 6,
  "mtp_bits": 4
}
```

Por tanto, la configuración nominal es **5 bpw para el modelo principal,
cabeza a 6 bits y MTP a 4 bits**. En `flash`, las 192 matrices MLP
(gate/up/down de las capas 0–63) se sustituyen por el donante NVIDIA NVFP4
pinado en `benchmarks/manifests/nvidia-qwen38-27b-nvfp4.json`. No estamos
ejecutando el antiguo artefacto de 3,5 bpw. Cambiar el modelo o el donante
exige actualizar y validar el pin; no basta con apuntar a cualquier carpeta
cuyo nombre diga «5 bpw».

### 2. Caché K8/V4

**K8/V4 describe la caché de atención, no los pesos del modelo**: claves a
8 bits y valores a 4 bits. Es una decisión separada que reduce la memoria del
contexto. El presupuesto nativo de **262.144 posiciones** incluye prompt,
salida máxima reservada y espacio de especulación; no se permite truncamiento
silencioso para hacer entrar una petición demasiado grande.

### 3. MTP

MTP es el mecanismo de propuesta especulativa: propone varios tokens y el modelo
principal verifica cuáles acepta. Su beneficio depende de la tasa de aceptación
y del trabajo generado; no debe interpretarse como una mejora automática de
precisión. La configuración activa utiliza seis propuestas fijas, pasadas al
cargador y al generador para reservar también el historial recurrente necesario.
Se adoptó después del [barrido MTP](benchmarks/2026-09-07-mtp-fixed-width-results.md),
por decisión del usuario; las limitaciones de calidad de ese screening permanecen documentadas.

## Por qué el contexto largo puede seguir siendo interactivo

Hay que distinguir dos operaciones:

- **Ingesta fría:** procesar por primera vez cientos de miles de tokens. Flash
  acelera esta operación, pero no elimina su coste.
- **Continuación caliente:** aprovechar el prefijo que ya está en GPU y procesar
  únicamente la parte nueva necesaria para continuar.

La pieza decisiva no es sólo un kernel más rápido: es **no rehacer el prefill
del historial completo en cada turno**. Para ello conservamos los IDs exactos
generados, incluido el terminador nativo real, en lugar de decodificar y volver
a tokenizar las respuestas anteriores.

En una muestra HTTP, la continuación tenía **254.098 tokens de entrada**:

| Medición | Resultado |
| --- | ---: |
| Tokens físicos reutilizados | 253.952 |
| Prefill físico nuevo | 145 tokens |
| Primer contenido HTTP | **0,593 s** |
| Decode posterior a la primera tanda | **97,19 tokens/s** |
| Respuesta completa | **10,412 s**, 950 tokens de salida contabilizados |
| Primer contenido de la ingesta fría previa | **129,208 s** |

El contador físico de prefill excluye la posición que se procesa para arrancar
el decode; por eso no es simplemente entrada menos caché. Esta muestra usó
**thinking off y greedy**, corpus fuente no repetido y ninguna herramienta.
No representa un percentil ni garantiza esos tiempos con thinking medium,
ingestas grandes, cualquier tarea o cualquier historial.

Cambiar instrucciones, schemas de herramientas, compactar el historial o perder
la caché puede exigir nuevo prefill. Pi reserva por defecto 16.384 tokens para
su compaction, por lo que no intenta llenar todas las posiciones nativas con
historial. No se garantiza que cualquier respuesta termine antes de 30 segundos.

## Persistencia y fiabilidad

SQLite conserva solicitudes, estados, respuestas y segmentos de tokens exactos.
El estado terminal se confirma antes de publicar una respuesta completada como
padre utilizable. Respuestas fallidas, canceladas o incompletas no son padres
válidos. La admisión permite una generación activa; otra recibe un conflicto en
lugar de quedar en una cola invisible.

**Persistencia en disco no significa persistencia de la caché GPU.** Después
de reiniciar se puede reconstruir el historial correcto, pero hay que volver a
hacer prefill. La cancelación descarta estado físico incierto para no reutilizar
una caché o estado recurrente de una generación interrumpida.

La compatibilidad de herramientas incluye los schemas `patternProperties` de
la extensión MCPorter de Pi. Se validan sus restricciones, no se eliminan para
evitar el error. Un transporte válido tampoco garantiza que el modelo elija
siempre la herramienta o los argumentos correctos.

## Qué no forma parte de esta v1

- No hay un megakernel monolítico propio ni un runtime nativo C++/CUDA completo.
- NVFP4 entra sólo en las 192 MLP del donante NVIDIA. GDN, atención, embeddings,
  `lm_head`, MTP, la torre de visión y la caché K8/V4 siguen en EXL3. XQA / KV
  NVFP4 no están en producción.
- Vídeo y URLs remotas de imágenes no se aceptan; la calidad visual sobre
  5 bpw + NVFP4 no está certificada.
- SGLang, vLLM y llama.cpp no están en la ruta de ejecución de Qwasar.
- No hay copias físicas persistentes de KV en RAM/disco para restaurar la GPU
  instantáneamente después de un reinicio.
- Flash no tiene certificación de paridad BF16 ni de superioridad general en
  calidad. Conservamos el prefill original como alternativa explícita.

## Verificación de los pesos residentes en la 5090

Comprobación realizada el **2026-09-06 a las 00:38, America/Montevideo**, sin
reiniciar ni alterar el trabajo del servicio:

| Evidencia | Valor observado |
| --- | --- |
| GPU del proceso | GPU 0, NVIDIA GeForce RTX 5090 |
| UUID | `GPU-e0509411-1220-82f4-3378-5d65e0c5232a` |
| PID del worker en GPU y en `/config` | **484424**, coincidente |
| Entorno del worker | `CUDA_VISIBLE_DEVICES=0` |
| Artefacto de su línea de comandos y configuración activa | `/home/rekeyea/models/Qwen3.8-27B-EXL3-5.0bpw` |
| Estado | `ready`, `fake=false` |
| Configuración del artefacto | `bits=5.0`, `head_bits=6`, `mtp_bits=4` |
| Memoria GPU atribuida al proceso | 28.310 MiB; incluye cachés y buffers, no sólo pesos |

Se recalculó el **hash del árbol completo del artefacto**, incluidos sus pesos,
y coincidió tanto con el manifiesto congelado como con el hash registrado por
el worker al cargar:

```text
0b9a439ffefa45c55a2a3cb0324de9fe1b23bce69a37a399cb952d355f02d92e
```

El backend comprueba ese hash y `bits=5.0` **antes de cargar el modelo**, y no
publica `ready` hasta completar la inicialización. Se verificó además que el PID
y su identidad de arranque permanecieron iguales durante la comprobación.

**Conclusión: el worker residente en la RTX 5090 corresponde al artefacto
EXL3 de 5 bpw fijado para Qwasar, no al de 3,5 bpw.** Esta es una verificación
operativa de proceso, dispositivo, ruta de carga, identidad e integridad del
artefacto; no un volcado y comparación byte a byte de tensores dentro de VRAM.
El consumo de memoria por sí solo no permite deducir los bpw. El proceso ya no
conserva mappings de los archivos de pesos en `/proc/PID/maps`.

Evidencia detallada local: `results/model-residency/2026-09-06.json`.
La RTX 3090 Ti mantiene su proceso SGLang separado, PID 3547; no se intervino.
PIDs y cifras de memoria describen esta comprobación, no valores permanentes.

## Referencias del proyecto

- [Manual de uso con Pi](manual-v1.md).
- [Verificación real de servicio y Pi](benchmarks/2026-09-05-v1-pi-acceptance.md).
- [Resultados de prefill y sus límites](benchmarks/2026-09-05-prefill-application-results.md).
- [Manifiesto del artefacto](../benchmarks/manifests/qwen38-27b-rtx5090-v1.json).
- [Identidad, carga y configuración del worker](../src/qwasar_runtime/engine.py).
