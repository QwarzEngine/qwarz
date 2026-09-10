# Flash prefill: código, herramientas y cierre de baseline provisional

Fecha: 2026-09-05. RTX 5090 exclusivamente; SGLang en la 3090 Ti permanece
intacto. EXL3 **5 bpw + MTP + K8/V4**, ventana nativa de 262.144 posiciones.

## Decisión

**Flash/8192 acelera el prefill, pero no mejora todo.** Se conserva como
perfil optimizado explícito, con el prefill Triton original disponible como
control/fallback por configuración. No se modificó el servidor donante ni
se cambió el comportamiento predeterminado del launcher.

La evidencia alcanza para cerrar esta etapa de medición y priorizar la
[v1 híbrida para pruebas manuales](../superpowers/plans/2026-09-05-v1-manual-service.md).
No alcanza para certificar calidad general, paridad BF16 o todas las
respuestas completas en menos de 30 segundos. No se requieren más barridos
de kernels antes de implementar ese servicio.

## Protocolo

El [plan predeclarado](2026-09-05-prefill-application-validation-plan.md)
fija dos presupuestos de contexto: 32.768 y 262.144. Ambos perfiles usan
el mismo corpus, tokenizer/template, runtime, sampler recomendado, thinking
medium y semillas. El pool completo está asignado en todos los casos.

- LRU: una carga seed y dos ramas calientes por contexto/perfil; 4.096
  tokens de salida reservados. **12 respuestas**; seis prompts pareados.
- Herramientas: dos sesiones por contexto/perfil, dos ciclos por sesión;
  512 tokens por llamada y 1.536 por respuesta. **8 sesiones, 16 ciclos,
  32 generaciones**. Historial exacto del runtime, sin reparar respuestas.
- Orden secuencial: LRU baseline, LRU Flash, herramientas Flash, herramientas
  baseline. Los tiempos son descriptivos; no un estudio aleatorizado de SLA.
- Inputs LRU efectivos: 28.656 y 258.032. Inputs iniciales de herramientas:
  27.632 y 257.008. Nunca se supera el límite nativo con la reserva de salida.
- TTFT mide el primer token, incluso reasoning; primer contenido final y
  respuesta completa se reportan por separado. No incluyen HTTP ni carga
  inicial del modelo. Preparación del prompt se registra aparte.

## Código: no sólo medir tokens

Se inspeccionaron las once respuestas no truncadas antes de ejecutar sus
módulos en `bwrap`, sin acceso al home, red o GPU, con timeout y límites de
recursos. El código se extrajo sin reparaciones. Se ejecutaron tanto los
unittests generados como los siete grupos independientes del checker LRU
congelado. Una respuesta truncada permanece como fallo, no desaparece.

| Calidad de respuesta completa | Baseline | Flash/8192 |
| --- | ---: | ---: |
| 32K: implementación + tests ejecutables | 1/3 | 3/3 |
| Cerca de 256K: implementación + tests ejecutables | 1/3 | 3/3 |
| Total | **2/6** | **6/6** |

Fallos baseline conservados:

- 32K seed: agotó el presupuesto, 4.091 tokens efectivos, código truncado.
- 32K rama 2: self-test espera una expulsión cuando todavía queda capacidad.
- Largo seed: self-test espera conservar tres entradas en capacidad dos.
- Largo rama 1: implementación enlazada no descuenta el tamaño al expulsar;
  falla también el checker independiente, no sólo su propio test.

Las seis respuestas Flash pasan sus 7–11 tests generados y los siete grupos
externos. **Es una sola tarea con tres muestras por bucket**, no una prueba
de que Flash mejore la precisión del modelo o el coding multiarchivo.

### Latencia LRU en ramas calientes

Medianas descriptivas de **dos** ramas por celda; incluyen fallos de calidad.

| Bucket / métrica | Baseline | Flash/8192 |
| --- | ---: | ---: |
| 32K: TTFT | 186,8 ms | **167,5 ms** |
| 32K: primer contenido final | **2,841 s** | 4,901 s |
| 32K: respuesta completa | **13,891 s** | 16,120 s |
| 32K: decode | 165,3 tok/s | 155,1 tok/s |
| Cerca de 256K: TTFT | 526,3 ms | **326,5 ms** |
| Cerca de 256K: primer contenido final | 6,483 s | **6,035 s** |
| Cerca de 256K: respuesta completa | 32,271 s | **30,396 s** |
| Cerca de 256K: decode | 86,7 tok/s | 84,8 tok/s |

Flash reduce el TTFT largo un **38,0%**, pero la segunda rama larga completa
tarda **32,800 s** frente a 31,748 s del baseline. La primera tarda 27,992 s.
En 32K la respuesta completa empeora un 16,0% en mediana. Cambian longitud,
reasoning y aceptación especulativa, aunque los seis prompts pareados y sus
contadores de prefill físico coinciden entre perfiles.

Los seeds largos LRU reutilizan 28.416 tokens y procesan 229.615: **no son
prefills completamente fríos**. Su TTFT baja de 190,431 a 124,541 segundos.

## Herramientas: mismo agregado, distintas regresiones

La herramienta es un `read_file` restringido a fixtures en memoria. La
comprobación exige la ruta correcta y un JSON final exacto, calculado con
registros distantes y el contenido efectivamente devuelto por la herramienta.
No es un agente general que edita archivos o ejecuta comandos del host.

| Ciclos completos correctos | Baseline | Flash/8192 |
| --- | ---: | ---: |
| 32K | 3/4 | **4/4** |
| Cerca de 256K | **3/4** | 2/4 |
| Total | **6/8** | **6/8** |

Los 16 pasos de llamada inicial usan la herramienta/ruta correcta. Los
fallos están en respuestas que, después de recibir el resultado, vuelven a
emitir `read_file` en lugar del JSON solicitado. No son fallos de sintaxis
de esas llamadas ni respuestas reparadas por el harness.

- Flash mejora el primer ciclo de la sesión 32K con semilla 43.
- Flash **regresa** en el segundo ciclo de la sesión larga con semilla 42:
  razona correctamente que el resultado es 66, pero repite la herramienta;
  el baseline entrega el JSON correcto.
- La primera respuesta larga con semilla 42 falla en ambos. Su prompt SHA
  `ab51961e8911f2dd948a6df1365606faf655920e97cddec3c10ac2b5ae4ff04f`
  coincide además con el caso histórico: aritmética 408 correcta en reasoning,
  seguida de una nueva llamada en vez del JSON. El error no nació con Flash.

Los prompts iniciales sí son iguales entre perfiles; los de respuestas
posteriores pueden diferir por el historial realmente generado. Por eso
estas son comparaciones del sistema completo, no aislamientos numéricos del
kernel ni prueba de que el problema sea exclusivamente el modelo.

### Frío frente a sesión caliente

Las cuatro cargas iniciales largas de herramientas verifican cero tokens
reutilizados y 257.007 tokens físicos de prefill. TTFT:

- Baseline: **198,673–198,711 s**.
- Flash: **131,941–132,001 s**, aproximadamente 33,6% menos.

Por tanto, importar 257K tokens de golpe sigue lejos de 30 segundos.

Espera completa del segundo ciclo, ya caliente:

| Contexto / semilla | Baseline | Flash/8192 |
| --- | ---: | ---: |
| 32K / 42 | 1,903 s, correcto | 2,977 s, correcto |
| 32K / 43 | 2,140 s, correcto | 2,781 s, correcto |
| Cerca de 256K / 42 | 5,337 s, correcto | 3,323 s, **incorrecto** |
| Cerca de 256K / 43 | 5,917 s, correcto | 5,390 s, correcto |

Una respuesta incorrecta más rápida no cuenta como mejora de experiencia.
Decode en pasos largos: baseline 98,4–107,5 tok/s; Flash 91,7–106,2 tok/s.
Se supera 50 tok/s en estos casos, sin garantizar calidad ni longitud final.

## Memoria, exactitud y límites

Pico observado por el allocator de PyTorch en estas pruebas: baseline
**24,550 GiB asignados / 25,715 GiB reservados**; Flash **25,799 / 26,984 GiB**.
No es una medición del pico total del proceso/dispositivo.

La ruta Flash mantiene pesos y K8/V4, no aritmética bit a bit. El
[oracle de atención anterior](2026-09-04-prefill-tuning-results.md) ya mostró
mayor error muestreado frente a FP32: relative L2 0,1561% frente a 0,0449%.
Ese oracle usa el KV cuantizado existente, no el modelo BF16 completo.
Los resultados actuales no permiten declarar ausencia de pérdida numérica.

Tampoco es posible garantizar una respuesta arbitrariamente larga en 30 s
sólo reduciendo TTFT: a unos 85 tok/s, 2.800 tokens de reasoning+código ya
requieren unos 33 s de generación. La v1 debe separar TTFT, contenido final,
presupuesto y cancelación, y mostrar los motivos de terminación.

## Artefactos y reproducción

- `results/20260905-application-lru-{baseline,flash}/`: seis muestras por perfil.
- `results/20260905-application-session-{baseline,flash}/`: cuatro sesiones por perfil.
- `results/20260905-application-grade-{baseline,flash}/`: código, checker, runner,
  hashes, logs aislados y resultados de ambas suites; originales intactos.
- `results/20260905-application-audit.py` y `.json`: reconstrucción de prompts,
  eventos/IDs incluidos los retenidos al EOS, presupuestos y provenance.

Desde la raíz del repo, para LRU elegir un directorio nuevo:

```bash
QWASAR_PROBE_KIND=decode QWASAR_WORKLOAD=lru \
QWASAR_MODEL_PATH=/home/rekeyea/models/Qwen3.8-27B-EXL3-5.0bpw \
QWASAR_DRAFT_METHOD=mtp QWASAR_CACHE_QUANT=8,4 \
QWASAR_CONTEXTS=32768,262144 QWASAR_REPETITIONS=2 \
QWASAR_MAX_NEW_TOKENS=4096 QWASAR_SAMPLER=recommended \
QWASAR_THINKING=medium QWASAR_OUTPUT_DIR=results/NEW_LRU_RUN \
./scripts/run_resident_probe.sh
```

Para el candidato añadir
`QWASAR_PREFILL_SETTINGS=benchmarks/profiles/prefill-flash-k8v4.json`.
Para sesiones cambiar a `QWASAR_WORKLOAD=retrieval_session`, salida 1.536 y
`QWASAR_TOOL_MAX_NEW_TOKENS=512`. No lanzar pruebas GPU simultáneas.

Las cuatro ejecuciones finalizaron con código cero y `completed.json`.
La auditoría independiente reconstruyó **44 generaciones**, verificó los
seis pares LRU y cuatro pares de prompts iniciales de sesiones y cerró con
cero inconsistencias. Integridad aprobada no significa calidad aprobada:
el audit conserva `all_observed_quality_pass: false` y
`acceptance_quality_certified: false`.
Hubo cero requeues; una truncación LRU baseline; ninguna truncación de tools.
El grader baseline devuelve fallo de calidad deliberadamente: 2/6 no se
convierte en aprobado por el marcador de colección. El grader Flash aprueba.

Validación del código: **223 tests CPU pasan**, launcher `bash -n` correcto;
revisión independiente de integración y grader sin bloqueadores. La escritura
de contadores puede ocultar una excepción anterior si también falla el disco;
es un límite menor pendiente, no observado aquí. Los contadores del perfil
son agregados: la distinción frío/caliente usa muestras y eventos individuales.

La carga experimental liberó la 5090. No se creó ningún commit ni se
modificaron los cambios preexistentes del donante. La v1 de servicio todavía
es trabajo pendiente, no un servidor que estas pruebas hayan implementado.
