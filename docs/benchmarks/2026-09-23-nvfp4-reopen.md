# Reevaluación selectiva de NVFP4 con visión y el stack actual

**Resultado:** no se recomienda promover este candidato. Atención NVFP4 reduce
TTFT frío un 5–10% en el piloto largo, pero no acelera decode y entrega
**2/6 tareas de código correctas frente a 5/6 del control**. Las entradas GDN
ahorran más prefill a costa de una caída clara de decode. Relajar calidad no
convierte estas implementaciones en una mejora sustancial global.

## Contrato de la campaña

El usuario permite estudiar una pérdida de calidad a cambio de una mejora
sustancial, pero decide el compromiso **después de ver resultados**. No hay
promoción automática ni un presupuesto de degradación preaprobado. Se conservan
256K, tools, visión y los contratos de estado. La RTX 3090 Ti no participa.

Control: cargador real `ExLlamaBackend(prefill="xqa")`, NVIDIA64 MLP, XQA,
KV target NVFP4, KV draft K8/V4, PRIMS, MTP6, rendezvous y hot64k. La torre
visual BF16 se carga antes del head, igual que en producción. Los candidatos
conservan los MLP NVIDIA y añaden proyecciones Minima calibradas; no se
recuantizan pesos EXL3 ni FP8.

SHA256 completos de Minima y los tres shards NVIDIA comprobados, no sólo tamaños.
Evidencia: `results/20260923-nvfp4-reopen/preflight/verified-donors.json`.
El control conserva el artefacto EXL3
`0b9a439ffefa45c55a2a3cb0324de9fe1b23bce69a37a399cb952d355f02d92e`.

## Variantes

| Perfil | Proyecciones adicionales NVFP4 | Conserva |
| --- | ---: | --- |
| control | 0 | Stack promovido completo |
| attention | 64 q/k/v/o | GDN, MTP, embeddings, head y visión |
| gdn_input | 96 qkv/z | Atención, GDN out/a/b y demás componentes |
| gdn_output | 48 out | Atención, GDN qkv/z/a/b y demás componentes |

Los perfiles `gdn_all` y `all` están implementados para futuros ensayos
condicionados, no se consideran medidos por existir en el adaptador.
Los estados recurrentes siguen siendo FP32.

## Validación numérica y protocolo

- 27 matrices representativas de capas tempranas, medias y finales; M=1/7/128/2048:
  **108 comprobaciones** contra FP32 con pesos decodificados directamente del
  checkpoint y activaciones sin cuantizar. RMS relativo máximo 0,17126.
- El límite 0,25 sólo discrimina errores mecánicos graves; **no es una puerta
  de calidad ni equivalencia con BF16**.
- 27 controles rojos alteran la escala de pesos por 1000 y todos son detectados.
- Tests CPU de cobertura exacta, exclusiones, escalas, padding, composición y
  restauración del hook, oráculos de tareas y calificación.
- Cribado: 42 muestras, 4K/32K, tres semillas por proceso y contexto.
  Orden control/attention/attention/control/gdn_input/gdn_output/control.
  Cada proceso carga desde cero. Calentamiento fuera de medida.
- El cribado usa tokens archivados para comparar implementación, **no calidad
  fuera de muestra**. Su salida tiene presupuesto 512 y queda truncada; no mide
  tiempo hasta una respuesta correcta.
- El piloto separado usa tres nuevas familias de código, tools con argumentos
  estructurados, tres imágenes sintéticas de 1920×1080 y recuperación larga.
  Orden control-s193/attention-s193/attention-s827/control-s827.
  Son pocas familias y semillas, no una certificación general.

Las latencias se miden sin profiler. El colector inicial `block-paths.json`
recorrió sólo módulos superiores y quedó vacío; se corrigió para el piloto,
sin cambiar los kernels ni reescribir las mediciones del cribado.

### Límites de las métricas

- Los porcentajes del cribado describen un único prompt recortado a dos
  longitudes, con varias semillas; no representan una mezcla de producción.
- `Engine` puede cerrar el razonamiento por presupuesto y abrir un segundo
  job. En esas respuestas, sus contadores de caché/aceptación pertenecen al
  último job, mientras la latencia cubre la petición completa. No se usa esa
  aceptación como si describiera todo el ciclo; un `cached_tokens` mayor que
  el prompt original no es reutilización de una petición anterior.
- Las continuaciones tienen prefijos generados distintos por brazo.
  Se registran tokens físicos reutilizados y nuevos; no son microbenchmarks
  con trabajo exactamente idéntico.
- Las imágenes son un smoke funcional sintético, no OCR, comprensión de
  documentos ni un ensayo del máximo de 16 imágenes en una sola petición.
- No se ejecuta una petición conjunta de imagen y 256K de texto. Sí se
  conserva la torre y la caché de embeddings visuales durante las pruebas
  largas del mismo proceso.
- Un límite de confianza descriptivo con tan pocas familias no demuestra
  no inferioridad. Si el bootstrap degenera, se informa como no estimable,
  no como incertidumbre cero.

## Cribado: resultados

Atención, medianas de seis muestras por brazo/contexto en el A/B/B/A:

| Contexto | TTFT control | TTFT atención NVFP4 | Decode control | Decode atención NVFP4 |
| --- | ---: | ---: | ---: | ---: |
| 4K | 620,7 ms | 582,4 ms (−6,2%) | 219,8 tok/s | 225,8 tok/s (+2,7%) |
| 32K | 5004,1 ms | 4622,8 ms (−7,6%) | 216,9 tok/s | 210,7 tok/s (−2,9%) |

El ahorro de prefill se reproduce entre procesos. El decode es variable:
las medianas a 32K de las dos cargas del candidato son 212,7 y 181,0 tok/s.
No se interpreta una diferencia pequeña de decode como ganancia estable.
La reducción de pico asignado es aproximadamente 0,077 GiB.

GDN, tres muestras por perfil/contexto frente a los controles que rodean
esos dos perfiles (procesos 03 y 06):

| Perfil | Δ TTFT 4K / 32K | Δ decode 4K / 32K |
| --- | ---: | ---: |
| Entradas qkv/z | −10,3% / −18,1% | −27,6% / −25,8% |
| Salidas out | −7,5% / −7,0% | +3,0% / −5,5% |

No se amplía a `gdn_all` ni a `all`. Las entradas presentan un intercambio
prefill/decode poco atractivo para el uso interactivo; las salidas no muestran
una ganancia grande ni uniforme. **Esto descarta estas implementaciones, no
el formato NVFP4 en general.**

El código instalado de `GatedDeltaNet.load_local` exige qkv/z/out EXL3 y a/b
FP16 para construir `BC_GatedDeltaNetSplit`. Cualquiera de los grafts GDN
desactiva esa ruta. El piloto confirma los 48 BC activos en control/attention;
el cribado GDN no registró su inventario dinámico por el fallo del colector
descrito arriba. No se atribuye toda la regresión al despacho sin un perfil
causal adicional.

La cuantización del candidato también cambia de receta/donante: estas cifras
no describen cualquier checkpoint NVFP4 ni una cuantización propia optimizada.

## Nota metodológica del piloto de recuperación

Antes de evaluar el primer candidato, el control s193 a 32K y 128K recuperó los
tres números y la suma correctamente, pero devolvió claves `AUDIT_MARKER_0`,
`AUDIT_MARKER_1`, `AUDIT_MARKER_2`, `sum`. El grader esperaba `values` como lista
y `sum`. La consulta dice “JSON with values (AUDIT_MARKER_0, _1, _2 in that order)
and their sum”, sin nombrar inequívocamente el campo `values`.

No se cambian prompts durante el A/B ni se borran fallos. Se conservará el
resultado estricto y se añadirá una puntuación semántica simétrica para
recuperación/continuación, que acepta únicamente esas dos representaciones
con los mismos tres valores y suma exactos. No es lícito interpretar este
desajuste de forma como fallo de recuperación o mejora de la cuantización.

## Reproducción y aislamiento

Entradas experimentales, sin cambios de defaults:

```sh
PYTHONPATH=src python3 -m qwasar_bench.nvfp4_study preflight --output NUEVO/preflight
# Sólo dentro de una ventana GPU gestionada:
python -m qwasar_bench.nvfp4_study oracle --output NUEVO/oracle
python -m qwasar_bench.nvfp4_study screen --output NUEVO/screen
python -m qwasar_bench.nvfp4_followup --output NUEVO/quality
# Después de liberar la GPU; el código generado corre sólo en bubblewrap:
python -m qwasar_bench.nvfp4_quality grade --input BRAZO --output NUEVO/grades/BRAZO
```

El Python GPU es el entorno donante de ExLlamaV3. Las tres ventanas usan
`results/20260908-hybrid-backends/managed.py` con etiquetas
`nvfp4-reopen-{oracle,screen,quality}-20260923`. Comprueban servicio libre,
paran de forma controlada y restauran la configuración exacta en `finally`.
Cada salida se crea con exclusión de nombres existentes.

## Piloto A/B/B/A: resultado final

60 respuestas, 30 por brazo: tres familias nuevas de código, tres llamadas a
herramienta, tres imágenes, tres consultas largas y sus tres continuaciones,
cada una con dos semillas. Todos los jobs terminan con estado `completed`;
eso **no equivale a que la respuesta sea correcta**.

### Calidad

| Categoría | Control | Atención NVFP4 | Interpretación |
| --- | ---: | ---: | --- |
| Código completo + tests + oráculo | **5/6** | **2/6** | Tres regresiones emparejadas, ninguna mejora |
| Visión, conteo de figuras | 6/6 | 6/6 | Smoke sintético, no calidad visual general |
| Recuperación, contenido semántico | 6/6 | 6/6 | Tres marcadores a distintas distancias y suma |
| Continuación, contenido semántico | 6/6 | 6/6 | Reutilización física registrada |
| Tools, argumentos exactos | 0/6 | 0/6 | Problema compartido de `string|null`, ver abajo |

Código del candidato:
- `interval-s193` no fusiona intervalos contiguos; fallan su test y el oráculo.
- `bimap-s193` pasa el oráculo, pero un test generado dispara una excepción
  esperada sin capturarla.
- `interval-s827` tiene un error de sintaxis.
- `topology-s827` termina sin código visible.

El único fallo de código del control es un test de intervalos con expectativa
incorrecta; su implementación pasa el oráculo. No se corrige ninguna entrega.
Tres de las cuatro entregas fallidas del candidato pasan por cierre de
razonamiento por presupuesto, igual que algunas del control. Se evalúa la
política real del servicio, no generación ilimitada.

La puntuación estricta de recuperación/continuación es 0/6 en cada categoría
y brazo, por el desajuste de campos descrito antes. Se conserva, junto a la
puntuación semántica, en `comparison.json`.

**Tools:** todos los brazos eligen `record` y preservan nombre, booleano e IDs,
pero `note: null` llega como `note: "null"`. El parser existente
`parse_tool_parameter` devuelve texto cuando cualquier tipo permitido es
`string`, incluida la unión `["string", "null"]`. La llamada pura
`parse_tool_parameter("null", [{"type": ["string", "null"]}])` reproduce el
resultado sin ejecutar el modelo. No es una regresión atribuible a NVFP4 y
no se modifica el parser en esta campaña, pero impide presentar este test
como tools 6/6.

Con la puntuación semántica de recuperación, el total es 23/30 frente a 20/30.
Ese promedio mezcla tareas y un fallo compartido de parser: no es una medida
de “10% menos inteligencia”. El bootstrap descriptivo por seis familias
produce un intervalo de diferencia de −31,25 a −2,0 pp; su tamaño y cobertura
no permiten estimar la degradación de producción ni certificar un umbral de
1–3 pp. El dato útil para la decisión es que hay fallos concretos de código
sin una ganancia grande que compense asumir ese riesgo.

### Rendimiento largo

Medianas de dos semillas, peticiones frías con idéntico hash de solicitud por
pareja. Contextos **efectivos**, distintos de las etiquetas nominales:

| Entrada efectiva | TTFT control | TTFT candidato | Δ TTFT | Decode control / candidato |
| --- | ---: | ---: | ---: | ---: |
| ~31,8K | 5,037 s | 4,544 s | −9,8% | 251,4 / 232,1 tok/s |
| 130.141 | 26,305 s | 24,416 s | −7,2% | 217,5 / 206,3 tok/s |
| 257.118 | 66,524 s | 63,178 s | −5,0% | 189,3 / 168,8 tok/s |

Son salidas cortas de recuperación y trayectorias distintas, no un benchmark
aislado de matrices. No se atribuye toda la variación de decode al GEMM.
El ahorro frío largo es real en estas parejas, pero modesto y no garantiza
igual porcentaje en otras tareas.

La continuación más larga reutiliza **257.024 tokens físicos** en ambos
brazos. TTFT mediano 578,9 → 601,1 ms, sin mejora; el trabajo nuevo difiere
entre brazos (aproximadamente 301–329 tokens) por sus respuestas anteriores.
En código, una salida más corta o vacía no se cuenta como tarea acelerada.

Pico asignado máximo del piloto: **28,883 GiB control / 28,806 GiB candidato**.
Ambos completan los ensayos, pero el allocator registra reintentos OOM:
3+3 en los controles y 1+0 en candidatos. No se cambia el allocator para
ocultar esas condiciones. La diferencia de unos 80 MiB no establece margen
suficiente para cualquier petición multimodal.

## Decisión y siguiente inversión

- Conservar producción sin cambios. El usuario decidirá cualquier promoción;
  estos datos no la recomiendan, incluso con tolerancia a alguna pérdida.
- No extender este graft a todo GDN ni interpretar el resultado como rechazo
  universal de NVFP4. Para una nueva oportunidad de GDN haría falta recuperar
  la ejecución fusionada y medir kernels por forma antes de otro barrido largo.
- Atención queda como referencia reproducible de un compromiso pequeño de
  prefill, no como ganador. Otro donante o calibración es un experimento nuevo.
- No se ejecutaron profiling causal de seams, cancelación/rewind adversarial,
  petición conjunta imagen+256K ni el máximo de 16 imágenes. No hay
  certificación multimodal completa ni comparación con BF16.

## Validación y artefactos

- `uv run --offline pytest -q`: **382 passed, 44 skipped**, por dependencias
  y condiciones de entorno. La política de colección puede excluir además
  módulos de campañas no importables.
- Tests específicos en el entorno donante, incluidos los de tensores Torch:
  **49 passed** (`test_nvfp4_projections.py`, `test_nvfp4_quality.py`,
  `test_hybrid_profile.py`).
- 108 controles GPU y 27 controles rojos pasan.
- Los programas generados se califican en bubblewrap sin red ni acceso al
  workspace, con límites de CPU/memoria; no se ejecutan directamente en el host.
- Las tres ventanas restauran `ready` con la misma configuración inicial.
  No se cambian pesos, dependencias instaladas, código del servicio ni defaults.

Evidencia local, sin sobrescribir campañas anteriores:

- `results/20260923-nvfp4-reopen/screen/`: 42 muestras e inventarios.
- `results/20260923-nvfp4-reopen/quality/`: 60 respuestas, solicitudes congeladas,
  imágenes y métricas de caché.
- `results/20260923-nvfp4-reopen/grades/`: código intacto, tests, logs y oráculos.
- `results/20260923-nvfp4-reopen/comparison.json`: agregado que rechaza
  calificaciones ausentes, hashes cambiados o parejas incompletas.
- `results/20260908-hybrid-backends/nvfp4-reopen-*-20260923/`: configuración
  anterior/posterior y estado de cada ventana GPU.
