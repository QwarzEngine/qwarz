# MTP fijo frente a adaptativo: primer A/B

Fecha: 2026-09-07. **Decisión: conservar MTP fijo de cuatro tokens en Qwasar.** El adaptativo con techo cuatro pierde velocidad de decode a 32K y no muestra una ventaja sostenida cerca de 256K. No se modificó la política del servicio.

## Qué cambió y qué se mantuvo

Único tratamiento: `generator.dynamic_draft=False` frente a `True`. En ambos casos: máximo cuatro propuestas, EXL3 5 bpw fijado por hash, cabeza 6 bits, MTP 4 bits, K8/V4, Flash/8192, RTX 5090 y pool de 262.144 posiciones. No se modificaron kernels, formato de pesos/caché, batch máximo, DFlash2 ni las dependencias del donante.

El adaptativo utiliza los parámetros existentes del donante: `alpha_up=1.30`, `alpha_down=0.65`, `skip_ema=0.3`, `probe_interval=16`. Ambos modos registran estadísticas por ronda. La longitud máxima permanece en cuatro; esta prueba no evalúa máximos de dos, tres o siete.

Se añadió la opción de benchmark `QWASAR_MTP_POLICY=fixed4|adaptive4`, disponible sólo para el probe decode con MTP/K8V4. Verifica el artefacto EXL3 5 bpw antes de cargar y registra configuración efectiva y estadísticas. La ruta de producción no activa esta opción.

## Protocolo

Una tarea LRU con implementación y tests, usando corpus fuente sin repetición artificial, `thinking=medium` y sampler greedy. Contextos nominales 32K y 262.144; entradas efectivas 28.656 y 258.032 tokens. Se reservan 4096 tokens de salida y 16 de scratch dentro del presupuesto nativo.

Cada política ejecuta una respuesta de preparación por contexto, seguida de tres ramas calientes con prompts distintos entre repeticiones pero **idénticos entre políticas**. Total: 16 generaciones, cuatro de preparación excluidas y 12 medidas, organizadas en seis pares. El orden fue fijo primero, adaptativo después; no hubo confirmación en orden inverso. No se propone adopción y no se interpreta este screening como prueba estadística de optimalidad.

La auditoría verifica coincidencia de hashes de pesos, runtime, corpus, template, harness, Flash y configuración; además de prompts exactos, semillas y presupuestos. En los seis pares calientes coinciden los contadores físicos: 239 tokens nuevos de prefill, con 28.416 o 257.792 reutilizados. Cero requeues y cero truncamientos en las 16 respuestas. La primera tanda se excluye del numerador de decode.

No es un ensayo HTTP ni una sesión continua con herramientas. El sampler de producción puede ser distinto. Las cifras de decode incluyen tokens de razonamiento y código, no sólo contenido visible final.

## Resultados calientes

Medianas de tres muestras por política y contexto:

| Contexto nominal | Decode fijo | Decode adaptativo | TTFT fijo → adaptativo | Primer contenido fijo → adaptativo | Respuesta completa fija → adaptativa |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32K | 155,39 tok/s | 136,32 tok/s | 171 → 179 ms | 5,30 → 6,20 s | 15,05 → 17,73 s |
| Casi 256K | 83,01 tok/s | 82,25 tok/s | 331 → 331 ms | 5,89 → 5,83 s | 32,10 → 27,04 s |

Por pares de prompt, las razones adaptativo/fijo de decode fueron:

- 32K: **0,8866 / 0,8773 / 0,8760**. Mediana 0,8773: aproximadamente **12,3% menos velocidad**.
- Casi 256K: **0,9435 / 0,9805 / 1,0044**. Mediana 0,9805: aproximadamente **1,9% menos**, con una muestra prácticamente empatada.

La mediana de razones pareadas no es necesariamente la razón entre las medianas de la tabla; ambas se conservan en los artefactos.

La respuesta completa larga baja de tiempo porque **las salidas también cambian**: la mediana de tokens generados pasa de 2641 a 2208. No atribuir esa reducción a un aumento de velocidad del motor. A 32K, las medianas de longitud son 2317 y 2398 tokens.

La ventana adaptativa media registrada queda entre **3,45 y 3,51** en los seis turnos medidos. Reducir propuestas no mejoró el coste por token aceptado. No se capturaron perfiles de hardware para atribuirlo a una combinación concreta de coste de draft, verificación, tamaños de kernels o coordinación.

Las estadísticas upstream omiten algunas rondas sin draft; su histograma no debe interpretarse como porcentaje de todos los pasos del target.

## Calidad y fidelidad

Se revisaron las respuestas y se ejecutaron dentro de bubblewrap, sin red ni acceso al proyecto, con límite de recursos. El grader aplica tanto sus unittests generados como siete comprobaciones independientes de LRU; no repara el código.

| Grupo | Fijo | Adaptativo |
| --- | ---: | ---: |
| Respuestas calientes que pasan ambas comprobaciones | **5/6** | **5/6** |
| Calientes a 32K | 3/3 | 3/3 |
| Calientes cerca de 256K | 2/3 | 2/3 |
| Todas, incluidas preparaciones | 7/8 | 6/8 |
| Oráculo independiente, todas las generaciones | 8/8 | 8/8 |

Los fallos corresponden a tests generados, aunque las implementaciones pasan el oráculo. El caso fijo largo de repetición tres espera una expulsión incompatible con los accesos de su propio test. Se conservan los fallos en el denominador y en las estadísticas de tiempo.

**Ninguno de los seis pares calientes produjo una secuencia greedy idéntica.** Se reconstruyeron los tokens de los eventos, incluidos los retenidos por terminadores. Cambiar el ancho de verificación puede cambiar la aritmética del backend, pero este ensayo no investigó la causa de las divergencias ni prueba equivalencia de distribución. Igual puntuación 5/6 no certifica paridad numérica o de calidad general.

## Operación y verificación

El primer intento fue rechazado por el guard de GPU ocupada inmediatamente después de `systemctl stop`: el worker anterior todavía aparecía en NVML. No generó muestras. La restauración automática funcionó. El segundo intento esperó tanto la salida del PID anterior como la liberación de la GPU antes de comenzar; no se desactivó el guard ni se mataron procesos ajenos.

Al terminar se restauró `qwasar.service`, se comparó todo el objeto de configuración con el anterior y coincidió. La comprobación HTTP posterior devolvió `ready`, `busy=false`, EXL3 5 bpw, MTP cuatro, K8/V4 y Flash. El historial durable se conserva; la parada sí pierde la caché GPU previa.

Pruebas del cambio de harness: **270 pasan y 35 se omiten** en el entorno CPU local; las omisiones incluyen dependencias/entornos no disponibles. Los controles específicos de política y probe pasan: 42 pruebas. Sintaxis del launcher válida. La ejecución real de las dos políticas y la auditoría de seis pares verifican además el funcionamiento en GPU. No se realizaron commits.

## Artefactos y siguiente decisión

- [Protocolo exacto](../../results/20260907-mtp-policy/attempt2/protocol.json).
- [Resumen pareado y validación de invariantes](../../results/20260907-mtp-policy/attempt2/paired-summary.json).
- [Resultados con calidad por muestra](../../results/20260907-mtp-policy/attempt2/audited-results.json).
- [Grading fijo](../../results/20260907-mtp-policy/attempt2/grade-fixed4/summary.json) y [adaptativo](../../results/20260907-mtp-policy/attempt2/grade-adaptive4/summary.json).
- [Configuración anterior](../../results/20260907-mtp-policy/attempt2/service-before.json) y [restaurada](../../results/20260907-mtp-policy/attempt2/service-after.json).
- [Script de auditoría](../../results/20260907-mtp-policy/analyze.py).

Conservar **fijo cuatro**. Esta conclusión descarta promover este adaptativo con sus parámetros actuales en este primer screening; no demuestra que cuatro sea el mejor máximo posible. Si continuamos con MTP, el siguiente ensayo separado sería variar únicamente la cantidad fija de propuestas y contrastar contra cuatro, antes de tocar atención o cuantización.
