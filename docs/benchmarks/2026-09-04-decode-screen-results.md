# Primera batería: EXL3, draft, caché y thinking

Fecha: 2026-09-04. Hardware: una RTX 5090. La 3090 Ti conserva su servicio SGLang y no participa. Este documento registra screening, no certificación de calidad frente a BF16 ni un SLA.

## Protocolo válido: token_safe_v2

Se usa un snapshot de código de ExLlamaV3 como contexto, sin repetición artificial, seguido por una tarea fija: implementar una caché LRU con tests. La tarea sirve para medir generación de código; no exige recuperar datos distantes del snapshot y no demuestra calidad de atención larga.

El cuerpo se tokeniza literalmente y se inserta entre prefijo/sufijo de chat. Sólo el framing introduce tokens de control. Una comprobación con el tokenizer real, usando texto que contiene delimitadores de chat, verifica exactamente tres inicios de mensaje y dos cierres.

Reservamos 4,096 tokens de salida y 16 de margen especulativo dentro de cada presupuesto: 28,656 tokens de entrada en el bucket 32K y 258,032 en el bucket 262,144. Cada bucket tiene un seed y una rama caliente. Las ramas cambian el número del pedido y reutilizan físicamente el prefijo; no son una conversación continua con herramientas.

Configuración: MTP, K8/V4, `medium`, temperatura 1.0, top-p 0.95, top-k 20, min-p 0, sin penalización de presencia. Ambas cuantizaciones usan el mismo sampler, semillas y corpus. La salida puede diferir, por lo que también cambian aceptación y longitud: son resultados del sistema completo, no un aislamiento del coste de GEMM.

El decode cuenta tokens realmente generados, incluyendo reasoning, y excluye todos los tokens del primer batch del numerador. TTFT y primer contenido se observan desde el host. El primero no equivale a texto final ni a herramienta ejecutable. La construcción del prompt se mide aparte; HTTP y carga inicial del modelo quedan fuera.

## Resultados válidos disponibles

| Target / rama caliente | Entrada efectiva | Decode tok/s | Primer token | Primer contenido final | Respuesta completa | Salida |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EXL3 3.5 bpw / 32K | 28,656 | 178.76 | 0.182 s | 7.514 s | 17.748 s | 3,145 |
| EXL3 3.5 bpw / 256K | 258,032 | 91.23 | 0.521 s | 10.923 s | 29.021 s | 2,604 |
| EXL3 5 bpw / 32K | 28,656 | 162.90 | 0.188 s | 3.498 s | 13.730 s | 2,211 |
| EXL3 5 bpw / 256K | 258,032 | 87.97 | 0.526 s | 5.711 s | 36.527 s | 3,172 |

Todas las respuestas de esta tabla terminan por stop normal, sin truncación. La rama larga de 3.5 bpw procesa 239 tokens físicos de prefill y reutiliza 257,792. Su aceptación de draft es 74.57%. El máximo reservado por el allocator CUDA de ese run es aproximadamente 21.40 GiB; no representa toda la memoria del dispositivo.

La ampliación del seed hacia 256K tarda 190.127 s hasta el primer token, reutilizando sólo 28,416 tokens y procesando 229,615 nuevos. Por tanto, estos resultados no prometen 30 segundos para una importación masiva o una sesión fría.

Los cuatro prompts coinciden por SHA-256 entre los dos targets. 5 bpw cabe con el pool completo y alcanza 87.97 tok/s en la rama larga. Reserva máxima del allocator: aproximadamente 25.85 GiB. El primer contenido llega antes en esta muestra, pero genera más tokens y termina después de 30 segundos. No atribuir la longitud del razonamiento a la cuantización sin más tareas/semillas.

## Comprobaciones de código

Las cuatro respuestas completas del run 3.5 bpw fueron inspeccionadas antes de ejecutarlas. Sus suites generadas pasan 7, 8, 7 y 8 tests, respectivamente para 32K seed/rama y 256K seed/rama. Cada respuesta pasa además los siete grupos de comprobaciones independientes de `benchmarks/fixtures/lru_checks.py`: capacidad, inserción, recency en get, recency en put, valores None, borrado/reinserción y capacidad uno.

Es un smoke test de una sola tarea, no pass@1 de coding general, no comparación con BF16 y no evaluación de agentes multiarchivo. Los JSON de generación permanecen intactos con `quality: not_scored`; esta verificación externa es una fase posterior.

En 5 bpw pasan las suites de 32K seed/rama (8/8 y 8/8) y de 256K seed (9/9). La rama 256K falla su suite: 17 tests, un error por `assertNotIs`, método inexistente de `unittest.TestCase`; debía usar `assertIsNot`. No se corrige el artefacto generado. Las cuatro implementaciones LRU pasan los siete grupos de controles externos, pero el requisito de entregar tests ejecutables falla en esa respuesta. Resultado de respuestas completas válidas: 3.5 bpw 4/4; 5 bpw 3/4. Es una muestra demasiado pequeña para ordenar su calidad general.

La velocidad mínima buscada queda demostrada sólo en estos casos. No se promueve 5 bpw a baseline de calidad ni se considera cumplido un máximo de 30 s para completar cualquier respuesta.

## Barrido de thinking: 5 bpw a 32K

Segunda batería con la misma tarea, MTP y K8/V4, 28,656 tokens de entrada, presupuesto de salida 4,096, un seed y una rama caliente por modo. Aquí el pool se asigna a 32,768 posiciones, no a 262,144 como en la comparación principal. Comparar modos dentro de esta batería; no extrapolar sus tiempos a 256K.

| Modo / rama caliente | Decode tok/s | Primer contenido final | Tiempo hasta terminar o cortar | Salida | Respuestas válidas seed + rama |
| --- | ---: | ---: | ---: | ---: | ---: |
| xhigh | 111.91 | No llegó | 36.699 s, truncada | 4,091 | 0/2 |
| medium | 163.13 | 3.493 s | 13.711 s | 2,211 | 1/2 |
| low | 160.68 | 7.138 s | 16.717 s | 2,660 | 2/2 |
| off | 163.37 | 0.187 s | 12.503 s | 2,016 | 2/2 |

Las dos muestras xhigh agotan el límite sin contenido final: son fallos, no respuestas utilizables. El runtime deja margen especulativo y corta en 4,091 tokens efectivos. El seed medium falla al importar el código por usar `_MISSING_SENTINEL` como argumento por defecto antes de definirlo. Su rama pasa ocho tests propios; las dos low pasan ocho cada una y las dos off diez cada una. Las cinco respuestas ejecutables pasan además los siete grupos externos. No se repararon las respuestas generadas.

Off usa su sampler recomendado (temperatura 0.7, top-p 0.8, penalización de presencia 1.5); los modos con thinking usan temperatura 1.0, top-p 0.95 y penalización 0. Top-k 20 y min-p 0 son comunes. Son políticas completas, no un experimento que aísle exclusivamente el flag de thinking. Low/xhigh también cambian el prefijo del template. Una tarea y dos muestras por modo no permiten concluir que off preserve la calidad de un agente complejo, ni que low siempre razone menos que medium. El seed medium difiere del screening con pool completo pese a compartir semilla: tampoco se asume determinismo bit a bit entre configuraciones de runtime.

Decisión provisional: conservar MTP + K8/V4 como candidato y evaluar medium/off en tareas reales; no usar xhigh como política predeterminada para una meta interactiva de 30 s sin una evaluación adicional del presupuesto. Los controles de thinking de esta batería pertenecen al probe, no cambian todavía el servidor del modelo.

## Screening inicial descartado para calidad

Los runs anteriores a `token_safe_v2` usaban un render de corpus que podía convertir delimitadores literales presentes en los archivos en tokens de control. Se conservan como diagnóstico de rendimiento, pero no se promueven como evidencia de calidad o comportamiento de thinking.

Dentro de ese protocolo antiguo, los prompts sí coinciden por hash entre las dos configuraciones:

| Contexto / dos ramas calientes | MTP + K8/V4, mediana tok/s | DFlash2 + KV NVFP4, mediana tok/s |
| --- | ---: | ---: |
| 128K, entrada 130,544 | 118.72 | 38.07 |
| 256K, entrada 261,616 | 74.21 | 21.66 |

Todos estos pesos son EXL3 3.5 bpw. NVFP4 en esta tabla es el formato de **caché**, no una cuantización NVFP4 de pesos. Cambiaron draft y caché a la vez: no se puede atribuir el efecto completo a uno de ellos. Además, eran salidas cortas, truncadas, con sampler greedy. No comparar esas cifras directamente con las filas v2 para deducir una mejora causal del framing o del sampler.

## Artefactos

- `results/20260904-mtp-k8v4-medium-safe/`: protocolo v2, EXL3 3.5, resultados y respuestas completas.
- `results/20260904-exl3-5bpw-mtp-medium-safe/`: protocolo v2, EXL3 5, run completo; un fallo de test generado en la rama larga.
- `results/20260904-5bpw-thinking-{xhigh,medium,low,off}-safe/`: cuatro runs v2 completos del barrido a 32K; completar un run no implica aprobar sus respuestas.
- `results/20260904-mtp-k8v4-long-screen/`: diagnóstico antiguo MTP, 128K/256K.
- `results/20260904-dflash-nvfp4-screen/`: diagnóstico antiguo DFlash2/NVFP4.
- `results/20260904-mtp-k8v4-32k-screen-v2/`: pese al sufijo del directorio, corresponde al framing antiguo; manda `run.json`, no el nombre.
- `results/20260904-mtp-k8v4-32k-screen/`: intento incompleto por serialización de una referencia `Job`, posteriormente corregido y cubierto por tests.

El target 5 bpw tiene sus tres shards y tokenizer verificados contra los hashes esperados; árbol local fijado en `0b9a439ffefa45c55a2a3cb0324de9fe1b23bce69a37a399cb952d355f02d92e`. Se actualizó el manifest; la revisión Git requerida para acceptance sigue pendiente y no se creó ningún commit.

Validación del harness al cierre: 66 tests de pytest pasan; ambos manifests son válidos con 12 casos cada uno; el launcher pasa `bash -n`. Los seis runs con framing seguro tienen marcador de finalización. La carga experimental terminó y liberó la 5090; SGLang en la 3090 Ti permaneció activo. Esto verifica la batería y su infraestructura, no certifica todavía el motor de producción.

## Reproducción

Desde la raíz de Qwasar, elegir un directorio de salida nuevo:

```bash
QWASAR_PROBE_KIND=decode \
QWASAR_DRAFT_METHOD=mtp \
QWASAR_CACHE_QUANT=8,4 \
QWASAR_CONTEXTS=32768,262144 \
QWASAR_REPETITIONS=1 \
QWASAR_MAX_NEW_TOKENS=4096 \
QWASAR_SAMPLER=recommended \
QWASAR_THINKING=medium \
./scripts/run_resident_probe.sh
```

Para 5 bpw, añadir `QWASAR_MODEL_PATH=/home/rekeyea/models/Qwen3.8-27B-EXL3-5.0bpw`. El launcher rechaza otra carga de cómputo en la 5090 y no sobrescribe runs del decode probe.

## Interpretación y próximos pasos

La meta de 50 tokens/s es viable en este screening corregido con EXL3 3.5 y 5, incluso usando el sampler recomendado. Una respuesta completa larga quedó dentro de 30 s, pero una muestra no demuestra un máximo o un percentil operacional.

Una prueba CPU del template con historial conservado obtiene un prefijo común de 53 de los 54 tokens del prompt `medium` al pasar a off; low/xhigh conservan sólo tres. Esto sugiere una vía para alternar medium/off sin invalidar el prefijo largo, a diferencia de cambiar instrucciones iniciales. Todavía requiere validación con una sesión real y el cliente serializando el historial sin modificaciones.

Falta ampliar tareas, semillas y turnos continuos; extender el barrido de thinking a contexto largo; hacer el cruce MTP/DFlash2 por K8V4/NVFP4; perfilar los kernels efectivos; y medir la cuantización de **pesos** NVFP4 con su receta calibrada frente a EXL3 y una referencia BF16. No elegir todavía un ganador universal de calidad por bpw nominal.
