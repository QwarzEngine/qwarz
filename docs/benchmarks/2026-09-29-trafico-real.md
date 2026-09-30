# Campaña 2026-09-29: el port residente contra tráfico real

Objetivo del usuario: "mejorar tanto como sea posible" la velocidad del
servicio sin perder precisión, probando las palancas que la comunidad
publicó entre junio y septiembre de 2026.

Punto de partida: desde `93b4fc3` el supervisor lanza el port residente
(`engine.forward.worker`) sin switch de vuelta, con la promoción cerrada. Los
benchmarks del port (greedy, prosa repetitiva) daban los pisos pre-port, pero
el tráfico real del día mostraba 1.6–112 tok/s y aceptación MTP 0.10–0.28.

Todo lo medido acá es sobre **requests reales** de `state/qwasar.db`
reproducidos por HTTP contra un server de prueba (`:8811`, DB scratch), con
el servicio de producción detenido durante cada ventana de GPU
(`results/20260929-real-traffic/`, local):

- `agentic`: 18 turnos consecutivos de una sesión de Claude Code (filas
  7606–7623, 42K→57K de contexto), `max_tokens` topeado en 1024, T=1 (lo que
  manda el cliente).
- `fresh`: sesión nueva de Claude Code (system+tools de 14K) dos veces, y un
  turno corto de otra sesión.
- `sessions`: siete sesiones del mismo cliente con idéntico system+tools y
  conversaciones distintas desde el primer mensaje.

## 1. Bugs de correctitud del port (arreglados)

| bug | síntoma | causa | arreglo |
|---|---|---|---|
| stash GDN realocado | tras un cierre de razonamiento por presupuesto la salida degenera en `!!!!` (NaN) | `_remember_window` realocaba los buffers del stash cuando el largo de la ventana cambiaba (2 tokens del `</think>` inyectado); el grafo del verify seguía escribiendo las direcciones viejas, ya liberadas, y el rewind leía un buffer obsoleto | stash de 8 filas alocado una vez y llenado como prefijo (`projections.py`) |
| pool de grafos obsoleto | el worker muere en el segundo turno con prefijo distinto (`use_count > 0 INTERNAL ASSERT`) | `reset()` soltaba todos los grafos y reusaba el mismo `graph_pool_handle` | pool nuevo en `reset()` |
| sin switch de rollback | producción corría el port sin gate | `worker.rs` elegía el módulo solo por `fake` | `QWASAR_WORKER=exllama` vuelve al worker ExLlamaV3 |

Con el stash corregido el port en greedy sobre el prompt de la fila 7606 pasa
de 0.12 de aceptación y texto roto a 0.55 y una respuesta completa.

## 2. Palancas medidas

(Tablas de resultados en §4.)

- **Reuso de sesión** (`Session._zero` → `rezero()`): una sesión nueva ya no
  libera KV, estados GDN ni grafos; el draft se recicla.
- **Marcas de prefijo** (`QWARZ_PREFIX_MARKS`, default 12): copia pinned del
  estado GDN (~151 MB; el KV de atención es append-only) en cada corte de
  prefill, más cortes extra en el fin del bloque system+tools y el inicio del
  último mensaje. Una sesión que comparte parte de la cinta restaura la marca
  más larga en vez de prefillear desde cero.
- **top_k 20 y clamp de vocabulario** (`QWARZ_TOP_K`): la receta de la model
  card para thinking (T=1, top_p 0.95, top_k 20), la misma que siempre aplicó
  el worker ExLlama; el port la había perdido. Las 243 filas de padding del
  head ya no se pueden muestrear.
- **Muestreo de la ventana en GPU**: una sola copia D2H por ventana en vez de
  ~35 syncs y 7 sorts de 248K.
- **Muestreo especulativo** (`QWARZ_SPEC_SAMPLING`, default on): el draft
  muestrea de q (misma transformación T/top-k/top-p, Gumbel-max en GPU) y el
  verify acepta con min(1, p/q), remuestreando del residual normalizado
  max(0, p−q) al primer rechazo. Conserva exactamente la ley del target
  (test Monte Carlo en `tests/test_q38_incremental.py`). Aceptación de la
  primera posición medida en tráfico real: greedy 0.672 → 0.782.
- **Grafos del rewind GDN** por `keep` (`QWARZ_REWIND_GRAPH`): tokens
  idénticos, +1% a 4K, +2.4% a 32K.
- **Grafo del draft** (`QWARZ_DRAFT_GRAPH`): KV del draft prealocado,
  escrituras indexadas por tensor y un kernel Triton de decode (split-K, largo
  en tensor; más rápido que SDPA a todos los largos, 94% de ancho de banda a
  256K). Tokens idénticos; +0.7–1.7%.
- **Detokenización incremental**: el streaming re-decodificaba todo el texto
  visible en cada ventana (2.8 ms/ventana a 32K tokens de salida).
- **Warmup al boot** (`QWARZ_WARMUP`): compila y captura los caminos del
  primer turno antes de anunciar ready.

## 3. Palancas descartadas o sin margen

- **MTP del port "peor" que el de ExLlama**: descartado. Con teacher forcing
  sobre el texto exacto que generó ExLlama (generador simple, mismo prompt),
  el draft del port acepta 0.45 vs 0.43 de ExLlama, con perfiles por posición
  equivalentes. La diferencia que reportan los servicios es de trayectorias.
- **Head de 64K**: no es la causa de nada; el head completo da la misma
  aceptación y es más lento.
- **Kernel NVFP4 del MLP**: el port ya usa b12x (~1.75 TB/s en microbench,
  ~78% efectivo dentro del verify). cutlass/cudnn son 1.6× más lentos; trtllm
  no soporta SM120.
- **GEMV EXL3 para M chico**: solo 2–4 bpw; el artefacto es 5 bpw.
- **Drafting por sufijo** (análisis offline): 15–20% de los tokens de salida
  agénticos están en copias ≥6 tokens, pero con ventanas de 7 filas el techo
  neto estimado es 3–4%.

## 4. Resultados

### Velocidad en tráfico real (replay, mismas requests)

| agentic (18 turnos) | port HEAD | ExLlama | port final (top-p de ExLlama) |
|---|---:|---:|---:|
| tok/s punta a punta | 39.4 (muere en el turno 1) | 120.9 | **139.3** |
| decode ponderado | 65.3 | 154.1 | **183.2** |
| aceptación mediana | 0.09 | 0.61 | 0.52 |
| TTFT sumado | — | 15.4 s | 14.6 s |

Contribuciones (brazos del mismo día, decode ponderado agéntico):
bugs + reuso de sesión 145.6 → grafos de rewind 149.9 → muestreo
especulativo 181.0 → grafo del draft + warmup 182.0 → KV8 del draft 184.3.
Sesiones nuevas del mismo cliente: marcas de prefijo 41.1 s → 31.2–32.1 s de
TTFT sumado (−22 a −24%), 11.264 tokens del bloque system+tools reusados.

Microbench greedy (prosa, tokens idénticos entre brazos): grafo de rewind
+1%/+2.4% (4K/32K); grafo del draft +1.7%/+0.7%; KV8 +0.9%/+5.4% (32K/131K).

Descartadas por medición: k=5 (−2%), k=7 (−5%), mapa calibrado con tráfico
(dentro del ruido), drafting por sufijo (MTP acepta 3.42 vs 2.24 donde hay
match), q del draft con otra temperatura/top-k (óptimo en T=1/k20).

### Calidad (gate de 37 celdas, mismo día)

| criterio | ExLlama | port final | gate |
|---|---:|---:|---|
| código | 24/32 | 16/32 | **falla** |
| JSON | 4/4 | 4/4 | pasa |
| aceptación mediana (código) | 0.671 | 0.679 | pasa (+0.8 pp) |
| pico asignado | 27.87 GiB | 27.69 GiB | pasa |

Diagnósticos de la brecha de código, ninguno encuentra un mecanismo:

- NLL del target por el camino de decode (ventanas MTP, stash, rewind,
  grafos) vs un prefill directo, sobre ~2.400 tokens: −0.002 a −0.005 nats.
- Sesgo del muestreador, E[log p(x) + H(p)] sobre ~6.900 tokens por modo:
  plano −0.0001 ± 0.003, especulativo −0.002 ± 0.003. Monte Carlo de la
  primera y la segunda posición de la ventana dentro del error.
- Prefill del port actual vs `93b4fc3`: logits idénticos (KL 0).
- Primer token tras los prompts largos de la matriz: ExLlama xqa vs ExLlama
  flash (dos rutas ya gateadas) correlacionan 0.67–0.72 sobre el vocabulario;
  el port vs ExLlama 0.62–0.75. El top-1 coincide en todos. Los prompts de
  relleno son numéricamente sensibles y el port no es un outlier.
- Semillas extra (lru/ring a 32K): especulativo 32/48, plano 36/48.

## 5. Causa raíz de la brecha de código: la semántica de top-p

La brecha se repitió con semillas nuevas (port 16/32 otra vez, ExLlama 22/32)
y **también con muestreo plano** (16/32): no era el muestreo especulativo.
Con el target equivalente (KL por posición a lo largo de una generación real
de código igual al piso ExLlama xqa-vs-flash: 0.001–0.007, top-1 32/32; NLL
de decode = prefill sobre 3K tokens sin deriva; mismo texto decodificado en
las 37 salidas) la diferencia tenía que estar en la distribución servida.

ExLlama aplica top-p después de top-k así (`SS_TopP`): renormaliza el top-k y
conserva un token mientras la masa acumulada **incluyéndolo** sea ≤ top_p,
siempre el primero. **El token que cruza 0.95 queda afuera.** El port usaba
la definición "conjunto mínimo con masa ≥ p", que **incluye** ese token: en
cada posición incierta muestreaba un token más de la cola. Sobre miles de
tokens eso son más "resbalones" en los tests que el modelo escribe (el
oráculo seguía en 7/7 en las celdas que fallaban).

Con `_nucleus` alineado a ExLlama (target, ventana y draft):

| variante | corridas | código acumulado |
|---|---|---:|
| port, top-p inclusivo (originales / +1000 / plano) | 16 / 16 / 16 | 48/96 (50%) |
| **port, top-p de ExLlama** (originales / +1000) | 23 / 15 | **38/64 (59%)** |
| ExLlama hoy (originales / +1000) | 24 / 22 | 46/64 (72%) |
| ExLlama, historial (6 corridas desde 09-11) | 24, 22, 20, 19, 19, 19 | 123/192 (64%) |

La matriz es muy ruidosa (8 puntos entre dos corridas del mismo motor). El
top-p inclusivo era peor que el historial de ExLlama (p≈0.03); con el top-p
alineado el port queda dentro de ese historial (p≈0.5) y a p≈0.14 del
ExLlama de hoy. La regla formal del gate (candidato ≥ control en una corrida)
no pasa: 23 vs 24 con las semillas originales
(`results/20260929-real-traffic/gate-decision-final.json`: código 23/24,
JSON 4/4, aceptación +1.7 pp, memoria +0.32 GiB).

Decode dentro de la matriz con el top-p corregido: 241 / 229 / 204 / 178
tok/s a 32K / 64K / 128K / 256K (control ExLlama del 22/09: 230 / 232 / 207 /
174), aceptación mediana 0.688.
