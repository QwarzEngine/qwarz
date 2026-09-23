# Promoción 2026-09-23: parche de determinismo GDN + grafo CUDA del draft loop

Los dos hallazgos de la campaña de nondeterminismo pasan al servicio de
producción (systemd `qwasar.service`, RTX 5090). Decisión del usuario:
"Promobamos nuestros hallazgos asi puedo probarlos".

## 1. Parche de determinismo GDN (en el venv del servicio)

Causa raíz (campaña `2026-09-23-nondeterminism-gdn`): los kernels
`cuda_recurrent_gated_delta_rule_kernel{,_128}` acumulaban 4 parciales SUBK
vía `atomicAdd` float en shared memory; el orden dependía del scheduling de
warps, así que los bits bajos variaban entre corridas. A 4K el ruido quedaba
bajo el umbral del argmax; a 32K+ volteaba ids (divergencia mediana: verify 9).

Cambio: parciales staged + reducción en orden fijo (bt=0..3) en los 4 sitios
atómicos (ambas plantillas de kernel, variantes MAMBA2/history incluidas).

Instalación en el venv (`qwen38-exl3-mia/.venv`):

| Archivo | sha256 | Estado |
|---|---|---|
| `exllamav3_ext.cpython-312-x86_64-linux-gnu.so` | `2ab1a278…` | **instalado** (build shadow certificado por la campaña) |
| `…so.orig-pre-gdn-determinism` | `cb15fb84…` | backup del original |
| `exllamav3/exllamav3_ext/gdn.cu` | `969eaec9…` | **fuente parchada** instalada |
| `gdn.cu.orig-pre-determinism` | `5e8327c3…` | backup del original |

Verificación previa (campañas): bit-estable a 4K y 32K (0 fuentes en 70
módulos × 64 tokens × 2 reps); E-E/G-G bit-idénticos en corridas completas de
541 verifies; ciclo no más lento (22.25 ms/verify).

## 2. Grafo CUDA del draft loop (en `qwasar_runtime`)

Puerto verbatim de la clase certificada en `results/20260922-draft-graph/`:
los seis pasos MTP se capturan en un `torch.cuda.CUDAGraph` por bucket de
páginas KV y cada verify es UNA replay + UN sync de host.

- Nuevo módulo: `src/qwasar_runtime/draft_graph.py`
  (`REVISION = "draft-graph-20260923-promoted"`).
- Cableado en `engine.py`: se instala tras `hot_head`, solo si rendezvous +
  hot64k están instalados; try/except que nunca rompe el boot; entrada en
  `/config` (`draft_graph`) y en la identidad de sesión.
- Kill switch: `QWASAR_DRAFT_GRAPH=0`. Diagnóstico:
  `QWASAR_DRAFT_GRAPH_VALIDATE=N` (cross-check eager de los primeros N
  verifies; off por defecto).
- Fallbacks automáticos por batch: prefill no terminado, batch != 1, ventana
  != 6 → camina el eager walk. Captura perezosa en el 3er verify.
- Efecto medido (campaña graph-band): −0.9..−1.1 ms/verify a 32K, +1.45%
  tok/s @32K, +5.58% @256K, memoria delta 0.
- Fidelidad: equivalencia de salida certificada (6/6 pares E-G produjeron
  completions idénticos; la verificación greedy hace que la salida sea la
  secuencia greedy del target independientemente de las propuestas). Los ids
  del draft pueden divergir E-G en near-ties a 32K+ (diferencia numérica
  determinista grafo-vs-eager, documentada en
  `results/20260923-gdn-followup/posthoc-notes.md`).

## 3. Evidencia post-promoción (servicio en vivo)

- `/config`: `draft_graph.installed=true`,
  `revision=draft-graph-20260923-promoted`; rendezvous + hot64k + XQA intactos.
- Determinismo en vivo: DOS requests greedy idénticos (temperature 0) a
  32.7K tokens de prompt → completions byte-idénticos (57 tokens). Repetido a
  35.4K con 395 tokens: byte-idénticos otra vez. Antes del parche, dos decodes
  greedy a 32K+ divergían (mediana: verify 9).
- Ritmo: 125–159 tok/s wall sobre HTTP incluyendo TTFT y prefills parciales;
  consistente con la banda 239–252 tok/s sostenidos del harness una vez
  excluido el TTFT. Sin errores en el journal; VRAM 30.9/32.6 GiB (normal).
- Tests: 524 passed, 3 skipped (5 nuevos en `tests/test_draft_graph.py`).

## 4. Rollback

Grafo del draft (solo código): `QWASAR_DRAFT_GRAPH=0 systemctl --user restart
qwasar.service` (o revertir el commit de `qwarz`).

Parche GDN (venv):
```bash
SP=~/Documents/llm/qwen38-exl3-mia/.venv/lib/python3.12/site-packages
cp $SP/exllamav3_ext.cpython-312-x86_64-linux-gnu.so{.orig-pre-gdn-determinism,}
cp $SP/exllamav3/exllamav3_ext/gdn.cu{.orig-pre-determinism,}
systemctl --user restart qwasar.service
```

## 5. Incidente post-promoción (mismo día, resuelto)

A los ~10 min de uso real (sesión de droid con prompts largos) el worker logueó
dos OOM transitorios (bloques de 96/272 MB con ~85 MB libres). El allocator
purgó su caché y reintentó: sin traceback, sin caída, pero el request en vuelo
se congeló. Diagnóstico: la pila opera por diseño a ~100 MiB del límite del
dispositivo; el allocator nativo fragmentado no pudo servir un bloque contiguo
grande. Las promociones no fueron la causa material (el grafo retiene unos MB;
el parche GDN usa shared memory) — el margen se lo comió el caché del
allocator tras tráfico de 32-35K (smoke tests incluidos).

Fix aplicado: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` en
`integrations/systemd/qwasar.service` (+ daemon-reload + restart). Verificado
en el environ del worker y sin costo: 290 tok/s warm, aceptación 0.84,
`draft_graph` instalado.

## 6. Pendiente (no bloquea)

- El parche GDN es upstreamable a MiaAI-Lab/exllamav3 (4 sitios, misma
  estructura de kernel; overhead medido ≈ 0).
- `results/20260923-promotion/smoke.py` queda como script de humo repetible.
- La gate de tok/s por trayectoria (aceptación) quedó documentada como flaw de
  diseño en `posthoc-notes.md`; no se retroajustó.
