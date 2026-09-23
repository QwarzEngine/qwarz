# Grafo del bucle draft: el ciclo se recupera, la pila no puede certificarlo bit-exacto (2026-09-22)

**Resultado:** el puerto del grafo CUDA del bucle draft de 6 pasos
(`results/20260911-draft-phase/draft_graph.py`) al stack promovido funciona
y vale lo que el perfil de costuras de hoy predijo: **−0,877 ms/verify a
32K (−3,9% de decode) y −0,454 a 256K (−1,5%)**, con la fase draft pegada a
su suelo de kernels (0,08–0,09 ms sobre el replay puro) — recupera ~todo
el idle del draft que midió la campaña de costuras (0,807/0,838 ms/verify).
Memoria en par (≤0,001 GiB), 169 kernels/replay, captura robusta (BC del
draft desviado a dispatch dentro de la captura; BCAttn/BC_GatedMLP sí
enganchan en producción sobre la caché K8/V4). **No se promueve:** la
puerta congelada exige ids bit-exactos y a 32K/256K la pila es
no-determinista contra sí misma — dos corridas greedy del MISMO brazo eager,
misma semilla, mismo proceso y sin grafo, divergen en el verify 4–5
(`control-run/control-verdict.json`); el grafo es bit-exacto a 4K (60/60) y
a 32K se desvía exactamente dentro de esa banda (4/60 intra-verify).

Es la extensión del hallazgo T>0 de esta mañana a greedy y a 32K+: **la
referencia bit-exacta no existe en contextos multichunk** (candidata: orden
de acumulación del split/combine de la atención, política Attention64).
Eso bloquea por igual el grafo del draft, la Fase 2 de costuras y cualquier
puerta de fidelidad bit-exacta futura a 32K+, y convierte la auditoría de
esa no-determinidad en la palanca previa. Con una pila bit-estable, la
campaña ya hecha se re-evalúa casi gratis (harness completo conservado) y
el grafo promociona sobre sus números medidos.

Informe completo, puerta formal, control y artefactos:
[`results/20260922-draft-graph/report.md`](../../results/20260922-draft-graph/report.md)
(artefactos locales, fuera de git).
