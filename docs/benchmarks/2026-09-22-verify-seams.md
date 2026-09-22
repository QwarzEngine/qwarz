# Perfil de costuras del verify: la puerta pasa, la Fase 2 queda acotada (2026-09-22)

**Resultado:** sobre el stack de producción intacto
(`xqa-KVNVFP4-MLPNV64-PRIMS-RDZ-MTP6-HOT64K`, delta cero), un perfil kineto
de celda completa con marcadores triton en las fases draft/verify mide
**3,08 ms/verify de costura a 32K y 3,44 a 256K** (GPU ociosa dentro del
span de verify + kernels no-GEMM fuera de los grafos), superando la puerta
congelada de 1,5 ms en ambos contextos. **La puerta formal pasa**, pero la
atribución por familia acota la Fase 2 congelada (extender el grafo XQA a la
costura de las 16 capas de atención) a un techo de **+2,2–2,4% de decode,
por debajo de su propia puerta de promoción (+3%)**: no se construye
aislada. El dinero medido de la costura está en el **idle del bucle draft
(0,81/0,84 ms/verify; el grafo del draft ya medido el 09-11 recupera
−0,35/−0,76)**, en el **walk+copias del rendezvous (1,3 ms/verify, territorio
del bucle nativo)** y en las **normas/residuales de las 64 capas
(0,65–0,77 ms, fusionables en Triton)**. Un paquete costura de atención +
fusión residual/norma proyecta +5,3% a 32K y +4,6% a 256K, y sí pasaría +3%
conjunto.

Chequeos de validez: bias del profiling 0,98 (span de ciclo vs muestra
entera), 1503 kernels por verify, sin clasificar 0,1%, y fidelidad greedy
bit-exacta 453/453 tokens del warmup contra la referencia no perturbada de
esta mañana (a 32K el greedy deriva en el token ~47, la banda cross-kernel ya
documentada). Hallazgo colateral: la referencia T>0 de esta mañana no es
reproducible ni contra su propio segundo intento en lru/json — los A/B a
temperatura deben comparar ciclo y calidad, no igualdad de completions.
El idle del draft (0,81 ms) confirma que el draft de producción corre eager:
el grafo del draft es la primera palanca siguiente recomendada.

Informe completo, puerta formal, bandas congeladas y artefactos:
[`results/20260922-verify-seams/report.md`](../../results/20260922-verify-seams/report.md)
(artefactos locales, fuera de git).
