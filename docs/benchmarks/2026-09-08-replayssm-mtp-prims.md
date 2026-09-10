# ReplaySSM y MTP reducido con Minima64 + FP8

Objetivo: al menos 10% de mejora en una métrica del modelo completo, conservando el perfil acordado.

Se verificó **+12,9–17,1% de decode en los JSON de 4K** con cabeza MTP 64K, embeddings del borrador residentes y copia agrupada de IDs. Confirmado con procesos full→hot y hot→full. Se mantienen Minima64, FP8 PRIMS, K8/V4, MTP6 y pool 262144. Costo de memoria: aproximadamente 0,856 GiB.

El resultado no se extiende a todos los turnos largos: Q128/KV258048 pasó de +10,8% a +0,7% al repetir. Coding inicial: 4/6 completa, 5/6 reducida; el caso largo repetido falló en ambas. 16/16 respuestas JSON válidas por brazo, incluyendo repeticiones de las mismas tareas.

ReplaySSM TF32x3 redujo 41,6% el tiempo del ciclo recurrente aislado y pasó oráculos/grafos, pero todavía requiere integrar cursor/rewind/snapshots. TF32 simple falló precisión. Ese speedup aislado no se usó para cumplir el objetivo.

Experimentos aislados; servicio de producción restaurado con su configuración exacta. Fuentes, resultados, límites y comandos: [informe completo MTP](../../results/20260908-mtp-prims/report.md), [verificación final](../../results/20260908-mtp-prims/verification.json), [ReplaySSM](../../results/20260908-replayssm/report.md).
