# MTP 64K: ampliación de contexto

Se ejecutaron 72 muestras emparejadas en 32K/128K/258K, con dos semillas y órdenes inversos, más un control diagnóstico de referencia. Se conservan Minima64+FP8 PRIMS, K8/V4, MTP6 y pool 262144.

En turnos de 128 tokens nuevos, decode mejoró 11–21% a 32K, 5.6–14.4% a 128K y 6.7–6.9% a 258K. La ingesta larga quedó prácticamente igual; memoria adicional ~0.856 GiB. No se sostiene 10% general en contexto largo.

JSON 12/12 por brazo. Código: 3/6 reducida frente a 1/6 completa; un control fresco de referencia aprobó el caso 32K que había truncado, sin aislar la causa de esa variación. Calidad no asegurada; variante experimental y servicio original restaurado.

[Informe, tablas por estrato y límites](../../results/20260908-mtp-long-context/report.md) · [Datos](../../results/20260908-mtp-long-context/aggregate.json) · [Verificación](../../results/20260908-mtp-long-context/verification.json).
