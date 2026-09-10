# Puerta de calidad ampliada (Fase 0) — 2026-09-10

**Estado:** puerta congelada. Baseline de control EXL3 (producción) medido y calificado: **código 11/16** (lru 4/4, ring 4/4, bucket 1/4, ttl 2/4), **JSON 2/2**, pico asignado 24,79 GiB, TTFT mediana 11,3 s, decode mediano 179,9 tok/s. Toda promoción de Fase 1 en adelante se decide contra este baseline.

## Por qué existe

Todas las rondas de septiembre evaluaron calidad de código con **una sola familia de tareas (LRU) × dos semillas**. Esa muestra produjo resultados contradictorios entre perfiles (4/6, 5/6, 6/6 en rondas distintas) sin poder distinguir degradación de calibración de ruido de muestreo. La puerta amplía la matriz y fija los criterios **antes** de cualquier promoción.

## Matriz de evaluación

18 celdas congeladas por SHA-256 (`prompts.json`, `prompt-manifest.json`):

| Carga | Celdas | Contextos | Detalle |
|---|---:|---:|---|
| Código LRU (control histórico) | 4 | 32K, 131K | especificación v1 y variante v2 |
| Código ring buffer | 4 | 32K, 131K | `RingBuffer` con overwrite |
| Código token bucket | 4 | 32K, 131K | `TokenBucket` con reloj inyectable |
| Código expiring cache | 4 | 32K, 131K | TTL + LRU combinados |
| JSON estructurado (control) | 2 | 32K, 131K | mismo formato que las rondas 24/24 |
| Calentamiento (fuera de puerta) | 1 | 4K | excluida del criterio |

Las variantes v2 reformulan constantes preservando la superficie de API (verificado por diff de identificadores). Cada celda: temperatura 1, top-p 0,95, top-k 20, thinking medium, semilla 42/43, presupuesto 4.096 tokens de salida.

## Oráculos y su validación

Cada familia nueva tiene un checker congelado en `results/20260910-quality-gate/fixtures/`, ejecutado dentro de bubblewrap con el mismo harness que el LRU histórico (`lru_grade.execute_code`); el alias `check_lru` satisface el RUNNER compartido. LRU conserva el checker fijado del repo.

- Los tres oráculos nuevos pasan contra implementaciones de referencia ingenuas (7/7 checks cada uno).
- **Pantalla de mutantes: 12/12 capturados** (`fixtures/mutant_screen.py`): inversión de orden, rechazo de `None`, sin evicción, `len` roto, refill sin techo, consumo en rechazo, allow-always, get que refresca TTL, sin expiración, evicción MRU, `len` que cuenta expirados, rechazo de valor `None`. Un oráculo que pasara un mutante sería demasiado débil para la puerta.
- La semántica de tiempo usa reloj inyectable (`time_fn`); las comparaciones de tokens se anclan a fracciones binarias exactas para evitar ruido FP64.

## Criterios de la puerta (fijados antes de medir)

Un candidato **pasa** si y solo si, contra el control EXL3 medido en esta misma matriz:

1. **Código: candidato ≥ control** en entregas completas (formato válido + suite generada con ≥8 tests descubiertos y sin fallos + oráculo independiente 7/7), sobre las 16 celdas. Truncados y fallos permanecen en el resultado; no se repara código.
2. **JSON: 2/2** en ambos brazos.
3. **Aceptación MTP**: mediana de `speculative_acceptance_rate` dentro de ±3 puntos porcentuales del control en las celdas de código parejas.
4. **Memoria**: pico asignado ≤ control + 0,5 GiB.

Un empate agregado no demuestra equivalencia: se reportan los fallos concretos por celda en ambos brazos. La decisión la aplica `results/20260910-quality-gate/aggregate.py` sobre las calificaciones y muestras crudas; el umbral de aceptación es simétrico porque una subida inexplicable también indica cambio de comportamiento.

## Artefactos

- Matriz congelada: `results/20260910-quality-gate/prompts.json` + `prompt-manifest.json`.
- Oráculos: `results/20260910-quality-gate/fixtures/*_checks.py` (hashes en el manifiesto).
- Pantalla de mutantes: `results/20260910-quality-gate/fixtures/mutant_screen.py` (12/12).
- Grader: `results/20260910-quality-gate/graders.py` (reutiliza sandbox y checker LRU del repo).
- Runner: `results/20260910-quality-gate/gate_runner.py`, ejecutado con `results/20260908-hybrid-backends/managed.py` (restauración verificada del servicio).
- Control: `results/20260910-quality-gate/control/` (en curso al momento de escribir).

## Resultado del control EXL3 (producción)

Corrida `control/` vía `managed.py` con restauración verificada del servicio.

| Familia | Entregas | Detalle de fallos |
|---|---:|---|
| lru | 4/4 | — |
| ring | 4/4 | — |
| bucket | 1/4 | dos tests generados con una expectativa errónea (oráculo 7/7 en ambos); un truncado |
| ttl (prompt v1, 4096) | 0/4 | 4/4 truncados: la especificación original + suite ≥8 no cabe en 4.096 tokens con thinking medium |
| ttl (prompt v2, 4096) | 1/4 | 3/4 truncados: thinking de 10–13K chars consume el presupuesto (las familias que pasan usan ~7,4K). **Defecto de presupuesto, no del modelo** |
| ttl (prompt v2, 5120) | ver tabla final | presupuesto elevado solo para TTL; el resto de celdas conserva 4.096 |

### Baseline de control congelado

Control EXL3 (producción), merge de `control` + `control-ttl2`:

| Familia | Control | Fallos |
|---|---:|---|
| lru | 4/4 | — |
| ring | 4/4 | — |
| bucket | 1/4 | 2× expectativa errónea en test generado (oráculo 7/7); 1× truncado (4089/4096) |
| ttl | 2/4 | 1× `NameError` por import de `unittest` omitido en el bloque entregado (rejected: execution_error); 1× expectativa errónea en test generado (oráculo 7/7) |
| **total código** | **11/16** | |
| json | 2/2 | — |

Métricas: TTFT mediana 11,3 s, decode mediano 179,9 tok/s, pico asignado 24,79 GiB. Artefacto: `results/20260910-quality-gate/control-baseline.json`.
| json | 2/2 | — |

Métricas de control: TTFT mediana ≈ 40,0 s (mezcla 32K/131K), decode mediana 147,5–176,0 tok/s por familia, pico asignado **24,79 GiB**. Aceptación MTP: no registrada por la sonda en esta configuración (campo `speculative_acceptance_rate` nulo); el criterio 3 se evalúa con los contadores del ciclo verify si el candidato los expone, si no se documenta como no comparable.

Lecciones del control:

1. **La matriz distingue dificultad por familia**: LRU y ring son cómodas; bucket introduce aritmética flotante con expectativas exactas; TTL exige síntesis. Ese gradiente es lo que la familia LRU sola no veía.
2. Los fallos del control son de **tests generados**, no del oráculo independiente: la implementación pasa las 7 comprobaciones pero el modelo escribe un test con expectativa errónea. Es el mismo modo de fallo documentado en septiembre.
3. **Presupuesto uniforme ≠ carga uniforme**: TTL consume ~1,5–1,8× el thinking de las demás familias; la puerta asigna 5.120 tokens solo a esa familia (medido, no adivinado) para no confundir truncamiento estructural con regresión de calidad.
4. Apareció un modo de fallo nuevo no visto en septiembre: `ttl-131072-1` entregó un bloque donde la clase de tests referencia `unittest` sin importarlo (`NameError` en la importación, categoría `execution_error`). El screening estático permite el import pero no obliga a incluirlo; queda como fallo legítimo del modelo, no del oráculo.

## Límites conocidos

- Dos contextos (32K/131K), no 258K: el gate prioriza matriz ancha sobre contexto máximo; la promoción final conserva la verificación de capacidad 262.144 ya existente en las rondas de perfil.
- Una especificación por familia con dos variantes de constantes; no es una tasa general de corrección de programación.
- Los oráculos de reloj cubren la semántica acordada en el prompt; interpretaciones distintas pero razonables del borde de TTL se documentan como fallos de especificación, no del modelo.
