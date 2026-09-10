# Eliminación de `cache_seqlens.item()` en el adaptador Flash — 2026-09-10

**Resultado:** el adaptador de prefill Flash ya no ejecuta ninguna copia device→host en el camino caliente. En 3.468 llamadas reales al adaptador (tres corridas de matriz de turnos) hubo **3.468 aciertos de longitud en host y 0 sincronizaciones**. El TTFT caliente no cambia de forma medible: las diferencias entre brazos (≤ 4%) son del mismo tamaño que la variación entre corridas del mismo brazo. Es una limpieza estructural verificada, no una mejora de latencia.

## Qué había

`torch_flash_prefill` necesita la longitud ya cacheada como entero de Python para recortar el scratch FP16 y construir la máscara causal lower-right. La obtenía con `cache_seqlens.item()`: una copia de 4 bytes device→host que obliga al host a esperar todo el trabajo GPU encolado. El [perfil de contexto caliente](2026-09-08-hot-context-profile.md) contó 16 de esas copias por forward de prefill (una por capa de atención); en la práctica son **17 por chunk**, porque la capa MTP del draft también pasa por el adaptador.

## Qué se cambió

- `prefill_flash.validate_capture` acepta `known_cache_len`; si viene, no llama a `.item()`. Sin él, el comportamiento anterior se conserva para replays offline (oráculo, microbench).
- `prefill_tuning.HostLengthTracker` envuelve `exllamav3.modules.attn.get_for_device` mientras dura el contexto de tuning. El generador construye `cache_seqlens` en CPU (`torch.tensor([prefill_start])`); al subirlo a GPU, el tracker lee el entero de la copia CPU (sin sincronizar) y lo asocia al tensor subido por identidad y `data_ptr`. La versión del tensor no sirve como guarda: los tensores en `inference_mode` no la exponen.
- `tuning_context` busca la longitud registrada para el `cache_seqlens` recibido y la pasa como `known_cache_len`. Si no hay registro cae a `.item()` y lo cuenta: `prefill-counters.json` ahora incluye `host_length_hits` y `host_length_syncs`.
- El hook se restaura al salir del contexto, también ante fallo. El donante instalado no se modifica.

## Medición

Matriz de turnos (`turn_matrix`), EXL3 5 bpw, MTP6, K8/V4, pool 262.144, perfil `prefill-flash-k8v4.json`, greedy, thinking medium, 256 tokens de salida, cinco ramas calientes por celda. Dos procesos por brazo, en orden control → tracker → tracker → control. El brazo de control usa una copia del código anterior (`/tmp/qwasar-control/src`, `prefill_flash` sha `72f6f822…`); el brazo nuevo usa `src` (`7d1fe6fe…`). Cada proceso se lanzó con `managed.py`, deteniendo y restaurando el servicio.

| Contexto / delta | control-1 | tracker-1 | tracker-2 | control-2 |
|---|---:|---:|---:|---:|
| 32K / 128 | 240 ms | 225 ms | 228 ms | 224 ms |
| 32K / 512 | 444 ms | 412 ms | 415 ms | 415 ms |
| 256K / 128 | 330 ms | 342 ms | 332 ms | 329 ms |
| 256K / 512 | 764 ms | 787 ms | 761 ms | 763 ms |

TTFT mediana de cinco ramas; la dispersión dentro de cada celda es de 1–10 ms. Ingesta fría: 32K 9,5–10,2 s y 258K 135,0–136,8 s en los cuatro procesos, sin separación por brazo.

Lectura: control-1 fue ~15–30 ms más lento a 32K que los otros tres procesos, incluido control-2; tracker-1 fue ~12–25 ms más lento a 256K que los otros tres, incluido tracker-2. El decode (ruta no tocada) se movió en la misma dirección en esos procesos. Son efectos de proceso/orden, no del cambio. **No se reclama ninguna reducción de TTFT.** Esto confirma la advertencia del perfil: los 131,8 ms de `cudaStreamSynchronize` se solapaban con trabajo GPU y no eran recuperables.

Contadores del brazo nuevo por proceso: `overridden_calls` 1.156, `host_length_hits` 1.156, `host_length_syncs` 0. Formas cubiertas: Q = 8192, 7936, 7168, 512, 256, 241, 129, 127, 113, 111. El brazo de control registra las mismas 1.156 llamadas sin los contadores nuevos.

Calidad: no evaluable en esta matriz (256 tokens truncan el razonamiento en todas las celdas de ambos brazos). La aritmética no cambia: el mismo `total` alimenta el mismo recorte y la misma máscara; los tests unitarios comprueban la equivalencia y que `.item()` no se invoca cuando hay longitud conocida.

## Qué habilita

El host ya no se detiene en cada capa de atención durante el prefill Flash. Es requisito del [diseño de atención directa K8/V4](../superpowers/specs/2026-09-08-direct-k8v4-attention-design.md) («sin sincronización `.item()` en el camino caliente») y deja el adaptador listo para captura de grafos o solapamiento CPU/GPU, cuya ganancia habrá que medir por separado.

## Artefactos

- Código: `src/qwasar_bench/prefill_flash.py`, `src/qwasar_bench/prefill_tuning.py`; tests en `tests/test_prefill_flash.py` y `tests/test_prefill_tuning.py`.
- Corridas: `results/20260910-item-sync/{control-1,tracked-1,tracked-2,control-2}/` con `matrix-summary.json`, `prefill-counters.json`, muestras y eventos; `*-managed/` con configuración del servicio antes/después y logs. `run_arm.sh` reproduce cada brazo.
- El servicio se restauró después de cada proceso con la configuración original y ahora importa el adaptador actualizado desde `src`.
