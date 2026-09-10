# Donante NVIDIA NVFP4 (Fase 1) — 2026-09-10

**Estado:** checkpoint verificado, adaptador con inversión de **ambas** escalas globales, oráculo de cuatro puertas pasado (`qg10-nvchk10-managed`, `nvidia-check10/report.json`). A/B NVIDIA-MLP64 sobre la matriz congelada relanzado (`candidate-nvidia64d`).

## Corrección (18:00): el primer A/B produjo ruido en 19/19 celdas con el oráculo en verde

`candidate-nvidia64b` completó la matriz pero todas las completions eran ruido (`'0.0.00\n0.0.02…'`, `'<'`, `''`), aceptación MTP 0–0,3, `stop_string` a los 2–5 tokens. Nadie lo había calificado.

Causa: el adaptador invertía `weight_scale_2` pero **no** `input_scale`. ModelOpt guarda ambos globales en forma multiplicativa; el kernel compartido (`NativeLinear`, convención SGLang) espera ambos en forma recíproca: `flashinfer.nvfp4_quantize(x, g)` multiplica `x` por `g` antes de cuantizar en bloques, y `alpha = 1/(g_in·g_w)` lo deshace.

| Tensor | NVIDIA (crudo) | Minima (forma del kernel) | Adaptador corregido |
|---|---:|---:|---:|
| `layers.0.mlp.gate_proj` input | 0,00126 | 440 | 1/0,00126 = **793** |
| `layers.0.mlp.gate_proj` weight | 0,000157 | 6400 | 6372 |

Con la escala cruda, el error es un factor `raw²` sobre los block-scales E4M3. La cuantización NVFP4 es invariante a la escala mientras el block-scale quede en el rango representable (~2⁻⁹…2⁸), así que sólo colapsan las matrices con `raw² < ~2⁻¹⁷`: **12/192**, todas en capas tempranas (`gate/up` de la capa 0 son 2⁻¹⁹·³ → activaciones exactamente cero). El resto sobrevive con precisión degradada. Eso alcanza para volver ruido al modelo y explica por qué el kernel nunca produjo NaN.

**Por qué el oráculo no lo vio:** la Puerta A cuantizaba las activaciones de referencia con el mismo `input_scale` que el kernel (`w4a4_activation(x, native.input_scale)`); referencia y kernel se corrompían igual y el cheque pasaba. Es la misma falla de «referencia autoconsistente» que ya se había identificado para los pesos, repetida en las activaciones. Además, el oráculo reconstruía los tensores del kernel localmente en vez de usar `tensors_for()` del adaptador, así que ni siquiera ejercitaba el código del cargador.

## Oráculo corregido (`nvidia_linear_check.py`, `nvidia-check10`)

- Los tensores del kernel salen de `nvidia_adapter.tensors_for()` (el mismo camino que el loader) y se asserta que ambos globales sean > 1.
- **Puerta C** (nueva, dura): kernel vs `x_fp16 @ W_modelopt` con activaciones **sin cuantizar**; umbral RMS 0,20. Resultado 48/48, máx **0,098** (el ruido esperado de cuantizar activaciones a FP4 en el rango calibrado; Minima64 corre con el mismo error y dio 4/6).
- **Puerta D** (nueva, dura): ida y vuelta W4A4 de `x` con el global del adaptador; umbral 0,20. 48/48, máx 0,096.
- **Contraste rojo:** cada puerta se repite con el `input_scale` crudo y debe dar error > 0,9 donde el desajuste excede el rango E4M3 (`2·log2(raw) < −17`). 8/8 checks discriminantes (capa 0 `gate/up`) dan **1,0**. En `down_proj` (2⁻⁷·⁴) y capas 21/42/63 el brazo rojo pasa numéricamente y se registra como no discriminante; por eso el umbral se ancló al rango E4M3 y no a «toda matriz debe fallar».
- Puertas A y B sin cambios: 48/48 RMS 0,00021; 12/12 coseno ≥ 0,992/0,984.

Corridas intermedias: `nvchk8` (contraste rojo asertado en `down_proj`, falso positivo del oráculo), `nvchk9` (umbral 2⁻¹² insuficiente en capa 21). Ambas conservadas.

Los tensores pasan en los oráculos; el A/B `candidate-nvidia64b` queda invalidado y se relanza como `candidate-nvidia64d` con el adaptador corregido. El intento `candidate-nvidia64c` se abortó por una interrupción del host antes de la primera muestra.

## Hallazgo: las convenciones de escalas de NVIDIA y Unsloth son inversas

Medido directamente en `model.language_model.layers.0.mlp.gate_proj` (misma matriz lógica en ambos checkpoints):

| Checkpoint | Reconstrucción | std | absmax |
|---|---|---:|---:|
| NVIDIA `q * block_scale * global` (multiplicativa) | **0,0101** | **0,422** | ✅ pesos plausibles |
| NVIDIA `q * block_scale / global` (como kernel SGLang) | 411.043 | 17,1e6 | ✗ explode |
| Unsloth `q * block_scale / global` (su convención) | **0,0102** | **0,420** | ✅ |
| Unsloth `q * block_scale * global` | 417.905 | 17,2e6 | ✗ explode |

ModelOpt guarda globales ~1,5e-4 (= 1/6372) y las usa multiplicando; SGLang/Unsloth guarda la recíproca y divide. El `NativeLinear` compartido (`alpha = 1/(input*global)`) fue escrito para Unsloth. **Adaptación:** el shim NVIDIA convierte `weight_global_scale → 1/global` al cargar, presentando la forma divisora que el kernel espera (`nvidia_adapter.py`). Un inversor de una línea; sin cambios en el kernel.

## Por qué el oráculo tipo Minima no alcanzaba

El oráculo heredado construía la referencia a partir de los **mismos operandos cuantizados** que el kernel. Una convención de escala errada produce referencia y kernel errados **de forma idéntica**: el cheque pasa siempre. Se reemplazó por una referencia **BF16 real del artefacto EXL3** (misma matriz lógica), que es lo que "recibir los pesos correctos" significa. Puerta: L2 relativo < 5% contra BF16 en 12 matrices × M=1/7/128/2048 (distingue convenciones erradas, que dan error ≫ 100% o NaN, del ruido de cuantización real).

## Cobertura y exclusiones

- **Importado:** 192 MLP (64 capas × gate/up/down), convención convertida como arriba.
- **No importado (intencional):** GDN y atención FP8 (frontera validada por ambos proveedores y nuestras propias mediciones de decode), lm_head FP8 (el head target no es el cuello de botella; el del draft se ataca en Fase 4), torre visual, tokenizer (byte-idéntico al pin, verificado).

## artefactos

- Descarga + verificación: `results/20260910-quality-gate/nvidia-download.json`.
- Adaptador: `results/20260910-quality-gate/nvidia_adapter.py` (`convert_modelopt_tensors` invierte ambos globales).
- Oráculo: `results/20260910-quality-gate/nvidia_linear_check.py` (puertas A/B/C/D + contraste rojo).
- Resultado vigente: `results/20260910-quality-gate/nvidia-check10/` (A 48/48, B 12/12, C 48/48, D 48/48, rojo 8/8). `nvidia-check7` fue el falso verde con el adaptador incompleto. Fallos previos: `qg10-nvchk{,2,3}` (geometría, NaN, `KeyError` `.weight` EXL3); `nvchk4` Gate B EXL3 inválida; `nvchk5` Unsloth sin capas 56–63; `nvchk6` `print` de `cos_us is None`; `nvchk8/9` calibración del contraste rojo.
- A/B inválido: `results/20260910-quality-gate/candidate-nvidia64b/` (ruido en 19/19, no calificar). Relanzado: `candidate-nvidia64d`.
