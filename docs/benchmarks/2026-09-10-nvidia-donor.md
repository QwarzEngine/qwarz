# Donante NVIDIA NVFP4 (Fase 1) — 2026-09-10

**Estado:** checkpoint verificado, adaptador con inversión de escala, **oráculo pasado** (`qg10-nvchk7-managed`, `nvidia-check7/report.json`). Puerta A: 48/48 W4A4, RMS máx 0,00021 (umbral 0,003). Puerta B: 12/12 packing, coseno mín vs Minima 0,992 / vs Unsloth 0,984. L2 de pesos vs Minima ≤ 12,6% (calibración, no kernel). Siguiente: A/B NVIDIA-MLP64 vs Minima64 sobre la matriz congelada.

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
- Adaptador: `results/20260910-quality-gate/nvidia_adapter.py`.
- Oráculo BF16: `results/20260910-quality-gate/nvidia_linear_check.py`.
- Resultado: `results/20260910-quality-gate/nvidia-check7/` (48/48 W4A4, 12/12 packing). Fallos previos: `qg10-nvchk{,2,3}` (geometría, NaN, `KeyError` `.weight` EXL3); `nvchk4` Gate B EXL3 inválida; `nvchk5` Unsloth sin capas 56–63; `nvchk6` `print` de `cos_us is None`.
