"""Guarded Flash Attention replay of the captured K8/V4 paged cache."""

from __future__ import annotations


def validate_capture(kwargs):
    import torch

    query = kwargs["q"]
    if len(query.shape) != 4 or query.shape[0] != 1 or query.shape[2:] != (24, 256):
        raise ValueError("Flash replay requires query shape [1, query_length, 24, 256]")
    if query.dtype != torch.half or query.device.type != "cuda" or not query.is_contiguous():
        raise ValueError("Flash replay requires contiguous CUDA fp16 queries")
    if not kwargs.get("causal") or kwargs.get("softcap", 0.) or kwargs.get("sinks") is not None:
        raise ValueError("Flash replay requires causal attention without softcap or sinks")
    window = kwargs.get("window_size")
    if window is not None and window != -1 and window not in ((-1, -1), [-1, -1]):
        raise ValueError("Flash replay requires an unrestricted attention window")
    if any(kwargs.get(name) is not None for name in ("k", "v", "k_new", "v_new")):
        raise ValueError("Flash replay requires an already-appended cache without new K/V")
    quantization = kwargs.get("qc")
    if not isinstance(quantization, (tuple, list)) or len(quantization) != 4 or tuple(quantization[2:]) != (8, 4):
        raise ValueError("Flash replay supports K8/V4 quantization only")
    if kwargs.get("n_kv_heads_override") != 4:
        raise ValueError("Flash replay requires four KV heads")
    appended = kwargs.get("pre_appended_len", 0)
    if type(appended) is not int or appended < 0:
        raise ValueError("pre_appended_len must be a nonnegative integer")
    key_cache, value_cache = kwargs["k_cache"], kwargs["v_cache"]
    key_scales, value_scales = quantization[:2]
    pages = key_cache.shape[0]
    if key_cache.shape != (pages, 256, 256) or value_cache.shape != (pages, 256, 128):
        raise ValueError("Packed cache geometry must match four 256-dimensional K8/V4 heads")
    if key_scales.shape != (pages, 256, 32) or value_scales.shape != key_scales.shape:
        raise ValueError("Quantized cache scales have incompatible geometry")
    table, lengths = kwargs["block_table"], kwargs["cache_seqlens"]
    if len(table.shape) != 2 or table.shape[0] != 1 or not 0 < table.shape[1] <= pages or lengths.shape != (1,):
        raise ValueError("Block table must fit the existing single-batch staging pool")
    for tensor in (key_cache, value_cache, key_scales, value_scales, table, lengths):
        if tensor.device != query.device or not tensor.is_contiguous():
            raise ValueError("Cache tensors must be contiguous on the query device")
    if any(tensor.dtype != torch.int32 for tensor in (key_cache, value_cache, table, lengths)):
        raise ValueError("Packed caches, block table and sequence lengths must use int32")
    if key_scales.dtype != torch.half or value_scales.dtype != torch.half:
        raise ValueError("Cache scales must use fp16")
    cached = kwargs.get("known_cache_len")
    if cached is None:
        # Device->host copy that forces the stream to drain. Production supplies the
        # host-known length through the tuning context; only offline replays land here.
        cached = int(lengths.item())
    elif type(cached) is not int:
        raise ValueError("known_cache_len must be an integer when supplied")
    total = cached + appended
    if cached < 0 or not 0 < query.shape[1] <= total <= table.shape[1] * 256:
        raise ValueError("Query and total KV lengths must fit the referenced page span")
    output = kwargs.get("out")
    if output is not None and (output.shape != query.shape or output.dtype != query.dtype or output.device != query.device):
        raise ValueError("Output must match query shape, dtype and device")
    return total


def torch_flash_prefill(**kwargs):
    import torch
    from torch.backends.cuda import SDPAParams, can_use_flash_attention
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.nn.attention.bias import causal_lower_right
    from torch.nn.functional import scaled_dot_product_attention

    total = validate_capture(kwargs)
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.util.tensor import g_tensor_cache

    query = kwargs["q"]
    key_cache, value_cache = kwargs["k_cache"], kwargs["v_cache"]
    key_scales, value_scales = kwargs["qc"][:2]
    scratch_shape = (key_cache.shape[0], 256, 4, 256)
    staged_key = g_tensor_cache.get(query.device, scratch_shape, torch.half, "qc_pf_k")
    staged_value = g_tensor_cache.get(query.device, scratch_shape, torch.half, "qc_pf_v")
    key = staged_key.view(1, -1, 4, 256)[:, :total].transpose(1, 2)
    value = staged_value.view(1, -1, 4, 256)[:, :total].transpose(1, 2)
    transposed_query = query.transpose(1, 2)
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        params = SDPAParams(transposed_query, key, value, None, 0., False, True)
        if not can_use_flash_attention(params):
            raise RuntimeError("Native Flash Attention does not support this captured geometry; no fallback permitted")
        ext.dequant_cache_paged_window(
            key_cache, key_scales, staged_key, value_cache, value_scales, staged_value,
            kwargs["cache_seqlens"], kwargs["block_table"], 256, kwargs.get("pre_appended_len", 0), 0.,
        )
        result = scaled_dot_product_attention(
            transposed_query, key, value,
            attn_mask=causal_lower_right(query.shape[1], total),
            dropout_p=0., is_causal=False, scale=kwargs.get("softmax_scale"), enable_gqa=True,
        ).transpose(1, 2)
    output = kwargs.get("out")
    if output is not None:
        output.copy_(result)
        return output
    return result.contiguous()
