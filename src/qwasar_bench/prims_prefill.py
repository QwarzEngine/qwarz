"""Large-query FP8 PRIMS route with resident K8/V4 and P-scale 256."""
import torch
import triton
import triton.language as tl


@triton.jit
def repage(inp, out, T, N, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    d = i % 256
    t = i // 256 % 128
    h = i // 32768 % 4
    p = i // 131072
    token = p * 128 + t
    x = tl.load(inp + (token * 4 + h) * 256 + d, (i < N) & (token < T), other=0).to(tl.float32)
    tl.store(out + i, tl.minimum(448., tl.maximum(-448., x)), i < N)


def install(min_q=8192):
    import flashinfer
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.util.tensor import g_tensor_cache
    import qwasar_bench.prefill_flash as flash

    original = flash.torch_flash_prefill
    buffers, wrappers, audit = {}, {}, {}
    counts = {"flash_calls": 0, "prims_calls": 0, "query_histogram": {}}

    def run(**kw):
        q = kw["q"]
        length = q.shape[1]
        if length < min_q:
            counts["flash_calls"] += 1
            return original(**kw)
        total = flash.validate_capture(kw)
        pages = (total + 127) // 128
        last = (total - 1) % 128 + 1
        device = q.device
        if device not in buffers:
            cap = kw["k_cache"].shape[0] * 2
            k8 = torch.empty((cap, 4, 128, 256), device=device, dtype=torch.float8_e4m3fn)
            v8 = torch.empty_like(k8)
            workspace = torch.empty(16 << 20, device=device, dtype=torch.uint8)
            buffers[device] = (k8, v8, workspace)
        k8, v8, workspace = buffers[device]
        shape = (kw["k_cache"].shape[0], 256, 4, 256)
        sk = g_tensor_cache.get(device, shape, torch.half, "qc_pf_k")
        sv = g_tensor_cache.get(device, shape, torch.half, "qc_pf_v")
        ext.dequant_cache_paged_window(
            kw["k_cache"], kw["qc"][0], sk, kw["v_cache"], kw["qc"][1], sv,
            kw["cache_seqlens"], kw["block_table"], 256, kw["pre_appended_len"], 0.)
        layer_key = int(kw["k_cache"].data_ptr())
        if layer_key not in audit:
            audit[layer_key] = torch.stack((
                q.abs().amax(),
                sk.view(-1, 4, 256)[:total].abs().amax(),
                sv.view(-1, 4, 256)[:total].abs().amax())).float()
        n = pages * 4 * 128 * 256
        repage[(triton.cdiv(n, 1024),)](sk, k8, total, n, 1024)
        repage[(triton.cdiv(n, 1024),)](sv, v8, total, n, 1024)
        q8 = q[0].clamp(-448, 448).to(torch.float8_e4m3fn)
        key = (length, total, kw["softmax_scale"])
        if wrappers.get("key") != key:
            if "wrapper" not in wrappers:
                wrappers["wrapper"] = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                    workspace, "HND", backend="cute-dsl-prims")
            wrapper = wrappers["wrapper"]
            qo = torch.tensor([0, length], device=device, dtype=torch.int32)
            pi = torch.tensor([0, pages], device=device, dtype=torch.int32)
            indices = torch.arange(pages, device=device, dtype=torch.int32)
            lasts = torch.tensor([last], device=device, dtype=torch.int32)
            wrapper.plan(
                qo, pi, indices, lasts, 24, 4, 256, 128, head_dim_vo=256, causal=True,
                sm_scale=kw["softmax_scale"], q_data_type=torch.float8_e4m3fn,
                kv_data_type=torch.float8_e4m3fn, o_data_type=torch.float16,
                block_tables=indices.view(1, -1))
            wrappers["key"] = key
        y = wrappers["wrapper"].run(
            q8, (k8, v8), q_scale=1., k_scale=1., v_scale=1 / 256, enable_pdl=False).unsqueeze(0)
        counts["prims_calls"] += 1
        counts["query_histogram"][str(length)] = counts["query_histogram"].get(str(length), 0) + 1
        if kw.get("out") is not None:
            kw["out"].copy_(y)
            return kw["out"]
        return y

    flash.torch_flash_prefill = run
    return counts, audit
