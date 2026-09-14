"""NVFP4 MLP inner module used by the production NVIDIA64 donor.

Scale conventions follow the SGLang/compressed-tensors W4A4 layout measured in
Phase 1/2. GEMMs go through FlashInfer; environment setup belongs to hybrid.py.
"""
import types


def setup_flashinfer():
    import torch
    import flashinfer
    return torch, flashinfer


def swizzle_scale(scale):
    import torch
    assert scale.ndim == 2 and scale.dtype == torch.float8_e4m3fn
    rows, cols = scale.shape
    rp, cp = (rows + 127) // 128 * 128, (cols + 3) // 4 * 4
    padded = torch.zeros((rp, cp), dtype=scale.dtype, device=scale.device)
    padded[:rows, :cols] = scale
    return padded.reshape(rp // 128, 4, 32, cp // 4, 4).permute(0, 3, 2, 1, 4).contiguous().reshape(rp, cp)


class NativeLinear:
    quant_type = "nvfp4_spike"
    bc = None

    def __init__(self, key, tensors, device, out_dtype=None, backend="cutlass"):
        import torch
        self.key = key
        self.weight = tensors["weight_packed"].to(device)
        self.weight_scale = swizzle_scale(tensors["weight_scale"]).to(device)
        self.input_scale = tensors["input_global_scale"].to(device).float()
        self.weight_global_scale = tensors["weight_global_scale"].to(device).float()
        self.alpha = 1 / (self.input_scale * self.weight_global_scale)
        self.out_features, self.in_features = self.weight.shape[0], self.weight.shape[1] * 2
        self.default_out_dtype = out_dtype or torch.float16
        self.backend = backend
        self.bias = None

    def forward(self, x, params=None, out_dtype=None):
        import flashinfer
        import torch
        shape = x.shape[:-1] + (self.out_features,)
        flat = x.reshape(-1, self.in_features)
        if flat.dtype not in (torch.float16, torch.bfloat16):
            flat = flat.to(torch.float16)
        xp, xs = flashinfer.nvfp4_quantize(flat, self.input_scale)
        dtype = out_dtype or self.default_out_dtype
        native_dtype = dtype if dtype in (torch.float16, torch.bfloat16) else torch.float16
        backend = (("b12x" if flat.shape[0] <= 128 else "cutlass")
                   if self.backend == "adaptive" else self.backend)
        out = flashinfer.mm_fp4(
            xp, self.weight.T, xs, self.weight_scale.T, self.alpha, native_dtype, backend=backend)
        return out.to(dtype).reshape(shape)

    def unload(self):
        pass

    def get_tensors(self, key):
        return {key + ".weight_packed": self.weight, key + ".weight_scale": self.weight_scale,
                key + ".input_global_scale": self.input_scale,
                key + ".weight_global_scale": self.weight_global_scale}


def graph_selected_mlps(model):
    import torch
    from exllamav3.modules.mlp import GatedMLP
    wrapped = []
    for module in model:
        if (not isinstance(module, GatedMLP) or not module.gates
                or module.gates[0].quant_type != "nvfp4_spike"):
            continue
        assert module.num_slices == 1 and not module.tp_reduce
        original = module.forward
        module._nvfp4_original_forward = original

        def wrap(original):
            cache, calls = {}, {}

            def forward(self, x, params, out_dtype=None):
                if (x.numel() // x.shape[-1] > 7 or params.get("q_mlp_slice") is not None
                        or "capture" in params):
                    return original(x, params, out_dtype)
                key = (tuple(x.shape), x.dtype, out_dtype)
                calls[key] = calls.get(key, 0) + 1
                if calls[key] < 3:
                    return original(x, params, out_dtype)
                if key not in cache:
                    static = x.clone()
                    torch.cuda.synchronize(x.device)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        output = original(static, params, out_dtype)
                    cache[key] = (graph, static, output)
                graph, static, output = cache[key]
                static.copy_(x)
                graph.replay()
                return output.clone()

            return forward

        module.forward = types.MethodType(wrap(original), module)
        wrapped.append(module.key)
    return wrapped
