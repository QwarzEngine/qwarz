"""Check sampled attention vectors against an FP32 oracle over the same K8/V4 cache."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import json
import math
from pathlib import Path


def query_indices(query_length):
    if query_length < 1:
        raise ValueError("query_length must be positive")
    return sorted({0, query_length // 4, query_length // 2, 3 * query_length // 4, query_length - 1})


@contextmanager
def without_tf32():
    import torch

    original = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original


def sampled_attention(query, key, value, indices, scale=None):
    import torch

    if any(len(tensor.shape) != 4 or tensor.shape[0] != 1 for tensor in (query, key, value)):
        raise ValueError("Oracle expects single-batch [1, sequence, heads, dimension] tensors")
    if key.shape != value.shape or query.shape[-1] != key.shape[-1]:
        raise ValueError("Key/value shapes and query/key dimensions must agree")
    query_length, query_heads, head_dim = query.shape[1:]
    total, kv_heads = key.shape[1:3]
    if not kv_heads or query_heads % kv_heads or total < query_length:
        raise ValueError("Query heads must group evenly onto KV heads and the cache must cover queries")
    if not indices or any(type(index) is not int or not 0 <= index < query_length for index in indices):
        raise ValueError("Sample indices must select existing queries")
    scale = 1. / math.sqrt(head_dim) if scale is None else scale
    if not math.isfinite(scale):
        raise ValueError("Attention scale must be finite")
    group_size = query_heads // kv_heads
    samples = torch.empty((len(indices), query_heads, head_dim), dtype=torch.float32, device=query.device)
    with without_tf32():
        for sample_index, query_index in enumerate(indices):
            valid_keys = total - query_length + query_index + 1
            for query_head in range(query_heads):
                kv_head = query_head // group_size
                scores = key[0, :valid_keys, kv_head].float() @ query[0, query_index, query_head].float()
                weights = torch.softmax(scores * scale, dim=0)
                samples[sample_index, query_head] = weights @ value[0, :valid_keys, kv_head].float()
    return samples


def error_metrics(output, reference):
    import torch

    if output.shape != reference.shape:
        raise ValueError("Output and oracle shapes must agree")
    output, reference = output.detach().float().cpu(), reference.detach().float().cpu()
    finite = bool(torch.isfinite(output).all() and torch.isfinite(reference).all())
    if not finite:
        return {"finite": False, "max_abs": None, "rmse": None, "relative_l2": None}
    difference = output - reference
    return {"finite": True, "max_abs": float(difference.abs().max()),
            "rmse": float(difference.square().mean().sqrt()),
            "relative_l2": float(torch.linalg.vector_norm(difference)) / max(float(torch.linalg.vector_norm(reference)), 1e-30)}


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists")
    import torch
    import triton
    from . import prefill_flash, prefill_microbench

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        parser.error("CUDA is required for the captured attention implementations")
    torch.cuda.set_device(args.device)
    module = importlib.import_module("exllamav3.modules.attention_fn.triton_paged")
    capture = torch.load(args.capture, map_location="cpu", weights_only=True)
    kwargs = prefill_microbench.to_device(prefill_microbench.prepare_kwargs(capture["kwargs"]), torch, args.device)
    total = prefill_flash.validate_capture(kwargs)
    query = kwargs["q"]
    indices = query_indices(query.shape[1])
    report = {
        "scope": "Sampled FP32 attention oracle over the same dequantized K8/V4 cache; not whole-model or BF16-model parity.",
        "reference_dtype": "float32", "reference_tf32": False,
        "capture_path": str(args.capture.resolve()), "capture_sha256": file_hash(args.capture),
        "capture_metadata": capture.get("metadata", {}), "query_shape": list(query.shape),
        "query_indices": indices, "query_heads": list(range(query.shape[2])),
        "sampled_vectors": len(indices) * query.shape[2], "total_kv_tokens": total,
        "valid_kv_tokens_per_query": [total - query.shape[1] + index + 1 for index in indices],
        "softmax_scale": kwargs.get("softmax_scale") if kwargs.get("softmax_scale") is not None else 1. / math.sqrt(query.shape[-1]),
        "baseline_staging": 1, "torch": torch.__version__, "triton": triton.__version__,
        "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(),
        "sources": {str(path): file_hash(path) for path in (
            Path(__file__), Path(module.__file__), Path(prefill_flash.__file__), Path(prefill_microbench.__file__),
        )},
    }
    try:
        with torch.inference_mode():
            with prefill_microbench.staging_context(module, 1):
                baseline = module.paged_attn_triton_prefill(**kwargs).detach().clone()
            flash = prefill_flash.torch_flash_prefill(**kwargs)
            from exllamav3.util.tensor import g_tensor_cache

            scratch_shape = (kwargs["k_cache"].shape[0], 256, 4, 256)
            key = g_tensor_cache.get(query.device, scratch_shape, torch.half, "qc_pf_k").view(1, -1, 4, 256)[:, :total]
            value = g_tensor_cache.get(query.device, scratch_shape, torch.half, "qc_pf_v").view(1, -1, 4, 256)[:, :total]
            reference = sampled_attention(query, key, value, indices, kwargs.get("softmax_scale")).cpu()
            baseline_samples, flash_samples = baseline[0, indices].float().cpu(), flash[0, indices].float().cpu()
            report["baseline"] = error_metrics(baseline_samples, reference)
            report["torch_flash"] = error_metrics(flash_samples, reference)
            report["per_query"] = [
                {"query_index": query_index,
                 "baseline": error_metrics(baseline_samples[sample], reference[sample]),
                 "torch_flash": error_metrics(flash_samples[sample], reference[sample])}
                for sample, query_index in enumerate(indices)
            ]
            report["status"] = "ok" if report["baseline"]["finite"] and report["torch_flash"]["finite"] else "nonfinite"
    except Exception as error:
        report.update(status="error", error_type=type(error).__name__, error=str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
