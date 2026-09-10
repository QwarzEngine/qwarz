from contextlib import contextmanager
from functools import wraps
import inspect
import json


def validate_screen_candidates(candidates):
    names = set()
    if not candidates:
        raise ValueError("candidates must not be empty")
    for candidate in candidates:
        name = candidate.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("candidate names must be nonempty and unique")
        names.add(name)
        chunk = candidate.get("chunk_size", 2048)
        if type(chunk) is not int or chunk <= 0 or chunk % 256:
            raise ValueError("chunk_size must be a positive multiple of the 256-token page")
        if name == "baseline" and (
            candidate.get("kwargs", {}) or candidate.get("implementation", "triton") != "triton"
            or candidate.get("staging", 1) != 1 or chunk != 2048
            or candidate.get("minimum_query", 256) != 256
        ):
            raise ValueError("baseline must retain default Triton settings and chunk_size=2048")
        validate_tuning_settings(candidate)


def validate_tuning_settings(settings):
    options = settings.get("kwargs", {})
    implementation = settings.get("implementation", "triton")
    if implementation not in ("triton", "torch_flash"):
        raise ValueError("unsupported prefill implementation")
    if implementation == "torch_flash" and (options or settings.get("staging", 1) != 1):
        raise ValueError("Flash uses staging without Triton overrides")
    allowed = {"block_m", "block_n", "num_warps", "num_stages", "num_splits"}
    if options.keys() - allowed or settings.get("staging", 1) not in (0, 1):
        raise ValueError("unsupported prefill tuning settings")
    if settings.get("staging", 1) == 0 and "block_n" in options:
        raise ValueError("direct quantized prefill overrides block_n internally")
    if any(type(value) is not int or value <= 0 for value in options.values()):
        raise ValueError("prefill options must be positive integers")


class HostLengthTracker:
    """Remembers the host value of each `cache_seqlens` tensor the attention module uploads.

    The generator builds `cache_seqlens` on the CPU and `Attention.forward` uploads it once per
    forward pass through `get_for_device`. Recording the integer at upload time lets the Flash
    adapter learn the cached length without a device->host `.item()` that drains the stream in
    every attention layer. Records hold the uploaded tensor itself so identity cannot be recycled,
    plus its storage pointer; `get_for_device` already requires the source to stay immutable for
    the lifetime of the params dict, and inference-mode tensors expose no version counter.
    """

    def __init__(self, capacity=8):
        self.capacity = capacity
        self.records = []

    def record(self, source, uploaded):
        if uploaded is None or getattr(source, "device", None) is None or source.device.type != "cpu":
            return
        if tuple(source.shape) != (1,):
            return
        if self.records and self.records[-1][0] is uploaded:
            return
        self.records.append((uploaded, uploaded.data_ptr(), int(source[0])))
        del self.records[:-self.capacity]

    def lookup(self, tensor):
        for uploaded, pointer, value in reversed(self.records):
            if uploaded is tensor:
                return value if uploaded.data_ptr() == pointer else None
        return None

    @contextmanager
    def hooked(self, attention_module):
        original = attention_module.get_for_device

        @wraps(original)
        def tracked(input_dict, key, device, *args, **kwargs):
            uploaded = original(input_dict, key, device, *args, **kwargs)
            if key == "cache_seqlens":
                self.record(input_dict.get(key), uploaded)
            return uploaded

        attention_module.get_for_device = tracked
        try:
            yield self
        finally:
            attention_module.get_for_device = original


def _attention_module():
    from exllamav3.modules import attn

    return attn


@contextmanager
def tuning_context(backend, settings, attention_module=None):
    validate_tuning_settings(settings)
    options = settings.get("kwargs", {})
    original = backend.paged_attn_triton_prefill
    signature = inspect.signature(original)
    staging = backend._qc_staging
    counters = {"overridden_calls": 0, "default_calls": 0, "eligible_query_lengths": {}}
    flash = settings.get("implementation") == "torch_flash"
    tracker = None
    if flash:
        counters.update({"host_length_hits": 0, "host_length_syncs": 0})
        tracker = HostLengthTracker()

    @wraps(original)
    def tuned(*args, **kwargs):
        query = args[0] if args else kwargs["q"]
        eligible = query.shape[1] >= settings.get("minimum_query", 256)
        prior = backend._qc_staging
        try:
            backend._qc_staging = settings.get("staging", staging) if eligible else staging
            if eligible:
                kwargs = kwargs | options
                counters["overridden_calls"] += 1
                length = str(query.shape[1])
                counters["eligible_query_lengths"][length] = counters["eligible_query_lengths"].get(length, 0) + 1
                if flash:
                    from qwasar_bench.prefill_flash import torch_flash_prefill

                    bound = signature.bind(*args, **kwargs)
                    bound.apply_defaults()
                    known = tracker.lookup(bound.arguments.get("cache_seqlens"))
                    counters["host_length_hits" if known is not None else "host_length_syncs"] += 1
                    return torch_flash_prefill(**bound.arguments, known_cache_len=known)
            else:
                counters["default_calls"] += 1
            return original(*args, **kwargs)
        finally:
            backend._qc_staging = prior

    backend.paged_attn_triton_prefill = tuned
    try:
        if flash:
            with tracker.hooked(attention_module if attention_module is not None else _attention_module()):
                yield counters
        else:
            yield counters
    finally:
        backend.paged_attn_triton_prefill = original
        backend._qc_staging = staging


@contextmanager
def workload_tuning_context(generator, settings, *, backend=None, counters_path=None, attention_module=None):
    if settings is None:
        yield None
        return
    validate_screen_candidates([settings])
    if backend is None:
        from exllamav3.modules.attention_fn import triton_paged

        backend = triton_paged
    original_chunk = generator.max_chunk_size
    counters = None
    try:
        generator.max_chunk_size = settings.get("chunk_size", 2048)
        with tuning_context(backend, settings, attention_module=attention_module) as counters:
            yield counters
    finally:
        generator.max_chunk_size = original_chunk
        if counters_path is not None and counters is not None:
            from qwasar_bench.exllamav3_probe import _write_text_atomic

            _write_text_atomic(counters_path, json.dumps(counters, indent=2) + "\n")


@contextmanager
def capture_context(backend, save, minimum_query=256):
    original = backend.paged_attn_triton_prefill
    signature = inspect.signature(original)
    captured = False

    @wraps(original)
    def capture(*args, **kwargs):
        nonlocal captured
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        if not captured and bound.arguments["q"].shape[1] >= minimum_query:
            save(dict(bound.arguments))
            captured = True
        return original(*args, **kwargs)

    backend.paged_attn_triton_prefill = capture
    try:
        yield
    finally:
        backend.paged_attn_triton_prefill = original
