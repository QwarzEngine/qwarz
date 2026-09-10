from types import SimpleNamespace
import importlib

import pytest


def module():
    try:
        return importlib.import_module("qwasar_bench.prefill_tuning")
    except ModuleNotFoundError:
        pytest.fail("prefill tuning helper missing")


def test_tuning_changes_only_eligible_prefill_and_restores():
    helper = module()
    calls = []

    def original(q, **kwargs):
        calls.append((backend._qc_staging, kwargs))
        return "output"

    backend = SimpleNamespace(paged_attn_triton_prefill=original, _qc_staging=1)
    settings = {"name": "tile", "staging": 0, "kwargs": {"block_m": 128, "num_splits": 2}}
    with helper.tuning_context(backend, settings) as counters:
        assert backend.paged_attn_triton_prefill(SimpleNamespace(shape=(1, 2048, 24, 256))) == "output"
        assert backend.paged_attn_triton_prefill(SimpleNamespace(shape=(1, 113, 24, 256))) == "output"
        assert counters["overridden_calls"] == 1
    assert calls == [(0, {"block_m": 128, "num_splits": 2}), (1, {})]
    assert backend.paged_attn_triton_prefill is original
    assert backend._qc_staging == 1


def test_tuning_restores_after_failure_and_rejects_unknown_options():
    helper = module()

    def original(q, **kwargs):
        raise RuntimeError("kernel failed")

    backend = SimpleNamespace(paged_attn_triton_prefill=original, _qc_staging=1)
    with pytest.raises(ValueError):
        with helper.tuning_context(backend, {"staging": 1, "kwargs": {"invented": 3}}):
            pass
    with pytest.raises(RuntimeError):
        with helper.tuning_context(backend, {"staging": 0, "kwargs": {}}):
            backend.paged_attn_triton_prefill(SimpleNamespace(shape=(1, 256, 24, 256)))
    assert backend._qc_staging == 1
    assert backend.paged_attn_triton_prefill is original


def test_capture_is_once_and_preserves_original_result():
    helper = module()
    captured = []

    def original(q, k=None, causal=False):
        return q

    backend = SimpleNamespace(paged_attn_triton_prefill=original)
    query = SimpleNamespace(shape=(1, 2048, 24, 256))
    with helper.capture_context(backend, captured.append, minimum_query=256):
        assert backend.paged_attn_triton_prefill(query, causal=True) is query
        backend.paged_attn_triton_prefill(query, causal=True)
    assert len(captured) == 1
    assert captured[0] == {"q": query, "k": None, "causal": True}
    assert backend.paged_attn_triton_prefill is original


@pytest.mark.parametrize("size", [0, -256, 1, 255, 257, 1000])
def test_invalid_chunk_rejected_before_execution(size):
    helper = module()
    with pytest.raises(ValueError):
        helper.validate_screen_candidates([{"name": "bad", "staging": 1, "kwargs": {}, "chunk_size": size}])


def test_valid_chunks_and_unique_candidate_names():
    helper = module()
    helper.validate_screen_candidates([{"name": "baseline", "staging": 1, "kwargs": {}},
                                       {"name": "large", "staging": 1, "kwargs": {}, "chunk_size": 4096}])
    with pytest.raises(ValueError):
        helper.validate_screen_candidates([{"name": "same"}, {"name": "same"}])


def test_direct_block_n_override_is_rejected():
    helper = module()
    backend = SimpleNamespace(paged_attn_triton_prefill=lambda **kwargs: None, _qc_staging=1)
    with pytest.raises(ValueError):
        with helper.tuning_context(backend, {"staging": 0, "kwargs": {"block_n": 32}}):
            pass


def test_flash_dispatch_keeps_small_queries_on_baseline(monkeypatch):
    helper = module()
    from qwasar_bench import prefill_flash

    calls = []
    monkeypatch.setattr(prefill_flash, "torch_flash_prefill", lambda **kwargs: calls.append(kwargs) or "flash")

    def original(q, causal=True):
        return "baseline"

    backend = SimpleNamespace(paged_attn_triton_prefill=original, _qc_staging=1)
    attention = SimpleNamespace(get_for_device=lambda params, key, device: params[key])
    with helper.tuning_context(backend, {"implementation": "torch_flash", "staging": 1, "kwargs": {}},
                               attention_module=attention):
        assert backend.paged_attn_triton_prefill(SimpleNamespace(shape=(1, 2048, 24, 256))) == "flash"
        assert backend.paged_attn_triton_prefill(SimpleNamespace(shape=(1, 113, 24, 256))) == "baseline"
    assert calls[0]["causal"] is True
    assert calls[0]["known_cache_len"] is None


class FakeTensor:
    def __init__(self, value, device="cpu"):
        self.value = value
        self.shape = (1,)
        self.device = SimpleNamespace(type=device)
        self.pointer = id(self)

    def data_ptr(self):
        return self.pointer

    def __getitem__(self, index):
        assert index == 0
        return self.value

    def item(self):
        pytest.fail("device->host synchronization must not happen on the hot path")


def flash_harness(helper, monkeypatch):
    from qwasar_bench import prefill_flash

    calls = []
    monkeypatch.setattr(prefill_flash, "torch_flash_prefill", lambda **kwargs: calls.append(kwargs) or "flash")

    def original(q, k=None, v=None, k_cache=None, v_cache=None, block_table=None, cache_seqlens=None, causal=True):
        return "baseline"

    uploads = {}

    def get_for_device(params, key, device):
        source = params[key]
        uploaded = uploads.get(id(source))
        if uploaded is None:
            uploaded = uploads[id(source)] = FakeTensor(source.value, device="cuda")
        return uploaded

    backend = SimpleNamespace(paged_attn_triton_prefill=original, _qc_staging=1)
    attention = SimpleNamespace(get_for_device=get_for_device)
    settings = {"implementation": "torch_flash", "staging": 1, "kwargs": {}}
    return backend, attention, settings, calls


def test_flash_receives_host_length_recorded_at_upload_without_item(monkeypatch):
    helper = module()
    backend, attention, settings, calls = flash_harness(helper, monkeypatch)
    hook = attention.get_for_device
    query = SimpleNamespace(shape=(1, 2048, 24, 256))
    with helper.tuning_context(backend, settings, attention_module=attention) as counters:
        assert attention.get_for_device is not hook
        lengths = FakeTensor(249856)
        params = {"cache_seqlens": lengths, "positions": lengths}
        uploaded = attention.get_for_device(params, "cache_seqlens", "cuda")
        assert attention.get_for_device(params, "positions", "cuda") is uploaded
        for _ in range(16):
            assert attention.get_for_device(params, "cache_seqlens", "cuda") is uploaded
            assert backend.paged_attn_triton_prefill(query, cache_seqlens=uploaded, causal=True) == "flash"
    assert attention.get_for_device is hook
    assert len(calls) == 16 and all(call["known_cache_len"] == 249856 for call in calls)
    assert all(call["cache_seqlens"] is uploaded for call in calls)
    assert counters["host_length_hits"] == 16 and counters["host_length_syncs"] == 0


def test_unrecorded_or_restorage_lengths_fall_back_and_are_counted(monkeypatch):
    helper = module()
    backend, attention, settings, calls = flash_harness(helper, monkeypatch)
    query = SimpleNamespace(shape=(1, 2048, 24, 256))
    with helper.tuning_context(backend, settings, attention_module=attention) as counters:
        foreign = FakeTensor(7, device="cuda")
        backend.paged_attn_triton_prefill(query, cache_seqlens=foreign, causal=True)
        params = {"cache_seqlens": FakeTensor(512)}
        uploaded = attention.get_for_device(params, "cache_seqlens", "cuda")
        backend.paged_attn_triton_prefill(query, cache_seqlens=uploaded, causal=True)
        uploaded.pointer += 4
        backend.paged_attn_triton_prefill(query, cache_seqlens=uploaded, causal=True)
    assert [call["known_cache_len"] for call in calls] == [None, 512, None]
    assert counters["host_length_hits"] == 1 and counters["host_length_syncs"] == 2


def test_tracker_ignores_device_or_batched_sources_and_bounds_its_memory():
    tracker = module().HostLengthTracker(capacity=2)
    tracker.record(FakeTensor(3, device="cuda"), FakeTensor(3, device="cuda"))
    batched = FakeTensor(3)
    batched.shape = (2,)
    tracker.record(batched, FakeTensor(3, device="cuda"))
    tracker.record(None, FakeTensor(3, device="cuda"))
    assert tracker.records == []
    uploads = [FakeTensor(index, device="cuda") for index in range(3)]
    for index, uploaded in enumerate(uploads):
        tracker.record(FakeTensor(index), uploaded)
    assert tracker.lookup(uploads[0]) is None
    assert tracker.lookup(uploads[1]) == 1 and tracker.lookup(uploads[2]) == 2
    assert tracker.lookup(FakeTensor(2, device="cuda")) is None


def test_tracker_hook_is_restored_even_when_the_workload_fails(monkeypatch):
    helper = module()
    backend, attention, settings, _ = flash_harness(helper, monkeypatch)
    hook = attention.get_for_device
    with pytest.raises(RuntimeError, match="boom"):
        with helper.tuning_context(backend, settings, attention_module=attention):
            raise RuntimeError("boom")
    assert attention.get_for_device is hook
    assert backend._qc_staging == 1


@pytest.mark.parametrize("settings", [{"implementation": "typo"},
                                    {"implementation": "torch_flash", "kwargs": {"num_warps": 4}},
                                    {"implementation": "torch_flash", "staging": 0}])
def test_rejects_inapplicable_flash_settings(settings):
    with pytest.raises(ValueError):
        module().validate_tuning_settings(settings)


@pytest.mark.parametrize("override", [
    {"kwargs": {"block_m": 128}}, {"implementation": "torch_flash"},
    {"staging": 0}, {"chunk_size": 4096}, {"minimum_query": 113},
])
def test_baseline_name_cannot_disguise_a_changed_control(override):
    with pytest.raises(ValueError, match="baseline"):
        module().validate_screen_candidates([{"name": "baseline", **override}])


def test_workload_without_settings_never_touches_generator_or_backend(tmp_path):
    path = tmp_path / "counters.json"
    with module().workload_tuning_context(object(), None, backend=object(), counters_path=path) as counters:
        assert counters is None
    assert not path.exists()


@pytest.mark.parametrize("fail", [False, True])
def test_workload_context_restores_chunk_and_backend_and_persists_counters(tmp_path, fail):
    import json
    from contextlib import nullcontext

    helper = module()
    generator = SimpleNamespace(max_chunk_size=1024)

    def original(q, block_m=None):
        return generator.max_chunk_size, backend._qc_staging, block_m

    backend = SimpleNamespace(paged_attn_triton_prefill=original, _qc_staging=1)
    settings = {"name": "candidate", "staging": 0, "kwargs": {"block_m": 128}, "chunk_size": 4096}
    path = tmp_path / "counters.json"
    with pytest.raises(RuntimeError, match="workload failed") if fail else nullcontext():
        with helper.workload_tuning_context(generator, settings, backend=backend, counters_path=path) as counters:
            assert backend.paged_attn_triton_prefill(SimpleNamespace(shape=(1, 2048, 24, 256))) == (4096, 0, 128)
            assert backend.paged_attn_triton_prefill(SimpleNamespace(shape=(1, 113, 24, 256))) == (4096, 1, None)
            if fail:
                raise RuntimeError("workload failed")
    assert generator.max_chunk_size == 1024
    assert backend.paged_attn_triton_prefill is original
    assert backend._qc_staging == 1
    assert json.loads(path.read_text()) == counters == {
        "overridden_calls": 1, "default_calls": 1, "eligible_query_lengths": {"2048": 1},
    }


def test_omitted_chunk_uses_declared_2048_default_then_restores_generator():
    generator = SimpleNamespace(max_chunk_size=1024)
    backend = SimpleNamespace(paged_attn_triton_prefill=lambda q: q, _qc_staging=1)
    with module().workload_tuning_context(generator, {"name": "baseline"}, backend=backend):
        assert generator.max_chunk_size == 2048
    assert generator.max_chunk_size == 1024
