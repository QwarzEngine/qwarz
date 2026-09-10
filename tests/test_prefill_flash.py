from __future__ import annotations

from types import SimpleNamespace

import pytest

from qwasar_bench import prefill_flash as flash


@pytest.fixture
def capture():
    torch = pytest.importorskip("torch")
    device = torch.device("cuda:0")

    def tensor(shape, dtype=torch.half):
        return SimpleNamespace(shape=shape, dtype=dtype, device=device,
                               is_contiguous=lambda: True)

    lengths = tensor((1,), torch.int32)
    lengths.item = lambda: 300
    return dict(q=tensor((1, 8, 24, 256)), k=None, v=None, k_new=None, v_new=None,
                k_cache=tensor((4, 256, 256), torch.int32),
                v_cache=tensor((4, 256, 128), torch.int32),
                block_table=tensor((1, 2), torch.int32), cache_seqlens=lengths,
                qc=(tensor((4, 256, 32)), tensor((4, 256, 32)), 8, 4),
                pre_appended_len=16, n_kv_heads_override=4, causal=True,
                window_size=(-1, -1), softcap=0., sinks=None, out=None)


def test_total_includes_preappended_chunk_even_for_query_subset(capture):
    assert flash.validate_capture(capture) == 316


@pytest.mark.parametrize(("key", "value"), [
    ("causal", False), ("softcap", 1.), ("sinks", object()),
    ("window_size", (128, -1)), ("k", object()), ("v", object()),
    ("k_new", object()), ("v_new", object()), ("pre_appended_len", -1),
    ("n_kv_heads_override", 8), ("qc", (None, None, 4, 4)),
])
def test_unsupported_inputs_are_rejected_before_dequantization(capture, key, value):
    capture[key] = value
    with pytest.raises(ValueError):
        flash.validate_capture(capture)


@pytest.mark.parametrize("shape", [(2, 8, 24, 256), (1, 8, 24, 128), (1, 8, 32, 256)])
def test_only_captured_model_head_geometry_is_supported(capture, shape):
    capture["q"].shape = shape
    with pytest.raises(ValueError):
        flash.validate_capture(capture)


def test_cpu_or_float32_query_is_rejected(capture):
    torch = pytest.importorskip("torch")
    capture["q"].dtype = torch.float32
    with pytest.raises(ValueError):
        flash.validate_capture(capture)
    capture["q"].dtype = torch.half
    capture["q"].device = torch.device("cpu")
    with pytest.raises(ValueError):
        flash.validate_capture(capture)


def test_window_cannot_exceed_shared_staging_pool(capture):
    capture["block_table"].shape = (1, 5)
    with pytest.raises(ValueError):
        flash.validate_capture(capture)


def test_total_cannot_exceed_referenced_page_span(capture):
    capture["cache_seqlens"].item = lambda: 512
    with pytest.raises(ValueError):
        flash.validate_capture(capture)


def test_known_host_length_avoids_device_synchronization(capture):
    capture["cache_seqlens"].item = lambda: pytest.fail("known length must not trigger .item()")
    capture["known_cache_len"] = 300
    assert flash.validate_capture(capture) == 316
    capture["known_cache_len"] = 512
    with pytest.raises(ValueError):
        flash.validate_capture(capture)


@pytest.mark.parametrize("value", [300.0, "300", True])
def test_known_host_length_must_be_an_integer(capture, value):
    capture["known_cache_len"] = value
    with pytest.raises(ValueError, match="known_cache_len"):
        flash.validate_capture(capture)


def test_packed_dimensions_must_match_k8v4(capture):
    capture["v_cache"].shape = (4, 256, 256)
    with pytest.raises(ValueError):
        flash.validate_capture(capture)


@pytest.fixture
def replay_backend(capture, monkeypatch):
    import sys
    from types import ModuleType

    torch = pytest.importorskip("torch")
    import torch.nn.attention.bias

    class Query:
        def __init__(self, tensor):
            self.tensor = tensor
            self.shape = tensor.shape
            self.dtype = tensor.dtype
            self.device = torch.device("cuda:0")

        def is_contiguous(self):
            return self.tensor.is_contiguous()

        def transpose(self, first, second):
            return self.tensor.transpose(first, second)

        def copy_(self, tensor):
            self.tensor.copy_(tensor)

    capture["q"] = Query(torch.zeros((1, 8, 24, 256), dtype=torch.half))
    capture["softmax_scale"] = .125
    state = {"available": True, "dequantized": False, "dispatched": False, "scratch": {}}

    def get(device, shape, dtype, key):
        assert device == torch.device("cuda:0")
        assert shape == (4, 256, 4, 256)
        assert dtype == torch.half
        assert key in ("qc_pf_k", "qc_pf_v")
        tensor = torch.empty(shape, dtype=dtype)
        state["scratch"][key] = tensor
        return tensor

    def dequantize(key_cache, key_scales, key_out, value_cache, value_scales, value_out,
                   lengths, table, page_size, appended, compand):
        assert key_cache is capture["k_cache"]
        assert value_cache is capture["v_cache"]
        assert key_scales is capture["qc"][0]
        assert value_scales is capture["qc"][1]
        assert lengths is capture["cache_seqlens"]
        assert table is capture["block_table"]
        assert (page_size, appended, compand) == (256, 16, 0.)
        key_out.view(-1, 4, 256).copy_(torch.arange(1024, dtype=torch.half).view(-1, 1, 1))
        value_out.copy_(key_out + 1000)
        state["dequantized"] = True

    def attention(query, key, value, *, attn_mask, dropout_p, is_causal, scale, enable_gqa):
        assert state["dequantized"]
        assert torch.backends.cuda.flash_sdp_enabled()
        assert not torch.backends.cuda.math_sdp_enabled()
        assert not torch.backends.cuda.mem_efficient_sdp_enabled()
        assert query.shape == (1, 24, 8, 256)
        assert key.shape == value.shape == (1, 4, 316, 256)
        assert key[0, 0, :, 0].tolist() == list(range(316))
        assert value[0, 0, :, 0].tolist() == list(range(1000, 1316))
        assert attn_mask.variant == torch.nn.attention.bias.CausalVariant.LOWER_RIGHT
        assert (attn_mask.seq_len_q, attn_mask.seq_len_kv) == (8, 316)
        assert (dropout_p, is_causal, scale, enable_gqa) == (0., False, .125, True)
        state["dispatched"] = True
        return torch.full_like(query, 7.)

    def can_use(params):
        assert params.enable_gqa
        assert params.attn_mask is None
        assert params.is_causal is False
        return state["available"]

    def forbid_mask(*args, **kwargs):
        raise AssertionError("Dense causal mask must never be materialized")

    package = ModuleType("exllamav3")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "exllamav3", package)
    monkeypatch.setitem(sys.modules, "exllamav3.ext", SimpleNamespace(
        exllamav3_ext=SimpleNamespace(dequant_cache_paged_window=dequantize)))
    monkeypatch.setitem(sys.modules, "exllamav3.util.tensor", SimpleNamespace(
        g_tensor_cache=SimpleNamespace(get=get)))
    monkeypatch.setattr(torch.backends.cuda, "can_use_flash_attention", can_use)
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", attention)
    monkeypatch.setattr(torch.nn.attention.bias.CausalBias, "_materialize", forbid_mask)
    return capture, state, Query


@pytest.mark.parametrize("provide_output", [False, True])
def test_flash_replay_preserves_page_order_causality_and_output_contract(replay_backend, provide_output):
    torch = pytest.importorskip("torch")
    capture, state, query_type = replay_backend
    if provide_output:
        capture["out"] = query_type(torch.empty((1, 8, 24, 256), dtype=torch.half))
    output = flash.torch_flash_prefill(**capture)
    if provide_output:
        assert output is capture["out"]
        output = output.tensor
    assert output.shape == (1, 8, 24, 256)
    assert output.is_contiguous()
    assert bool((output == 7.).all())
    assert set(state["scratch"]) == {"qc_pf_k", "qc_pf_v"}
    assert state["dispatched"]


def test_flash_replay_with_known_length_never_reads_the_device_tensor(replay_backend):
    capture, state, _ = replay_backend
    capture["cache_seqlens"].item = lambda: pytest.fail("hot path must not synchronize")
    capture["known_cache_len"] = 300
    output = flash.torch_flash_prefill(**capture)
    assert output.shape == (1, 8, 24, 256)
    assert state["dispatched"]


def test_unavailable_flash_rejects_before_dequantization_without_fallback(replay_backend):
    capture, state, _ = replay_backend
    state["available"] = False
    with pytest.raises(RuntimeError, match="no fallback"):
        flash.torch_flash_prefill(**capture)
    assert not state["dequantized"]
    assert not state["dispatched"]
