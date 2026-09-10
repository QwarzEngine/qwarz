from __future__ import annotations

import math

import pytest

from qwasar_bench import prefill_oracle as oracle


def test_default_samples_include_causal_boundaries_without_duplicate_indices():
    assert oracle.query_indices(8) == [0, 2, 4, 6, 7]
    assert oracle.query_indices(1) == [0]
    with pytest.raises(ValueError):
        oracle.query_indices(0)


def test_fp32_oracle_respects_bottom_right_causality_and_gqa_grouping():
    torch = pytest.importorskip("torch")
    query = torch.zeros((1, 2, 4, 2), dtype=torch.float16)
    key = torch.zeros((1, 4, 2, 2), dtype=torch.float16)
    value = torch.tensor([[[[1., 2.], [10., 20.]],
                           [[3., 4.], [30., 40.]],
                           [[5., 6.], [50., 60.]],
                           [[99., 100.], [990., 1000.]]]], dtype=torch.float16)
    result = oracle.sampled_attention(query, key, value, [0, 1], scale=1.)
    assert result.dtype == torch.float32
    assert result.shape == (2, 4, 2)
    torch.testing.assert_close(result, torch.tensor([
        [[3., 4.], [3., 4.], [30., 40.], [30., 40.]],
        [[27., 28.], [27., 28.], [270., 280.], [270., 280.]],
    ]), rtol=1e-6, atol=1e-6)


def test_fp32_oracle_matches_independent_double_precision_softmax():
    torch = pytest.importorskip("torch")
    query = torch.tensor([[[[2., 0.]]]], dtype=torch.float16)
    key = torch.tensor([[[[0., 0.]], [[1., 0.]], [[2., 0.]]]], dtype=torch.float16)
    value = torch.tensor([[[[1., 10.]], [[2., 20.]], [[4., 40.]]]], dtype=torch.float16)
    denominator = 1. + math.exp(1.) + math.exp(2.)
    expected = (1. + 2. * math.exp(1.) + 4. * math.exp(2.)) / denominator
    result = oracle.sampled_attention(query, key, value, [0], scale=.5)
    assert result.flatten().tolist() == pytest.approx([expected, expected * 10.], rel=1e-6)


def test_invalid_head_grouping_or_sample_position_is_rejected():
    torch = pytest.importorskip("torch")
    query = torch.zeros((1, 2, 3, 2))
    key = torch.zeros((1, 4, 2, 2))
    with pytest.raises(ValueError):
        oracle.sampled_attention(query, key, key, [0])
    query = torch.zeros((1, 2, 4, 2))
    with pytest.raises(ValueError):
        oracle.sampled_attention(query, key, key, [2])


def test_tf32_setting_is_restored_even_when_reference_computation_fails():
    torch = pytest.importorskip("torch")
    original = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        with pytest.raises(RuntimeError):
            with oracle.without_tf32():
                assert torch.backends.cuda.matmul.allow_tf32 is False
                raise RuntimeError("reference failed")
        assert torch.backends.cuda.matmul.allow_tf32 is True
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original


def test_error_metrics_use_fp32_reference_and_reject_nonfinite_outputs():
    torch = pytest.importorskip("torch")
    reference = torch.tensor([3., 4.])
    metrics = oracle.error_metrics(torch.tensor([3., 5.]), reference)
    assert metrics["max_abs"] == 1.
    assert metrics["rmse"] == pytest.approx(math.sqrt(.5))
    assert metrics["relative_l2"] == pytest.approx(.2)
    assert metrics["finite"] is True
    assert oracle.error_metrics(torch.tensor([math.inf, 4.]), reference)["finite"] is False


def test_cli_preserves_kernel_failure_with_sampling_scope_and_capture_hash(monkeypatch, tmp_path):
    import hashlib
    import json
    import sys
    from types import SimpleNamespace

    torch = pytest.importorskip("torch")
    from qwasar_bench import prefill_flash, prefill_microbench

    capture_path = tmp_path / "capture.pt"
    torch.save({"kwargs": {"q": torch.zeros((1, 8, 24, 256), dtype=torch.half),
                           "qc": (None, None, 8, 4), "causal": True},
                "metadata": {"fixture": "CPU"}}, capture_path)
    output_path = tmp_path / "oracle.json"

    def fail_kernel(**kwargs):
        raise RuntimeError("kernel failure")

    module = SimpleNamespace(__file__=__file__, _qc_staging=0,
                             paged_attn_triton_prefill=fail_kernel)
    monkeypatch.setitem(sys.modules, "exllamav3.modules.attention_fn.triton_paged", module)
    monkeypatch.setitem(sys.modules, "triton", SimpleNamespace(__version__="CPU-test"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "CPU fixture")
    monkeypatch.setattr(prefill_microbench, "to_device", lambda value, torch, device: value)
    monkeypatch.setattr(prefill_flash, "validate_capture", lambda kwargs: 316)
    assert oracle.main(["--capture", str(capture_path), "--output", str(output_path)]) == 1
    report = json.loads(output_path.read_text())
    assert report["status"] == "error"
    assert report["error"] == "kernel failure"
    assert report["query_indices"] == [0, 2, 4, 6, 7]
    assert report["valid_kv_tokens_per_query"] == [309, 311, 313, 315, 316]
    assert report["sampled_vectors"] == 120
    assert report["capture_sha256"] == hashlib.sha256(capture_path.read_bytes()).hexdigest()
    assert module._qc_staging == 0
