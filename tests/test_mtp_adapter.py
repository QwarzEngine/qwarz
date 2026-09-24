import types

import pytest

from qwasar_bench.mtp_adapter import validate_alignment
from qwasar_bench.mtp_tasks import FAMILIES, manifest
from qwasar_bench.mtp_analysis import online_summary
from qwasar_bench.mtp_study import warmup_tokens


def test_warmup_uses_real_cache_capacity_and_crosses_prims_threshold():
    assert warmup_tokens(types.SimpleNamespace(max_num_tokens=262144)) == 8200
    assert warmup_tokens(types.SimpleNamespace(max_num_tokens=8192)) == 7168
    with pytest.raises(ValueError, match="too small"):
        warmup_tokens(types.SimpleNamespace(max_num_tokens=1024))


def test_resumed_analysis_verifies_original_training_hashes(tmp_path):
    import json
    from qwasar_bench.mtp_adapter import file_hash
    from qwasar_bench.mtp_analysis import training_source
    original = tmp_path / "original"
    training = original / "training"
    training.mkdir(parents=True)
    names = ("selection.json", "heldout.json", "baseline.json")
    for name in names:
        (training / name).write_text("{}")
    resumed = tmp_path / "resumed"
    resumed.mkdir()
    (resumed / "origin.json").write_text(json.dumps({
        "source": str(original), "training_hashes": {
            name: file_hash(training / name) for name in names}}))
    assert training_source(resumed) == training
    (training / "selection.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="evidence changed"):
        training_source(resumed)


def test_quality_analysis_checks_selected_adapter_before_grading(tmp_path):
    import json
    from qwasar_bench.mtp_analysis import quality_summary
    for index in range(4):
        run = tmp_path / f"quality-{index}-candidate-s{index}"
        run.mkdir()
        (run / "arm.json").write_text('{"arm":"candidate"}')
        (run / "configuration.json").write_text(json.dumps({
            "config": {"experimental_mtp_adapter": {"sha256": "wrong"}}}))
    with pytest.raises(ValueError, match="checkpoint differs"):
        quality_summary(tmp_path, tmp_path / "grades", "selected")


def test_document_split_has_no_family_overlap():
    data = manifest()
    assert len(data["tasks"]) == 64
    assert not set(FAMILIES["train"]) & set(FAMILIES["validation"])
    assert not set(FAMILIES["train"]) & set(FAMILIES["test"])
    assert not set(FAMILIES["validation"]) & set(FAMILIES["test"])
    assert data == manifest()


def test_online_report_requires_matching_prompts_and_both_repetitions():
    rows = []
    for arm in ("control", "candidate"):
        for i in range(2):
            rows.append({"context": 4096, "arm": arm, "rep": 0, "prompt_sha256": "same",
                         "decode_tokens_per_second": 200, "draft_acceptance": .7,
                         "decode_ms_per_verify_estimate": 20, "ttft_ms": 600, "elapsed_ms": 3000})
    assert online_summary(rows)[4096]["control_n"] == 2
    with pytest.raises(ValueError, match="unpaired"):
        online_summary(rows[:-1])
    rows[-1]["prompt_sha256"] = "changed"
    with pytest.raises(ValueError, match="unpaired"):
        online_summary(rows)


def test_shift_alignment_is_checked_not_assumed():
    assert validate_alignment([9, 4, 6], [100, 101, 102], [9, 4, 6, 8], 100) == 3
    assert validate_alignment([9, 4, 6], [100, 101, 102], [9], 100) == 1
    with pytest.raises(ValueError, match="token"):
        validate_alignment([9, 4], [100, 101], [4, 9], 100)
    with pytest.raises(ValueError, match="position"):
        validate_alignment([9, 4], [99, 100], [9, 4], 100)


def test_zero_adapter_is_bit_exact_and_trainable():
    torch = pytest.importorskip("torch")
    from qwasar_bench.mtp_adapter import ResidualAdapter
    adapter = ResidualAdapter(16, 4)
    x = torch.randn(2, 6, 16).half()
    assert torch.equal(adapter(x), x)
    adapter(x).float().square().sum().backward()
    assert adapter.b.grad is not None and adapter.b.grad.abs().sum() > 0


def test_install_only_changes_draft_and_restores_on_failure():
    torch = pytest.importorskip("torch")
    from qwasar_bench.mtp_adapter import ResidualAdapter, install
    norm = types.SimpleNamespace(forward=lambda x: x)
    original = norm.forward
    target_forward = lambda x: x
    head = object()
    target = types.SimpleNamespace(modules=[object(), head], logit_layer_idx=1, forward=target_forward)
    draft = types.SimpleNamespace(final_norm=norm, config=types.SimpleNamespace(hidden_size=16))
    generator = types.SimpleNamespace(draft_model=draft, model=target, mtp_draft=True, num_draft_tokens=6)
    with pytest.raises(RuntimeError, match="test failure"):
        with install(generator, ResidualAdapter(16, 4)):
            x = torch.randn(1, 16).half()
            assert torch.equal(norm.forward(x), x)
            assert target.forward is target_forward and target.modules[-1] is head
            raise RuntimeError("test failure")
    assert norm.forward is original


def test_collector_clones_mutable_inputs_and_pairs_depths():
    torch = pytest.importorskip("torch")
    from qwasar_bench.mtp_adapter import PairCollector
    shared = torch.zeros(1, 1, 4)

    def draft_forward(ids, params):
        shared.fill_(ids.item())
        return shared

    def target_forward(input_ids, params):
        params["export_states"] = [torch.arange(input_ids.numel() * 4).reshape(1, -1, 4).float()]
        return torch.zeros(1, input_ids.numel(), 8)

    draft = types.SimpleNamespace(forward=draft_forward)
    target = types.SimpleNamespace(forward=target_forward)
    generator = types.SimpleNamespace(draft_model=draft, model=target)
    with PairCollector(generator) as collector:
        ids, lengths = torch.tensor([[7]]), torch.tensor([10])
        draft.forward(ids, {"cache_seqlens": lengths})
        ids.fill_(9)
        lengths.add_(1)
        draft.forward(ids, {"cache_seqlens": lengths})
        target.forward(torch.tensor([[7, 9, 5]]),
                       {"pinned_staging": True, "cache_seqlens": torch.tensor([10])})
        data = collector.tensors()
        assert data["input_id"].tolist() == [7, 9]
        assert data["depth"].tolist() == [1, 2]
        assert data["position"].tolist() == [10, 11]
        assert data["draft"][:, 0].tolist() == [7, 9]
        assert data["teacher"].shape == (2, 4)
        collector.reset()
        assert collector.windows == 0
    assert draft.forward is draft_forward and target.forward is target_forward


def test_adapter_checkpoint_roundtrip_and_nonfinite_rejection(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from safetensors.torch import save_file
    from qwasar_bench.mtp_adapter import ResidualAdapter
    adapter = ResidualAdapter(16, 4)
    with torch.no_grad():
        adapter.b.normal_(0, .01)
    path = tmp_path / "adapter.safetensors"
    save_file(adapter.state_dict(), str(path))
    loaded = ResidualAdapter.load(path, device="cpu")
    x = torch.randn(3, 16).half()
    assert torch.equal(adapter(x), loaded(x))
    with torch.no_grad():
        adapter.b.fill_(float("nan"))
    save_file(adapter.state_dict(), str(path))
    with pytest.raises(ValueError, match="nonfinite"):
        ResidualAdapter.load(path, device="cpu")
