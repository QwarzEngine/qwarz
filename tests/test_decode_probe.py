from __future__ import annotations

import pytest

from qwasar_bench import decode_probe
from qwasar_bench.exllamav3_probe import _model_init_argv


@pytest.mark.parametrize("method", ["mtp", "none", "dflash2"])
def test_draft_flags_are_explicit(method):
    arguments = _model_init_argv(
        model_path="/target", draft_model_path="/draft", cache_size=262144,
        cache_quant="8,4", gpu_split_gb=30, draft_method=method,
    )
    assert ("--mtp" in arguments) == (method == "mtp")
    assert ("--draft_model_dir" in arguments) == (method == "dflash2")
    assert arguments[-2:] == ["--cache_quant", "8,4"]


def test_reserve_includes_generation_and_speculative_scratch():
    assert decode_probe.prompt_budget(262144, 512, 16) == 261616
    with pytest.raises(ValueError):
        decode_probe.prompt_budget(512, 512, 16)
    with pytest.raises(ValueError):
        decode_probe.prompt_budget(262145, 512, 16)


def test_decode_rate_excludes_every_token_in_first_streaming_batch():
    result = decode_probe.decode_metrics(
        started=1.0, first=3.0, finished=5.0,
        generated=104, first_batch_tokens=4,
    )
    assert result["ttft_ms"] == 2000.0
    assert result["decode_tokens_per_second"] == 50.0
    assert result["elapsed_ms"] == 4000.0
    assert decode_probe.decode_metrics(
        started=1.0, first=2.0, finished=2.0,
        generated=4, first_batch_tokens=4,
    )["decode_tokens_per_second"] is None


def test_thinking_is_not_visible_content():
    assert decode_probe.visible_content("still reasoning", "medium") == ""
    assert decode_probe.visible_content("thought</think>\nanswer", "medium") == "answer"
    assert decode_probe.visible_content("answer", "off") == "answer"


@pytest.mark.parametrize("thinking", ["xhigh", "medium", "low", "off"])
def test_recommended_sampler_is_mode_specific(thinking):
    settings = decode_probe.sampler_settings(thinking)
    assert settings["temperature"] == (0.7 if thinking == "off" else 1.0)
    assert settings["top_p"] == (0.8 if thinking == "off" else 0.95)
    assert settings["pres_p"] == (1.5 if thinking == "off" else 0.0)
    assert settings["top_k"] == 20
    assert settings["min_p"] == 0.0


def test_event_serialization_excludes_live_job_reference():
    import json

    event = {"stage": "started", "job": object(), "serial": 3}
    assert json.loads(json.dumps(decode_probe.serialize_event(event))) == {
        "stage": "started", "serial": 3,
    }


def test_logits_are_excluded_even_when_stop_token_is_held():
    event = {"logits": object(), "held": {"logits": object(), "token_ids": [[0]]}}
    assert decode_probe.serialize_event(event) == {"held": {"token_ids": [[0]]}}


def test_token_framing_preserves_literal_corpus_ids_and_exact_budget():
    assert decode_probe.frame_corpus([1, 2], [10, 11, 12, 13], [3], 6) == [
        1, 2, 10, 11, 12, 3,
    ]
    with pytest.raises(ValueError, match="corpus"):
        decode_probe.frame_corpus([1, 2], [10], [3], 6)
    with pytest.raises(ValueError, match="framing"):
        decode_probe.frame_corpus([1, 2], [10], [3], 2)


def test_literal_encoding_restores_tokenizer_setting():
    from types import SimpleNamespace

    class Backend:
        encode_special_tokens = False

        def encode(self, text, add_special_tokens):
            assert self.encode_special_tokens is True
            assert add_special_tokens is False
            return SimpleNamespace(ids=[27, 91, 29])

    backend = Backend()
    assert decode_probe.encode_corpus(SimpleNamespace(tokenizer=backend), "literal") == [27, 91, 29]
    assert backend.encode_special_tokens is False


@pytest.mark.parametrize("requeued", [False, True])
def test_sample_records_content_separately_and_batch_timing(monkeypatch, tmp_path, requeued):
    import sys
    from types import SimpleNamespace

    class Tokens:
        def __init__(self, count):
            self.shape = (1, count)

        def tolist(self):
            return [[1] * self.shape[-1]]

    def make_job(**kwargs):
        assert kwargs["return_logits"] is True
        return SimpleNamespace(accepted_draft_tokens=52, rejected_draft_tokens=10,
                               draft_stats=[(4, 4, 3, 3.0), (7, 3, 2, 2.35)],
                               time_prefill=2.0, sequences=[SimpleNamespace(
                                   sequence_ids=SimpleNamespace(torch=lambda: SimpleNamespace(
                                       flatten=lambda: SimpleNamespace(tolist=lambda: [1] * 260 + [9, 0]))))])

    cuda = SimpleNamespace(synchronize=lambda: None, reset_peak_memory_stats=lambda: None,
                           max_memory_allocated=lambda: 100, max_memory_reserved=lambda: 200)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=cuda, tensor=lambda values, **kwargs: values, long="long"))
    monkeypatch.setitem(sys.modules, "exllamav3", SimpleNamespace(Job=make_job))
    monkeypatch.setitem(sys.modules, "exllamav3.generator.sampler.presets", SimpleNamespace(
        GreedySampler=lambda: None, ComboSampler=lambda **kwargs: None))
    class Logits:
        def __getitem__(self, key):
            assert key == (0, 0, slice(None))
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def clone(self):
            return "first-full-vocabulary-cpu-tensor"

    rounds = [
        [{"stage": "streaming", "token_ids": Tokens(4), "text": "thought</think>",
          "requeue": requeued, "logits": Logits()}],
        [{"stage": "streaming", "token_ids": Tokens(100), "text": "answer",
          "new_tokens": 104, "cached_tokens": 256, "eos": True, "eos_reason": "stop_token"}],
    ]
    generator = SimpleNamespace(enqueue=lambda job: None, record_draft_stats=True,
                                num_remaining_jobs=lambda: len(rounds),
                                iterate=lambda: rounds.pop(0))
    clock = iter([1.0, 3.0, 5.0, 5.01])
    monkeypatch.setattr(decode_probe.time, "perf_counter", lambda: next(clock))
    args = SimpleNamespace(sampler="greedy", thinking="medium", max_new_tokens=512)
    sequence = []
    logits = []
    sample = decode_probe.run_sample(generator, SimpleNamespace(eos_token_id=0),
                                    [1] * 260, args, 42, tmp_path / "events.jsonl",
                                    sequence_sink=sequence, logits_sink=logits)
    assert logits == ["first-full-vocabulary-cpu-tensor"]
    assert sequence == [1] * 260 + [9, 0]
    assert sample["first_content_ms"] == 4000.0
    assert sample["ttft_ms"] == 2000.0
    assert sample["decode_tokens_per_second"] == pytest.approx(100 / 2.01)
    assert sample["requeue_count"] == int(requeued)
    assert sample["cache_metrics_valid"] is not requeued
    assert sample["physical_prefill_tokens"] == (None if requeued else 3)
    assert sample["cached_tokens"] == (None if requeued else 256)
    assert sample["draft_acceptance"] == (None if requeued else 52 / 62)
    assert sample["host_prefill_ms"] == (None if requeued else 2000.0)
    assert sample["streaming_batches"] == 2
    assert sample["truncated"] is False
    assert sample["draft_stats"] == (None if requeued else [(4, 4, 3, 3.0), (7, 3, 2, 2.35)])


@pytest.fixture
def workload_cli(tmp_path, monkeypatch):
    import json
    import sys
    from types import SimpleNamespace
    from qwasar_bench import fidelity_probe, session_probe, turn_matrix

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"max_position_embeddings": 262144}')
    (model / "chat_template.jinja").write_text("fixture")
    package = tmp_path / "backend"
    package.mkdir()
    (package / "__init__.py").write_text("")
    generator = SimpleNamespace(max_chunk_size=1024, num_draft_tokens=4)
    observed = []
    state = {"fail": False}

    def original(q, block_m=None):
        return generator.max_chunk_size, backend._qc_staging, block_m

    backend = SimpleNamespace(_qc_staging=1, paged_attn_triton_prefill=original)

    def workload(*args, **kwargs):
        observed.append(backend.paged_attn_triton_prefill(SimpleNamespace(shape=(1, 2048, 24, 256))))
        if state["fail"]:
            raise RuntimeError("workload failed")
        return {"completion": "fixture"}

    monkeypatch.setitem(sys.modules, "exllamav3", SimpleNamespace(__file__=str(package / "__init__.py")))
    monkeypatch.setitem(sys.modules, "exllamav3.modules.attention_fn", SimpleNamespace(triton_paged=backend))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="CPU-test", version=SimpleNamespace(cuda=None)))
    monkeypatch.setattr(decode_probe, "capture_environment", lambda **kwargs: SimpleNamespace(to_dict=lambda: {}))
    monkeypatch.setattr(decode_probe, "_load_exllamav3_runtime", lambda **kwargs: (generator, object(), None))
    monkeypatch.setattr(decode_probe, "encode_corpus", lambda *args: [1, 2])
    monkeypatch.setattr(decode_probe, "build_prompt", lambda *args: [1, 2])
    monkeypatch.setattr(decode_probe, "run_sample", workload)
    monkeypatch.setattr(session_probe, "run_session", workload)
    monkeypatch.setattr(turn_matrix, "run_matrix", workload)
    monkeypatch.setattr(fidelity_probe, "run_fidelity", workload)

    def arguments(workload_name, use_settings):
        output = tmp_path / "output"
        argv = ["decode_probe", "--model", str(model), "--draft-model", str(model),
                "--draft-method", "none", "--cache-quant", "8,4", "--output", str(output),
                "--corpus-root", str(package), "--workload", workload_name,
                "--contexts", "4096,8192", "--repetitions", "2", "--max-new-tokens", "64",
                "--tool-max-new-tokens", "64"]
        if use_settings:
            settings = tmp_path / "settings.json"
            settings.write_text(json.dumps({"name": "candidate", "staging": 0,
                                            "kwargs": {"block_m": 128}, "chunk_size": 4096}))
            argv.extend(["--prefill-settings", str(settings)])
        monkeypatch.setattr(sys, "argv", argv)
        return output

    return arguments, generator, backend, original, observed, state


@pytest.mark.parametrize(("workload", "expected_calls"), [("lru", 6), ("retrieval_session", 4), ("turn_matrix", 1)])
@pytest.mark.parametrize("use_settings", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_prefill_settings_cover_every_workload_iteration_and_restore_on_exit(workload_cli, workload, expected_calls, use_settings, fail):
    from contextlib import nullcontext
    import json

    arguments, generator, backend, original, observed, state = workload_cli
    output = arguments(workload, use_settings)
    state["fail"] = fail
    with pytest.raises(RuntimeError, match="workload failed") if fail else nullcontext():
        decode_probe.main()
    count = 1 if fail else expected_calls
    assert observed == ([(4096, 0, 128)] if use_settings else [(1024, 1, None)]) * count
    assert generator.max_chunk_size == 1024
    assert backend._qc_staging == 1
    assert backend.paged_attn_triton_prefill is original
    assert (output / "completed.json").exists() is not fail
    if use_settings:
        assert json.loads((output / "prefill-counters.json").read_text()) == {
            "overridden_calls": count, "default_calls": 0,
            "eligible_query_lengths": {"2048": count},
        }
    else:
        assert not (output / "prefill-counters.json").exists()


def test_cache_fidelity_explicitly_rejects_prefill_settings_before_running(workload_cli):
    arguments, generator, backend, original, observed, _ = workload_cli
    output = arguments("cache_fidelity", True)
    with pytest.raises(ValueError, match="cache_fidelity"):
        decode_probe.main()
    assert not output.exists()
    assert observed == []
    assert generator.max_chunk_size == 1024
    assert backend.paged_attn_triton_prefill is original


def test_cache_fidelity_without_settings_retains_original_control(workload_cli):
    arguments, generator, backend, original, observed, _ = workload_cli
    output = arguments("cache_fidelity", False)
    decode_probe.main()
    assert observed == [(1024, 1, None)]
    assert generator.max_chunk_size == 1024
    assert backend.paged_attn_triton_prefill is original
    assert (output / "completed.json").exists()
    assert not (output / "prefill-counters.json").exists()


def test_explicit_cache_pool_does_not_change_prompt_context_budgets(workload_cli, monkeypatch):
    import sys
    import json

    arguments, generator, *_ = workload_cli
    output = arguments("lru", False)
    sys.argv.extend(["--cache-size", "262144"])
    observed = []
    def load(**kwargs):
        observed.append(kwargs["cache_size"])
        return generator, object(), None
    monkeypatch.setattr(decode_probe, "_load_exllamav3_runtime", load)
    decode_probe.main()
    metadata = json.loads((output / "run.json").read_text())
    assert observed == [262144]
    assert metadata["cache_size"] == 262144
    assert metadata["arguments"]["contexts"] == "4096,8192"


@pytest.mark.parametrize("size", ["4096", "262145", "0"])
def test_invalid_cache_pool_is_rejected_before_output(workload_cli, size):
    import sys

    arguments, *_ = workload_cli
    output = arguments("lru", False)
    sys.argv.extend(["--cache-size", size])
    with pytest.raises(SystemExit):
        decode_probe.main()
    assert not output.exists()
