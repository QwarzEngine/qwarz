from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest


def test_first_divergence_and_logit_margin_are_distinguished_from_roundoff():
    from qwasar_bench.fidelity_probe import compare_outputs

    result = compare_outputs([1, 2, 3], [1, 2, 4], [0.0, 10.0, 3.0], [0.001, 10.001, 3.001])
    assert result["prefix_agreement_tokens"] == 2
    assert result["first_divergent_token"] == {"index": 2, "warm": 3, "cold": 4}
    assert result["first_logits"]["max_abs_difference"] == pytest.approx(0.001)
    assert result["first_logits"]["cold_top1_token"] == 1
    assert result["first_logits"]["cold_top1_margin"] == pytest.approx(7.0)
    assert result["first_logits"]["perturbation_bound_below_cold_margin"] is True
    assert result["first_logits"]["top1_agrees"] is True


def test_sequence_length_difference_and_changed_top1():
    from qwasar_bench.fidelity_probe import compare_outputs

    result = compare_outputs([0], [1, 0], [1.0, 0.99], [0.99, 1.0])
    assert result["sequences_equal"] is False
    assert result["first_logits"]["top1_agrees"] is False
    assert result["first_logits"]["perturbation_bound_below_cold_margin"] is False
    result = compare_outputs([0], [0, 1], [2.0, 1.0], [2.0, 1.0])
    assert result["first_divergent_token"] == {"index": 1, "warm": None, "cold": 1}


def test_matching_negative_infinity_padding_is_excluded_from_difference_statistics():
    from qwasar_bench.fidelity_probe import compare_outputs

    result = compare_outputs([1], [1], [0.0, 2.0, -float("inf")], [0.25, 2.25, -float("inf")])
    logits = result["first_logits"]
    assert logits["vocabulary_size"] == 3
    assert logits["finite_logits_compared"] == 2
    assert logits["matching_negative_infinity_mask_tokens"] == 1
    assert logits["max_abs_difference"] == 0.25
    assert logits["mean_abs_difference"] == 0.25
    assert logits["warm_top1_token"] == logits["cold_top1_token"] == 1


@pytest.mark.parametrize("warm,cold", [
    ([0.0, 2.0, -float("inf")], [0.0, 2.0, 0.0]),
    ([0.0, 2.0, float("inf")], [0.0, 2.0, float("inf")]),
    ([0.0, 2.0, float("nan")], [0.0, 2.0, float("nan")]),
])
def test_invalid_nonfinite_logits_remain_errors(warm, cold):
    from qwasar_bench.fidelity_probe import compare_outputs

    with pytest.raises(ValueError, match="nonfinite"):
        compare_outputs([1], [1], warm, cold)


def test_raw_history_retains_held_stop_token_without_reencoding(tmp_path):
    from qwasar_bench.fidelity_probe import read_generated_ids

    path = tmp_path / "events.jsonl"
    path.write_text('\n'.join(json.dumps({"event": event}) for event in [
        {"token_ids": [[12, 13]]},
        {"eos": True, "new_tokens": 3, "held": {"token_ids": [[99]]}},
    ]))
    assert read_generated_ids(path, 3) == [12, 13, 99]
    with pytest.raises(ValueError, match="count"):
        read_generated_ids(path, 4)


def test_fresh_generator_replaces_cache_metadata_and_rejects_active_jobs():
    from qwasar_bench.fidelity_probe import fresh_generator

    class PageTable:
        def __init__(self, generator, cache):
            self.referenced_pages = {}
            self.unreferenced_pages = {0: SimpleNamespace(kv_position=0)}

    class RecurrentCache(dict):
        def __init__(self, model, max_size):
            super().__init__()
            self.pagetable = None

    cache = SimpleNamespace(free_list=[0], num_slots=1)
    generator = SimpleNamespace(num_remaining_jobs=lambda: 0, cache=cache,
                                model=SimpleNamespace(loaded_tp=False), cpu_page_cache=None,
                                recurrent_cache_size=123)
    generator.pagetable = PageTable(generator, cache)
    old_table = generator.pagetable
    generator.recurrent_cache = RecurrentCache(generator.model, 123)
    generator.recurrent_cache["stale"] = 42
    old_recurrent = generator.recurrent_cache
    assert fresh_generator(generator) is generator
    assert generator.pagetable is not old_table
    assert generator.recurrent_cache is not old_recurrent
    assert generator.recurrent_cache == {}
    assert generator.recurrent_cache.pagetable is generator.pagetable
    generator.num_remaining_jobs = lambda: 1
    with pytest.raises(ValueError, match="idle"):
        fresh_generator(generator)


def test_prompt_hash_verification_rejects_any_changed_token():
    from qwasar_bench.fidelity_probe import verify_prompt

    sample = {"input_tokens": 2, "prompt_sha256": hashlib.sha256(b"[10, 11]").hexdigest()}
    verify_prompt([10, 11], sample)
    with pytest.raises(ValueError, match="SHA"):
        verify_prompt([10, 12], sample)


def test_reconstructs_exact_original_prompt_and_raw_tool_history(tmp_path):
    from qwasar_bench.fidelity_probe import reconstructed_problem

    class Tokenizer:
        def __init__(self):
            self.tokenizer = self
            self.encode_special_tokens = False

        def encode(self, text, **kwargs):
            ids = [999] if text == "<|im_end|>" else [ord(char) for char in text]
            return SimpleNamespace(ids=ids, flatten=lambda: SimpleNamespace(tolist=lambda: ids))

        def hf_render_chat_template(self, messages, **kwargs):
            if len(messages) == 2:
                return "A__QWASAR_SNAPSHOT_BODY__Z"
            return "X__QWASAR_ASSISTANT_ANCHOR__<|im_end|>T__QWASAR_MESSAGE_BODY__U"

    corpus = "\x07" * 300000
    (tmp_path / "corpus.txt").write_text(corpus)
    (tmp_path / "run.json").write_text(json.dumps({
        "corpus_sha256": hashlib.sha256(corpus.encode()).hexdigest(),
        "arguments": {"thinking": "medium"},
    }))
    label = "session-262144-0"
    (tmp_path / f"{label}-fixture.json").write_text(json.dumps({
        "records": ["R", "S", "T"], "cases": [{"query": "query", "expected": {"result": 9}}],
        "files": {"fixture.json": "tool"},
    }))
    original = ([65] + [7] * 12850 + [82] + [7] * 115651 + [83]
                + [7] * 115651 + [84] + [7] * 12851 + [90])
    completed = original + [20, 999]
    answer = completed + [84, 116, 111, 111, 108, 85]
    for step, prompt in enumerate((original, answer)):
        sample = {"input_tokens": len(prompt), "prompt_sha256": hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
                  "generated_tokens": 2, "sequence_tokens": len(prompt) + 2, "selected_path": "fixture.json"}
        (tmp_path / f"{label}-sample-{step}.json").write_text(json.dumps(sample))
    (tmp_path / f"{label}-events-0.jsonl").write_text(json.dumps({
        "event": {"token_ids": [[20, 999]], "eos": True},
    }))
    assert reconstructed_problem(Tokenizer(), [7] * 300000, tmp_path) == (completed, answer, {"result": 9})
    (tmp_path / "corpus.txt").write_text("changed")
    with pytest.raises(ValueError, match="corpus"):
        reconstructed_problem(Tokenizer(), [7] * 300000, tmp_path)


def test_fidelity_runs_identical_answer_warm_then_cold_and_checks_physical_cache(monkeypatch, tmp_path):
    import sys
    from qwasar_bench import fidelity_probe

    monkeypatch.setattr(fidelity_probe, "reconstructed_problem", lambda *args: ([1, 99], [1, 99, 5], {"result": 9}))
    resets = []
    monkeypatch.setattr(fidelity_probe, "fresh_generator", lambda generator: resets.append(True) or generator)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(save=lambda values, path: path.write_text("tensor")))
    prompts = []

    def sample(generator, tokenizer, prompt, args, seed, event_path, *, sequence_sink=None, logits_sink=None):
        prompts.append(list(prompt))
        if logits_sink is not None:
            logits_sink.append([1.0, 2.0])
            sequence_sink.extend(prompt + [1, 99])
        return {"cached_tokens": 2 if len(prompts) == 2 else 0, "cache_metrics_valid": True,
                "completion": 'thought</think>{"result": 9}', "truncated": False,
                "prompt_sha256": "same"}

    monkeypatch.setattr(fidelity_probe, "run_sample", sample)
    args = SimpleNamespace(output=tmp_path, source_run=tmp_path, max_new_tokens=512, thinking="medium", sampler="greedy")
    result = fidelity_probe.run_fidelity(SimpleNamespace(num_draft_tokens=4), object(), [], args)
    assert prompts == [[1, 99], [1, 99, 5], [1, 99, 5]]
    assert len(resets) == 2
    assert result["comparison"]["sequences_equal"] is True
    assert result["cold"]["cached_tokens"] == 0
    assert result["warm"]["quality_pass"] is True
    assert (tmp_path / "fidelity-summary.json").exists()
