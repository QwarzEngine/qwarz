from types import SimpleNamespace

import pytest

from qwasar_bench import turn_matrix


def test_history_budget_reserves_largest_delta_and_generation():
    assert turn_matrix.history_budget(32768, 8192, 1536, 16) == 32768
    assert turn_matrix.history_budget(262144, 8192, 1536, 16) == 252400
    with pytest.raises(ValueError):
        turn_matrix.history_budget(262145, 8192, 1536, 16)


def test_percentiles_interpolate_and_do_not_accept_empty_samples():
    assert turn_matrix.percentile([10, 20, 30, 40, 50], .5) == 30
    assert turn_matrix.percentile([10, 20, 30, 40, 50], .95) == pytest.approx(48)
    with pytest.raises(ValueError):
        turn_matrix.percentile([], .95)


def test_summary_keeps_failures_in_latency_and_quality_denominators():
    samples = [dict(nominal_context=262144, appended_tokens=128, elapsed_ms=elapsed,
                    first_content_ms=None if not passed else elapsed - 1,
                    ttft_ms=1, quality_pass=passed, warm_valid=True, physical_prefill_tokens=129)
               for elapsed, passed in [(10, True), (100, False), (20, True)]]
    cell = turn_matrix.summarize(samples)[0]
    assert cell['samples'] == 3 and cell['quality_passes'] == 2
    assert cell['elapsed_ms_p50'] == 20
    assert cell['elapsed_ms_p95'] == pytest.approx(92)
    assert cell['valid_elapsed_ms_p95'] == pytest.approx(19.5)
    assert cell['all_observed_valid_under_30s'] is False


def test_summary_preserves_immediate_eos_failures_without_first_token():
    cell = turn_matrix.summarize([dict(nominal_context=32768, appended_tokens=128,
                                      elapsed_ms=10, first_content_ms=None, ttft_ms=None,
                                      quality_pass=False, warm_valid=False, physical_prefill_tokens=128)])[0]
    assert cell['samples'] == 1 and cell['quality_passes'] == 0
    assert cell['ttft_ms_p50'] is None
    assert cell['missing_first_token'] == 1


def test_matrix_delta_is_exact_and_does_not_change_history(monkeypatch):
    monkeypatch.setattr(turn_matrix, 'tail_parts', lambda tokenizer, thinking: ([91], [92]))
    monkeypatch.setattr(turn_matrix, 'encode_corpus', lambda tokenizer, text: list(text.encode()))
    history = [10, 11, 12]
    corpus = list(range(1000, 2000))
    first = turn_matrix.build_turn(None, history, corpus, 128, 'task', 'medium', 'trial-0')
    second = turn_matrix.build_turn(None, history, corpus, 128, 'task', 'medium', 'trial-1')
    assert len(first) == len(history) + 128
    assert first[:3] == second[:3] == history
    assert first[3:] != second[3:]
    assert first[-1] == 92
    filler = [token for token in first if token >= 1000]
    assert filler == corpus[:len(filler)]
    with pytest.raises(ValueError, match='delta'):
        turn_matrix.build_turn(None, history, corpus, 3, 'task', 'medium', 'trial-0')


def test_profile_selection_never_selects_invalid_or_truncated_output():
    valid = dict(quality_pass=True, warm_valid=True, truncated=False, elapsed_ms=10)
    invalid = dict(quality_pass=False, warm_valid=True, truncated=False, elapsed_ms=100)
    truncated = dict(quality_pass=True, warm_valid=True, truncated=True, elapsed_ms=200)
    assert turn_matrix.slowest_valid([valid, invalid, truncated]) is valid
    assert turn_matrix.slowest_valid([invalid, truncated]) is None


def test_cold_or_requeued_correct_answers_do_not_qualify_as_warm_success():
    row = dict(nominal_context=262144, appended_tokens=128, elapsed_ms=10,
               first_content_ms=1, ttft_ms=1, quality_pass=True, warm_valid=False,
               truncated=False, physical_prefill_tokens=None)
    cell = turn_matrix.summarize([row])[0]
    assert cell['quality_passes'] == 1
    assert cell['warm_valid_samples'] == 0
    assert cell['all_observed_valid_under_30s'] is False
    assert turn_matrix.slowest_valid([row]) is None
    assert turn_matrix.warm_qualified(dict(cache_metrics_valid=True, cached_tokens=3840), 4096, 2048)
    assert not turn_matrix.warm_qualified(dict(cache_metrics_valid=True, cached_tokens=0), 4096, 2048)
    assert not turn_matrix.warm_qualified(dict(cache_metrics_valid=False, cached_tokens=None), 4096, 2048)
    assert not turn_matrix.warm_qualified(dict(cache_metrics_valid=True, cached_tokens=1024), 4096, 2048)


@pytest.mark.parametrize('profile_enabled', [0, 1])
def test_matrix_orchestration_preserves_base_and_reports_failed_trials(tmp_path, monkeypatch, profile_enabled):
    import json
    from qwasar_bench import fidelity_probe, profile_probe

    monkeypatch.setattr(turn_matrix, 'control_ids', lambda tokenizer, text: [999] if text == '<|im_end|>' else [])
    monkeypatch.setattr(turn_matrix, 'encode_corpus', lambda tokenizer, text: [10])
    monkeypatch.setattr(turn_matrix, 'build_turn', lambda tokenizer, history, corpus, delta, *args: history + [20] * delta)
    fixture = turn_matrix.make_fixture(42)
    calls = []
    histories = []
    resets = []

    def reset(generator):
        resets.append(len(calls))
        return generator

    def profile(generator, tokenizer, prompt, arguments, seed, event_path, output_directory):
        assert resets == [5]
        return {'sample': sample(generator, tokenizer, prompt, arguments, seed, event_path), 'profile': {}}

    monkeypatch.setattr(fidelity_probe, 'fresh_generator', reset)
    monkeypatch.setattr(profile_probe, 'profile_sample', profile)

    def sample(generator, tokenizer, prompt, arguments, seed, event_path, *, sequence_sink=None):
        calls.append(prompt)
        if sequence_sink is not None:
            sequence_sink.extend(prompt + [999])
            histories.append(list(sequence_sink))
            completion = 'READY'
        else:
            assert prompt[:len(histories[0])] == histories[0]
            completion = '</think>' + (json.dumps(fixture['cases'][0]['expected']) if seed == 42 else '{}')
        return dict(completion=completion, truncated=False, elapsed_ms=10,
                    first_content_ms=2, ttft_ms=1, physical_prefill_tokens=128,
                    cache_metrics_valid=True, cached_tokens=len(histories[0]) - 256,
                    generated_tokens=1, prompt_sha256='fake')

    monkeypatch.setattr(turn_matrix, 'run_sample', sample)
    args = SimpleNamespace(contexts='32768', appended_tokens='128,512', repetitions=2,
                           max_new_tokens=1536, thinking='medium', sampler='recommended',
                           output=tmp_path, profile_matrix=profile_enabled)
    tokenizer = SimpleNamespace(hf_render_chat_template=lambda *args, **kwargs: '__QWASAR_MATRIX_SOURCE__')
    result = turn_matrix.run_matrix(SimpleNamespace(num_draft_tokens=4), tokenizer,
                                    [30] * 50000, args)
    assert len(calls) == (7 if profile_enabled else 5)
    assert [len(prompt) - len(histories[0]) for prompt in calls[1:5]] == [128, 128, 512, 512]
    assert [cell['samples'] for cell in result['cells']] == [2, 2]
    assert [cell['quality_passes'] for cell in result['cells']] == [1, 1]
    assert result['profile_candidate'].endswith('-0')
    assert json.loads((tmp_path / 'seed-32768-history.json').read_text()) == histories[0]
