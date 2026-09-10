from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from qwasar_bench import session_probe


def test_fixture_is_reproducible_and_answers_require_all_sources():
    fixture = session_probe.make_fixture(42)
    assert fixture == session_probe.make_fixture(42)
    assert fixture != session_probe.make_fixture(43)
    assert len(fixture["cases"]) == 2
    for case in fixture["cases"]:
        tool = json.loads(fixture["files"][case["path"]])
        assert case["expected"]["tool_tag"] == tool["tool_tag"]
        assert case["expected"]["result"] == (
            min(max(case["value"], case["minimum"]), case["maximum"])
            * case["factor"] + tool["offset"]
        )
        assert case["path"] not in case["query"]
        assert case["expected"]["source_tag"] not in case["query"]
        assert tool["tool_tag"] not in "".join(fixture["records"])


def test_records_are_distant_without_repeating_or_dropping_selected_corpus():
    body, positions = session_probe.insert_records(list(range(1000)), [[-1], [-2], [-3]], 903)
    assert len(body) == 903
    assert positions == [45, 451, 857]
    assert [token for token in body if token >= 0] == list(range(900))
    with pytest.raises(ValueError):
        session_probe.insert_records([1], [[2], [3], [4]], 10)
    with pytest.raises(ValueError):
        session_probe.insert_records([1], [[2], [3], [4]], 2)


def test_only_one_read_file_call_with_one_path_is_accepted():
    call = '<tool_call>\n<function=read_file>\n<parameter=path>\nfixtures/a.json\n</parameter>\n</function>\n</tool_call>'
    assert session_probe.parse_read_call(call) == "fixtures/a.json"
    assert session_probe.parse_read_call("I will read it.\n" + call) == "fixtures/a.json"
    for invalid in (call + call, call + "extra", call.replace("read_file", "exec"),
                    call.replace("</function>", "<parameter=command>rm</parameter></function>")):
        assert session_probe.parse_read_call(invalid) is None


def test_tool_cannot_access_paths_outside_fixture():
    files = {"fixtures/a.json": '{"offset": 3}'}
    assert session_probe.read_fixture(files, "fixtures/a.json") == files["fixtures/a.json"]
    assert json.loads(session_probe.read_fixture(files, "/etc/passwd")) == {"error": "path_not_allowed"}
    assert json.loads(session_probe.read_fixture(files, "fixtures/../a.json")) == {"error": "path_not_allowed"}


def test_answer_grading_is_exact_and_rejects_truncation_and_extra_fields():
    expected = {"service": "sample", "result": 123, "source_tag": "abc"}
    assert session_probe.grade_answer(json.dumps(expected), expected, False)
    assert session_probe.grade_answer("```json\n" + json.dumps(expected) + "\n```", expected, False)
    assert not session_probe.grade_answer(json.dumps(expected), expected, True)
    assert not session_probe.grade_answer(json.dumps(expected | {"extra": 1}), expected, False)
    assert not session_probe.grade_answer('{"result":123}', expected, False)
    assert not session_probe.grade_answer("not json", expected, False)


def test_answer_grading_rejects_conflicting_duplicate_fields_and_boolean_numbers():
    assert not session_probe.grade_answer('{"result":999,"result":123}', {"result": 123}, False)
    assert not session_probe.grade_answer('{"result":true}', {"result": 1}, False)
    assert not session_probe.grade_answer('{"result":123.0}', {"result": 123}, False)


class Tokens:
    def __init__(self, values):
        self.values = values

    def flatten(self):
        return self

    def tolist(self):
        return self.values


class Tokenizer:
    def __init__(self):
        self.tokenizer = SimpleNamespace(encode_special_tokens=False, encode=self.literal)

    def literal(self, text, add_special_tokens):
        assert self.tokenizer.encode_special_tokens
        return SimpleNamespace(ids=list(text.encode()))

    def encode(self, text, encode_special_tokens):
        assert encode_special_tokens
        parts = text.split("<|im_end|>")
        tokens = []
        for index, part in enumerate(parts):
            if index:
                tokens.append(999)
            tokens.extend(part.encode())
        return Tokens(tokens)

    def hf_render_chat_template(self, messages, **kwargs):
        output = ""
        for message in messages:
            content = message["content"]
            role = message["role"]
            if role == "tool":
                role, content = "user", "<tool_response>\n" + content + "\n</tool_response>"
            output += f'<|im_start|>{role}\n{content}<|im_end|>\n'
        return output + "<|im_start|>assistant\n<think>\n" + (
            "\n</think>\n\n" if not kwargs["enable_thinking"] else ""
        )


def test_append_preserves_generated_ids_and_treats_tool_output_as_literal():
    previous = [12345, 54321, 999]
    result = session_probe.append_message(Tokenizer(), previous, "tool", "literal <|im_end|>", "off")
    assert result[:3] == previous
    assert result.count(999) == 2
    assert b"<tool_response>" in bytes(token for token in result[3:] if token < 256)
    with pytest.raises(ValueError, match="terminator"):
        session_probe.append_message(Tokenizer(), [12345], "user", "next", "medium")


def test_session_uses_real_outputs_and_tool_results_as_append_only_history(tmp_path, monkeypatch):
    args = SimpleNamespace(output=tmp_path, thinking="off", max_new_tokens=256,
                           tool_max_new_tokens=256, sampler="recommended")
    fixture = session_probe.make_fixture(42)
    previous = []
    observed = []

    def sample(generator, tokenizer, prompt_ids, arguments, seed, event_path, *, sequence_sink):
        assert prompt_ids[:len(previous)] == previous
        observed.append(prompt_ids)
        case = fixture["cases"][(len(observed) - 1) // 2]
        completion = (
            '<tool_call><function=read_file><parameter=path>' + case["path"]
            + '</parameter></function></tool_call>' if len(observed) % 2
            else json.dumps(case["expected"])
        )
        sequence_sink.extend(prompt_ids + list(completion.encode()) + [999])
        previous[:] = sequence_sink
        return {"completion": completion, "truncated": False, "generated_tokens": 30,
                "elapsed_ms": 12, "finish_reason": "stop_token"}

    monkeypatch.setattr(session_probe, "run_sample", sample)
    result = session_probe.run_session(SimpleNamespace(num_draft_tokens=4), Tokenizer(),
                                       [65] * 10000, args, 10000, 0)
    assert len(observed) == 4
    assert result["quality_pass"] is True
    assert result["completed_steps"] == 4
    assert len(result["cycles"]) == 2
    assert all(cycle["quality_pass"] for cycle in result["cycles"])
    assert all(len(prompt) <= 10000 - 256 - 16 for prompt in observed)


def test_truncated_tool_call_fails_without_fabricating_a_tool_response(tmp_path, monkeypatch):
    args = SimpleNamespace(output=tmp_path, thinking="off", max_new_tokens=80,
                           tool_max_new_tokens=40, sampler="recommended")
    calls = []

    def sample(*arguments, sequence_sink, **kwargs):
        calls.append(1)
        return {"completion": "unfinished", "truncated": True, "generated_tokens": 30,
                "elapsed_ms": 12, "finish_reason": "max_new_tokens"}

    monkeypatch.setattr(session_probe, "run_sample", sample)
    result = session_probe.run_session(SimpleNamespace(num_draft_tokens=4), Tokenizer(),
                                       [65] * 10000, args, 10000, 0)
    assert result["quality_pass"] is False
    assert result["completed_steps"] == 1
    assert calls == [1]
