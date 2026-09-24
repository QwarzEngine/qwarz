import json
import stat
import threading
from unittest.mock import patch

import pytest

from qwasar_runtime.engine import Engine, FakeBackend
from qwasar_runtime.parsing import StreamParser


TOOLS = [{"type": "function", "function": {"name": "ask_user_question", "parameters": {
    "type": "object", "properties": {"question": {"type": "string"}, "options": {
        "type": "array", "items": {"type": "string"}}}, "required": ["question"]}}}]
RAW = ('<tool_call><function=ask_user_question><parameter=question>Choose</parameter>'
       '<parameter=options>[{"secret":"PRIVATE_VALUE","other":true}]</parameter></function></tool_call>')
COERCED = ('<tool_call><function=ask_user_question><parameter=question>Choose</parameter>'
           '<parameter=options>[{"label":"Go ahead"}]</parameter></function></tool_call>')


def run_invalid(directory, raw=RAW):
    backend = FakeBackend()
    original = backend.start

    def start(tokens, request):
        serial = original(tokens, request)
        backend.chunks = list('Public prefix.\n' + raw)
        return serial

    backend.start = start
    engine = Engine(backend, diagnostic_directory=directory)
    events = list(engine.generate('resp_test', {"messages": [{"role": "user", "content": "PROMPT_SECRET"}],
        "tools": TOOLS, "thinking": "off"}, None, threading.Event()))
    return engine, backend, events


def test_validation_error_locates_nested_field_without_values():
    parser = StreamParser('off', TOOLS, 'resp_test')
    parser.feed(RAW)
    message = parser.finish()
    assert len(message['tool_calls']) == 1
    assert json.loads(message['tool_calls'][0]['function']['arguments'])['options'] == [
        {'secret': 'PRIVATE_VALUE', 'other': True}]
    assert len(parser.validation_failures) == 1
    failure = parser.validation_failures[0]
    assert failure['validation']['path'] == ['options', 0]
    assert failure['validation']['actual_type'] == 'object'
    assert failure['validation']['expected_type'] == 'string'
    assert 'PRIVATE_VALUE' not in failure['error']['message']


def test_engine_captures_exact_evidence_without_exposing_it_in_protocol(tmp_path):
    directory = tmp_path / 'diagnostics'
    engine, backend, events = run_invalid(directory)
    terminal = events[-1]
    assert terminal['status'] == 'completed'
    assert terminal['snapshot'] is not None
    assert terminal['error'] is None
    assert len(terminal['message']['tool_calls']) == 1
    call = terminal['message']['tool_calls'][0]
    assert call['function']['name'] == 'ask_user_question'
    assert json.loads(call['function']['arguments'])['options'] == [
        {'secret': 'PRIVATE_VALUE', 'other': True}]
    assert terminal['metrics']['incomplete_reason'] is None
    assert terminal['metrics']['tool_error'] == {'stage': 'schema_validation', 'tool': 'ask_user_question',
                                                'call_index': 0, 'error': 'expected string'}
    # Pass-through delivers the argument values to the client, but the
    # protocol-visible metrics stay value-free.
    assert 'PRIVATE_VALUE' not in json.dumps(terminal['metrics'])
    record = json.loads(next(directory.glob('*.json')).read_text())
    assert record['raw_tool_calls'] == RAW
    assert record['delivery'] == 'passed_to_client'
    assert record['tool_schemas']['ask_user_question'] == TOOLS[0]['function']['parameters']
    assert record['parsed_arguments']['options'] == [{'secret': 'PRIVATE_VALUE', 'other': True}]
    assert record['validation']['path'] == ['options', 0]
    assert record['stage'] == 'schema_validation'
    assert record['response_id'] == 'resp_test'
    assert record['runtime_identity'] == backend.identity
    assert 'PROMPT_SECRET' not in json.dumps(record)
    assert 'Public prefix.' not in json.dumps(record)
    assert len(record['parser_sha256']) == 64
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert stat.S_IMODE(next(directory.glob('*.json')).stat().st_mode) == 0o600
    assert backend.resets == 0
    backend.start = FakeBackend.start.__get__(backend)
    followup = list(engine.generate('resp_next', {'messages': [{'role': 'user', 'content': 'hello'}]},
                                   None, threading.Event()))
    assert followup[-1]['status'] == 'completed'
    assert len(list(directory.glob('*.json'))) == 1


def test_capture_failure_does_not_break_delivery(tmp_path):
    with patch('qwasar_runtime.diagnostics.save_diagnostic', side_effect=OSError('PRIVATE_PATH')):
        _, _, events = run_invalid(tmp_path)
    assert events[-1]['status'] == 'completed'
    assert events[-1]['error'] is None
    assert events[-1]['metrics']['tool_error']['stage'] == 'schema_validation'
    assert 'PRIVATE_PATH' not in json.dumps(events[-1]['metrics'])


def test_malformed_native_call_is_captured(tmp_path):
    _, _, events = run_invalid(tmp_path, '<tool_call>broken</tool_call>')
    record = json.loads(next(tmp_path.glob('*.json')).read_text())
    assert record['stage'] == 'native_parse'
    assert record['raw_tool_calls'] == '<tool_call>broken</tool_call>'
    assert events[-1]['status'] == 'incomplete'


def test_unclosed_native_call_is_incomplete(tmp_path):
    engine, backend, events = run_invalid(
        tmp_path, '<tool_call><function=ask_user_question><parameter=question>Choose')
    terminal = events[-1]
    assert terminal['status'] == 'incomplete'
    assert terminal['error'] is None
    assert terminal['snapshot'] is None
    assert terminal['message']['tool_calls'] == []
    assert terminal['metrics']['incomplete_reason'] == 'unclosed_tool_call'
    assert terminal['metrics']['tool_error']['stage'] == 'incomplete_native_parse'
    assert backend.resets == 0
    record = json.loads(next(tmp_path.glob('*.json')).read_text())
    assert record['stage'] == 'incomplete_native_parse'
    assert 'Choose' in record['raw_tool_calls']
    assert 'PROMPT_SECRET' not in json.dumps(events)


def test_writer_limits_size_and_retention(tmp_path):
    from qwasar_runtime.diagnostics import save_diagnostic

    for index in range(4):
        save_diagnostic(tmp_path, {'raw_tool_calls': 'x' * 2000, 'index': index}, max_bytes=1024, keep=2)
    files = list(tmp_path.glob('*.json'))
    assert len(files) == 2
    for path in files:
        assert path.stat().st_size <= 1024
        record = json.loads(path.read_text())
        assert record['evidence_omitted'] == 'size_limit'
        assert record['original_bytes'] > 1024


def test_writer_rejects_symlink_directory(tmp_path):
    from qwasar_runtime.diagnostics import save_diagnostic

    target = tmp_path / 'target'
    target.mkdir()
    link = tmp_path / 'link'
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(OSError):
        save_diagnostic(link, {})
    assert not list(target.iterdir())


def test_replay_compares_recorded_arguments_and_validator_result(tmp_path):
    from qwasar_runtime.diagnostics import replay_diagnostic

    run_invalid(tmp_path)
    report = replay_diagnostic(next(tmp_path.glob('*.json')))
    assert report['parser_matches_capture'] is True
    assert report['parsed_arguments_match'] is True
    assert report['validation_matches'] is True
    assert report['validation']['path'] == ['options', 0]
    assert report['accepted'] is False
    assert 'PRIVATE_VALUE' not in json.dumps(report)


def test_replay_reports_changed_parser_and_argument_evidence(tmp_path):
    from qwasar_runtime.diagnostics import replay_diagnostic

    run_invalid(tmp_path)
    path = next(tmp_path.glob('*.json'))
    record = json.loads(path.read_text())
    record['parser_sha256'] = 'old-parser'
    record['parsed_arguments'] = {}
    path.write_text(json.dumps(record))
    report = replay_diagnostic(path)
    assert report['parser_matches_capture'] is False
    assert report['parsed_arguments_match'] is False


def test_label_options_are_coerced_instead_of_failing(tmp_path):
    _, backend, events = run_invalid(tmp_path, COERCED)
    terminal = events[-1]
    assert terminal['status'] == 'completed'
    assert terminal['error'] is None
    arguments = json.loads(terminal['message']['tool_calls'][0]['function']['arguments'])
    assert arguments['options'] == ['Go ahead']
    assert backend.resets == 0
    assert not list(tmp_path.glob('*.json'))


def test_capture_can_be_disabled():
    _, _, events = run_invalid(None)
    assert events[-1]['error'] is None
    assert events[-1]['status'] == 'completed'
    assert events[-1]['metrics']['tool_error']['stage'] == 'schema_validation'


def test_malformed_second_call_does_not_publish_first_call(tmp_path):
    raw = ('<tool_call><function=ask_user_question><parameter=question>Choose</parameter>'
           '</function></tool_call>' + '<tool_call>broken</tool_call>')
    _, _, events = run_invalid(tmp_path, raw)
    record = json.loads(next(tmp_path.glob('*.json')).read_text())
    assert record['call_index'] == 1
    assert record['raw_tool_calls'] == raw
    assert events[-1]['status'] == 'incomplete'
    assert events[-1]['message']['tool_calls'] == []


def test_schema_invalid_second_call_publishes_both_calls(tmp_path):
    raw = ('<tool_call><function=ask_user_question><parameter=question>Choose</parameter>'
           '</function></tool_call>' + RAW)
    _, _, events = run_invalid(tmp_path, raw)
    terminal = events[-1]
    assert terminal['status'] == 'completed'
    assert len(terminal['message']['tool_calls']) == 2
    assert terminal['metrics']['tool_error'] == {'stage': 'schema_validation', 'tool': 'ask_user_question',
                                                'call_index': 1, 'error': 'expected string'}
    record = json.loads(next(tmp_path.glob('*.json')).read_text())
    assert record['call_index'] == 1
    assert record['stage'] == 'schema_validation'
    assert record['delivery'] == 'passed_to_client'
