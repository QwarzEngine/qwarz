import json

import pytest

from qwasar_runtime.parsing import StreamParser, validate_schema, validate_tools, validate_value


TOOLS = [{"type": "function", "function": {"name": "mcporter_call", "parameters": {
    "type": "object", "properties": {"args": {"description": "JSON arguments object",
    "patternProperties": {"^.*$": {}}, "type": "object"}}, "required": ["args"]}}}]


def test_pi_mcporter_schema_and_native_nested_arguments():
    validate_tools(TOOLS)
    parser = StreamParser("off", TOOLS, "pattern_regression")
    arguments = {"args": {"count": 3, "nested": {"enabled": True}, "paths": ["a.py"], "empty": None}}
    parser.feed('<tool_call><function=mcporter_call><parameter=args>' + json.dumps(arguments["args"]) + '</parameter></function></tool_call>')
    assert json.loads(parser.finish()["tool_calls"][0]["function"]["arguments"]) == arguments


def test_matching_patterns_are_not_additional_properties():
    schema = {"type": "object", "patternProperties": {"count": {"type": "integer"}}, "additionalProperties": False}
    validate_schema(schema)
    validate_value({"my_count_value": 4}, schema)
    with pytest.raises(ValueError):
        validate_value({"my_count_value": "4"}, schema)
    with pytest.raises(ValueError):
        validate_value({"unknown": 4}, schema)


def test_all_overlapping_patterns_and_explicit_properties_apply():
    schema = {"type": "object", "properties": {"count": {"maximum": 5}},
              "patternProperties": {"^c": {"type": "integer"}, "t$": {"minimum": 2}}}
    validate_schema(schema)
    validate_value({"count": 3}, schema)
    for invalid in (1, 6, "3"):
        with pytest.raises(ValueError):
            validate_value({"count": invalid}, schema)


def test_additional_schema_only_validates_unmatched_properties():
    schema = {"patternProperties": {"^number_": {"type": "integer"}},
              "additionalProperties": {"type": "string"}}
    validate_schema(schema)
    validate_value({"number_one": 1, "label": "one"}, schema)
    with pytest.raises(ValueError):
        validate_value({"label": 1}, schema)


@pytest.mark.parametrize("patterns", [[], {"[": {}}, {".*": {"unsupported": True}}, {".*": 42}])
def test_invalid_pattern_schemas_rejected_at_admission(patterns):
    with pytest.raises(ValueError):
        validate_schema({"type": "object", "patternProperties": patterns})


def test_boolean_pattern_schema_is_enforced():
    schema = {"patternProperties": {"^blocked": False, "^allowed": True}, "additionalProperties": False}
    validate_schema(schema)
    validate_value({"allowed_key": {"anything": True}}, schema)
    with pytest.raises(ValueError):
        validate_value({"blocked_key": 1}, schema)


def test_native_pattern_string_parameter_is_not_coerced_to_number():
    tools = [{"type": "function", "function": {"name": "labels", "parameters": {
        "type": "object", "patternProperties": {"^label_": {"type": "string"}}, "additionalProperties": False}}}]
    parser = StreamParser("off", tools, "pattern_string")
    parser.feed('<tool_call><function=labels><parameter=label_number>123</parameter></function></tool_call>')
    assert json.loads(parser.finish()["tool_calls"][0]["function"]["arguments"]) == {"label_number": "123"}
