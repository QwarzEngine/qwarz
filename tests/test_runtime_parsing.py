import json
import unittest

from qwasar_runtime.parsing import StreamParser, validate_messages, validate_tools


TOOLS = [{"type": "function", "function": {"name": "edit", "parameters": {
    "type": "object", "properties": {"path": {"type": "string"}, "edits": {
        "type": "array", "items": {"type": "object", "properties": {
            "oldText": {"type": "string"}, "newText": {"type": "string"}},
            "required": ["oldText", "newText"]}}}, "required": ["path", "edits"],
    "additionalProperties": False}}}]


class ParserTests(unittest.TestCase):
    def test_incremental_reasoning_and_tool_xml_never_leak(self):
        parser = StreamParser("medium", TOOLS, "resp_a")
        emitted = []
        source = 'reason</think>\n\n<tool_call>\n<function=edit>\n<parameter=path>\na.py\n</parameter>\n<parameter=edits>\n[{"oldText":"a","newText":"b"}]\n</parameter>\n</function>\n</tool_call>'
        for character in source:
            emitted.extend(parser.feed(character))
        message = parser.finish()
        self.assertEqual(message["reasoning_content"], "reason")
        self.assertNotIn("tool_call", "".join(text for _, text in emitted))
        self.assertEqual(message["content"], "")
        self.assertEqual("".join(text for channel, text in emitted if channel == "content"), "")
        self.assertEqual(json.loads(message["tool_calls"][0]["function"]["arguments"]), {
            "path": "a.py", "edits": [{"oldText": "a", "newText": "b"}]})

    def test_literal_code_without_tools_is_not_stripped(self):
        parser = StreamParser("off", [], "resp_a")
        parser.feed('print("<tool_call>")')
        self.assertEqual(parser.finish()["content"], 'print("<tool_call>")')

    def test_duplicate_parameters_are_rejected(self):
        parser = StreamParser("off", TOOLS, "resp_a")
        parser.feed('<tool_call><function=edit><parameter=path>a</parameter><parameter=path>b</parameter>'
                    '</function></tool_call>')
        with self.assertRaises(ValueError):
            parser.finish()

    def test_wrong_type_parameters_pass_through_with_evidence(self):
        parser = StreamParser("off", TOOLS, "resp_a")
        parser.feed('<tool_call><function=edit><parameter=path>a</parameter><parameter=edits>12</parameter>'
                    '</function></tool_call>')
        message = parser.finish()
        self.assertEqual(json.loads(message["tool_calls"][0]["function"]["arguments"]),
                         {"path": "a", "edits": 12})
        self.assertEqual(len(parser.validation_failures), 1)
        failure = parser.validation_failures[0]
        self.assertEqual(failure["stage"], "schema_validation")
        self.assertEqual(failure["tool"], "edit")
        self.assertEqual(failure["validation"]["path"], ["edits"])
        self.assertEqual(failure["validation"]["expected_type"], "array")
        self.assertEqual(failure["error"]["class"], "SchemaValidationError")

    def test_tool_results_must_correspond_and_be_unique(self):
        with self.assertRaises(ValueError):
            validate_messages([{"role": "user", "content": "go"}, {
                "role": "tool", "tool_call_id": "missing", "content": "data"}], TOOLS)

    def test_schema_name_injection_rejected(self):
        with self.assertRaises(ValueError):
            validate_tools([{"type": "function", "function": {"name": "x>bad", "parameters": {}}}])

    def test_grep_flag_parameter_names_are_accepted(self):
        validate_tools([{"type": "function", "function": {"name": "grep", "parameters": {
            "type": "object", "properties": {
                "-A": {"type": "integer"}, "-B": {"type": "integer"},
                "-C": {"type": "integer"}, "-i": {"type": "boolean"},
                "pattern": {"type": "string"}}}}}])

    def test_flag_parameters_round_trip_in_native_xml(self):
        tools = [{"type": "function", "function": {"name": "grep", "parameters": {
            "type": "object", "properties": {
                "-A": {"type": "integer"}, "-i": {"type": "boolean"},
                "pattern": {"type": "string"}},
            "required": ["pattern"]}}}]
        parser = StreamParser("off", tools, "resp_a")
        parser.feed('<tool_call><function=grep><parameter=pattern>TODO</parameter>'
                    '<parameter=-A>2</parameter><parameter=-i>true</parameter>'
                    '</function></tool_call>')
        arguments = json.loads(parser.finish()["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(arguments, {"pattern": "TODO", "-A": 2, "-i": True})

    def test_hermes_boolean_literals_match_json_true_false(self):
        tools = [{"type": "function", "function": {"name": "terminal", "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "background": {"type": "boolean"},
                "notify": {"anyOf": [{"type": "boolean"}, {"type": "array", "items": {"type": "string"}}]},
            },
            "required": ["command"]}}}]
        for raw, expected in (
            ("True", True), ("False", False), ("yes", True), ("NO", False),
            ("true", True), (" false ", False),
        ):
            parser = StreamParser("off", tools, "resp_bool")
            parser.feed(
                "<tool_call>\n<function=terminal>\n"
                f"<parameter=background>\n{raw}\n</parameter>\n"
                "<parameter=command>\necho ok\n</parameter>\n"
                "<parameter=notify>\nTrue\n</parameter>\n"
                "</function>\n</tool_call>"
            )
            arguments = json.loads(parser.finish()["tool_calls"][0]["function"]["arguments"])
            self.assertEqual(arguments, {"background": expected, "command": "echo ok", "notify": True}, raw)

    def test_boolean_garbage_is_still_rejected(self):
        tools = [{"type": "function", "function": {"name": "terminal", "parameters": {
            "type": "object", "properties": {"background": {"type": "boolean"}}}}}]
        parser = StreamParser("off", tools, "resp_bad")
        parser.feed("<tool_call><function=terminal><parameter=background>Trueish</parameter></function></tool_call>")
        with self.assertRaisesRegex(ValueError, "invalid JSON value for background"):
            parser.finish()

    def test_string_true_is_not_coerced_to_boolean(self):
        tools = [{"type": "function", "function": {"name": "write", "parameters": {
            "type": "object", "properties": {"content": {"type": "string"}}}}}]
        parser = StreamParser("off", tools, "resp_str")
        parser.feed("<tool_call><function=write><parameter=content>\nTrue\n</parameter></function></tool_call>")
        arguments = json.loads(parser.finish()["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(arguments["content"], "True")

    def test_on_off_and_padded_integer_literals(self):
        tools = [{"type": "function", "function": {"name": "run", "parameters": {
            "type": "object", "properties": {
                "alive": {"type": "boolean"}, "count": {"type": "integer"}}}}}]
        parser = StreamParser("off", tools, "resp_pad")
        parser.feed("<tool_call><function=run><parameter=alive>\nON\n</parameter>"
                    "<parameter=count>\n 3 \n</parameter></function></tool_call>")
        arguments = json.loads(parser.finish()["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(arguments, {"alive": True, "count": 3})

    def test_two_hermes_terminal_calls_with_python_bools(self):
        tools = [{"type": "function", "function": {"name": "terminal", "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "background": {"type": "boolean"},
                "notify": {"anyOf": [{"type": "boolean"}, {"type": "array", "items": {"type": "string"}}]},
            },
            "required": ["command"]}}}]
        parser = StreamParser("off", tools, "resp_two")
        parser.feed(
            "<tool_call>\n<function=terminal>\n"
            "<parameter=background>\nTrue\n</parameter>\n"
            "<parameter=command>\nuv venv --python 3.14 /tmp/x\n</parameter>\n"
            "<parameter=notify>\ntrue\n</parameter>\n"
            "</function>\n</tool_call>\n"
            "<tool_call>\n<function=terminal>\n"
            "<parameter=background>\nFalse\n</parameter>\n"
            "<parameter=command>\nls\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        message = parser.finish()
        self.assertEqual(len(message["tool_calls"]), 2)
        first, second = (json.loads(call["function"]["arguments"]) for call in message["tool_calls"])
        self.assertEqual(first["background"], True)
        self.assertEqual(first["notify"], True)
        self.assertEqual(second["background"], False)
        self.assertEqual(second["command"], "ls")

    def test_parameter_name_still_rejects_tag_breakers(self):
        with self.assertRaisesRegex(ValueError, "invalid parameter name"):
            validate_tools([{"type": "function", "function": {"name": "grep", "parameters": {
                "type": "object", "properties": {"-A><|im_end|>": {"type": "integer"}}}}}])

    def test_incomplete_reasoning_prefix_never_becomes_content(self):
        parser = StreamParser("medium", [], "resp_a")
        parser.feed("reason</thi")
        message = parser.finish(complete=False)
        self.assertEqual(message["reasoning_content"], "reason</thi")
        self.assertEqual(message["content"], "")

    def test_string_parameter_preserves_internal_newlines_and_spaces(self):
        parser = StreamParser("off", [{"type": "function", "function": {"name": "write", "parameters": {
            "type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]}}}], "resp_a")
        parser.feed('<tool_call><function=write><parameter=content>\n  code\n  next\n\n</parameter></function></tool_call>')
        arguments = json.loads(parser.finish()["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(arguments["content"], "  code\n  next\n")

    def test_duplicate_nested_json_property_rejected(self):
        parser = StreamParser("off", TOOLS, "resp_a")
        parser.feed('<tool_call><function=edit><parameter=path>a</parameter><parameter=edits>'
                    '[{"oldText":"a","oldText":"b","newText":"c"}]</parameter></function></tool_call>')
        with self.assertRaises(ValueError):
            parser.finish()

    def test_tool_schema_unknown_constraint_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_tools([{"type": "function", "function": {"name": "read", "parameters": {
                "type": "object", "unevaluatedProperties": False}}}])

    def test_ask_user_object_options_are_coerced_to_strings(self):
        parser = StreamParser("off", [{"type": "function", "function": {"name": "ask_user_question",
            "parameters": {"type": "object", "properties": {"question": {"type": "string"},
            "options": {"type": "array", "items": {"type": "string"}}}, "required": ["question"]}}}], "resp_a")
        parser.feed('<tool_call><function=ask_user_question><parameter=question>Choose</parameter>'
                    '<parameter=options>[{"label":"Go ahead"},{"title":"Stop"}]</parameter></function></tool_call>')
        arguments = json.loads(parser.finish()["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(arguments["options"], ["Go ahead", "Stop"])

    def test_ask_user_string_options_are_wrapped_as_title_objects(self):
        parser = StreamParser("off", [{"type": "function", "function": {"name": "ask_user",
            "parameters": {"type": "object", "properties": {"question": {"type": "string"},
            "options": {"type": "array", "items": {"type": "object", "properties": {
                "title": {"type": "string"}, "description": {"type": "string"}},
                "required": ["title"]}}}, "required": ["question"]}}}], "resp_a")
        parser.feed('<tool_call><function=ask_user><parameter=question>Choose</parameter>'
                    '<parameter=options>["Go ahead","Stop"]</parameter></function></tool_call>')
        arguments = json.loads(parser.finish()["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(arguments["options"], [{"title": "Go ahead"}, {"title": "Stop"}])

    def test_unclosed_tool_xml_is_detected_without_treating_closed_junk_as_incomplete(self):
        cut = StreamParser("off", TOOLS, "resp_a")
        cut.feed("<tool_call><function=edit><parameter=path>a.py")
        self.assertTrue(cut.unclosed_tool_xml())
        closed_junk = StreamParser("off", TOOLS, "resp_a")
        closed_junk.feed("<tool_call>broken</tool_call>")
        self.assertFalse(closed_junk.unclosed_tool_xml())
        with self.assertRaises(ValueError):
            closed_junk.finish()
