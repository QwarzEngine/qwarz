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

    def test_duplicate_and_wrong_type_parameters_rejected(self):
        for arguments in ('<parameter=path>a</parameter><parameter=path>b</parameter>',
                          '<parameter=path>a</parameter><parameter=edits>12</parameter>'):
            parser = StreamParser("off", TOOLS, "resp_a")
            parser.feed('<tool_call><function=edit>' + arguments + '</function></tool_call>')
            with self.assertRaises(ValueError):
                parser.finish()

    def test_tool_results_must_correspond_and_be_unique(self):
        with self.assertRaises(ValueError):
            validate_messages([{"role": "user", "content": "go"}, {
                "role": "tool", "tool_call_id": "missing", "content": "data"}], TOOLS)

    def test_schema_name_injection_rejected(self):
        with self.assertRaises(ValueError):
            validate_tools([{"type": "function", "function": {"name": "x>bad", "parameters": {}}}])

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
