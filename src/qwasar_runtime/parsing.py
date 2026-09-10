from __future__ import annotations

import copy
import hashlib
import json
import math
import re


NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_-]{0,127}\Z")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate JSON property: {name}")
        result[name] = value
    return result


def parse_json(text):
    return json.loads(text, object_pairs_hook=unique_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def validate_schema(schema, depth=0):
    if type(schema) is bool:
        return
    if not isinstance(schema, dict) or depth > 32:
        raise ValueError("invalid or excessively nested schema")
    supported = {"type", "properties", "patternProperties", "required", "additionalProperties", "items", "minItems", "maxItems",
        "uniqueItems", "minLength", "maxLength", "pattern", "minimum", "maximum", "exclusiveMinimum",
        "exclusiveMaximum", "enum", "const", "anyOf", "oneOf", "allOf", "$ref", "$defs", "definitions",
        "$schema", "$id", "title", "description", "default", "examples", "deprecated", "readOnly", "writeOnly"}
    unknown = schema.keys() - supported
    if unknown:
        raise ValueError("unsupported schema keywords: " + ", ".join(sorted(unknown)))
    types = schema.get("type", [])
    types = [types] if isinstance(types, str) else types
    if not isinstance(types, list) or any(kind not in ("object", "array", "string", "integer", "number", "boolean", "null") for kind in types):
        raise ValueError("invalid schema type")
    for keyword in ("properties", "patternProperties", "$defs", "definitions"):
        if keyword in schema:
            if not isinstance(schema[keyword], dict):
                raise ValueError(f"{keyword} must be an object")
            if keyword == "patternProperties":
                for pattern in schema[keyword]:
                    try:
                        re.compile(pattern)
                    except (re.error, TypeError) as error:
                        raise ValueError("invalid patternProperties pattern") from error
            for child in schema[keyword].values():
                validate_schema(child, depth + 1)
    for keyword in ("additionalProperties", "items"):
        if keyword in schema:
            validate_schema(schema[keyword], depth + 1)
    for keyword in ("anyOf", "oneOf", "allOf"):
        if keyword in schema:
            if not isinstance(schema[keyword], list) or not schema[keyword]:
                raise ValueError(f"{keyword} must be a nonempty array")
            for child in schema[keyword]:
                validate_schema(child, depth + 1)
    required = schema.get("required", [])
    if not isinstance(required, list) or any(not isinstance(name, str) for name in required) or len(required) != len(set(required)):
        raise ValueError("required must contain unique property names")
    if "$ref" in schema and (not isinstance(schema["$ref"], str) or not schema["$ref"].startswith("#/")):
        raise ValueError("only local schema references are supported")
    if "pattern" in schema:
        try:
            re.compile(schema["pattern"])
        except (re.error, TypeError) as error:
            raise ValueError("invalid schema pattern") from error


def property_schemas(schema, name):
    matches = []
    if name in schema.get("properties", {}):
        matches.append(schema["properties"][name])
    matches.extend(candidate for pattern, candidate in schema.get("patternProperties", {}).items()
                   if re.search(pattern, name))
    return matches or [schema.get("additionalProperties", True)]


class SchemaValidationError(ValueError):
    def __init__(self, message, path, value, schema):
        super().__init__(message)
        self.path = list(path)
        self.expected_type = schema.get("type") if isinstance(schema, dict) else None
        self.actual_type = ("null" if value is None else "boolean" if type(value) is bool else
                            "object" if isinstance(value, dict) else "array" if isinstance(value, list) else
                            "string" if isinstance(value, str) else "integer" if type(value) is int else
                            "number" if type(value) is float else "unknown")


def validate_value(value, schema, root=None, path=()):
    try:
        _validate_value(value, schema, root, path)
    except SchemaValidationError:
        raise
    except ValueError as error:
        raise SchemaValidationError(str(error), path, value, schema) from error


def _validate_value(value, schema, root, path):
    root = schema if root is None else root
    if schema is True:
        return
    if schema is False:
        raise ValueError("value prohibited by schema")
    if not isinstance(schema, dict):
        raise ValueError("schema must be an object or boolean")
    if "$ref" in schema:
        reference = schema["$ref"]
        if not reference.startswith("#/"):
            raise ValueError("only local schema references are supported")
        target = root
        for component in reference[2:].split("/"):
            target = target[component.replace("~1", "/").replace("~0", "~")]
        validate_value(value, target, root, path)
    for keyword in ("anyOf", "oneOf", "allOf"):
        if keyword in schema:
            matches = 0
            for candidate in schema[keyword]:
                try:
                    validate_value(value, candidate, root, path)
                    matches += 1
                except ValueError:
                    pass
            if (keyword == "anyOf" and not matches or keyword == "oneOf" and matches != 1
                    or keyword == "allOf" and matches != len(schema[keyword])):
                raise ValueError(f"value does not satisfy {keyword}")
    expected = schema.get("type")
    types = {"string": isinstance(value, str), "object": isinstance(value, dict),
             "array": isinstance(value, list), "integer": type(value) is int,
             "number": type(value) in (int, float) and math.isfinite(value),
             "boolean": type(value) is bool, "null": value is None}
    if expected is not None and not any(types.get(kind, False) for kind in
                                        (expected if isinstance(expected, list) else [expected])):
        raise ValueError(f"expected {expected}")
    if "enum" in schema and canonical(value) not in [canonical(item) for item in schema["enum"]]:
        raise ValueError("value outside enum")
    if "const" in schema and canonical(value) != canonical(schema["const"]):
        raise ValueError("value differs from const")
    if isinstance(value, dict):
        if set(schema.get("required", [])) - value.keys():
            raise ValueError("missing required parameters")
        for name, item in value.items():
            for candidate in property_schemas(schema, name):
                validate_value(item, candidate, root, (*path, name))
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", math.inf):
            raise ValueError("array length outside schema")
        if schema.get("uniqueItems") and len({canonical(item) for item in value}) != len(value):
            raise ValueError("array items must be unique")
        for index, item in enumerate(value):
            validate_value(item, schema.get("items", True), root, (*path, index))
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0) or len(value) > schema.get("maxLength", math.inf):
            raise ValueError("string length outside schema")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            raise ValueError("string does not match pattern")
    if type(value) in (int, float):
        if not math.isfinite(value) or value < schema.get("minimum", -math.inf) or value > schema.get("maximum", math.inf):
            raise ValueError("number outside schema")
        if value <= schema.get("exclusiveMinimum", -math.inf) or value >= schema.get("exclusiveMaximum", math.inf):
            raise ValueError("number outside exclusive limits")


def validate_tools(tools):
    if not isinstance(tools, list):
        raise ValueError("tools must be an array")
    functions = {}
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise ValueError("only function tools are supported")
        function = tool.get("function", {})
        if not isinstance(function, dict):
            raise ValueError("function must be an object")
        name = function.get("name", "")
        if not isinstance(name, str) or not NAME.fullmatch(name) or name in functions:
            raise ValueError("tool names must be valid and unique")
        schema = function.get("parameters", {"type": "object", "properties": {}})
        if not isinstance(schema, dict) or schema.get("type", "object") != "object":
            raise ValueError("tool parameters must use an object schema")
        validate_schema(schema)
        for parameter in schema.get("properties", {}):
            if not NAME.fullmatch(parameter):
                raise ValueError("invalid parameter name")
        functions[name] = schema
    return functions


def validate_messages(messages, tools):
    validate_tools(tools)
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty array")
    output, pending, used = [], {}, set()
    for index, original in enumerate(messages):
        if not isinstance(original, dict):
            raise ValueError("messages must be objects")
        message = copy.deepcopy(original)
        role = message.get("role")
        if role not in ("system", "user", "assistant", "tool") or role == "system" and index:
            raise ValueError("invalid message role or system position")
        content = message.get("content")
        content = "" if content is None else content
        if isinstance(content, list):
            if any(not isinstance(part, dict) or part.get("type") != "text" or not isinstance(part.get("text"), str) for part in content):
                raise ValueError("only text message parts are supported")
            content = "".join(part["text"] for part in content)
        if not isinstance(content, str):
            raise ValueError("message content must be text")
        normalized = {"role": role, "content": content}
        if role == "tool":
            call_id = message.get("tool_call_id")
            if call_id not in pending:
                raise ValueError("tool result has no pending call")
            if message.get("name", pending[call_id]) != pending[call_id]:
                raise ValueError("tool result name mismatch")
            normalized["tool_call_id"] = call_id
            del pending[call_id]
        elif pending:
            raise ValueError("all tool results must precede the next message")
        if role == "assistant":
            reasoning = message.get("reasoning_content") or ""
            if not isinstance(reasoning, str):
                raise ValueError("reasoning_content must be text")
            normalized.update(reasoning_content=reasoning, tool_calls=[])
            calls = message.get("tool_calls")
            calls = [] if calls is None else calls
            if not isinstance(calls, list):
                raise ValueError("tool_calls must be an array")
            for call in calls:
                if not isinstance(call, dict):
                    raise ValueError("tool call must be an object")
                call_id = call.get("id")
                function = call.get("function", {})
                if not isinstance(function, dict):
                    raise ValueError("tool call function must be an object")
                name = function.get("name", "")
                if not isinstance(call_id, str) or not call_id or call_id in used or not isinstance(name, str) or not NAME.fullmatch(name):
                    raise ValueError("invalid or duplicate tool call")
                arguments = function.get("arguments", "{}")
                arguments = parse_json(arguments) if isinstance(arguments, str) else arguments
                if not isinstance(arguments, dict):
                    raise ValueError("tool arguments must be an object")
                if any(not isinstance(parameter, str) or not NAME.fullmatch(parameter) for parameter in arguments):
                    raise ValueError("invalid historical parameter name")
                normalized["tool_calls"].append({"id": call_id, "type": "function", "function": {
                    "name": name, "arguments": canonical(arguments)}})
                used.add(call_id)
                pending[call_id] = name
        output.append(normalized)
    if pending or not any(message["role"] == "user" for message in output):
        raise ValueError("messages need a user query and all pending tool results")
    return output


def message_key(message):
    return canonical({key: value for key, value in message.items() if key != "reasoning_content"})


class StreamParser:
    def __init__(self, thinking, tools, response_id, tool_choice="auto"):
        self.channel = "content" if thinking == "off" else "reasoning"
        self.functions = validate_tools(tools)
        self.response_id, self.tool_choice = response_id, tool_choice
        self.pending, self.xml = "", ""
        self.reasoning, self.content = "", ""
        self.leading = ""
        self.in_tools = False
        self.tool_diagnostic = None

    def feed(self, text):
        if self.in_tools:
            self.xml += text
            return []
        self.pending += text
        output = []
        while self.pending:
            marker = "</think>" if self.channel == "reasoning" else "<tool_call>" if self.functions else None
            position = self.pending.find(marker) if marker else -1
            if position >= 0:
                self._emit(self.pending[:position], output)
                self.pending = self.pending[position + len(marker):]
                if self.channel == "reasoning":
                    self.channel = "content"
                else:
                    self.in_tools = True
                    self.leading = ""
                    self.xml = marker + self.pending
                    self.pending = ""
                    break
            else:
                held = 0
                if marker:
                    for length in range(1, min(len(marker), len(self.pending) + 1)):
                        if self.pending.endswith(marker[:length]):
                            held = length
                visible = self.pending[:-held] if held else self.pending
                self._emit(visible, output)
                self.pending = self.pending[-held:] if held else ""
                break
        return output

    def _emit(self, text, output):
        if not text:
            return
        if self.channel == "reasoning":
            self.reasoning += text
        else:
            if self.functions and not self.content:
                if text.isspace():
                    self.leading += text
                    return
                text = self.leading + text
                self.leading = ""
            self.content += text
        output.append((self.channel, text))

    def finish(self, complete=True):
        if complete and self.channel == "reasoning":
            raise ValueError("generation ended before reasoning closed")
        calls = []
        if complete and self.in_tools:
            self.tool_diagnostic = {"raw_tool_calls": self.xml, "tool_schemas": self.functions,
                "tool": None, "parsed_arguments": {}, "stage": "native_parse", "call_index": 0}
            remaining = self.xml.strip()
            while remaining:
                self.tool_diagnostic.update(tool=None, parsed_arguments={}, stage="native_parse", call_index=len(calls))
                match = re.match(r"<tool_call>\s*<function=([A-Za-z_][A-Za-z0-9_-]*)>(.*?)</function>\s*</tool_call>", remaining, re.S)
                if not match:
                    raise ValueError("malformed native tool call")
                name, parameters = match.group(1, 2)
                self.tool_diagnostic["tool"] = name
                if name not in self.functions or self.tool_choice == "none":
                    raise ValueError("undeclared or prohibited tool")
                if isinstance(self.tool_choice, dict) and name != self.tool_choice["function"]["name"]:
                    raise ValueError("tool does not match tool_choice")
                schema, arguments = self.functions[name], {}
                self.tool_diagnostic["parsed_arguments"] = arguments
                while parameters.strip():
                    parameter = re.match(r"\s*<parameter=([A-Za-z_][A-Za-z0-9_-]*)>(.*?)</parameter>", parameters, re.S)
                    if not parameter or parameter[1] in arguments:
                        raise ValueError("malformed or duplicate parameter")
                    key, value = parameter.group(1, 2)
                    if value.startswith("\n"):
                        value = value[1:]
                    if value.endswith("\n"):
                        value = value[:-1]
                    kind = schema.get("properties", {}).get(key, {}).get("type")
                    if any(isinstance(candidate, dict) and candidate.get("type") == "string"
                           for candidate in property_schemas(schema, key)):
                        kind = "string"
                    if kind != "string":
                        try:
                            value = parse_json(value)
                        except (ValueError, TypeError):
                            if kind is not None:
                                raise ValueError(f"invalid JSON value for {key}")
                    arguments[key] = value
                    parameters = parameters[parameter.end():]
                self.tool_diagnostic["stage"] = "schema_validation"
                validate_value(arguments, schema)
                call_id = "call_" + hashlib.sha256(f"{self.response_id}:{len(calls)}".encode()).hexdigest()[:24]
                calls.append({"id": call_id, "type": "function", "function": {
                    "name": name, "arguments": canonical(arguments)}})
                remaining = remaining[match.end():].strip()
            self.tool_diagnostic = None
        if complete and (self.tool_choice == "required" or isinstance(self.tool_choice, dict)) and not calls:
            raise ValueError("required tool call was not generated")
        return {"role": "assistant", "content": self.content + self.leading + (self.pending if self.channel == "content" and not self.in_tools else ""),
                "reasoning_content": self.reasoning + (self.pending if self.channel == "reasoning" else ""), "tool_calls": calls}
