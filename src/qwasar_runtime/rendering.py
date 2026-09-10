from __future__ import annotations

import copy
from dataclasses import dataclass
import re
import uuid

from .parsing import canonical, message_key, parse_json


def encode(tokenizer, text, literal=False):
    if literal:
        backend = tokenizer.tokenizer
        previous = backend.encode_special_tokens
        try:
            backend.encode_special_tokens = True
            return backend.encode(text, add_special_tokens=False).ids
        finally:
            backend.encode_special_tokens = previous
    return tokenizer.encode(text, encode_special_tokens=True).flatten().tolist()


@dataclass
class Prompt:
    tokens: list[int]
    assistant_start: int
    messages: list[dict]
    segments: list[dict]
    request: dict
    header: str


def render(tokenizer, messages, tools, thinking, tool_choice, segments):
    replacements = {}
    prefix = "QWASAR_" + uuid.uuid4().hex + "_"

    def placeholder(value, tokens=None):
        marker = prefix + str(len(replacements)) + "_END"
        replacements[marker] = encode(tokenizer, value, literal=True) if tokens is None else tokens
        return marker

    protected = copy.deepcopy(messages)
    retained = []
    frame_markers = []
    available = list(segments)
    for message in protected:
        segment = next((item for item in available if message_key(item["message"]) == message_key(message)), None)
        if message["role"] == "assistant" and segment:
            available.remove(segment)
            retained.append(segment)
            marker = placeholder("", segment["tokens"])
            message.update(content=marker, reasoning_content="", tool_calls=[])
            frame_markers.append(marker)
        else:
            for field in ("content", "reasoning_content"):
                if message.get(field):
                    message[field] = placeholder(message[field])
            for call in message.get("tool_calls", []):
                function = call["function"]
                arguments = parse_json(function["arguments"])
                function["arguments"] = {name: placeholder(value if isinstance(value, str) else canonical(value))
                                         for name, value in arguments.items()}
    protected_tools = []
    for tool in tools:
        marker = placeholder(canonical(tool))
        protected_tools.append({marker: marker})
    if tool_choice == "required" or isinstance(tool_choice, dict):
        instruction = "You must call a tool in this response."
        if isinstance(tool_choice, dict):
            instruction = "You must call the " + tool_choice["function"]["name"] + " tool in this response."
        if protected[0]["role"] == "system":
            protected[0]["content"] += "\n\n" + instruction
        else:
            protected.insert(0, {"role": "system", "content": instruction})
    rendered = tokenizer.hf_render_chat_template(
        protected, tools=protected_tools, add_generation_prompt=True,
        enable_thinking=thinking != "off", reasoning_effort=thinking if thinking != "off" else "medium",
        preserve_thinking=True,
    )
    for protected_tool in protected_tools:
        marker = next(iter(protected_tool))
        pattern = r'\{\s*"' + re.escape(marker) + r'"\s*:\s*"' + re.escape(marker) + r'"\s*\}'
        rendered, replaced = re.subn(pattern, marker, rendered)
        if replaced != 1:
            raise ValueError("template must preserve each tool schema placeholder exactly once")
    for marker in frame_markers:
        position = rendered.index(marker)
        start = rendered.rfind("<|im_start|>assistant\n", 0, position)
        end = rendered.index("<|im_end|>", position) + len("<|im_end|>")
        if start < 0:
            raise ValueError("template lost assistant frame")
        rendered = rendered[:start] + marker + rendered[end:]
    tokens = []
    pattern = re.compile("(" + re.escape(prefix) + r"\d+_END)")
    for part in pattern.split(rendered):
        if part in replacements:
            tokens.extend(replacements[part])
        else:
            tokens.extend(encode(tokenizer, part))
    start_id = encode(tokenizer, "<|im_start|>")
    if len(start_id) != 1:
        raise ValueError("native assistant delimiter must be a single token")
    assistant_start = len(tokens) - 1 - tokens[::-1].index(start_id[0])
    return tokens, assistant_start, retained
