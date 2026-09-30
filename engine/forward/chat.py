"""Chat messages for the resident worker.

Droid posts an OpenAI chat completion: messages, tools, and a token budget.
This renders that turn with the artifact chat template, then reads the
generated ids back into assistant text and tool calls. Every messages turn
reasons at high. Image rows are embedded by the worker and spliced as
dynamic ids. Raw ``ids`` requests never come through this module.
"""
from __future__ import annotations

import time

import os

from engine.forward.cycle import _stop_requested
from engine.forward.sample import chooser
from qwasar_runtime.engine import incomplete_reason_from_error, reasoning_token_cap, tool_error_summary
from qwasar_runtime.parsing import StreamParser, canonical, image_parts, validate_messages
from qwasar_runtime.rendering import REASONING_CLOSE, encode, render
from qwasar_runtime.vision import MAX_IMAGES, decode_image_data


# The model card's sampler for thinking mode: temperature 1.0, top_p 0.95,
# top_k 20, min_p 0. The ExLlama worker has always applied top_k 20.
TOP_K = int(os.environ.get("QWARZ_TOP_K", "20"))
# Sampled drafts with the min(1, p/q) test instead of greedy drafts with an
# exact-match test. Both keep the target's law; this one accepts more.
SPECULATIVE = os.environ.get("QWARZ_SPEC_SAMPLING", "1") != "0"


def decode_tokens(tokenizer, ids):
    ids = [int(token) for token in ids]
    if not ids:
        return ""
    own = getattr(tokenizer, "decode_ids", None)
    if own is not None:
        return own(ids)
    backend = getattr(tokenizer, "tokenizer", None)
    decode = getattr(backend, "decode", None)
    if decode is None:
        raise RuntimeError("tokenizer cannot decode generated tokens")
    return decode(ids, skip_special_tokens=False)


def _stream_piece(previous, text, hold_replacement):
    """Return only the characters the parser has not seen yet.

    A byte-level decode rewrites a trailing U+FFFD once the next token
    completes the character. Republishing the whole string pastes the
    reasoning and the closing tag into the answer on every later token.
    """
    stable = text
    if hold_replacement:
        while stable.endswith("\ufffd"):
            stable = stable[:-1]
    if stable.startswith(previous):
        return stable[len(previous):], stable
    index = 0
    limit = min(len(previous), len(stable))
    while index < limit and previous[index] == stable[index]:
        index += 1
    return stable[index:], stable


class IncrementalText:
    """Decode only the tail since the last complete-character anchor.

    Re-decoding every visible token on every window grew with the answer
    (2.8 ms per window at 32K tokens, with the GPU idle). A byte-level decode
    of two pieces split at a complete character equals the decode of the
    whole, so the anchor moves only when the tail ends on one.
    """

    SPAN = 32

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.ids = []
        self.anchor = 0
        self.fed = ""

    def push(self, ids):
        self.ids.extend(int(token) for token in ids)
        tail = decode_tokens(self.tokenizer, self.ids[self.anchor:])
        piece, self.fed = _stream_piece(self.fed, tail, True)
        if (
            len(self.ids) - self.anchor >= self.SPAN
            and self.fed == tail
            and not tail.endswith("\ufffd")
        ):
            self.anchor = len(self.ids)
            self.fed = ""
        return piece

    def flush(self):
        if self.anchor >= len(self.ids):
            return ""
        tail = decode_tokens(self.tokenizer, self.ids[self.anchor:])
        piece, self.fed = _stream_piece(self.fed, tail, False)
        return piece


def prepare_chat(tokenizer, request, parent, context, embedder=None):
    if not isinstance(request, dict):
        raise ValueError("request must be an object")
    tools = request.get("tools") or []
    requested = request.get("thinking") or "xhigh"
    if requested not in ("off", "low", "medium", "xhigh"):
        raise ValueError("thinking must be off, low, medium, or xhigh")
    # The client effort is validated and then ignored. Messages always reason high.
    thinking = "xhigh"
    choice = request.get("tool_choice") or "auto"
    messages = validate_messages(request.get("messages"), tools)
    maximum = request.get("max_new", request.get("max_tokens", 32768))
    if type(maximum) is not int or maximum < 1 or maximum > 32768:
        raise ValueError("max_tokens must be 1..32768")
    temperature = _sampling(request, "temperature", 0, 2)
    top_p = _sampling(request, "top_p", 0, 1, exclusive_low=True)
    seed = request.get("seed", 42)
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be nonnegative integer")
    override = request.get("reasoning_budget_tokens")
    if override is not None and (type(override) is not int or override < 0 or override > 32768):
        raise ValueError("reasoning_budget_tokens must be 0..32768")
    image_tokens, images = _resolve_images(messages, request.get("images"), embedder)
    segments = _segments(parent)
    visible_tools = [] if choice == "none" else tools
    tokens, assistant_start, retained = render(
        tokenizer, messages, visible_tools, thinking, choice, segments, image_tokens=image_tokens,
    )
    stop = encode(tokenizer, "<|im_end|>")
    if len(stop) != 1:
        raise ValueError("native end token must be a single id")
    room = context - len(tokens)
    if room < 1:
        raise ValueError(
            f"prompt ({len(tokens)}) exceeds the resident context ({context})"
        )
    if maximum > room:
        maximum = room
    cap = reasoning_token_cap(thinking, maximum, override)
    return {
        "tokens": tokens,
        "assistant_start": assistant_start,
        "messages": messages,
        "segments": retained,
        "header": canonical({"tools": tools, "thinking": thinking, "tool_choice": choice}),
        "max_new": maximum,
        "stop": stop[0],
        "thinking": thinking,
        "tools": visible_tools,
        "tool_choice": choice,
        "reasoning_cap": cap,
        "close_ids": encode(tokenizer, REASONING_CLOSE),
        "temperature": None if "temperature" not in request else float(request["temperature"]),
        "top_p": 0.95 if top_p is None else float(top_p),
        "top_k": TOP_K,
        "vocab": _vocab(tokenizer),
        "cuts": prefix_cuts(tokens),
        "seed": seed,
        "images": images,
    }


IM_START = 248045


def prefix_cuts(tokens, page=256):
    """Page floors of the two prefixes a later session is likely to share.

    The end of the system block (system prompt and tools) is shared by every
    new session of the same client; the start of the last message is the
    history an edited or regenerated turn keeps.
    """
    starts = [index for index, token in enumerate(tokens) if token == IM_START]
    cuts = set()
    if len(starts) >= 2:
        cuts.add(starts[1] // page * page)
    if len(starts) >= 3:
        cuts.add(starts[-2] // page * page)
    return sorted(cut for cut in cuts if cut > 0)


def _vocab(tokenizer):
    """Ids the tokenizer can decode. The output head pads past them."""
    size = getattr(tokenizer, "actual_vocab_size", None)
    return int(size) if isinstance(size, int) and size > 0 else None


def _sampling(request, name, low, high, exclusive_low=False):
    if name not in request:
        return None
    value = request[name]
    if type(value) is bool or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if value < low or value > high or (exclusive_low and value == low):
        raise ValueError("invalid sampling range")
    return value


def _resolve_images(messages, supplied, embedder):
    parts = list(image_parts(messages))
    if not parts:
        return {}, []
    unique = []
    for part in parts:
        if part["sha256"] not in unique:
            unique.append(part["sha256"])
    if len(unique) > MAX_IMAGES:
        raise ValueError(f"at most {MAX_IMAGES} images per request")
    supplied = supplied if isinstance(supplied, dict) else {}
    tokens = {}
    records = []
    for part in parts:
        sha = part["sha256"]
        if sha in tokens:
            continue
        entry = supplied.get(sha)
        if not isinstance(entry, dict):
            raise ValueError(f"image {sha} has no inline data")
        raw = decode_image_data(sha, part["media_type"], entry.get("data"))
        if embedder is None:
            raise ValueError(f"image {sha} has no embedding")
        record = embedder(sha, part["media_type"], raw)
        row = record["tokens"] if isinstance(record, dict) else list(record)
        tokens[sha] = list(row)
        if isinstance(record, dict):
            records.append(record)
    return tokens, records


def _target_choose(runner, prepared):
    """Scripted windows stay greedy. A real target, or an explicit temperature, samples."""
    temperature = prepared.get("temperature")
    has_window = callable(getattr(runner, "score_window", None))
    has_logits = callable(getattr(runner, "score_logits", None))
    if has_window and not has_logits and (temperature is None or temperature == 0):
        return None
    if temperature is None:
        temperature = 1.0
    return chooser(
        float(temperature), float(prepared.get("top_p", 0.95)), int(prepared.get("seed", 42)),
        top_k=int(prepared.get("top_k", TOP_K)), vocab=prepared.get("vocab"), speculative=SPECULATIVE,
    )


def run_turn(tokenizer, session, prepared, response_id, publish, cancel=None, embed_ids=None, inv_freq=None):
    """Generate one chat turn and publish each accepted piece before the next window."""
    parser = StreamParser(
        prepared["thinking"], prepared["tools"], response_id, prepared["tool_choice"],
    )
    stop = prepared["stop"]
    cap = prepared["reasoning_cap"]
    close_ids = list(prepared["close_ids"])
    maximum = prepared["max_new"]
    started = time.perf_counter()
    state = {
        "seen": 0,
        "visible": [],
        "text": IncrementalText(tokenizer),
        "streamed": {"content": 0, "reasoning": 0},
        "reasoning_tokens": 0,
        "close_reason": None,
        "budget_halt": False,
        "first": None,
        "first_content": None,
        "last": None,
        "first_batch": 0,
        "emitted": 0,
    }

    def emit_text(channel, text, now):
        if not text:
            return
        state["streamed"][channel] += len(text)
        if channel == "content" and text.strip() and state["first_content"] is None:
            state["first_content"] = now
        publish({"type": "delta", "id": response_id, "channel": channel, "text": text})

    def feed(text, now):
        if not text:
            return
        for channel, piece in parser.feed(text):
            emit_text(channel, piece, now)

    def on_accepted(produced):
        new = produced[state["seen"]:]
        if not new:
            return None
        hit_stop = new[-1] == stop
        visible = new[:-1] if hit_stop else list(new)
        now = time.perf_counter()
        if state["first"] is None:
            state["first"] = now
            state["first_batch"] = len(new)
        state["emitted"] += len(new)
        state["last"] = now
        if parser.channel == "reasoning":
            state["reasoning_tokens"] += len(new)
        if visible:
            state["visible"].extend(int(token) for token in visible)
            feed(state["text"].push(visible), now)
        state["seen"] = len(produced)
        if _stop_requested(cancel):
            return None
        hit_budget = cap > 0 and state["reasoning_tokens"] >= cap
        if state["close_reason"] is None and parser.channel == "reasoning" and (hit_budget or hit_stop):
            if hit_stop:
                state["seen"] -= 1
            feed(REASONING_CLOSE, now)
            state["seen"] += len(close_ids)
            state["emitted"] += len(close_ids)
            state["close_reason"] = "budget" if hit_budget else "unterminated"
            leftover = maximum - (len(produced) - (1 if hit_stop else 0) + len(close_ids))
            state["budget_halt"] = leftover <= 0
            return {"drop_last": hit_stop, "inject": list(close_ids), "halt": leftover <= 0}
        return None

    result = session.generate(
        prepared["tokens"], maximum, stop_ids={stop}, choose=_target_choose(session.runner, prepared),
        on_accepted=on_accepted, cancel=cancel, embed_ids=embed_ids, inv_freq=inv_freq,
        cuts=prepared.get("cuts"),
    )
    finished = time.perf_counter()
    if state["visible"]:
        feed(state["text"].flush(), finished)
    produced = list(result["tokens"])
    status = "completed"
    incomplete = None
    if result["status"] == "cancelled":
        status = "cancelled"
    elif state["budget_halt"]:
        status = "incomplete"
        incomplete = "reasoning_budget"
    elif not produced or produced[-1] != stop:
        status = "incomplete"
        incomplete = "max_new_tokens"
    if status != "cancelled" and parser.in_tools and parser.unclosed_tool_xml():
        status = "incomplete"
        if incomplete is None:
            incomplete = "max_new_tokens" if not produced or produced[-1] != stop else "unclosed_tool_call"
    tool_error = None
    try:
        message = parser.finish(complete=status == "completed")
    except ValueError as error:
        diagnostic = getattr(parser, "tool_diagnostic", None)
        message = parser.finish(complete=False)
        if status != "cancelled":
            status = "incomplete"
            if incomplete is None:
                eos = "max_new_tokens" if not produced or produced[-1] != stop else "stop_token"
                incomplete = incomplete_reason_from_error(error, {"eos_reason": eos})
        if isinstance(diagnostic, dict):
            diagnostic = {**diagnostic, "error": {"class": type(error).__name__, "message": str(error)}}
        tool_error = tool_error_summary(diagnostic)
    for channel, field in (("content", "content"), ("reasoning", "reasoning_content")):
        tail = message[field][state["streamed"][channel]:]
        if tail:
            emit_text(channel, tail, finished)
    accepted = int(result.get("accepted") or 0)
    rejected = int(result.get("rejected") or 0)
    cached = result.get("cached_tokens")
    valid = status != "cancelled"
    prompt = len(prepared["tokens"])
    first = state["first"]
    last = state["last"]
    if status == "cancelled":
        finish = "cancelled"
    elif incomplete in ("max_new_tokens", "reasoning_budget"):
        finish = "max_new_tokens"
    else:
        finish = "stop_token"
    metrics = {
        "ttft_ms": (first - started) * 1000 if first is not None else None,
        "first_content_ms": (state["first_content"] - started) * 1000 if state["first_content"] is not None else None,
        "elapsed_ms": (finished - started) * 1000,
        "decode_tokens_per_second": (state["emitted"] - state["first_batch"]) / (last - first)
            if first is not None and last is not None and last > first and state["emitted"] > state["first_batch"] else None,
        "first_batch_tokens": state["first_batch"],
        "requeue_count": 0,
        "cache_metrics_valid": valid,
        "cached_tokens": cached if valid else None,
        "physical_prefill_tokens": max(prompt - 1 - int(cached or 0), 0) if valid and cached is not None else None,
        "accepted_draft_tokens": accepted if valid else None,
        "rejected_draft_tokens": rejected if valid else None,
        "draft_acceptance": accepted / (accepted + rejected) if valid and accepted + rejected else None,
        "host_prefill_ms": None,
        "finish_reason": finish,
        "reasoning_closed": state["close_reason"],
        "reasoning_tokens": state["reasoning_tokens"],
        "incomplete_reason": incomplete,
        "tool_error": tool_error,
    }
    return {
        "message": message,
        "status": status,
        "tokens": produced,
        "cached_tokens": cached,
        "metrics": metrics,
    }


def finish_turn(tokenizer, prepared, produced, response_id):
    produced = [int(token) for token in produced]
    stopped = bool(produced) and produced[-1] == prepared["stop"]
    visible = produced[:-1] if stopped else produced
    text = decode_tokens(tokenizer, visible)
    parser = StreamParser(
        prepared["thinking"], prepared["tools"], response_id, prepared["tool_choice"],
    )
    deltas = []
    streamed = {"content": 0, "reasoning": 0}
    for channel, piece in parser.feed(text):
        if not piece:
            continue
        deltas.append({"channel": channel, "text": piece})
        streamed[channel] += len(piece)
    incomplete = None
    try:
        message = parser.finish(complete=stopped)
    except ValueError as error:
        message = parser.finish(complete=False)
        stopped = False
        incomplete = "unterminated_reasoning" if "reasoning" in str(error) else "unclosed_tool_call"
    if not stopped and incomplete is None:
        incomplete = "max_new_tokens"
    for channel, field in (("content", "content"), ("reasoning", "reasoning_content")):
        tail = message[field][streamed[channel]:]
        if tail:
            deltas.append({"channel": channel, "text": tail})
    status = "completed" if stopped else "incomplete"
    return {
        "message": message,
        "status": status,
        "deltas": deltas,
        "metrics": {
            "finish_reason": "stop_token" if incomplete != "max_new_tokens" else "max_new_tokens",
            "incomplete_reason": incomplete,
        },
    }


def _segments(parent):
    if not isinstance(parent, dict) or parent.get("version") != 1:
        return []
    raw = parent.get("segments")
    if not isinstance(raw, list):
        return []
    for segment in raw:
        tokens = segment.get("tokens") if isinstance(segment, dict) else None
        message = segment.get("message") if isinstance(segment, dict) else None
        if (not isinstance(tokens, list) or not tokens or not isinstance(message, dict)
                or any(type(token) is not int or token < 0 for token in tokens)):
            return []
    return raw
