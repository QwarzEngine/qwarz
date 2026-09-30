"""Resident session, MTP6, vision ids, and the 37-cell cutover lock."""
from engine.forward.gate import GATES, cells, promotion_allowed
from engine.forward.mtp import DRAFT_TOKENS, acceptance, emitted_tokens, verify_window
from engine.forward.schedule import suffix_bounds
from engine.forward.session import MemoryRunner, Session, SessionBusy
from engine.forward.vision import MM_TOKEN_BASE, assemble_rows, canonical_tape, embed_spans
from engine.forward.worker import Worker


class _ExportRunner:
    """Records each cut and the post-norm rows that cut exported."""

    def __init__(self):
        self.calls = []
        self.exported = []
        self.hidden = None

    def forward(self, token_ids, cache_len):
        self.calls.append((cache_len, list(token_ids)))
        self.hidden = [("row", cache_len + index) for index in range(len(token_ids))]
        self.exported.append(self.hidden)


class _RecordingDraft:
    def __init__(self):
        self.seen = []
        self.position = 0

    def prefill(self, token_ids, paired):
        self.seen.append((list(token_ids), list(paired)))
        self.position += len(list(token_ids))


def test_cold_prefill_matches_the_measured_cuts():
    assert [end - start for start, end in suffix_bounds(0, 64)] == [63]
    assert suffix_bounds(0, 512) == [(0, 256), (256, 511)]
    assert [end - start for start, end in suffix_bounds(0, 32768)] == [8192, 8192, 8192, 7936, 255]
    long = [end - start for start, end in suffix_bounds(0, 261888)]
    assert long == [8192] * 31 + [7680, 255]
    assert suffix_bounds(8192, 32768) == suffix_bounds(0, 32768)[1:]
    assert suffix_bounds(0, 400) == [(0, 256), (256, 399)]

    from engine.forward.cycle import DraftCycle, prefill_held

    for length in (64, 400, 512, 32768, 261888):
        prompt = list(range(length))
        runner = _ExportRunner()
        draft = _RecordingDraft()
        cache_len, held = prefill_held(runner, DraftCycle(runner, draft=draft), prompt)
        spans = suffix_bounds(0, length)
        assert runner.calls == [(start, prompt[start:end]) for start, end in spans]
        assert cache_len == length - 1 == spans[-1][1]
        assert held == prompt[-1]
        assert all(prompt[-1] not in span for _cache, span in runner.calls)
        assert draft.position == cache_len
        assert [ids for ids, _paired in draft.seen] == [prompt[start:end] for start, end in spans]
        assert draft.seen[0][1][0] is None
        assert draft.seen[0][1][1:] == runner.exported[0][:-1]
        for previous, (ids, paired), exported in zip(runner.exported, draft.seen[1:], runner.exported[1:]):
            assert paired[0] == previous[-1]
            assert paired[1:] == exported[:-1]
            assert ids[0] == prompt[previous[-1][1] + 1]


def test_second_turn_forwards_only_the_suffix_and_cancel_restores_the_commit():
    session = Session(MemoryRunner())
    first = list(range(64))
    done = session.turn(first)
    assert done["cached_tokens"] == 0
    assert done["cursor"] == 64
    assert session.runner.calls == [(0, first[:63]), (63, first[63:])]

    session.runner.calls.clear()
    extended = first + list(range(100, 110))
    again = session.turn(extended)
    assert again["cached_tokens"] == 64
    assert [start for start, _end in again["forwarded"]][0] == 64
    assert session.runner.state == len(extended)

    session.runner.calls.clear()
    seen = {"n": 0}

    def cancel_on_second():
        seen["n"] += 1
        return seen["n"] == 2

    stopped = session.turn(extended + [7, 8, 9], cancelled=cancel_on_second)
    assert stopped["status"] == "cancelled"
    assert session.cursor == len(extended)
    assert session.runner.state == len(extended)
    assert session.busy is False


def test_a_second_generate_is_rejected_while_one_is_open():
    session = Session(MemoryRunner())
    session.turn(list(range(8)))
    worker = Worker(session)
    worker.active = "resp_open"
    rejected = worker.push('{"op":"generate","id":"resp_b","request":{"ids":[1,2,3,4]}}')
    assert rejected[0]["error"]["code"] == "runtime_busy"
    assert rejected[0]["error"]["http_status"] == 409
    assert session.cursor == 8
    assert worker.active == "resp_open"
    print("OBSERVATION overlapping generate is runtime_busy HTTP 409 and the cursor stays on the open turn")


def test_branch_before_the_commit_restarts_from_zero():
    session = Session(MemoryRunner())
    session.turn([1, 2, 3, 4])
    session.runner.calls.clear()
    redone = session.turn([9, 8, 7, 6])
    assert redone["cached_tokens"] == 0
    assert session.runner.calls[0][0] == 0


def test_shift_rows_carries_the_last_post_norm_row_on_tensors():
    import pytest
    torch = pytest.importorskip("torch")
    from engine.forward.cycle import shift_rows

    first = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    second = torch.tensor([[[7.0, 8.0], [9.0, 10.0]]])
    paired, carry = shift_rows(None, first)
    nxt, nxt_carry = shift_rows(carry, second)
    blank = torch.zeros(1, 1, 2, dtype=torch.float16)
    assert torch.equal(paired[:, :1], blank)
    assert torch.equal(paired[:, 1:], first.to(dtype=torch.float16)[:, :-1])
    assert torch.equal(carry, first.to(dtype=torch.float16)[:, -1:])
    assert torch.equal(nxt[:, :1], carry)
    assert torch.equal(nxt[:, 1:], second.to(dtype=torch.float16)[:, :-1])
    assert torch.equal(nxt_carry, second.to(dtype=torch.float16)[:, -1:])


def test_emitted_tokens_keep_the_prefix_and_the_bonus():
    draft = [10, 11, 12, 13, 14, 15]
    full, matched = emitted_tokens(draft, [10, 11, 12, 13, 14, 15, 50])
    assert matched == 6
    assert full == [10, 11, 12, 13, 14, 15, 50]
    partial, matched = emitted_tokens(draft, [10, 11, 99, 13, 14, 15, 50])
    assert matched == 2
    assert partial == [10, 11, 99]
    corrected, matched = emitted_tokens(draft, [198, 11, 12, 13, 14, 15, 50])
    assert matched == 0
    assert corrected == [198]


def test_mtp6_keeps_the_matching_prefix_and_rewinds_the_rest():
    session = Session(MemoryRunner())
    session.turn([1, 2, 3, 4])
    assert DRAFT_TOKENS == 6
    draft = [10, 11, 12, 13, 14, 15]
    verified = [10, 11, 99, 13, 14, 15]
    outcome = verify_window(session, draft, verified, bonus=50)
    assert outcome["accepted"] == 2
    assert outcome["rejected"] == 4
    assert outcome["kept"] == [10, 11, 99]
    assert outcome["acceptance"] == acceptance(2, 4)
    assert session.cursor == 7
    assert session.runner.state == 7
    assert session.tape[-3:] == [10, 11, 99]
    speculative = session.runner.calls[-2]
    assert speculative == (4, draft)


def test_turn_passes_each_span_its_rows_and_the_whole_rope_table():
    class Rows:
        def __init__(self, rows):
            self.rows = rows

        def dim(self):
            return 2

        def __getitem__(self, item):
            return self.rows[item]

    class Recorder:
        def __init__(self):
            self.calls = []
            self.state = 0

        def forward(self, token_ids, cache_len, embedded=None, inv_freq=None):
            if cache_len != self.state:
                raise RuntimeError(f"forward at {cache_len} with resident cursor {self.state}")
            self.calls.append((cache_len, list(token_ids), embedded, inv_freq))
            self.state += len(token_ids)

        def capture(self):
            return self.state

        def restore(self, snapshot):
            self.state = snapshot

        def reset(self):
            self.state = 0

    ids = list(range(10))
    rows = [f"r{token}" for token in ids]
    session = Session(Recorder())
    done = session.turn(ids, embedded=Rows(rows), inv_freq="table")
    assert done["forwarded"] == [(0, 9), (9, 10)]
    assert session.runner.calls == [
        (0, ids[:9], rows[:9], "table"),
        (9, ids[9:], rows[9:], "table"),
    ]
    session.runner.calls.clear()
    follow = session.turn(ids + [20, 21, 22])
    assert follow["cached_tokens"] == 10
    assert follow["forwarded"][0][0] == 10
    assert all(row is None and table is None for _cache, _span, row, table in session.runner.calls)


def test_image_ids_skip_the_text_table_and_canonicalize():
    image = MM_TOKEN_BASE + 7
    assert canonical_tape([4, image, 5]) == [4, -1, 5]
    assert embed_spans([4, image, 5, image + 1]) == [
        ("text", (4,)), ("image", image), ("text", (5,)), ("image", image + 1),
    ]
    text, images = [], []
    assemble_rows([4, image], text.append, images.append)
    assert text == [4]
    assert images == [image]


def test_promotion_stays_closed_until_every_cell_passes():
    assert len(cells()) == 37
    assert len(set(cells())) == 37
    assert promotion_allowed({}) is False
    report = {"cells": {label: {"passed": True} for label in cells()}, "gates": {name: True for name in GATES}}
    assert promotion_allowed(report) is True
    missing = dict(report["cells"])
    missing.pop("warmup")
    assert promotion_allowed({"cells": missing, "gates": report["gates"]}) is False
    report["gates"]["acceptance_within_5pp"] = False
    assert promotion_allowed(report) is False


class _ScriptedDraft:
    def __init__(self):
        self.position = 0
        self._next = 10

    def prefill(self, token_ids, _hidden):
        self.position += len(list(token_ids))

    def step(self, _token, _state):
        drafted = self._next
        self._next += 1
        self.position += 1
        return drafted, ("step", drafted)

    def truncate(self, position):
        if position < 0 or position > self.position:
            raise RuntimeError(f"draft cache cannot rewind to {position}")
        self.position = position


class _ScriptedRunner(MemoryRunner):
    def __init__(self, samples):
        super().__init__()
        self.samples = list(samples)
        self.hidden = None
        self.draft = _ScriptedDraft()

    def forward(self, token_ids, cache_len, embedded=None, inv_freq=None):
        super().forward(token_ids, cache_len)
        self.hidden = [("row", cache_len + index) for index in range(len(token_ids))]

    def score_window(self):
        return self.samples


def _scripted_session(samples):
    from engine.forward.cycle import DraftCycle

    runner = _ScriptedRunner(samples)
    session = Session(runner)
    session.cycle = DraftCycle(runner, draft=runner.draft)
    return session


def test_a_runner_that_can_rewind_skips_the_replay_forward():
    class Rewinding(_ScriptedRunner):
        def __init__(self):
            super().__init__([10, 11, 99, 13, 14, 15, 50])
            self.rewinds = []

        def rewind_recurrent(self, keep):
            last_cache, last_ids = self.calls[-1]
            self.state = last_cache + keep
            self.rewinds.append(keep)

    session = Session(Rewinding())
    from engine.forward.cycle import DraftCycle

    session.cycle = DraftCycle(session.runner, draft=session.runner.draft)
    done = session.generate([1, 2, 3, 4], 3)
    assert done["tokens"] == [10, 11, 99]
    assert done["cached_tokens"] == 0
    assert session.runner.rewinds == [3]
    assert session.runner.calls == [
        (0, [1, 2, 3]),
        (3, [4, 10, 11, 12, 13, 14, 15]),
    ]
    assert session.runner.state == 6


def test_draft_decode_sees_the_whole_cache():
    import pytest
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or "RTX 5090" not in torch.cuda.get_device_name(0):
        pytest.skip("needs the visible RTX 5090")
    from engine.forward.draft import draft_attend

    length = 4
    query = torch.zeros(1, 1, 8, 64, device="cuda", dtype=torch.float16)
    key = torch.zeros(1, length, 2, 64, device="cuda", dtype=torch.float16)
    value = torch.zeros(1, length, 2, 64, device="cuda", dtype=torch.float16)
    value[:, :, :, 0] = torch.tensor([1, 2, 3, 4], device="cuda", dtype=torch.float16).view(1, length, 1)
    seen = draft_attend(query, key, value)
    assert abs(float(seen[0, 0, 0, 0]) - 2.5) < 1e-3


def test_worker_advertises_mtp6_and_refuses_promotion_without_the_matrix():
    session = _scripted_session([42, 0, 0, 0, 0, 0, 0])
    worker = Worker(session)
    ready = worker.ready()
    assert ready["config"]["linked"] is False
    assert ready["config"]["streams"] == 1
    assert ready["config"]["vision"] is True
    assert ready["config"]["draft_method"] == "mtp"
    assert ready["config"]["draft_tokens"] == 6
    assert ready["config"]["promotion_allowed"] is False
    assert worker.push('{"op":"promote","report":{}}') == [{"type": "promote", "allowed": False}]
    done = worker.push('{"op":"generate","id":"resp_a","request":{"ids":[1,2,3,4]}}')
    assert done[0]["status"] == "completed"
    assert done[0]["token_ids"] == [42]
    assert done[0]["usage"]["completion_tokens"] == 1
    assert done[0]["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
    assert session.runner.calls[0] == (0, [1, 2, 3])
    assert 4 not in session.runner.calls[0][1]
    follow = worker.push('{"op":"generate","id":"resp_b","request":{"ids":[1,2,3,4,5,6]}}')
    assert follow[0]["token_ids"] == [42]
    assert follow[0]["usage"]["prompt_tokens_details"]["cached_tokens"] == 4
    assert follow[0]["usage"]["completion_tokens"] == 1


def test_protocol_generate_emits_the_mtp6_window():
    samples = [10, 11, 12, 13, 14, 15, 77]
    session = _scripted_session(samples)
    worker = Worker(session)
    done = worker.push('{"op":"generate","id":"resp_full","request":{"ids":[1,2,3,4],"max_new":7}}')
    assert done[0]["token_ids"] == samples
    assert done[0]["usage"]["completion_tokens"] == 7
    assert session.cursor == 4 + 7 - 1
    assert session.tape == [1, 2, 3, 4, 10, 11, 12, 13, 14, 15]
    assert session.cycle.accepted == 6
    assert session.cycle.rejected == 0


def test_a_short_accept_keeps_the_cursor_on_the_returned_tokens():
    samples = [10, 11, 12, 13, 14, 15, 77]
    session = _scripted_session(samples)
    worker = Worker(session)
    done = worker.push('{"op":"generate","id":"resp_one","request":{"ids":[1,2,3,4],"max_new":1}}')
    assert done[0]["token_ids"] == [10]
    assert done[0]["usage"]["completion_tokens"] == 1
    assert session.cursor == 4
    assert session.tape == [1, 2, 3, 4]
    assert session.cycle.draft.position == session.cursor
    follow = worker.push('{"op":"generate","id":"resp_suffix","request":{"ids":[1,2,3,4,5,6]}}')
    assert follow[0]["status"] == "completed"
    assert follow[0]["usage"]["prompt_tokens_details"]["cached_tokens"] == 4
    print("OBSERVATION max_new=1 full accept leaves the cursor at 4 on the prompt and the later prefix continues with cached_tokens 4")

    fresh = _scripted_session(samples)
    Worker(fresh).push('{"op":"generate","id":"resp_hold","request":{"ids":[1,2,3,4],"max_new":1}}')
    fresh.runner.calls.clear()
    turned = fresh.turn([1, 2, 3, 4, 10, 20])
    assert turned["cached_tokens"] == 4
    assert turned["forwarded"][0][0] == 4
    assert fresh.runner.calls[0][0] == 4


def test_a_diverged_draft_does_not_stick_the_worker():
    from engine.forward.cycle import DraftCycle

    session = _scripted_session([42, 0, 0, 0, 0, 0, 0])
    session.cycle.draft.prefill = lambda token_ids, hidden: None
    worker = Worker(session)
    try:
        worker.push('{"op":"generate","id":"resp_bad","request":{"ids":[1,2,3,4]}}')
    except RuntimeError as error:
        assert "diverged" in str(error)
    else:
        raise AssertionError("a draft that does not advance must fail")
    assert worker.active is None
    assert session.busy is False
    session.cycle = DraftCycle(session.runner, draft=_ScriptedDraft())
    done = worker.push('{"op":"generate","id":"resp_next","request":{"ids":[1,2,3,4]}}')
    assert done[0]["status"] == "completed"
    assert done[0]["token_ids"] == [42]
    assert worker.active is None


def test_a_stop_token_ends_the_window_before_the_rest_is_repaired():
    from engine.forward.cycle import continue_drafted

    session = _scripted_session([10, 11, 12, 13, 14, 15, 77])
    tokens, cycle = continue_drafted(
        session.runner, [1, 2, 3, 4], 7, cycle=session.cycle, stop_ids={12},
    )
    assert tokens == [10, 11, 12]
    assert cycle.cache_len == 6
    assert cycle.draft.position == 6


def test_windows_record_their_phase_times():
    from engine.forward.cycle import continue_drafted

    session = _scripted_session([10, 11, 12, 13, 14, 15, 77])
    tokens, cycle = continue_drafted(
        session.runner, [1, 2, 3, 4], 7, cycle=session.cycle, stop_ids={12},
    )
    assert tokens == [10, 11, 12]
    assert cycle.windows == 1
    assert len(cycle.window_s) == 1
    assert all(delta >= 0.0 for delta in cycle.phase_s)
    assert abs(sum(cycle.phase_s) - cycle.window_s[0]) < 0.05


def test_generate_reports_the_turn_window_delta():
    session = _scripted_session([10, 11, 99, 13, 14, 15, 50])
    first = session.generate([1, 2, 3, 4], 3)
    assert first["status"] == "completed"
    assert first["windows"] == 1
    assert len(first["window_s"]) == 1
    assert all(delta >= 0.0 for delta in first["phase_s"])
    assert abs(sum(first["phase_s"]) - first["window_s"][0]) < 0.05
    assert first["host_prefill_s"] is None
    # The continuation shares the whole tape, so it reuses the cycle and the
    # reported delta is only this turn's windows.
    second = session.generate([1, 2, 3, 4, 10, 11, 99, 8, 8, 8], 3)
    assert second["status"] == "completed"
    assert second["windows"] == 3
    assert len(second["window_s"]) == 3
    assert abs(sum(second["phase_s"]) - sum(second["window_s"])) < 0.05
    assert session.cycle.windows == first["windows"] + second["windows"]


def test_chat_without_ids_returns_text_and_keeps_the_prefix():
    import json

    from qwasar_runtime.engine import FakeTokenizer

    class _Talk(FakeTokenizer):
        def decode_ids(self, ids):
            return "".join(chr(token) for token in ids if token < 128)

    end = _Talk.special["<|im_end|>"]
    close = [_Talk.special["</think>"], ord("\n"), ord("\n")]
    session = _scripted_session([end, 0, 0, 0, 0, 0, 0])
    worker = Worker(session, _Talk())
    done = worker.push(json.dumps({
        "op": "generate", "id": "resp_chat",
        "request": {"messages": [{"role": "user", "content": "hi"}], "thinking": "off", "max_tokens": 8},
    }))
    assert done[0]["type"] == "started"
    assert done[-1]["type"] == "terminal"
    assert any(event["type"] == "delta" for event in done)
    assert done[-1]["status"] == "completed"
    assert done[-1]["message"]["content"] == "\n\n"
    assert done[-1]["message"]["reasoning_content"] == ""
    assert done[-1]["token_ids"] == close + [end]
    assert done[-1]["snapshot"]["version"] == 1
    assert "xhigh" in done[-1]["snapshot"]["header"]
    assert session.cursor == len(done[-1]["snapshot"]["tape"])


def test_chat_metrics_carry_the_measured_window_phases():
    import json

    from qwasar_runtime.engine import FakeTokenizer

    class _Talk(FakeTokenizer):
        def decode_ids(self, ids):
            return "".join(chr(token) for token in ids if token < 128)

    end = _Talk.special["<|im_end|>"]
    session = _scripted_session([end, 0, 0, 0, 0, 0, 0])
    worker = Worker(session, _Talk())
    done = worker.push(json.dumps({
        "op": "generate", "id": "resp_phases",
        "request": {"messages": [{"role": "user", "content": "hi"}], "thinking": "off", "max_tokens": 8},
    }))
    metrics = done[-1]["metrics"]
    # Two windows: the first ends in the stop token and injects the reasoning
    # close, the second emits the stop again and ends the turn.
    assert metrics["windows"] == 2
    assert metrics["window_ms"] is not None
    for phase in ("draft_ms", "verify_ms", "sample_ms", "replay_ms"):
        assert metrics[phase] is not None
        assert metrics[phase] >= 0.0
    # The phase totals cover every window, so they reach at least the median.
    assert sum(metrics[phase] for phase in ("draft_ms", "verify_ms", "sample_ms", "replay_ms")) >= metrics["window_ms"]
    assert metrics["host_prefill_ms"] is None
    cursor = session.cursor
    follow_messages = done[-1]["snapshot"]["messages"] + [{"role": "user", "content": "next"}]
    follow = worker.push(json.dumps({
        "op": "generate", "id": "resp_next", "parent": done[-1]["snapshot"],
        "request": {"messages": follow_messages, "thinking": "off", "max_tokens": 1},
    }))
    assert follow[-1]["usage"]["prompt_tokens_details"]["cached_tokens"] == cursor


def test_a_decoded_tool_call_is_returned_as_a_function_call():
    import json

    from engine.forward.chat import finish_turn
    from qwasar_runtime.engine import FakeTokenizer

    class _Talk(FakeTokenizer):
        def decode_ids(self, ids):
            return "".join(chr(token) for token in ids if token < 128)

    xml = (
        "<tool_call>\n<function=read>\n<parameter=path>\na.txt\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    end = _Talk.special["<|im_end|>"]
    prepared = {
        "stop": end,
        "thinking": "off",
        "tools": [{"type": "function", "function": {"name": "read", "parameters": {
            "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}],
        "tool_choice": "auto",
    }
    turn = finish_turn(_Talk(), prepared, [ord(character) for character in xml] + [end], "resp_tool")
    assert turn["status"] == "completed"
    assert turn["message"]["content"] == ""
    assert turn["message"]["tool_calls"][0]["function"]["name"] == "read"
    assert json.loads(turn["message"]["tool_calls"][0]["function"]["arguments"]) == {"path": "a.txt"}
    assert turn["deltas"] == []


def test_chat_text_is_a_delta_and_raw_ids_stay_a_single_terminal():
    from qwasar_runtime.engine import FakeTokenizer

    class _Talk(FakeTokenizer):
        def decode_ids(self, ids):
            return "".join(chr(token) for token in ids if token < 128)

    session = _scripted_session([80, 0, 0, 0, 0, 0, 0])
    worker = Worker(session, _Talk())
    done = worker.push(
        '{"op":"generate","id":"resp_pong","request":{"messages":[{"role":"user","content":"hi"}],'
        '"thinking":"off","max_tokens":1}}'
    )
    assert done[0]["type"] == "started"
    assert done[1] == {"type": "delta", "id": "resp_pong", "channel": "reasoning", "text": "P"}
    assert done[-1]["message"]["reasoning_content"] == "P"
    assert done[-1]["message"]["content"] == "\n\n"
    assert done[-1]["status"] == "incomplete"
    assert done[-1]["metrics"]["incomplete_reason"] == "reasoning_budget"
    assert done[-1]["metrics"]["finish_reason"] == "max_new_tokens"
    assert done[-1]["snapshot"] is None
    raw = Worker(_scripted_session([42, 0, 0, 0, 0, 0, 0])).push(
        '{"op":"generate","id":"resp_raw","request":{"ids":[1,2,3,4]}}'
    )
    assert len(raw) == 1
    assert raw[0]["type"] == "terminal"
    assert raw[0]["token_ids"] == [42]
    assert raw[0]["message"]["content"] == ""
    assert raw[0]["message"]["reasoning_content"] == ""
    sha = "ab" * 32
    refused = worker.push(
        '{"op":"generate","id":"resp_img","request":{"messages":[{"role":"user","content":[{"type":"image","sha256":"'
        + sha + '","media_type":"image/png"}]}]}}'
    )
    assert refused[0]["error"]["message"] == f"image {sha} has no inline data"
    assert worker.active is None


def _bind(runner):
    from engine.forward.cycle import DraftCycle

    session = Session(runner)
    session.cycle = DraftCycle(runner, draft=runner.draft)
    return session


def _talk():
    from qwasar_runtime.engine import FakeTokenizer

    class _Mapped(FakeTokenizer):
        def decode_ids(self, ids):
            names = {value: key for key, value in self.special.items()}
            parts = []
            for token in ids:
                if token in names:
                    parts.append(names[token])
                elif 0 <= token < 128:
                    parts.append(chr(token))
            return "".join(parts)

    return _Mapped()


def _chat(worker, response_id, request, parent=None, emit=None):
    import json

    payload = {"op": "generate", "id": response_id, "request": request}
    if parent is not None:
        payload["parent"] = parent
    return worker.push(json.dumps(payload), emit=emit)


def test_messages_stay_high_when_the_client_disables_thinking():
    import pytest

    from engine.forward.chat import prepare_chat
    from qwasar_runtime.engine import reasoning_token_cap

    assert reasoning_token_cap("xhigh", 32768) == 8192
    talk = _talk()
    think = talk.special["<think>"]
    close = talk.special["</think>"]
    base = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 32768}
    for thinking in (None, "off", "low", "medium", "xhigh"):
        request = dict(base)
        if thinking is not None:
            request["thinking"] = thinking
        prepared = prepare_chat(talk, request, None, 262144)
        assert prepared["thinking"] == "xhigh"
        assert prepared["reasoning_cap"] == 8192
        assert "xhigh" in prepared["header"]
        assert prepared["tokens"][-2] == think
        assert prepared["tokens"][-1] == ord("\n")
        assert close not in prepared["tokens"]
    zero = prepare_chat(talk, {**base, "thinking": "off", "reasoning_budget_tokens": 0}, None, 262144)
    assert zero["thinking"] == "xhigh"
    assert zero["reasoning_cap"] == 0
    assert close not in zero["tokens"]
    assert prepare_chat(talk, {**base, "max_tokens": 10, "reasoning_budget_tokens": 2}, None, 262144)["reasoning_cap"] == 2
    with pytest.raises(ValueError):
        prepare_chat(talk, {**base, "thinking": "banana"}, None, 262144)
    with pytest.raises(ValueError):
        prepare_chat(talk, {**base, "temperature": True}, None, 262144)
    session = _scripted_session([ord("A"), 0, 0, 0, 0, 0, 0])
    done = _chat(Worker(session, talk), "resp_high", {
        "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4, "temperature": 0,
    })
    assert done[1]["channel"] == "reasoning"
    assert done[1]["text"] == "A"
    assert done[-1]["token_ids"][0] == ord("A")
    assert done[-1]["message"]["reasoning_content"].startswith("A")


def test_chat_deltas_are_emitted_before_the_next_token():
    import time

    class _Live(_ScriptedRunner):
        def __init__(self, live):
            super().__init__([ord("A"), 0, 0, 0, 0, 0, 0])
            self.live = live
            self.windows = 0

        def score_window(self):
            self.windows += 1
            if self.windows >= 2:
                assert any(event.get("type") == "delta" for event in self.live)
                time.sleep(0.002)
            return [ord("A"), 0, 0, 0, 0, 0, 0]

    live = []
    runner = _Live(live)
    done = _chat(Worker(_bind(runner), _talk()), "resp_stream", {
        "messages": [{"role": "user", "content": "hi"}], "thinking": "off", "max_tokens": 8,
    }, emit=live.append)
    assert [event["type"] for event in done] == ["terminal"]
    assert live[0]["type"] == "started"
    deltas = [event for event in live if event["type"] == "delta"]
    assert len(deltas) > 1
    assert deltas[0]["channel"] == "reasoning"
    terminal = done[0]
    for channel, field in (("reasoning", "reasoning_content"), ("content", "content")):
        joined = "".join(event["text"] for event in deltas if event["channel"] == channel)
        assert joined == terminal["message"][field]
        assert "<|im_end|>" not in joined
    metrics = terminal["metrics"]
    assert isinstance(metrics["ttft_ms"], float)
    assert isinstance(metrics["decode_tokens_per_second"], float)
    assert metrics["reasoning_tokens"] == 4
    assert metrics["finish_reason"] == "max_new_tokens"
    assert metrics["draft_acceptance"] == 0.0
    assert metrics["reasoning_closed"] == "budget"
    assert terminal["message"]["reasoning_content"] == "AAAA"
    assert terminal["message"]["content"] == "\n\nA"


def test_a_split_character_does_not_replay_reasoning_into_the_answer():
    talk = _talk()
    end = talk.special["<|im_end|>"]
    close = talk.special["</think>"]

    class _Wave(type(talk)):
        def decode_ids(self, ids):
            names = {value: key for key, value in self.special.items()}
            parts = []
            index = 0
            while index < len(ids):
                token = ids[index]
                if token == 1 and index + 1 < len(ids) and ids[index + 1] == 2:
                    parts.append("👋")
                    index += 2
                    continue
                if token == 1:
                    parts.append("\ufffd")
                elif token in names:
                    parts.append(names[token])
                elif 0 <= token < 128:
                    parts.append(chr(token))
                index += 1
            return "".join(parts)

    samples = [ord(character) for character in "Think"]
    samples += [close, ord("H"), ord("i"), 1, 2, ord("!"), end]

    class _Queue(_ScriptedRunner):
        def __init__(self):
            super().__init__([0, 0, 0, 0, 0, 0, 0])
            self.queue = list(samples)

        def score_window(self):
            first = self.queue.pop(0) if self.queue else end
            return [first, 0, 0, 0, 0, 0, 0]

    done = _chat(Worker(_bind(_Queue()), _Wave()), "resp_wave", {
        "messages": [{"role": "user", "content": "hi"}], "max_tokens": 32,
    })
    message = done[-1]["message"]
    assert message["reasoning_content"] == "Think"
    assert message["content"] == "Hi👋!"
    assert "</think>" not in message["content"]
    assert "\ufffd" not in message["content"]
    assert message["content"].count("Think") == 0
    deltas = [event for event in done if event["type"] == "delta"]
    for channel, field in (("reasoning", "reasoning_content"), ("content", "content")):
        assert "".join(event["text"] for event in deltas if event["channel"] == channel) == message[field]


def test_an_image_is_spliced_and_bad_images_are_rejected():
    import base64
    import hashlib
    import struct

    from engine.forward.vision import text_table_ids
    from qwasar_runtime.vision import MAX_IMAGE_BYTES

    assert text_table_ids([1, MM_TOKEN_BASE + 4, 2]) == [1, 0, 2]
    talk = _talk()
    end = talk.special["<|im_end|>"]
    think_end = talk.special["</think>"]
    raw = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", 32, 32)
    sha = hashlib.sha256(raw).hexdigest()
    image = {"media_type": "image/png", "data": base64.b64encode(raw).decode()}
    message = {"role": "user", "content": [
        {"type": "text", "text": "see"},
        {"type": "image", "sha256": sha, "media_type": "image/png"},
    ]}

    class _Vision(_ScriptedRunner):
        def __init__(self):
            super().__init__([0, 0, 0, 0, 0, 0, 0])
            self.queue = [think_end, ord("o"), ord("k"), end, end, end, end]
            self.seen = []

        def forward(self, token_ids, cache_len, embedded=None, inv_freq=None):
            self.seen.append((list(token_ids), embedded, inv_freq))
            super().forward(token_ids, cache_len, embedded, inv_freq)

        def score_window(self):
            first = self.queue.pop(0) if self.queue else end
            return [first, 0, 0, 0, 0, 0, 0]

    runner = _Vision()
    session = _bind(runner)
    worker = Worker(session, talk)
    done = _chat(worker, "resp_image", {"messages": [message], "images": {sha: image}, "max_tokens": 16})
    assert done[-1]["status"] == "completed"
    assert done[-1]["error"] is None
    assert done[-1]["message"]["content"] == "ok"
    assert "not available" not in str(done[-1])
    found = False
    for ids, embedded, inv in runner.seen:
        assert inv == "mrope"
        if not embedded:
            continue
        vision = [token for kind, token in embedded if kind == "vision"]
        text = [token for kind, token in embedded if kind == "text"]
        if not vision:
            continue
        found = True
        assert all(token >= MM_TOKEN_BASE for token in vision)
        assert all(token in ids for token in vision)
        assert all(token < MM_TOKEN_BASE for token in text)
    assert found
    plain = _chat(Worker(_scripted_session([end, 0, 0, 0, 0, 0, 0]), talk), "resp_plain", {
        "messages": [{"role": "user", "content": "see"}], "max_tokens": 8,
    })
    assert done[-1]["usage"]["prompt_tokens"] > plain[-1]["usage"]["prompt_tokens"]
    tape = done[-1]["snapshot"]["tape"]
    assert -1 in tape
    assert any(token >= MM_TOKEN_BASE for token in session.tape)
    assert canonical_tape(session.tape) == tape
    cursor = session.cursor
    follow_messages = done[-1]["snapshot"]["messages"] + [{"role": "user", "content": "next"}]
    follow = _chat(worker, "resp_follow", {
        "messages": follow_messages, "images": {sha: image}, "max_tokens": 8,
    }, parent=done[-1]["snapshot"])
    assert follow[-1]["usage"]["prompt_tokens_details"]["cached_tokens"] == cursor

    def rejected(request):
        failed = _chat(Worker(_scripted_session([end, 0, 0, 0, 0, 0, 0]), talk), "resp_bad", request)
        assert failed[0]["status"] == "failed"
        assert failed[0]["type"] == "terminal"
        return failed[0]["error"]["message"]

    bad_sha = "cd" * 32
    assert rejected({"messages": [{"role": "user", "content": [
        {"type": "image", "sha256": bad_sha, "media_type": "image/bmp"}]}]}) == "unsupported image media type"
    assert rejected({"messages": [message], "images": {sha: {"media_type": "image/png", "data": "@@@"}}}) == (
        "image data is not valid base64"
    )
    assert rejected({"messages": [{"role": "user", "content": [
        {"type": "image", "sha256": bad_sha, "media_type": "image/png"}]}],
        "images": {bad_sha: image}}) == f"image data does not match sha256 {bad_sha}"
    huge = "aa" * 32
    assert rejected({"messages": [{"role": "user", "content": [
        {"type": "image", "sha256": huge, "media_type": "image/png"}]}],
        "images": {huge: {"media_type": "image/png", "data": base64.b64encode(b"\x00" * (MAX_IMAGE_BYTES + 1)).decode()}},
    }) == f"image must contain 1..{MAX_IMAGE_BYTES} bytes"
    many = [f"{index:064x}" for index in range(17)]
    assert rejected({"messages": [{"role": "user", "content": [
        {"type": "image", "sha256": item, "media_type": "image/png"} for item in many]}]}) == (
        "at most 16 images per request"
    )
    assert rejected({"messages": [message]}) == f"image {sha} has no inline data"


def test_sampling_repeats_for_a_seed_and_can_diverge():
    class _Logits(_ScriptedRunner):
        def __init__(self, row):
            super().__init__([0, 0, 0, 0, 0, 0, 0])
            self.row = row

        def score_logits(self):
            return [list(self.row) for _ in range(7)]

    def row(left, right):
        values = [-1e9] * 82
        values[80] = left
        values[81] = right
        return values

    talk = _talk()

    def first_token(values, seed, temperature, top_p=1.0):
        done = _chat(Worker(_bind(_Logits(values)), talk), "resp_sample", {
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1, "temperature": temperature, "top_p": top_p, "seed": seed,
        })
        return done[-1]["token_ids"][0]

    peak = row(1.0, 0.0)
    assert first_token(peak, 1, 0) == 80
    assert first_token(peak, 2, 0) == 80
    flat = row(0.0, 0.0)
    assert first_token(flat, 7, 1) == first_token(flat, 7, 1)
    drawn = {first_token(flat, seed, 1) for seed in range(1, 25)}
    assert drawn <= {80, 81}
    assert len(drawn) >= 2
    assert {first_token(peak, seed, 1, top_p=0.5) for seed in range(1, 9)} == {80}


def test_tensor_sampling_matches_the_list_draw():
    import pytest
    torch = pytest.importorskip("torch")
    from engine.forward.sample import sample_id
    import random

    peak = [-1e9] * 82
    peak[80] = 1.0
    peak[81] = 0.0
    flat = [-1e9] * 82
    flat[80] = 0.0
    flat[81] = 0.0
    for values, temperature, top_p, seed in (
        (peak, 0, 1.0, 1),
        (peak, 1, 0.5, 3),
        (flat, 1, 1.0, 7),
        (flat, 1, 1.0, 8),
    ):
        listed = sample_id(values, temperature, top_p, random.Random(seed))
        tensor = sample_id(torch.tensor(values), temperature, top_p, random.Random(seed))
        assert tensor == listed


def test_a_tool_call_survives_the_resident_chat_turn():
    import json

    talk = _talk()
    end = talk.special["<|im_end|>"]
    xml = (
        "<tool_call>\n<function=read>\n<parameter=path>\na.txt\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    samples = [talk.special["</think>"]] + [ord(character) for character in xml] + [end]

    class _Queue(_ScriptedRunner):
        def __init__(self):
            super().__init__([0, 0, 0, 0, 0, 0, 0])
            self.queue = list(samples)

        def score_window(self):
            return [self.queue.pop(0), 0, 0, 0, 0, 0, 0]

    done = _chat(Worker(_bind(_Queue()), talk), "resp_tool", {
        "messages": [{"role": "user", "content": "read it"}],
        "tools": [{"type": "function", "function": {"name": "read", "parameters": {
            "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}],
        "max_tokens": 128,
    })
    assert done[-1]["status"] == "completed"
    call = done[-1]["message"]["tool_calls"][0]["function"]
    assert call["name"] == "read"
    assert json.loads(call["arguments"]) == {"path": "a.txt"}
    assert "<|im_end|>" not in done[-1]["message"]["content"]


def test_the_high_cap_closes_reasoning_and_the_answer_continues():
    talk = _talk()
    close = [talk.special["</think>"], ord("\n"), ord("\n")]

    def turn(max_tokens, **extra):
        done = _chat(Worker(_scripted_session([ord("Z"), 0, 0, 0, 0, 0, 0]), talk), "resp_cap", {
            "messages": [{"role": "user", "content": "hi"}], "thinking": "off", "max_tokens": max_tokens, **extra,
        })
        return done[-1]

    continued = turn(10)
    assert continued["status"] == "incomplete"
    assert continued["message"]["reasoning_content"] == "ZZZZZ"
    assert continued["message"]["content"] == "\n\nZZ"
    assert continued["metrics"]["reasoning_tokens"] == 5
    assert continued["metrics"]["reasoning_closed"] == "budget"
    assert continued["metrics"]["incomplete_reason"] == "max_new_tokens"
    assert continued["metrics"]["finish_reason"] == "max_new_tokens"
    assert continued["snapshot"] is None
    stopped = turn(6)
    assert stopped["message"]["reasoning_content"] == "ZZZ"
    assert stopped["message"]["content"] == "\n\n"
    assert stopped["token_ids"] == [ord("Z")] * 3 + close
    assert stopped["metrics"]["incomplete_reason"] == "reasoning_budget"
    assert stopped["metrics"]["finish_reason"] == "max_new_tokens"
    assert stopped["metrics"]["draft_acceptance"] == 0.0
    tight = turn(10, reasoning_budget_tokens=2)
    assert tight["message"]["reasoning_content"] == "ZZ"
    assert tight["metrics"]["reasoning_tokens"] == 2
    assert tight["message"]["content"] == "\n\nZZZZZ"
    open_cap = turn(4, reasoning_budget_tokens=0)
    assert open_cap["message"]["reasoning_content"] == "ZZZZ"
    assert open_cap["message"]["content"] == ""
    assert open_cap["metrics"]["incomplete_reason"] == "max_new_tokens"
    assert open_cap["metrics"]["reasoning_closed"] is None


def test_cancel_during_a_chat_turn_restores_the_commit():
    talk = _talk()
    end = talk.special["<|im_end|>"]
    samples = [end, end, ord("A"), ord("A"), ord("A"), ord("A"), ord("A")]

    class _Queue(_ScriptedRunner):
        def __init__(self):
            super().__init__([0, 0, 0, 0, 0, 0, 0])
            self.queue = list(samples)

        def score_window(self):
            first = self.queue.pop(0) if self.queue else end
            return [first, 0, 0, 0, 0, 0, 0]

    session = _bind(_Queue())
    worker = Worker(session, talk)
    first = _chat(worker, "resp_commit", {
        "messages": [{"role": "user", "content": "hi"}], "thinking": "off", "max_tokens": 8,
    })
    assert first[-1]["status"] == "completed"
    cursor = session.cursor
    tape = list(session.tape)
    follow_messages = first[-1]["snapshot"]["messages"] + [{"role": "user", "content": "next"}]
    request = {"messages": follow_messages, "thinking": "off", "max_tokens": 8}

    def emit(event):
        if event.get("type") == "delta":
            worker.push('{"op":"cancel","id":"resp_cancel"}')

    cancelled = _chat(worker, "resp_cancel", request, parent=first[-1]["snapshot"], emit=emit)
    assert [event["type"] for event in cancelled] == ["terminal"]
    assert cancelled[0]["status"] == "cancelled"
    assert cancelled[0]["snapshot"] is None
    assert cancelled[0]["message"]["reasoning_content"] == "A"
    assert cancelled[0]["usage"]["prompt_tokens_details"]["cached_tokens"] == cursor
    assert session.cursor == cursor
    assert session.tape == tape
    assert session.busy is False
    assert worker.active is None
    again = _chat(worker, "resp_after", request, parent=first[-1]["snapshot"])
    assert again[-1]["usage"]["prompt_tokens_details"]["cached_tokens"] == cursor


def test_a_second_chat_while_one_is_streaming_is_busy():
    import json

    talk = _talk()
    end = talk.special["<|im_end|>"]
    session = _scripted_session([end, 0, 0, 0, 0, 0, 0])
    worker = Worker(session, talk)
    held = {}

    def emit(event):
        if event["type"] == "started":
            held["active"] = worker.active
            held["second"] = worker.push(json.dumps({
                "op": "generate", "id": "resp_b",
                "request": {"messages": [{"role": "user", "content": "other"}], "max_tokens": 4},
            }))

    done = _chat(worker, "resp_a", {
        "messages": [{"role": "user", "content": "hi"}], "thinking": "off", "max_tokens": 8,
    }, emit=emit)
    assert held["active"] == "resp_a"
    assert held["second"][0]["error"]["code"] == "runtime_busy"
    assert held["second"][0]["error"]["http_status"] == 409
    assert done[-1]["status"] == "completed"
    assert worker.active is None


def test_archived_matrix_is_the_37_cells():
    from pathlib import Path
    import pytest
    path = Path("results/20260910-quality-gate/prompts-phase2.json")
    if not path.is_file():
        pytest.skip("archived matrix prompts are not in this tree")
    from engine.forward.matrix import load_archived
    rows = load_archived(path)
    assert len(rows) == 37
    assert all(isinstance(row["prompt_ids"], list) and row["prompt_ids"] for row in rows)


def test_resident_model_keeps_token_198_and_a_suffix_cursor():
    import pytest
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or "RTX 5090" not in torch.cuda.get_device_name(0):
        pytest.skip("needs the visible RTX 5090")
    from engine.forward.embed import prompt_ids
    from engine.forward.resident import ModelRunner
    from engine.forward.session import Session

    ids = prompt_ids(torch, 64).view(-1).tolist()
    session = Session(ModelRunner(pages=1))
    done = session.turn(ids)
    assert done["cursor"] == 64
    assert session.runner.predict() == 198
    session.runner.calls.clear()
    follow = session.turn(ids + [11, 12, 13, 14])
    assert follow["cached_tokens"] == 64
    assert follow["forwarded"][0][0] == 64
    assert session.runner.calls[0][0] == 64


def test_mtp6_window_matches_greedy_on_the_short_prompt():
    import pytest
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or "RTX 5090" not in torch.cuda.get_device_name(0):
        pytest.skip("needs the visible RTX 5090")
    from engine.forward.embed import prompt_ids
    from engine.forward.matrix import continue_greedy
    from engine.forward.resident import ModelRunner

    ids = prompt_ids(torch, 64).view(-1).tolist()
    runner = ModelRunner(pages=1, retain=True)
    reference = continue_greedy(runner, ids, stop_ids=set(), max_new=8)
    assert reference[0] == 198
    runner.reset()
    runner.calls.clear()
    session = Session(runner)
    done = Worker(session).push(
        '{"op":"generate","id":"resp_short","request":{"ids":%s,"max_new":8}}' % ids
    )
    assert done[0]["token_ids"] == reference
    assert done[0]["token_ids"][0] == 198
    print(
        f"OBSERVATION 64-id generate token_ids={done[0]['token_ids']} "
        "equal the one-token greedy reference and start with 198"
    )
    spans = suffix_bounds(0, len(ids))
    assert [(cache, len(span)) for cache, span in runner.calls[: len(spans)]] == [
        (start, end - start) for start, end in spans
    ]
    assert all(ids[-1] not in span for _cache, span in runner.calls[: len(spans)])
    cycle = session.cycle
    assert cycle.accepted + cycle.rejected > 0
    assert cycle.rate() == cycle.accepted / (cycle.accepted + cycle.rejected)


def test_mtp6_draft_matches_exllama_on_the_short_prompt():
    import gc
    import os
    import pytest
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or "RTX 5090" not in torch.cuda.get_device_name(0):
        pytest.skip("needs the visible RTX 5090")
    os.environ["QWASAR_HOT64K"] = "0"
    os.environ["QWASAR_RDZ"] = "0"
    os.environ["QWASAR_DRAFT_GRAPH"] = "0"
    from engine.forward.draft import MTPDraft
    from engine.forward.embed import prompt_ids
    from engine.forward.resident import ModelRunner
    from engine.forward.token import rms

    ids = prompt_ids(torch, 64).view(-1).tolist()
    runner = ModelRunner(pages=1)
    residual = runner.forward(ids[:63], 0)
    hidden = rms(runner._model.tensor("model.language_model.norm.weight"), residual)
    drafted = MTPDraft(runner._model, runner._table).propose(ids[:63], hidden, ids[63])
    del runner, residual, hidden
    gc.collect()
    torch.cuda.empty_cache()

    from exllamav3 import Job
    from exllamav3.generator.sampler.presets import GreedySampler
    from qwasar_runtime.engine import ExLlamaBackend

    backend = ExLlamaBackend(os.path.expanduser("~/models/Qwen3.8-27B-EXL3-5.0bpw"), 32768, prefill="xqa")
    generator = backend.generator
    captured = []
    original = generator.iterate_draftmodel_mtp_gen

    def wrap(results):
        drafted_ids = original(results)
        if drafted_ids is not None and not captured:
            captured.append(drafted_ids[0].detach().cpu().tolist())
        return drafted_ids

    generator.iterate_draftmodel_mtp_gen = wrap
    generator.enqueue(Job(
        input_ids=torch.tensor([ids], dtype=torch.long), max_new_tokens=1,
        sampler=GreedySampler(), seed=20260928, stop_conditions=[],
    ))
    while generator.num_remaining_jobs():
        generator.iterate()
    assert captured and captured[0] == drafted


def test_vision_tower_frames_dynamic_image_ids():
    import io
    import os
    import pytest
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or "RTX 5090" not in torch.cuda.get_device_name(0):
        pytest.skip("needs the visible RTX 5090")
    from PIL import Image
    from engine.forward.vision import MM_TOKEN_BASE, assemble_rows
    from engine.forward.vision_tower import VISION_END, VISION_START, embed_image, png_bytes

    image = Image.open(io.BytesIO(png_bytes(64, 64)))
    embedded = embed_image(os.path.expanduser("~/models/Qwen3.8-27B-EXL3-5.0bpw"), image)
    tokens = embedded["tokens"]
    assert tokens[0] == VISION_START and tokens[-1] == VISION_END
    assert embedded["dynamic_ids"]
    assert all(token >= MM_TOKEN_BASE for token in embedded["dynamic_ids"])
    seen = []
    assemble_rows(tokens, lambda token: seen.append(("text", token)), lambda token: seen.append(("image", token)))
    assert [item for item in seen if item[0] == "image"] == [("image", token) for token in embedded["dynamic_ids"]]
    assert seen[0] == ("text", VISION_START)
    assert seen[-1] == ("text", VISION_END)


def test_vision_rows_match_exllama_on_a_short_prompt():
    import gc
    import io
    import os
    import pytest
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or "RTX 5090" not in torch.cuda.get_device_name(0):
        pytest.skip("needs the visible RTX 5090")
    os.environ["QWASAR_HOT64K"] = "0"
    os.environ["QWASAR_RDZ"] = "0"
    os.environ["QWASAR_DRAFT_GRAPH"] = "0"
    from PIL import Image
    from exllamav3.tokenizer.mm_embedding import MMEmbedding
    from engine.forward.embed import prompt_ids
    from engine.forward.resident import ModelRunner
    from engine.forward.vision_tower import embed_image, mix_rows, mrope_freqs, png_bytes

    image = Image.open(io.BytesIO(png_bytes(64, 64)))
    embedded = embed_image(os.path.expanduser("~/models/Qwen3.8-27B-EXL3-5.0bpw"), image)
    text = prompt_ids(torch, 8).view(-1).tolist()
    ids = text[:4] + embedded["tokens"] + text[4:]
    if len(ids) == 63:
        ids.append(text[0])
    pages = len(ids) // 256 + 1
    runner = ModelRunner(pages=pages, retain=True)
    runner._ensure()
    residual = mix_rows(runner._table, ids, embedded["rows"], embedded["dynamic_ids"])
    freqs = mrope_freqs(
        ids, embedded["first_index"], embedded["last_index"],
        embedded["grid_thw"], embedded["merge_size"],
    )
    runner.forward(ids, 0, embedded=residual, inv_freq=freqs)
    token = runner.predict()
    rows = embedded["rows"].detach().cpu()
    frame = {
        "grid_thw": embedded["grid_thw"],
        "merge_size": embedded["merge_size"],
        "first_index": embedded["first_index"],
        "last_index": embedded["last_index"],
        "dynamic": len(embedded["dynamic_ids"]),
    }
    del runner, residual, freqs, embedded
    gc.collect()
    torch.cuda.empty_cache()

    from exllamav3 import Job
    from exllamav3.generator.sampler.presets import GreedySampler
    from qwasar_runtime.engine import ExLlamaBackend

    oracle = MMEmbedding(imp={
        "metadata": {},
        "full_length": len(ids),
        "mm_length": frame["dynamic"],
        "first_index": frame["first_index"],
        "last_index": frame["last_index"],
        "text_alias": "",
        "grid_thw": frame["grid_thw"],
        "mrope_merge_size": frame["merge_size"],
        "embeddings": rows.cuda().contiguous(),
        "deepstack_embeddings": None,
    })
    backend = ExLlamaBackend(os.path.expanduser("~/models/Qwen3.8-27B-EXL3-5.0bpw"), 32768, prefill="xqa")
    seen = []
    backend.generator.enqueue(Job(
        input_ids=torch.tensor([ids], dtype=torch.long),
        max_new_tokens=1,
        sampler=GreedySampler(),
        seed=20260928,
        stop_conditions=[],
        embeddings=[oracle],
    ))
    while backend.generator.num_remaining_jobs():
        for result in backend.generator.iterate():
            produced = result.get("token_ids")
            if produced is not None:
                seen.extend(int(item) for item in produced.view(-1).tolist())
    assert seen and seen[0] == token


def test_mtp6_window_matches_greedy_across_prefill_chunks():
    import pytest
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or "RTX 5090" not in torch.cuda.get_device_name(0):
        pytest.skip("needs the visible RTX 5090")
    from engine.forward.embed import prompt_ids
    from engine.forward.matrix import continue_greedy
    from engine.forward.resident import ModelRunner

    # 400 ids is the short two-chunk prompt whose greedy margin survives a 7-token verify.
    ids = prompt_ids(torch, 400).view(-1).tolist()
    runner = ModelRunner(pages=4, retain=True)
    reference = continue_greedy(runner, ids, stop_ids=set(), max_new=2)
    runner.reset()
    runner.calls.clear()
    session = Session(runner)
    done = Worker(session).push(
        '{"op":"generate","id":"resp_chunks","request":{"ids":%s,"max_new":2}}' % ids
    )
    spans = suffix_bounds(0, len(ids))
    assert [(cache, len(span)) for cache, span in runner.calls[: len(spans)]] == [
        (start, end - start) for start, end in spans
    ]
    assert spans[-1][1] == len(ids) - 1
    assert all(ids[-1] not in span for _cache, span in runner.calls[: len(spans)])
    assert done[0]["token_ids"] == reference
    print(f"OBSERVATION 400-id max_new=2 token_ids={done[0]['token_ids']} equal the greedy reference")
    assert session.cycle.accepted + session.cycle.rejected > 0


def test_session_turn_matches_the_vision_forward():
    import io
    import os
    import pytest
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or "RTX 5090" not in torch.cuda.get_device_name(0):
        pytest.skip("needs the visible RTX 5090")
    from PIL import Image
    from engine.forward.embed import prompt_ids
    from engine.forward.resident import ModelRunner
    from engine.forward.schedule import suffix_bounds
    from engine.forward.session import Session
    from engine.forward.vision import MM_TOKEN_BASE
    from engine.forward.vision_tower import embed_image, mix_rows, mrope_freqs, png_bytes

    image = Image.open(io.BytesIO(png_bytes(64, 64)))
    embedded = embed_image(os.path.expanduser("~/models/Qwen3.8-27B-EXL3-5.0bpw"), image)
    text = prompt_ids(torch, 8).view(-1).tolist()
    ids = text[:4] + embedded["tokens"] + text[4:]
    pages = len(ids) // 256 + 1
    runner = ModelRunner(pages=pages, retain=True)
    runner._ensure()
    rows = mix_rows(runner._table, ids, embedded["rows"], embedded["dynamic_ids"])
    freqs = mrope_freqs(
        ids, embedded["first_index"], embedded["last_index"],
        embedded["grid_thw"], embedded["merge_size"],
    )
    runner.forward(ids, 0, embedded=rows, inv_freq=freqs)
    token = runner.predict()
    assert token == 198
    print("OBSERVATION vision fixture predicts 198")
    runner.reset()
    runner.calls.clear()
    session = Session(runner)
    done = session.turn(ids, embedded=rows, inv_freq=freqs)
    expected = list(suffix_bounds(0, len(ids)))
    if expected[-1][1] < len(ids):
        expected.append((len(ids) - 1, len(ids)))
    assert done["forwarded"] == expected
    assert session.runner.predict() == token
    runner.calls.clear()
    follow = session.turn(ids + text[:3])
    assert follow["cached_tokens"] == len(ids)
    assert follow["forwarded"][0][0] == len(ids)
    assert all(item < MM_TOKEN_BASE for _cache, span in runner.calls for item in span)


def test_shipped_worker_entry_returns_token_ids():
    import json
    import os
    import subprocess
    import sys
    import time
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(root), str(root / "src")])}
    process = subprocess.Popen(
        [sys.executable, "-u", "-m", "engine.forward.worker", "--fake", "--model", "unused",
         "--prefill", "xqa", "--context-size", "1024"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env=env, cwd=root,
    )
    try:
        command = b""
        for _ in range(50):
            if process.poll() is not None:
                break
            try:
                command = Path(f"/proc/{process.pid}/cmdline").read_bytes().replace(b"\0", b" ")
            except FileNotFoundError:
                command = b""
            if b"engine.forward.worker" in command:
                break
            time.sleep(0.01)
        assert b"engine.forward.worker" in command, (command, process.stderr.read())
        assert b"qwasar_runtime.worker" not in command
        ready_line = process.stdout.readline()
        assert ready_line, process.stderr.read()
        ready = json.loads(ready_line)
        assert ready["type"] == "ready"
        assert ready["protocol"] == 1
        config = ready["config"]
        assert config["streams"] == 1
        assert config["vision"] is True
        assert config["draft_tokens"] == 6
        assert config["draft_method"] == "mtp"
        assert config["promotion_allowed"] is False
        assert config["engine"] == "q38"
        assert "engine_not_linked" not in ready_line
        process.stdin.write(json.dumps({
            "op": "generate", "id": "resp_one", "request": {"ids": [1, 2, 3, 4], "max_new": 1},
        }) + "\n")
        process.stdin.flush()
        done_line = process.stdout.readline()
        assert done_line, process.stderr.read()
        done = json.loads(done_line)
        assert done["status"] == "completed"
        assert done["token_ids"] == [42]
        assert done["usage"]["completion_tokens"] == 1
        assert done["error"] is None
        assert "engine_not_linked" not in done_line
        process.stdin.write(json.dumps({
            "op": "generate", "id": "resp_suffix", "request": {"ids": [1, 2, 3, 4, 5, 6]},
        }) + "\n")
        process.stdin.flush()
        follow = json.loads(process.stdout.readline())
        assert follow["status"] == "completed"
        assert follow["usage"]["prompt_tokens_details"]["cached_tokens"] == 4
        print("OBSERVATION shipped worker max_new=1 leaves the cursor on the returned token with cached_tokens 4")
        process.stdin.write(json.dumps({"op": "shutdown"}) + "\n")
        process.stdin.flush()
        assert process.wait(timeout=5) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_open_turn_rejects_a_nested_session_turn():
    session = Session(MemoryRunner())
    session.busy = True
    try:
        session.turn([1, 2])
    except SessionBusy as error:
        assert error.http_status == 409
        assert error.code == "runtime_busy"
    else:
        raise AssertionError("expected the resident session to reject the second turn")
    try:
        session.generate([1, 2, 3, 4], 1)
    except SessionBusy as error:
        assert error.http_status == 409
        assert error.code == "runtime_busy"
    else:
        raise AssertionError("expected the resident session to reject the nested generate")
