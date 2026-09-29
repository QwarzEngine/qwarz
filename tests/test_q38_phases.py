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
