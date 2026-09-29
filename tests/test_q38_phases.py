"""Resident session, MTP6, vision ids, and the 37-cell cutover lock."""
from engine.forward.gate import GATES, cells, promotion_allowed
from engine.forward.mtp import DRAFT_TOKENS, acceptance, emitted_tokens, verify_window
from engine.forward.schedule import suffix_bounds
from engine.forward.session import MemoryRunner, Session, SessionBusy
from engine.forward.vision import MM_TOKEN_BASE, assemble_rows, canonical_tape, embed_spans
from engine.forward.worker import Worker


def test_cold_prefill_matches_the_measured_cuts():
    assert [end - start for start, end in suffix_bounds(0, 64)] == [63]
    assert [end - start for start, end in suffix_bounds(0, 32768)] == [8192, 8192, 8192, 7936, 255]
    long = [end - start for start, end in suffix_bounds(0, 261888)]
    assert long == [8192] * 31 + [7680, 255]
    assert suffix_bounds(8192, 32768) == suffix_bounds(0, 32768)[1:]


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


def test_worker_advertises_mtp6_and_refuses_promotion_without_the_matrix():
    worker = Worker(Session(MemoryRunner()))
    ready = worker.ready()
    assert ready["config"]["linked"] is False
    assert ready["config"]["draft_tokens"] == 6
    assert ready["config"]["promotion_allowed"] is False
    assert worker.push('{"op":"promote","report":{}}') == [{"type": "promote", "allowed": False}]
    done = worker.push('{"op":"generate","id":"resp_a","request":{"ids":[1,2,3,4]}}')
    assert done[0]["status"] == "completed"
    assert done[0]["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
    follow = worker.push('{"op":"generate","id":"resp_b","request":{"ids":[1,2,3,4,5,6]}}')
    assert follow[0]["usage"]["prompt_tokens_details"]["cached_tokens"] == 4


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
    from engine.forward.cycle import continue_drafted
    from engine.forward.embed import prompt_ids
    from engine.forward.matrix import continue_greedy
    from engine.forward.resident import ModelRunner

    ids = prompt_ids(torch, 64).view(-1).tolist()
    runner = ModelRunner(pages=1, retain=True)
    reference = continue_greedy(runner, ids, stop_ids=set(), max_new=8)
    assert reference[0] == 198
    runner.reset()
    drafted, cycle = continue_drafted(runner, ids, 8)
    assert drafted == reference
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
