"""Cold MTP6 generation against the pre-port service floors.

Temperature 0 is the shipped argmax (`choose` omitted). Thinking stays off
because the prompt is raw ids, not the chat template. Each length is warmed
before its timed completion: the baseline was a warm service, and a cold
`cached_tokens` of 0 is a fresh cursor, not a first-ever compile.
"""
import json
from pathlib import Path

import pytest

TEXT = "The river was low that summer, and the stones showed all the way to the far bank. "
LENGTHS = (261888, 256, 4096, 32768)
MAX_NEW = 64
SEED = 20260928


def _floors():
    rows = json.loads(Path("engine/oracle/service-baseline.json").read_text())["rows"]
    return {row["prompt_tokens"]: row for row in rows}


def _prose(tokenizer, length):
    piece = tokenizer.encode(TEXT, add_bos=False, add_eos=False).view(-1).tolist()
    ids = []
    while len(ids) < length:
        ids.extend(piece)
    return ids[:length]


def _require_5090():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or "RTX 5090" not in torch.cuda.get_device_name(0):
        pytest.skip("needs the visible RTX 5090")
    return torch


def _release(runner, torch):
    import gc

    del runner
    gc.collect()
    torch.cuda.empty_cache()


def _timed(runner, ids, row):
    import time

    import torch

    state = {"first": None, "last": None, "first_batch": 0, "emitted": 0, "seen": 0}
    torch.cuda.synchronize()
    started = time.perf_counter()

    def on_accepted(produced):
        new = produced[state["seen"]:]
        if not new:
            return None
        now = time.perf_counter()
        if state["first"] is None:
            state["first"] = now
            state["first_batch"] = len(new)
        state["emitted"] += len(new)
        state["last"] = now
        state["seen"] = len(produced)
        return None

    from engine.forward.session import Session

    done = Session(runner).generate(ids, MAX_NEW, on_accepted=on_accepted)
    decode = (state["emitted"] - state["first_batch"]) / (state["last"] - state["first"])
    prefill = (len(ids) - 1) / done["host_prefill_s"]
    ttft_ms = (state["first"] - started) * 1000
    decode_floor = 0.95 * row["decode_tokens_per_second"]
    ttft_cap = 1.05 * row["ttft_ms"]
    prefill_floor = 0.95 * row["prefill_tokens_per_second"]
    print(
        f"n={len(ids)} decode={decode:.2f} floor={decode_floor:.2f} "
        f"ttft_ms={ttft_ms:.2f} cap={ttft_cap:.2f} "
        f"prefill={prefill:.1f} floor={prefill_floor:.1f} "
        f"prefill_s={done['host_prefill_s']:.3f} "
        f"acc={done['accepted']}/{done['rejected']} "
        f"first_batch={state['first_batch']} emitted={state['emitted']} "
        f"cached={done['cached_tokens']} status={done['status']} "
        f"tokens={len(done['tokens'])}",
        flush=True,
    )
    assert done["status"] == "completed"
    assert done["cached_tokens"] == 0
    assert len(done["tokens"]) == MAX_NEW
    assert state["emitted"] == MAX_NEW
    assert 0 < state["first_batch"] < MAX_NEW
    assert decode >= decode_floor
    assert ttft_ms <= ttft_cap
    assert prefill >= prefill_floor
    return done


def test_cold_generate_clears_the_service_floors():
    torch = _require_5090()
    from engine.forward.resident import ModelRunner
    from engine.forward.session import Session
    from engine.forward.worker import _load_tokenizer

    floors = _floors()
    name = torch.cuda.get_device_name(0)
    print(f"device {name}", flush=True)
    assert "RTX 5090" in name
    torch.manual_seed(SEED)
    runner = ModelRunner(pages=1024, retain=True)
    try:
        tokenizer = _load_tokenizer(str(runner.model_dir))
        prompts = {length: _prose(tokenizer, length) for length in LENGTHS}
        for length in LENGTHS:
            ids = prompts[length]
            warms = 2 if length == 4096 else 1
            for warm in range(warms):
                runner.rezero()
                torch.cuda.synchronize()
                print(f"warm{length}.{warm}", flush=True)
                Session(runner).generate(ids, 16)
            runner.rezero()
            torch.cuda.synchronize()
            _timed(runner, ids, floors[length])
            if length == 261888:
                for _ in range(3):
                    runner.rezero()
                    runner.forward(prompts[256][:255], 0)
                torch.cuda.synchronize()
    finally:
        _release(runner, torch)


def test_greedy_tokens_match_the_frozen_forward():
    torch = _require_5090()
    from engine.forward.embed import prompt_ids
    from engine.forward.resident import ModelRunner
    from engine.forward.session import Session
    from engine.forward.token import GREEDY

    torch.manual_seed(SEED)
    runner = ModelRunner(pages=1024, retain=True)
    try:
        for length, want in GREEDY.items():
            runner.rezero()
            torch.cuda.synchronize()
            ids = prompt_ids(torch, length).view(-1).tolist()
            done = Session(runner).generate(ids, 1)
            token = done["tokens"][0]
            print(
                f"greedy n={length} token={token} want={want} cached={done['cached_tokens']}",
                flush=True,
            )
            assert done["cached_tokens"] == 0
            assert done["status"] == "completed"
            assert token == want
    finally:
        _release(runner, torch)
