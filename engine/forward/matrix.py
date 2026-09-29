"""The 37-cell matrix the cutover gate reads.

Prompt ids live in the archived phase-2 file. A cell is eligible only when
its label is one of those 37. This module does not start the service and
does not mark the gate passed.
"""
from __future__ import annotations

import json
from pathlib import Path

from engine.forward.gate import cells

ARCHIVED = Path(__file__).resolve().parents[2] / "results/20260910-quality-gate/prompts-phase2.json"


def load_archived(path=ARCHIVED):
    rows = json.loads(Path(path).read_text())
    labels = [row["label"] for row in rows]
    if set(labels) != set(cells()) or len(labels) != len(cells()):
        raise ValueError("archived prompts are not the 37-cell matrix")
    return rows


def cell(label, path=ARCHIVED):
    for row in load_archived(path):
        if row["label"] == label:
            return row
    raise KeyError(label)


def continue_greedy(runner, prompt, stop_ids, max_new):
    """Score the held prompt token, then append greedy ids until stop or the cap."""
    from engine.forward.session import Session

    session = Session(runner)
    session.turn(prompt)
    produced = []
    for _ in range(max_new):
        token = runner.predict()
        if token in stop_ids:
            break
        produced.append(token)
        session.turn(session.tape + [token])
    return produced
