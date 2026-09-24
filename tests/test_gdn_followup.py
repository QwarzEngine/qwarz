import json

import pytest

from qwasar_bench.gdn_followup import (ab_gate, bitexact_gate, cycle_gate,
                                       evaluate, load_cells)


def write_run(run_dir, cell, rep, arm, ids, div_at=None, ms=22.0, acc=.77,
              mismatches=0, completion="same"):
    ids = [list(row) for row in ids]
    if div_at is not None:
        for i in range(div_at, len(ids)):
            ids[i] = [9] * len(ids[0])
    label = f"{cell}-r{rep}-{arm}"
    (run_dir / f"{label}-draft-ids.json").write_text(json.dumps(ids))
    (run_dir / f"{label}-sample.json").write_text(json.dumps({
        "completion_sha256": completion,
        "ms_per_verify": ms, "decode_tokens_per_second": 240.0,
        "draft_acceptance": acc, "peak_allocated": 28 * 2**30,
        "draft_graph": {"validation": {"mismatches": mismatches}}}))


def bx_dir(tmp_path, diverge=False, mismatches=0):
    run_dir = tmp_path / "bx-shadow"
    run_dir.mkdir(parents=True)
    for rep in range(2):
        for arm in ("E", "G"):
            write_run(run_dir, "lru-32768-0", rep, arm, [[1, 2, 3]] * 40,
                      div_at=12 if diverge and arm == "G" and rep == 1 else None,
                      mismatches=mismatches)
    return load_cells(run_dir)


def test_bitexact_zero_tolerance(tmp_path):
    runs = bx_dir(tmp_path)
    assert bitexact_gate(runs, "lru-32768-0")["gate_pass"]
    runs = bx_dir(tmp_path / "d", diverge=True)
    gate = bitexact_gate(runs, "lru-32768-0")
    assert not gate["gate_pass"] and gate["pairs"][1]["first_divergence"] == 12
    runs = bx_dir(tmp_path / "m", mismatches=2)
    assert not bitexact_gate(runs, "lru-32768-0")["gate_pass"]


def test_cycle_gate_band(tmp_path):
    runs = bx_dir(tmp_path)
    gate = cycle_gate(runs, "lru-32768-0", 0.4, 1.5)
    assert gate["median_ms"] == 0.0 and not gate["gate_pass"]
    assert cycle_gate(runs, "lru-32768-0", -0.1, 0.1)["gate_pass"]


def test_ab_gate(tmp_path):
    shadow = tmp_path / "s"
    venv = tmp_path / "v"
    shadow.mkdir()
    venv.mkdir()
    write_run(shadow, "lru-32768-0", 0, "P", [[1]] * 5, acc=.770, ms=22.2)
    write_run(venv, "lru-32768-0", 0, "P", [[1]] * 5, acc=.765, ms=22.8)
    write_run(venv, "lru-32768-0", 1, "P", [[1]] * 5, acc=.782, ms=23.1)
    gate = ab_gate(load_cells(shadow), load_cells(venv), "lru-32768-0", .02, .01)
    assert gate["acceptance_pass"] and gate["cycle_pass"]
    # Shadow acceptance far outside the venv band fails.
    write_run(shadow, "lru-32768-0", 0, "P", [[1]] * 5, acc=.70, ms=22.2)
    gate = ab_gate(load_cells(shadow), load_cells(venv), "lru-32768-0", .02, .01)
    assert not gate["acceptance_pass"]


def test_load_cells_fail_closed(tmp_path):
    with pytest.raises(ValueError):
        load_cells(tmp_path / "empty")
    d = tmp_path / "d"
    d.mkdir()
    write_run(d, "c", 0, "P", [[1]] * 3)
    (d / "c-r0-P-draft-ids.json").unlink()  # missing ids log must fail
    with pytest.raises(ValueError):
        load_cells(d)


def test_load_cells_global_rep_counter(tmp_path):
    """The runner increments reps across cells; grouping must not require
    per-cell contiguity."""
    d = tmp_path / "d"
    d.mkdir()
    write_run(d, "warmup", 0, "E", [[1]] * 3)
    write_run(d, "warmup", 1, "G", [[1]] * 3)
    write_run(d, "lru-32768-0", 2, "E", [[1]] * 3)
    write_run(d, "lru-32768-0", 3, "G", [[1]] * 3)
    runs = load_cells(d)
    assert [r["rep"] for r in runs[("lru-32768-0", "E")]] == [2]


def test_evaluate_requires_frozen_prediction(tmp_path):
    (tmp_path / "prediction.json").write_text(json.dumps({"gates": {}}))
    with pytest.raises(ValueError):
        evaluate(tmp_path)
