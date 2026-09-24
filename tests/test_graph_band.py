import json

import pytest

from qwasar_bench.graph_band import band, evaluate, first_divergence, gain


def make_runs(tmp_path, arms, diverge_at=None, tokens=6):
    """Synthetic runner output: diverge_at maps (occurrence, arm) -> verify
    index where its ids start differing (models the eager band)."""
    seen = {}
    for rep, arm in enumerate(arms):
        occurrence = seen.get(arm, 0)
        seen[arm] = occurrence + 1
        div = None if diverge_at is None else diverge_at.get((occurrence, arm))
        ids = [[1, 2, 3, 4, 5, 6] for _ in range(20)]
        if div is not None:
            for i in range(div, 20):
                ids[i] = [9, 9, 9, 9, 9, 9]
        label = f"cell-r{rep}-{arm}"
        (tmp_path / f"{label}-draft-ids.json").write_text(json.dumps(ids))
        (tmp_path / f"{label}-sample.json").write_text(json.dumps({
            # Upstream quirk: graph_runner overwrites sample["label"] with the
            # cell label; the run label only survives in the filename.
            "label": "cell", "completion_sha256": "same" if div is None else f"div{div}",
            "ms_per_verify": 22.0 if arm != "G" else 21.2,
            "decode_tokens_per_second": 170.0 if arm != "G" else 176.0,
            "draft_acceptance": .4, "peak_allocated": 28 * 2**30}))
    from qwasar_bench.graph_band import load_run
    return load_run(tmp_path)


def test_first_divergence_and_length_mismatch():
    a = [[1]] * 10
    b = [[1]] * 4 + [[2]] + [[1]] * 5
    assert first_divergence(a, b) == 4
    assert first_divergence(a, a) == float("inf")
    # Identical prefix, different lengths: divergence at the shorter end.
    assert first_divergence(a, a[:7]) == 7


def test_band_gate_within_eager_band_passes(tmp_path):
    diverge = {(r, "E"): 4 + (r % 2) for r in range(6)}
    diverge.update({(r, "G"): 4 for r in range(6)})
    runs = make_runs(tmp_path, list("EGEGEGEGEGEG"), diverge)
    result = band(runs)
    # Same-div pairs are identical to each other: 9/15 E-E and 18/36 G-E diverge.
    assert result["ee"]["diverged"] == 9 and result["ge"]["diverged"] == 18
    assert result["ee"]["first_divergence_median"] == 4.0
    assert result["gate_pass"]


def test_band_gate_fails_when_graph_diverges_earlier(tmp_path):
    diverge = {(r, "E"): 8 for r in range(6)}
    diverge.update({(r, "G"): 1 for r in range(6)})
    runs = make_runs(tmp_path, list("EGEGEGEGEGEG"), diverge)
    assert not band(runs)["gate_pass"]


def test_empty_eager_band_requires_perfect_graph(tmp_path):
    runs = make_runs(tmp_path, list("EGEGEGEGEGEG"), None)
    assert band(runs)["gate_pass"]
    runs = make_runs(tmp_path, list("EGEGEGEGEGEG"), {(0, "G"): 7})
    assert not band(runs)["gate_pass"]


def test_band_fail_closed_on_missing_runs(tmp_path):
    runs = make_runs(tmp_path, list("EGEG"))
    with pytest.raises(ValueError, match="expected 6 E and 6 G"):
        band(runs)


def test_gain_cycle_net_and_missing_arms(tmp_path):
    runs = make_runs(tmp_path, list("PGEEGPPGEEGP"))
    result = gain(runs)
    assert result["cycle_ms"] == pytest.approx(0.8)
    assert result["net_tok_s_median"] > 0
    assert result["mem_delta_gib"] == pytest.approx(0.0)
    small = tmp_path / "small"
    small.mkdir()
    runs = make_runs(small, list("PGEE"))
    with pytest.raises(ValueError, match="expected 4"):
        gain(runs)


def test_evaluate_requires_frozen_prediction(tmp_path):
    (tmp_path / "prediction.json").write_text(json.dumps({"frozen_before_measurement": False}))
    with pytest.raises(ValueError, match="not frozen"):
        evaluate(tmp_path)
