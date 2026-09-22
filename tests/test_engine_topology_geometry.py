import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "results/20260917-engine-topology"))
from geometry import ALLOWED_SPLITS, auto_splits, candidate_splits, decide, describe, row_geometry, waves


def test_q7_matches_donor_attention64_launch():
    geo = row_geometry(7)
    assert geo == {"block_m": 8, "block_h": 2, "programs": 12}


def test_q1_matches_donor_attention64_launch():
    geo = row_geometry(1)
    assert geo == {"block_m": 1, "block_h": 16, "programs": 4}


def test_donor_auto_on_5090_is_28_for_q7():
    assert auto_splits(170, 12, 258183) == 28


def test_candidates_include_one_wave_and_donor_auto():
    splits = candidate_splits(170, 12)
    assert 14 in splits and 15 in splits and 28 in splits
    assert set(ALLOWED_SPLITS) <= set(splits)


def test_current_28_splits_is_two_waves():
    info = describe(170, 7, 28)
    assert info["ctas"] == 336
    assert info["waves"] == 2
    assert info["is_donor_auto"] is True
    assert waves(12, 14, 170) == 1


def test_decide_promotes_only_allowed_10pct_with_stable_q1():
    def row(shape, splits, ratio, status="ok"):
        return {"shape": shape, "num_splits": splits, "ratio": ratio, "status": status,
                "timings": {"candidate": {"median_ms": 0.7 * ratio}, "baseline": {"median_ms": 0.7}}}

    rows = [row("q7-256k", 16, 0.88), row("q1-256k", 16, 1.01),
            row("q7-256k", 28, 1.00), row("q1-256k", 28, 1.00)]
    selection = decide(rows)
    assert selection["promote_policy"] is True
    assert selection["winner"]["num_splits"] == 16

    blocked = decide([row("q7-256k", 14, 0.85), row("q1-256k", 14, 1.00)])
    assert blocked["promote_policy"] is False
    assert blocked["extend_allowed"] is True

    reject = decide([row("q7-256k", 16, 0.97), row("q1-256k", 16, 1.00)])
    assert reject["promote_policy"] is False
    assert "superblock" in reject["next"]
