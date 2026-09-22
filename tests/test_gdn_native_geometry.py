import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "results/20260917-gdn-native"))
from geometry import GROUP, HEAD_DIM, K_HEADS, SMEM_BYTES, V_HEADS, decide, describe


def test_qwen38_gdn_shape():
    assert (K_HEADS, V_HEADS, HEAD_DIM, GROUP) == (16, 48, 128, 3)
    assert describe()["cta"] == 48
    assert SMEM_BYTES == 67584
    assert SMEM_BYTES <= 101376


def test_decide_requires_numeric_and_fifteen_percent():
    miss = decide(0.20, 0.16, 0.01)
    assert miss["promote_recurrent"] is False
    win = decide(0.20, 0.16, 0.0004)
    assert win["promote_recurrent"] is True
    assert abs(win["gain"] - 0.20) < 1e-12
