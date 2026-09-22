import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "results/20260918-xqa-fase3"))
from geometry import decide, numeric_ok


def test_numeric_ok_uses_max_l2():
    assert numeric_ok([{"relative_l2": 0.01}, {"relative_l2": 0.024}]) is True
    assert numeric_ok([{"relative_l2": 0.01}, {"relative_l2": 0.03}]) is False


def test_decide_promotes_only_with_numeric_and_gain():
    numeric = [{"relative_l2": 0.01}]
    keep = decide(numeric, [{"prefix": 32768, "wall_ms": 25.0}])
    assert keep["promote"] is False
    move = decide(numeric, [{"prefix": 32768, "wall_ms": 20.0}])
    assert move["promote"] is True
    assert "quality matrix" in move["next"]
    bad = decide([{"relative_l2": 0.04}], [{"prefix": 32768, "wall_ms": 18.0}])
    assert bad["promote"] is False
