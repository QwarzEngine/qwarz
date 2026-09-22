import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "results/20260918-walker-overhead"))
from geometry import decide, summarize


def test_summarize_decode_median():
    rows = [
        {"kind": "prefill", "wall_ms": 80.0, "gpu_ms": 70.0, "host_ms": 10.0},
        {"kind": "decode", "wall_ms": 20.0, "gpu_ms": 18.0, "host_ms": 2.0,
         "draft_host_ms": 0.4, "target_host_ms": 1.4, "other_host_ms": 0.2},
        {"kind": "decode", "wall_ms": 22.0, "gpu_ms": 16.0, "host_ms": 6.0,
         "draft_host_ms": 0.5, "target_host_ms": 5.0, "other_host_ms": 0.5},
        {"kind": "decode", "wall_ms": 21.0, "gpu_ms": 17.0, "host_ms": 4.0,
         "draft_host_ms": 0.4, "target_host_ms": 3.2, "other_host_ms": 0.4},
    ]
    summary = summarize(rows)
    assert summary["n"] == 3
    assert summary["wall_ms"] == 21.0
    assert summary["host_ms"] == 4.0


def test_decide_rewrites_only_when_host_is_real():
    keep = decide({"wall_ms": 32.0, "gpu_ms": 30.5, "host_ms": 1.5, "host_frac": 1.5 / 32.0})
    assert keep["rewrite"] is False
    move = decide({
        "wall_ms": 36.0, "gpu_ms": 28.0, "host_ms": 8.0, "host_frac": 8.0 / 36.0,
        "draft_host_ms": 1.0, "target_host_ms": 6.5, "other_host_ms": 0.5,
    })
    assert move["rewrite"] is True
    assert "iterate_gen" in move["next"]
