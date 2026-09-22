import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "results/20260917-engine-topology"))
from superblock_sweep import query_len
from superblocks import (
    EXPECTED_PATTERN, assert_qwen38_topology, attn_kind, bc_bypassed, classify, decide,
)


class Attention:
    def __init__(self):
        self.bc_attn_step = lambda: "bc"


class GatedDeltaNet:
    def __init__(self):
        self.bc_split = True


class Other:
    pass


class TransformerBlock:
    def __init__(self, attn):
        self.attn = attn


def _model(pattern):
    kinds = {"Attention": Attention, "GatedDeltaNet": GatedDeltaNet}
    blocks = [TransformerBlock(kinds[name]()) for name in pattern]
    return SimpleNamespace(modules=blocks)


def test_query_len_reads_seq_on_token_ids_and_hidden():
    class Tensor:
        def __init__(self, shape):
            self.shape = shape
        def dim(self):
            return len(self.shape)

    assert query_len(Tensor((1, 7))) == 7
    assert query_len(Tensor((1, 7, 5120))) == 7
    assert query_len(Tensor((2, 4096))) == 4096


def test_classify_sixteen_qwen38_superblocks():
    pattern = list(EXPECTED_PATTERN) * 16
    groups = assert_qwen38_topology(classify(_model(pattern)))
    assert len(groups) == 16
    assert groups[0]["pattern"] == EXPECTED_PATTERN
    assert attn_kind(groups[0]["blocks"][3]) == "Attention"


def test_classify_rejects_drifted_pattern():
    pattern = ["Attention", "GatedDeltaNet", "GatedDeltaNet", "GatedDeltaNet"] * 16
    try:
        assert_qwen38_topology(classify(_model(pattern)))
    except ValueError as error:
        assert "pattern drifted" in str(error)
    else:
        raise AssertionError("expected pattern drift")


def test_bc_bypass_restores_attention_and_gdn():
    attn, gdn = Attention(), GatedDeltaNet()
    mlp = SimpleNamespace(forward="graph", _nvfp4_original_forward="eager")
    gdn_block = TransformerBlock(gdn)
    gdn_block.mlp = mlp
    blocks = [gdn_block, TransformerBlock(attn)]
    original = attn.bc_attn_step
    with bc_bypassed(blocks):
        assert attn.bc_attn_step(("x",), {}) is None
        assert gdn.bc_split is False
        assert mlp.forward == "eager"
    assert attn.bc_attn_step is original
    assert gdn.bc_split is True
    assert mlp.forward == "graph"


def test_decide_rejects_slower_eager_and_promotes_only_with_numeric():
    reject = decide(bc_ms=10.0, eager_ms=12.0, graph_ms=11.5, eager_vs_bc_l2=0.001, graph_vs_eager_l2=0.0)
    assert reject["promote"] is False
    assert "slower than BC" in reject["next"]

    win = decide(bc_ms=10.0, eager_ms=10.1, graph_ms=8.5, eager_vs_bc_l2=0.001, graph_vs_eager_l2=1e-7)
    assert win["promote"] is True
    assert abs(win["gain_vs_bc"] - 0.15) < 1e-12
