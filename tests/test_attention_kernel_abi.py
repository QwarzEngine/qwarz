import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "results/20260911-attention-kernel"))
from backend import INT32_ARG_INDEX, SPLIT_ARG_NAMES, pack_split_args


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "results/20260911-attention-kernel/contract.json"


def test_split_abi_has_fifteen_args_and_graph_patches():
    abi = json.loads(CONTRACT.read_text())["abi_reconfirmed"]
    assert len(abi["split_args"]) == 15
    assert abi["split_args"] == [
        "q", "k_cache", "v_cache", "block_table", "cache_seqlens", "out",
        "partial_o", "partial_ml", "k_scales", "v_scales", "h32",
        "split_len", "num_pages_per_seq", "num_splits", "sinks",
    ]
    assert abi["combine_args"] == [
        "partial_o", "partial_ml", "out", "h32", "num_splits", "sinks",
    ]
    assert abi["graph_patch_split"] == {
        "3": "block_table",
        "4": "cache_seqlens",
        "11": "split_len int32",
        "12": "num_pages_per_seq int32",
        "13": "num_splits int32",
    }
    assert abi["graph_patch_combine"] == {"4": "num_splits int32"}
    assert "two null pointer slots" in abi["wrapper"]


def test_pack_split_args_uses_int32_at_graph_patch_indices():
    packed = pack_split_args(
        {name: 0x1000 + i for i, name in enumerate(SPLIT_ARG_NAMES)},
        {"split_len": 1024, "num_pages_per_seq": 1008, "num_splits": 28},
    )
    assert [name for name, _, _ in packed] == list(SPLIT_ARG_NAMES)
    assert len(packed) == 17
    for index in INT32_ARG_INDEX:
        assert packed[index][1] == "int32"
    assert packed[11] == ("split_len", "int32", 1024)
    assert packed[12] == ("num_pages_per_seq", "int32", 1008)
    assert packed[13] == ("num_splits", "int32", 28)
    assert packed[15][0] == "global_scratch"
    assert packed[16][0] == "profile_scratch"


def test_q7_row_layout_matches_donor_attention64():
    geometry = json.loads(CONTRACT.read_text())["geometry"]["q7_layout"]
    assert geometry == {
        "BLOCK_M": 8, "BLOCK_H": 2, "BLOCK_ROWS": 16, "programs": 12, "BLOCK_N": 64,
    }
