import ast
from types import SimpleNamespace

import pytest

from qwasar_bench.attention_tuning import rewrite_attention, validate_policy, select_options
from qwasar_bench.attention_tuning import graph_attention_context


SOURCE = '''def configure(self, q_len):
    hd = 256
    block_m = 8
    block_n = max(16, 8192 // hd)
    block_h = max(16 // block_m, 1)
    programs = 12
    target = 340
    splits_cap = max(1, min(target // programs, 128))
    k_split = _compile_kernel(None, _paged_attn_decode_split_kernel, {}, {}, 4, 2)
    return block_n, block_h, splits_cap, k_split
'''


def test_graph_configuration_receives_tiles_splits_and_compile_options():
    policy = {"default": {"block_n": 64, "block_h": 4, "num_splits": 64,
                           "num_warps": 8, "num_stages": 1}, "queries": {"1": {}}}
    ns = {"_qwasar_options": lambda q: select_options(policy, q),
          "_compile_kernel": lambda *args: args[-2:], "_paged_attn_decode_split_kernel": object()}
    exec(rewrite_attention(SOURCE, "graph"), ns)
    assert ns["configure"](None, 7) == (64, 4, 64, (8, 1))
    assert ns["configure"](None, 1) == (32, 2, 28, (4, 2))


def test_empty_policy_keeps_all_original_defaults():
    ns = {"_qwasar_options": lambda q: {}, "_compile_kernel": lambda *args: args[-2:],
          "_paged_attn_decode_split_kernel": object()}
    exec(rewrite_attention(SOURCE, "graph"), ns)
    assert ns["configure"](None, 7) == (32, 2, 28, (4, 2))


def test_refuses_unknown_backend_source_instead_of_silent_noop():
    with pytest.raises(ValueError, match="expected"):
        rewrite_attention(SOURCE.replace("splits_cap =", "renamed_cap ="), "graph")


@pytest.mark.parametrize("policy", [
    {"default": {"block_n": 3}}, {"default": {"num_splits": True}},
    {"default": {"num_warps": 3}}, {"default": {"unknown": 1}},
    {"queries": {"17": {}}}, {"extra": {}}, {"default": {"num_splits": 0}},
])
def test_rejects_invalid_policy(policy):
    with pytest.raises(ValueError): validate_policy(policy)


def test_standalone_matches_graph_head_grouping_and_accepts_normal_kwargs():
    source = '''def decode(q, block_n=None, num_splits=None, num_warps=4, num_stages=2):
        q_len = q
        block_m = 8
        block_h = max(16 // block_m, 1)
        return block_h, block_n, num_splits, num_warps, num_stages
'''
    ns = {"_qwasar_options": lambda q: {"block_h": 4}}
    exec(rewrite_attention(source, "standalone"), ns)
    assert ns["decode"](7, block_n=64, num_splits=64, num_warps=8, num_stages=1) == (4, 64, 64, 8, 1)


def test_rejects_head_group_changes_until_native_launch_grid_is_supported():
    with pytest.raises(ValueError, match="native launch grid"):
        with graph_attention_context({"default": {"block_h": 4}}):
            pass
