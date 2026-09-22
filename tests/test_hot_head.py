"""CPU-only contract tests for the promoted 65536-token MTP proposer head."""
from pathlib import Path

from qwasar_runtime import hot_head


def test_enabled_respects_kill_switch():
    assert hot_head.enabled({}) is True
    assert hot_head.enabled({"QWASAR_HOT64K": "1"}) is True
    assert hot_head.enabled({"QWASAR_HOT64K": "0"}) is False


def test_map_is_pinned_and_structurally_valid():
    blocks = hot_head.load_blocks()
    assert len(blocks) == 4096
    hot_head.validate_blocks(blocks)
    # 512 complete, aligned 128-token groups = 65536 proposable tokens
    groups = sorted({block // 8 for block in blocks})
    assert len(groups) == 512
    assert hot_head.SUBSET_VOCAB == 65536


def test_map_hash_is_pinned():
    import hashlib

    digest = hashlib.sha256(hot_head.MAP_PATH.read_bytes()).hexdigest()
    assert digest == hot_head.MAP_SHA256
    assert hot_head.MAP_PATH == Path(hot_head.__file__).with_name("hot_blocks_64k.txt")


def test_validate_blocks_rejects_malformed_maps():
    for bad in ([], [0], list(range(1, 9)), list(range(7)) + [8],
                list(range(8))[::-1], [8 * 15519 + i for i in range(8)]):
        try:
            hot_head.validate_blocks(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted malformed block map: {bad}")
