import hashlib
import json
from pathlib import Path

from qwasar_runtime import hybrid


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "results/20260911-attention-kernel/contract.json"


def test_task0_contract_matches_installed_donor_and_flash_policy():
    contract = json.loads(CONTRACT.read_text())
    assert contract["decode_policy"] == hybrid.DECODE_POLICY
    assert hybrid.DECODE_POLICY == {"default": {"block_n": 64, "num_warps": 4, "num_stages": 1}}
    assert contract["quantization_identity"] == hybrid.QUANTIZATION
    for source in contract["sources"]:
        path = Path(source["path"])
        assert path.is_file(), path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"]
        assert source["match_20260908"] is True
    abi = contract["abi_reconfirmed"]
    assert abi["split_args"][3:5] == ["block_table", "cache_seqlens"]
    assert abi["split_args"][11:14] == ["split_len", "num_pages_per_seq", "num_splits"]
    assert abi["graph_patch_split"]["13"] == "num_splits int32"
    assert abi["graph_patch_combine"]["4"] == "num_splits int32"
