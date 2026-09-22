import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "results/20260917-attention-cubin-v2"))
from backend import BLOCK_N, HEAD_DIM, ROWS, SHARED_BYTES, STAGES


def test_v2_shared_fits_5090_and_holds_fp16_tile():
    expected = (
        ROWS * HEAD_DIM * 2
        + STAGES * BLOCK_N * 64 * 4
        + STAGES * BLOCK_N * 8 * 2
        + STAGES * BLOCK_N * 32 * 4
        + STAGES * BLOCK_N * 8 * 2
        + BLOCK_N * HEAD_DIM * 2
        + ROWS * BLOCK_N * 4
        + ROWS * BLOCK_N * 4
        + ROWS * HEAD_DIM * 4
        + ROWS * 4 * 2
        + STAGES * BLOCK_N * 4
    )
    assert STAGES == 1
    assert SHARED_BYTES == expected == 92544
    assert SHARED_BYTES <= 101376
