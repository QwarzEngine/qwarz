"""The incremental streamer publishes exactly what a full re-decode would."""
import random
from pathlib import Path

import pytest

TOKENIZER = Path.home() / "models/Qwen3.8-27B-EXL3-5.0bpw/tokenizer.json"
SAMPLES = [
    "Hola, ¿cómo estás? Añadí un índice único — sin pérdida de precisión. 🚀✨ 日本語のテキストも。",
    '{"file_path":"/home/u/á/ñ.rs","content":"fn main() {\\n    println!(\\"héllo 😀\\");\\n}"}',
    "Let me think about the ring buffer:\n\n```python\nclass Ring:\n    def __init__(self, n):\n"
    "        self.buf = [None] * n  # ≈ O(1) ⚡\n```\n\n" * 12,
]


class _Tok:
    def __init__(self, backend):
        self.backend = backend

    def decode_ids(self, ids):
        return self.backend.decode(ids, skip_special_tokens=False)


def _full(tokenizer, windows):
    from engine.forward.chat import _stream_piece, decode_tokens

    visible, prev, out = [], "", []
    for window in windows:
        visible.extend(window)
        piece, prev = _stream_piece(prev, decode_tokens(tokenizer, visible), True)
        out.append(piece)
    piece, prev = _stream_piece(prev, decode_tokens(tokenizer, visible), False)
    out.append(piece)
    return out


def _incremental(tokenizer, windows):
    from engine.forward.chat import IncrementalText

    text = IncrementalText(tokenizer)
    out = [text.push(window) for window in windows]
    out.append(text.flush())
    return out


def test_incremental_text_matches_the_full_decode():
    tokenizers = pytest.importorskip("tokenizers")
    if not TOKENIZER.is_file():
        pytest.skip("artifact tokenizer is not on this machine")
    tokenizer = _Tok(tokenizers.Tokenizer.from_file(str(TOKENIZER)))
    rng = random.Random(7)
    for sample in SAMPLES:
        ids = tokenizer.backend.encode(sample, add_special_tokens=False).ids
        for _ in range(20):
            windows, index = [], 0
            while index < len(ids):
                step = rng.randint(1, 7)
                windows.append(ids[index:index + step])
                index += step
            assert _incremental(tokenizer, windows) == _full(tokenizer, windows)
            assert "".join(_incremental(tokenizer, windows)) == tokenizer.decode_ids(ids)


def _reference_row(values, temperature, top_p, top_k, rng):
    import torch

    top, indices = values.float().topk(top_k)
    weights = ((top - top[0]) / temperature).exp()
    cumulative = weights.cumsum(0) / weights.sum()
    # ExLlama's SS_TopP after SS_TopK: keep while cumulative <= top_p, first always.
    keep = max(1, int((cumulative <= top_p).sum()))
    kept = weights[:keep]
    draw = rng.random() * float(kept.sum())
    position = min(int(torch.searchsorted(kept.cumsum(0), torch.tensor(draw))), keep - 1)
    return int(indices[position])


def test_window_sampling_matches_the_row_reference_and_clamps_padding():
    torch = pytest.importorskip("torch")
    from engine.forward.sample import chooser, sample_window

    generator = torch.Generator().manual_seed(3)
    rows = torch.randn(7, 1000, generator=generator) * 4
    # Padding rows past the tokenizer must never be drawn, even when largest.
    rows[:, 990:] = 100.0
    for temperature, top_p in ((1.0, 0.95), (0.6, 0.8), (1.3, 1.0)):
        a, b = __import__("random").Random(11), __import__("random").Random(11)
        got = sample_window(rows, temperature, top_p, 20, a, vocab=990)
        want = [_reference_row(row[:990], temperature, top_p, 20, b) for row in rows]
        assert got == want
    assert sample_window(rows, 0, 0.95, 20, None, vocab=990) == rows[:, :990].argmax(-1).tolist()
    choose = chooser(1.0, 0.95, 5, top_k=20, vocab=990)
    assert callable(choose.window) and all(token < 990 for token in choose.window(rows))
    assert not hasattr(chooser(1.0, 0.95, 5), "window")


def test_speculative_sampling_keeps_the_target_distribution():
    torch = pytest.importorskip("torch")
    import random
    from collections import Counter
    from engine.forward.sample import _target_rows, speculative_window

    generator = torch.Generator().manual_seed(1)
    rows = torch.randn(3, 64, generator=generator) * 2
    p0 = _target_rows(rows, 1.0, 0.95, 20, None)[0]
    # A draft law that disagrees with the target on purpose.
    q_logits = torch.randn(64, generator=generator) * 2
    q_top, q_ids = q_logits.topk(20)
    q_w = (q_top - q_top[0]).exp()
    q = {int(t): float(w / q_w.sum()) for t, w in zip(q_ids, q_w)}
    rng, draft_rng = random.Random(5), random.Random(6)
    counts, trials = Counter(), 40000
    for _ in range(trials):
        tokens, weights = zip(*q.items())
        first = draft_rng.choices(tokens, weights)[0]
        second = draft_rng.choices(tokens, weights)[0]
        samples = speculative_window(rows, [first, second], [q, q], 1.0, 0.95, 20, rng)
        counts[samples[0]] += 1
    for token, prob in p0.items():
        assert abs(counts[token] / trials - prob) < 0.012, (token, counts[token] / trials, prob)
    assert sum(counts[t] for t in counts if t not in p0) == 0
