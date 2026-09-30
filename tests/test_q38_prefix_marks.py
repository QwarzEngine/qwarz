"""A new session that shares part of the resident tape restores a prefix mark."""
from engine.forward.cycle import DraftCycle
from engine.forward.session import Session


class _Draft:
    def __init__(self):
        self.position = 0

    def prefill(self, token_ids, _hidden):
        self.position += len(list(token_ids))

    def step(self, _token, _state):
        self.position += 1
        return 7, None

    def truncate(self, position):
        if position < 0 or position > self.position:
            raise RuntimeError("cannot rewind")
        self.position = position

    def reset(self):
        self.position = 0


class _Runner:
    """Integer recurrent state; a mark is the state value at its cut."""

    def __init__(self):
        self.state = 0
        self.calls = []
        self.hidden = None
        self.states = {"gdn": True}
        self.zeroed = 0
        self.resets = 0

    def forward(self, token_ids, cache_len, embedded=None, inv_freq=None):
        if cache_len != self.state:
            raise RuntimeError(f"forward at {cache_len} with cursor {self.state}")
        ids = list(token_ids)
        self.calls.append((cache_len, len(ids)))
        self.state += len(ids)
        self.hidden = [("row", cache_len + index) for index in range(len(ids))]

    def score_window(self):
        return [0] * 7

    def capture(self):
        return self.state

    def capture_host(self, into=None):
        return self.state

    def restore(self, snapshot):
        self.state = snapshot

    def rezero(self):
        self.state = 0
        self.zeroed += 1

    def reset(self):
        self.state = 0
        self.resets += 1


def _session():
    runner = _Runner()
    session = Session(runner)
    session.cycle = DraftCycle(runner, draft=_Draft())
    return session, runner


def test_a_shared_prefix_restores_the_longest_mark_and_forwards_only_the_suffix():
    session, runner = _session()
    first = list(range(1000, 1000 + 9000))
    session.generate(first, 2)
    assert set(session.marks) >= {8192}
    runner.calls.clear()
    # Same system prompt for 8500 tokens, then a different user message.
    second = first[:8500] + list(range(50_000, 50_300))
    done = session.generate(second, 2)
    assert done["cached_tokens"] == 8192
    assert runner.calls[0][0] == 8192
    assert runner.resets == 0 and runner.zeroed == 0
    assert all(position <= 8192 or position > 8500 for position in session.marks)


def test_an_identical_prompt_resumes_at_its_held_token():
    session, runner = _session()
    prompt = list(range(10, 10 + 700))
    session.generate(prompt, 3)
    runner.calls.clear()
    done = session.generate(prompt, 3)
    assert done["cached_tokens"] == len(prompt) - 1
    assert all(call[0] >= len(prompt) - 1 for call in runner.calls)


def test_no_shared_mark_zeroes_without_freeing_the_runner():
    session, runner = _session()
    session.generate(list(range(10, 800)), 2)
    done = session.generate(list(range(90_000, 90_400)), 2)
    assert done["cached_tokens"] == 0
    assert runner.zeroed == 1 and runner.resets == 0
    assert not any(position > 400 for position in session.marks)


def test_prefix_cuts_mark_the_system_block_and_the_last_message():
    from engine.forward.chat import IM_START, prefix_cuts
    from engine.forward.cycle import split_spans

    tokens = [IM_START] + [1] * 13000 + [IM_START] + [2] * 900 + [IM_START] + [3] * 40 + [IM_START, 9]
    cuts = prefix_cuts(tokens)
    assert cuts == [13001 // 256 * 256, 13902 // 256 * 256]
    spans = split_spans([(0, 8192), (8192, 13943)], cuts)
    assert spans == [(0, 8192), (8192, cuts[0]), (cuts[0], cuts[1]), (cuts[1], 13943)]


def test_a_new_session_restores_the_system_block_cut():
    session, runner = _session()
    system = list(range(1000, 1000 + 13000))
    first = system + list(range(60_000, 60_500))
    session.generate(first, 2, cuts=[12800])
    runner.calls.clear()
    second = system + list(range(70_000, 70_300))
    done = session.generate(second, 2, cuts=[12800])
    assert done["cached_tokens"] == 12800


def test_suffix_index_copies_what_followed_the_last_occurrence():
    from engine.forward.cycle import SuffixIndex

    history = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4]
    index = SuffixIndex(history, n=5)
    assert index.propose(5, 6) == [6, 7, 8, 9, 10, 11]
    assert index.propose(99, 6) is None
    index.extend([5, 6])
    assert index.propose(7, 3) == [8, 9, 10]
