"""Frozen oracle for a new half-open interval task; RUNNER uses check_lru."""


def check_lru(cls):
    item = cls()
    assert item.intervals() == [] and not item.contains(0)
    item.add(1, 4)
    item.add(4, 8)
    item.add(-3, -1)
    assert item.intervals() == [(-3, -1), (1, 8)]
    assert item.contains(1) and item.contains(7) and not item.contains(8)
    item.remove(2, 6)
    assert item.intervals() == [(-3, -1), (1, 2), (6, 8)]
    item.remove(-9, 2)
    assert item.intervals() == [(6, 8)]
    item.add(7, 10)
    snapshot = item.intervals()
    snapshot.clear()
    assert item.intervals() == [(6, 10)]
    for method in (item.add, item.remove):
        for pair in ((3, 3), (4, 2), (True, 3), (1.0, 3)):
            try:
                method(*pair)
            except (ValueError, TypeError):
                pass
            else:
                raise AssertionError(f"accepted invalid interval: {pair}")
    assert item.intervals() == [(6, 10)]
    return 7
