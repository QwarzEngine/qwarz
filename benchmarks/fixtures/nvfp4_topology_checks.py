"""Frozen oracle for lexicographically minimal topological sorting."""


def check_lru(cls):
    item = cls()
    assert item.order() == []
    for name in ("z", "a", "m"):
        item.add_node(name)
    assert item.order() == ["a", "m", "z"]
    item.add_edge("z", "a")
    assert item.order() == ["m", "z", "a"]
    item.add_edge("m", "z")
    item.add_edge("m", "z")
    assert item.order() == ["m", "z", "a"]
    item.add_edge("a", "b")
    assert item.order() == ["m", "z", "a", "b"]
    first = item.order()
    first.clear()
    assert item.order() == ["m", "z", "a", "b"]
    item.add_edge("b", "m")
    for _ in range(2):
        try:
            item.order()
        except ValueError:
            pass
        else:
            raise AssertionError("cycle not detected")
    other = cls()
    other.add_edge("x", "x")
    try:
        other.order()
    except ValueError:
        pass
    else:
        raise AssertionError("self cycle not detected")
    return 7
