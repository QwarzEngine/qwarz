"""Frozen oracle for a bijective map; failed collisions must be atomic."""


def check_lru(cls):
    item = cls()
    assert len(item) == 0
    item.put("a", 1)
    item.put("b", 2)
    assert item.get("a") == 1 and item.inverse(2) == "b"
    item.put("a", 3)
    assert len(item) == 2 and item.inverse(3) == "a"
    for lookup, key in ((item.inverse, 1), (item.get, "missing")):
        try:
            lookup(key)
        except KeyError:
            pass
        else:
            raise AssertionError("missing lookup did not raise")
    try:
        item.put("a", 2)
    except ValueError:
        pass
    else:
        raise AssertionError("collision not rejected")
    assert item.get("a") == 3 and item.get("b") == 2 and item.inverse(3) == "a"
    item.put("a", 3)
    item.put(None, None)
    assert item.inverse(None) is None and item.get(None) is None and len(item) == 3
    item.delete("a")
    assert len(item) == 2
    try:
        item.inverse(3)
    except KeyError:
        pass
    else:
        raise AssertionError("stale inverse")
    item.put("c", 3)
    assert item.inverse(3) == "c"
    return 7
