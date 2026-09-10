def check_lru(cache_type):
    missing = object()

    def lookup(cache, key):
        try:
            return cache.get(key)
        except KeyError:
            return missing

    for capacity in (0, -1, 1.5, "2", True):
        try:
            cache_type(capacity)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(f"accepted invalid capacity: {capacity!r}")

    cache = cache_type(2)
    assert len(cache) == 0
    cache.put("a", 1)
    cache.put("b", 2)
    assert len(cache) == 2 and lookup(cache, "a") == 1

    cache.put("c", 3)
    assert lookup(cache, "b") != 2
    assert lookup(cache, "a") == 1 and lookup(cache, "c") == 3

    cache.put("a", 10)
    cache.put("d", 4)
    assert lookup(cache, "c") != 3
    assert lookup(cache, "a") == 10 and len(cache) == 2

    cache.put("a", None)
    assert lookup(cache, "a") is None and len(cache) == 2
    cache.put("e", 5)
    assert lookup(cache, "d") != 4 and lookup(cache, "a") is None

    cache.delete("a")
    assert len(cache) == 1
    cache.put("a", 42)
    assert len(cache) == 2 and lookup(cache, "a") == 42

    cache = cache_type(1)
    cache.put("a", 1)
    cache.put("a", 2)
    assert len(cache) == 1 and lookup(cache, "a") == 2
    cache.put("b", 3)
    assert lookup(cache, "a") != 2 and lookup(cache, "b") == 3
    assert len(cache) == 1
    return 7
