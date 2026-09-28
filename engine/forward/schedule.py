"""Page-aligned prefill cuts shared by a cold prompt and a resident suffix.

ExLlama prefills every token except the last one. Chunks are at most 8192
tokens and end on a 256-token page, and the final partial page is its own
forward. A later turn starts at the resident cursor and uses the same cuts.
"""

CHUNK = 8192
PAGE = 256


def suffix_bounds(start, length, chunk=CHUNK, page=PAGE):
    """Token spans [start, length) to prefill, excluding the final id."""
    if start < 0 or length < 0:
        raise ValueError("prefill bounds need nonnegative positions")
    seqlen = length - 1
    if start >= seqlen:
        return []
    last_page = (seqlen // page) * page
    pos = start
    bounds = []
    while pos < seqlen:
        end = ((pos + chunk) // page) * page
        end = min(end, seqlen)
        if pos < last_page <= end:
            end = last_page
        if end <= pos:
            break
        bounds.append((pos, end))
        pos = end
    return bounds
