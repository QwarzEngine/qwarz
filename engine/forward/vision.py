"""Image token ids for the resident tape.

Dynamic image ids start at 1000000000, the same base ExLlama uses. They are
process-local, so a persisted tape stores -1 in their place and the text
embedding table is never indexed with them.
"""
from __future__ import annotations

MM_TOKEN_BASE = 1_000_000_000


def canonical_tape(tokens):
    return [token if token < MM_TOKEN_BASE else -1 for token in tokens]


def text_table_ids(ids):
    """Stand-ins for a text-table gather. Image ids become 0 and are never indexed."""
    return [0 if int(token) >= MM_TOKEN_BASE else int(token) for token in ids]


def embed_spans(ids):
    """Split a token row into text spans and single image ids."""
    spans = []
    text = []

    def flush():
        if text:
            spans.append(("text", tuple(text)))
            text.clear()

    for token in ids:
        if token >= MM_TOKEN_BASE:
            flush()
            spans.append(("image", token))
        else:
            text.append(token)
    flush()
    return spans


def assemble_rows(ids, text_row, image_row):
    """Build hidden rows. Image ids go to image_row, never to text_row."""
    rows = []
    for kind, payload in embed_spans(ids):
        if kind == "text":
            rows.extend(text_row(token) for token in payload)
        else:
            rows.append(image_row(payload))
    return rows
