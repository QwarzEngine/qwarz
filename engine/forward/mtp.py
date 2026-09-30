"""MTP6 acceptance.

The draft proposes at most six tokens. The target verifies that window in
one forward, keeps the matching prefix, and replaces the first mismatch
with its own token. A fully accepted window also keeps the bonus token the
target sampled after the draft. Rejected positions are rewound before the
kept tokens are written again.
"""
from __future__ import annotations

import os

# Proposals per window. The verify window (DRAFT_TOKENS + 1) must stay within
# the XQA short-query path and the GDN rewind stash, both at most 8 rows.
DRAFT_TOKENS = min(max(int(os.environ.get("QWARZ_DRAFT_TOKENS", "6")), 1), 7)


def accepted_prefix(draft, verified):
    if not draft or len(draft) > DRAFT_TOKENS:
        raise ValueError(f"MTP proposes 1..{DRAFT_TOKENS} tokens")
    count = 0
    for proposed, sampled in zip(draft, verified):
        if proposed != sampled:
            break
        count += 1
    return count


def acceptance(accepted, rejected):
    total = accepted + rejected
    if total == 0:
        return None
    return accepted / total


def emitted_tokens(draft, samples):
    """Keep the matching draft prefix, then the target sample at the first mismatch.

    `samples` has one argmax per draft token plus the bonus argmax after the
    last draft token. A full match keeps that bonus. The held prompt token is
    not part of the result.
    """
    draft = list(draft)
    samples = list(samples)
    if len(samples) != len(draft) + 1:
        raise ValueError("the target returns one sample per draft token plus the bonus")
    matched = accepted_prefix(draft, samples)
    if matched == len(draft):
        return draft + [samples[-1]], matched
    return draft[:matched] + [samples[matched]], matched


def verify_window(session, draft, verified, bonus):
    """Speculatively write the draft, then rewind and commit the kept tokens."""
    draft = list(draft)
    verified = list(verified)
    if len(verified) < len(draft):
        raise ValueError("the target verified fewer tokens than the draft proposed")
    kept_count = accepted_prefix(draft, verified)
    before = session.runner.capture()
    start = session.cursor
    session.runner.forward(draft, start)
    session.runner.restore(before)
    kept = draft[:kept_count]
    if kept_count < len(draft):
        kept.append(verified[kept_count])
    else:
        kept.append(bonus)
    session.runner.forward(kept, start)
    session.cursor = start + len(kept)
    session.tape = session.tape[:start] + kept
    rejected = len(draft) - kept_count
    return {
        "accepted": kept_count,
        "rejected": rejected,
        "acceptance": acceptance(kept_count, rejected),
        "kept": kept,
    }
