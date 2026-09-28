"""MTP6 acceptance.

The draft proposes at most six tokens. The target verifies that window in
one forward, keeps the matching prefix, and replaces the first mismatch
with its own token. A fully accepted window also keeps the bonus token the
target sampled after the draft. Rejected positions are rewound before the
kept tokens are written again.
"""
from __future__ import annotations

DRAFT_TOKENS = 6


def accepted_prefix(draft, verified):
    if not draft or len(draft) > DRAFT_TOKENS:
        raise ValueError("MTP6 proposes 1..6 tokens")
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
