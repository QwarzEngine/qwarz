"""Cutover stays closed until the same 37-cell matrix passes.

The cells are the expanded quality matrix: four coding families, four
contexts, two seeds, four cold JSON cells, and the warmup cell. ExLlama
remains the serving worker until every cell is present and the four
promotion checks are true. This module does not start or stop a service.
"""
from __future__ import annotations

FAMILIES = ("lru", "ring", "bucket", "ttl")
CONTEXTS = (32768, 65536, 131072, 258048)
GATES = (
    "coding_not_worse",
    "json_all",
    "acceptance_within_5pp",
    "memory_within_half_gib",
)


def cells():
    labels = [f"{family}-{context}-{seed}" for family in FAMILIES for context in CONTEXTS for seed in (0, 1)]
    labels.extend(f"json-{context}-0-cold" for context in CONTEXTS)
    labels.append("warmup")
    return tuple(labels)


def promotion_allowed(report):
    if not isinstance(report, dict):
        return False
    got = report.get("cells")
    if not isinstance(got, dict) or set(got) != set(cells()):
        return False
    if not all(isinstance(row, dict) and row.get("passed") is True for row in got.values()):
        return False
    gates = report.get("gates")
    if not isinstance(gates, dict):
        return False
    return all(gates.get(name) is True for name in GATES)
