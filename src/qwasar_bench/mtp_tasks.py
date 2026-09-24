"""Task-family split fixed before capture. No benchmark completions used for training."""
from __future__ import annotations

import hashlib
import json

FAMILIES = {
    "train": {
        "csv": "Implement a small CSV reader supporting quoted commas, escaped quotes, and empty fields.",
        "ini": "Implement an INI parser with sections, comments, and duplicate-key errors.",
        "heap": "Implement a binary min-heap with push, pop, peek, and explicit empty errors.",
        "rle": "Implement run-length encoding and decoding of arbitrary Unicode strings.",
        "trie": "Implement a trie with insert, membership, prefix search, and deletion.",
        "brackets": "Implement a bracket checker that ignores brackets inside quoted strings and handles escapes.",
        "unionfind": "Implement disjoint sets with path compression and union by size.",
        "bitset": "Implement a growable bit set with set, clear, membership, union, and intersection.",
    },
    "validation": {
        "percent": "Implement percent encoding and decoding of UTF-8 URL path components.",
        "roman": "Implement strict conversion between integers and canonical Roman numerals.",
        "histogram": "Implement a streaming histogram with exact counts and deterministic tie ordering.",
        "matrix": "Implement transpose and matrix multiplication with dimension checks.",
    },
    "test": {
        "luhn": "Implement Luhn checksum validation and check-digit generation.",
        "pagination": "Implement cursor pagination over immutable records sorted by timestamp and unique ID.",
        "dedup": "Implement stable deduplication of dictionaries by a selected tuple of fields.",
        "duration": "Implement parsing and formatting of signed durations with hours, minutes, and seconds.",
    },
}


def tasks():
    result = []
    for split, families in FAMILIES.items():
        for family, task in families.items():
            for variant in range(6 if split == "train" else 2):
                language = "English" if variant % 2 == 0 else "Spanish"
                prompt = (f"{task} Use Python and only its standard library. "
                          f"Explain the design in {language}. Include executable examples, "
                          f"edge cases and tests. Scenario identifier: {9230 + variant}. "
                          "Prioritize a correct, complete implementation.")
                result.append({"id": f"{split}-{family}-{variant}", "split": split,
                               "family": family, "prompt": prompt, "seed": 193 + variant})
    assert len({row["family"] for row in result}) == sum(map(len, FAMILIES.values()))
    return result


def manifest():
    rows = tasks()
    return {"tasks": rows, "sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
            "split_unit": "task family", "train": 48, "validation": 8, "test": 8,
            "calibration_benchmark_completions_used": False}
