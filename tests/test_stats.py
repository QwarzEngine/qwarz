import json
from pathlib import Path
import sqlite3

import pytest

from qwasar_runtime.stats import median, parse_duration, report, select_turns, summarize, turn_from_row


NOW = 1_000_000.0


def _row(created, prompt, completion, decode, ttft, physical=0, prefill_ms=0, accepted=0, rejected=0, status="completed", valid=True):
    prefill_tok_s_inputs = physical, prefill_ms
    result = {
        "chat": {
            "created": created,
            "usage": {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "prompt_tokens_details": {"cached_tokens": max(prompt - physical - 1, 0)},
            },
            "qwasar_metrics": {
                "ttft_ms": ttft,
                "first_content_ms": ttft + 100,
                "decode_tokens_per_second": decode,
                "physical_prefill_tokens": physical,
                "host_prefill_ms": prefill_ms,
                "accepted_draft_tokens": accepted if valid else None,
                "rejected_draft_tokens": rejected if valid else None,
                "draft_acceptance": (accepted / (accepted + rejected)) if valid and accepted + rejected else None,
                "cache_metrics_valid": valid,
                "reasoning_tokens": 10,
            },
        }
    }
    assert prefill_tok_s_inputs
    return status, result


def _database(tmp_path: Path, rows) -> Path:
    database = tmp_path / "qwasar.db"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE responses(status TEXT, result TEXT)")
    connection.executemany(
        "INSERT INTO responses(status, result) VALUES(?, ?)",
        [(status, json.dumps(result)) for status, result in rows],
    )
    connection.commit()
    return database


def test_duration_units():
    assert parse_duration("90s") == 90
    assert parse_duration("30m") == 1800
    assert parse_duration("2h") == 7200
    assert parse_duration("1d") == 86400
    assert parse_duration("15") == 15
    with pytest.raises(ValueError):
        parse_duration("-1m")
    with pytest.raises(ValueError):
        parse_duration("soon")


def test_window_keeps_the_newest_turns_and_stops_at_since():
    rows = [
        _row(NOW - 10, 1000, 50, 200, 400),
        _row(NOW - 50, 8000, 40, 150, 800, status="cancelled"),
        _row(NOW - 4000, 32000, 30, 90, 2000),
    ]
    chosen, scanned = select_turns(rows, last=10, since_seconds=3600, now=NOW)
    assert scanned == 3
    assert [turn["prompt_tokens"] for turn in chosen] == [1000]
    assert chosen[0]["status"] == "completed"


def test_status_and_prompt_filters_define_the_measurement_window():
    rows = [
        _row(NOW, 1000, 10, 100, 100),
        _row(NOW, 200_000, 10, 50, 100, status="incomplete"),
        _row(NOW, 50_000, 10, 80, 100, status="failed"),
    ]
    chosen, _ = select_turns(rows, last=10, statuses=("completed", "incomplete"), min_prompt=10_000, max_prompt=100_000, now=NOW)
    assert chosen == []
    chosen, _ = select_turns(rows, last=10, statuses=("incomplete",), min_prompt=100_000, now=NOW)
    assert len(chosen) == 1
    assert chosen[0]["status"] == "incomplete"


def test_decode_and_prefill_use_their_own_floors():
    turns = [
        turn_from_row(*_row(NOW, 20_000, 1, 100, 500, physical=10, prefill_ms=100, accepted=3, rejected=1)),
        turn_from_row(*_row(NOW, 20_000, 40, 180, 1500, physical=8_000, prefill_ms=2000, accepted=30, rejected=10)),
    ]
    summary = summarize(turns, min_completion=8, min_prefill=256)
    assert summary["decode_tok_s"]["n"] == 1
    assert summary["decode_tok_s"]["median"] == 180
    assert summary["ttft_ms"]["n"] == 2
    assert summary["prefill_tok_s"]["n"] == 1
    assert summary["prefill_tok_s"]["median"] == pytest.approx(4000)
    assert summary["acceptance"]["weighted"] == pytest.approx(33 / 44)


def test_invalid_cache_metrics_do_not_enter_acceptance():
    turn = turn_from_row(*_row(NOW, 1000, 20, 120, 300, accepted=9, rejected=1, valid=False))
    summary = summarize([turn], min_completion=1, min_prefill=256)
    assert summary["decode_tok_s"]["median"] == 120
    assert summary["acceptance"]["n"] == 0
    assert summary["acceptance"]["weighted"] is None


def test_report_reads_the_newest_stored_interactions(tmp_path):
    database = _database(tmp_path, [
        _row(NOW - 900, 8_000, 2, 10, 100, physical=10, prefill_ms=10, accepted=1, rejected=1),
        _row(NOW - 500, 180_000, 80, 140, 900, physical=2_000, prefill_ms=1000, accepted=40, rejected=40),
        _row(NOW - 5, 4_000, 30, 210, 600, physical=3_000, prefill_ms=500, accepted=20, rejected=10),
    ])
    text = report(database, last=2, min_completion=8, min_prefill=256, now=NOW)
    assert "newest 2" in text
    assert "210 tok/s" in text
    assert "140 tok/s" in text
    assert "10.0 tok/s" not in text
    assert "<4K" in text
    assert "128K–256K" in text
    payload = json.loads(report(database, last=1, as_json=True, now=NOW))
    assert payload["window"]["matched"] == 1
    assert payload["turns"][0]["decode_tok_s"] == 210
    assert median([1, 2, 3, 4]) == 2.5


def test_missing_database_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="no interaction database"):
        report(tmp_path / "missing.db")
