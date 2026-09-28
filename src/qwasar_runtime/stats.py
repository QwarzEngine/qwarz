"""Summaries of the interactions already stored by the service.

The window is the newest stored turns, optionally cut by age, status and
prompt length. Decode, TTFT, prefill and acceptance are then measured inside
that window, each with its own sample rule.
"""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import time


DEFAULT_LAST = 20
DEFAULT_STATUS = ("completed", "incomplete")
SCAN_CAP = 20000
BUCKETS = (
    (4_096, "<4K"),
    (32_768, "4K–32K"),
    (65_536, "32K–64K"),
    (131_072, "64K–128K"),
    (262_144, "128K–256K"),
)


def parse_duration(text: str) -> float:
    """Seconds in a window such as ``90s``, ``30m``, ``2h`` or ``1d``."""
    value = text.strip().lower()
    if not value:
        raise ValueError("duration is empty")
    suffix = value[-1]
    scale = {"s": 1, "m": 60, "h": 3600, "d": 86400}.get(suffix)
    number = value[:-1] if scale else value
    if scale is None:
        scale = 1
    try:
        amount = float(number)
    except ValueError as error:
        raise ValueError(f"duration {text!r} is not a number of seconds, minutes, hours or days") from error
    if amount < 0:
        raise ValueError("duration must be zero or positive")
    return amount * scale


def parse_statuses(text: str) -> tuple[str, ...] | None:
    if text.strip().lower() == "all":
        return None
    statuses = tuple(part.strip() for part in text.split(",") if part.strip())
    if not statuses:
        raise ValueError("status list is empty")
    allowed = {"completed", "incomplete", "cancelled", "failed", "in_progress"}
    unknown = [status for status in statuses if status not in allowed]
    if unknown:
        raise ValueError("unknown status: " + ", ".join(unknown))
    return statuses


def load_results(database: Path, limit: int) -> list[tuple[str, dict]]:
    if limit < 1:
        raise ValueError("last must be at least 1")
    if not database.is_file():
        raise ValueError(f"no interaction database at {database}")
    uri = f"file:{database}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=5)
    try:
        rows = connection.execute(
            "SELECT status, result FROM responses ORDER BY rowid DESC LIMIT ?",
            (min(limit, SCAN_CAP),),
        ).fetchall()
    finally:
        connection.close()
    loaded = []
    for status, text in rows:
        try:
            loaded.append((status, json.loads(text)))
        except (TypeError, json.JSONDecodeError):
            continue
    return loaded


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def turn_from_row(status: str, result: dict) -> dict:
    chat = result.get("chat") if isinstance(result, dict) else None
    if not isinstance(chat, dict):
        chat = {}
    metrics = chat.get("qwasar_metrics") if isinstance(chat.get("qwasar_metrics"), dict) else {}
    usage = chat.get("usage") if isinstance(chat.get("usage"), dict) else {}
    details = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}
    prompt = _number(usage.get("prompt_tokens"))
    completion = _number(usage.get("completion_tokens"))
    physical = _number(metrics.get("physical_prefill_tokens"))
    prefill_ms = _number(metrics.get("host_prefill_ms"))
    prefill_tok_s = physical / (prefill_ms / 1000) if physical and prefill_ms and prefill_ms > 0 else None
    accepted = _number(metrics.get("accepted_draft_tokens"))
    rejected = _number(metrics.get("rejected_draft_tokens"))
    return {
        "status": status,
        "created": _number(chat.get("created")),
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "ttft_ms": _number(metrics.get("ttft_ms")),
        "first_content_ms": _number(metrics.get("first_content_ms")),
        "decode_tok_s": _number(metrics.get("decode_tokens_per_second")),
        "prefill_tok_s": prefill_tok_s,
        "physical_prefill_tokens": physical,
        "accepted_draft_tokens": accepted,
        "rejected_draft_tokens": rejected,
        "draft_acceptance": _number(metrics.get("draft_acceptance")),
        "cache_metrics_valid": metrics.get("cache_metrics_valid") is True,
        "reasoning_tokens": _number(metrics.get("reasoning_tokens")),
        "cached_tokens": _number(details.get("cached_tokens")),
    }


def select_turns(rows, *, last, since_seconds=None, statuses=DEFAULT_STATUS, min_prompt=None, max_prompt=None, now=None):
    """Newest matching turns, stopping once a stored timestamp falls outside ``since``."""
    if last < 1:
        raise ValueError("last must be at least 1")
    cutoff = None if since_seconds is None else (time.time() if now is None else now) - since_seconds
    chosen = []
    scanned = 0
    for status, result in rows:
        scanned += 1
        turn = turn_from_row(status, result)
        if cutoff is not None and turn["created"] is not None and turn["created"] < cutoff:
            break
        if statuses is not None and turn["status"] not in statuses:
            continue
        prompt = turn["prompt_tokens"]
        if min_prompt is not None and (prompt is None or prompt < min_prompt):
            continue
        if max_prompt is not None and (prompt is None or prompt > max_prompt):
            continue
        chosen.append(turn)
        if len(chosen) >= last:
            break
    return chosen, scanned


def median(values):
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _sample(values):
    if not values:
        return {"n": 0, "median": None, "min": None, "max": None}
    return {"n": len(values), "median": median(values), "min": min(values), "max": max(values)}


def bucket_name(prompt_tokens):
    if prompt_tokens is None:
        return "unknown"
    for limit, name in BUCKETS:
        if prompt_tokens < limit:
            return name
    return "256K+"


def summarize(turns, *, min_completion, min_prefill):
    decode = [turn["decode_tok_s"] for turn in turns if turn["decode_tok_s"] is not None and (turn["completion_tokens"] or 0) >= min_completion]
    ttft = [turn["ttft_ms"] for turn in turns if turn["ttft_ms"] is not None]
    first_content = [turn["first_content_ms"] for turn in turns if turn["first_content_ms"] is not None]
    prefill = [
        turn["prefill_tok_s"] for turn in turns
        if turn["prefill_tok_s"] is not None and (turn["physical_prefill_tokens"] or 0) >= min_prefill
    ]
    accepted = rejected = 0.0
    acceptance_rates = []
    for turn in turns:
        if not turn["cache_metrics_valid"]:
            continue
        if turn["accepted_draft_tokens"] is None or turn["rejected_draft_tokens"] is None:
            continue
        accepted += turn["accepted_draft_tokens"]
        rejected += turn["rejected_draft_tokens"]
        if turn["draft_acceptance"] is not None:
            acceptance_rates.append(turn["draft_acceptance"])
    drafted = accepted + rejected
    buckets = {}
    for turn in turns:
        name = bucket_name(turn["prompt_tokens"])
        bucket = buckets.setdefault(name, [])
        bucket.append(turn)
    by_prompt = []
    for _, name in (*BUCKETS, (None, "256K+")):
        group = buckets.get(name)
        if not group:
            continue
        group_decode = [turn["decode_tok_s"] for turn in group if turn["decode_tok_s"] is not None and (turn["completion_tokens"] or 0) >= min_completion]
        group_ttft = [turn["ttft_ms"] for turn in group if turn["ttft_ms"] is not None]
        group_accepted = sum(turn["accepted_draft_tokens"] or 0 for turn in group if turn["cache_metrics_valid"])
        group_drafted = group_accepted + sum(turn["rejected_draft_tokens"] or 0 for turn in group if turn["cache_metrics_valid"] and turn["accepted_draft_tokens"] is not None)
        by_prompt.append({
            "bucket": name,
            "n": len(group),
            "decode_tok_s": median(group_decode),
            "ttft_ms": median(group_ttft),
            "acceptance": (group_accepted / group_drafted) if group_drafted else None,
        })
    return {
        "turns": len(turns),
        "decode_tok_s": _sample(decode),
        "ttft_ms": _sample(ttft),
        "first_content_ms": _sample(first_content),
        "prefill_tok_s": _sample(prefill),
        "acceptance": {
            "n": len(acceptance_rates),
            "weighted": (accepted / drafted) if drafted else None,
            "median": median(acceptance_rates),
            "accepted_tokens": accepted,
            "drafted_tokens": drafted,
        },
        "prompt_tokens": _sample([turn["prompt_tokens"] for turn in turns if turn["prompt_tokens"] is not None]),
        "completion_tokens": _sample([turn["completion_tokens"] for turn in turns if turn["completion_tokens"] is not None]),
        "reasoning_tokens": _sample([turn["reasoning_tokens"] for turn in turns if turn["reasoning_tokens"] is not None]),
        "by_prompt": by_prompt,
        "min_completion": min_completion,
        "min_prefill": min_prefill,
    }


def _fmt_number(value, digits=0):
    if value is None:
        return "—"
    if digits == 0:
        return f"{value:,.0f}"
    return f"{value:,.{digits}f}"


def _fmt_ms(value):
    if value is None:
        return "—"
    if value >= 1000:
        return f"{value / 1000:.2f} s"
    return f"{value:.0f} ms"


def _fmt_tok_s(value):
    if value is None:
        return "—"
    if value >= 100:
        return f"{value:,.0f} tok/s"
    return f"{value:.1f} tok/s"


def _fmt_ratio(value):
    if value is None:
        return "—"
    return f"{value * 100:.0f}%"


def _fmt_age(created, now):
    if created is None:
        return "—"
    delta = max(0, int(now - created))
    if delta < 60:
        return f"{delta}s"
    if delta < 3600:
        return f"{delta // 60}m"
    if delta < 86400:
        return f"{delta // 3600}h"
    return f"{delta // 86400}d"


def _line(label, body):
    return f"{label:<14}{body}"


def format_report(summary, turns, *, last, since_seconds, statuses, scanned, truncated, now):
    if since_seconds is None:
        window = f"newest {last}"
    else:
        window = f"newest {last} within {_fmt_age(now - since_seconds, now)}"
    status_text = "any status" if statuses is None else ", ".join(statuses)
    lines = [
        f"Qwarz stats — {window} ({status_text})",
        f"{summary['turns']} turns in the window, {scanned} scanned"
        + (" (scan cap reached)" if truncated else ""),
        "",
        _line("decode", f"{_fmt_tok_s(summary['decode_tok_s']['median'])} median  n={summary['decode_tok_s']['n']}  min {_fmt_tok_s(summary['decode_tok_s']['min'])}  max {_fmt_tok_s(summary['decode_tok_s']['max'])}  (completion ≥ {summary['min_completion']})"),
        _line("ttft", f"{_fmt_ms(summary['ttft_ms']['median'])} median  n={summary['ttft_ms']['n']}  min {_fmt_ms(summary['ttft_ms']['min'])}  max {_fmt_ms(summary['ttft_ms']['max'])}"),
        _line("first content", f"{_fmt_ms(summary['first_content_ms']['median'])} median  n={summary['first_content_ms']['n']}"),
        _line("prefill", f"{_fmt_tok_s(summary['prefill_tok_s']['median'])} median  n={summary['prefill_tok_s']['n']}  (new prompt tokens ≥ {summary['min_prefill']})"),
        _line("acceptance", f"{_fmt_ratio(summary['acceptance']['weighted'])} weighted  n={summary['acceptance']['n']}  median {_fmt_ratio(summary['acceptance']['median'])}  {_fmt_number(summary['acceptance']['accepted_tokens'])} accepted / {_fmt_number(summary['acceptance']['drafted_tokens'])} drafted"),
        _line("prompt", f"{_fmt_number(summary['prompt_tokens']['median'])} median tokens"),
        _line("output", f"{_fmt_number(summary['completion_tokens']['median'])} median tokens"),
        _line("reasoning", f"{_fmt_number(summary['reasoning_tokens']['median'])} median tokens"),
    ]
    if len(summary["by_prompt"]) > 1:
        lines.extend(["", "by prompt size"])
        for bucket in summary["by_prompt"]:
            lines.append(
                f"  {bucket['bucket']:<10} n={bucket['n']:<4} decode {_fmt_tok_s(bucket['decode_tok_s']):<12} ttft {_fmt_ms(bucket['ttft_ms']):<8} acceptance {_fmt_ratio(bucket['acceptance'])}"
            )
    if turns:
        lines.extend(["", f"{'age':<6}{'prompt':>10}{'out':>8}{'decode':>12}{'ttft':>10}{'accept':>8}  status"])
        for turn in turns:
            lines.append(
                f"{_fmt_age(turn['created'], now):<6}{_fmt_number(turn['prompt_tokens']):>10}{_fmt_number(turn['completion_tokens']):>8}{_fmt_tok_s(turn['decode_tok_s']):>12}{_fmt_ms(turn['ttft_ms']):>10}{_fmt_ratio(turn['draft_acceptance']):>8}  {turn['status']}"
            )
    else:
        lines.extend(["", "No interactions match this window."])
    return "\n".join(lines)


def report(database, *, last=DEFAULT_LAST, since=None, status="completed,incomplete", min_completion=1, min_prompt=None, max_prompt=None, min_prefill=256, as_json=False, now=None):
    if min_completion < 0 or min_prefill < 0:
        raise ValueError("measurement floors must be zero or positive")
    since_seconds = None if since is None else parse_duration(since)
    statuses = parse_statuses(status)
    fetch = SCAN_CAP if since_seconds is not None else min(SCAN_CAP, max(last * 50, last))
    rows = load_results(Path(database), fetch)
    turns, scanned = select_turns(
        rows,
        last=last,
        since_seconds=since_seconds,
        statuses=statuses,
        min_prompt=min_prompt,
        max_prompt=max_prompt,
        now=now,
    )
    summary = summarize(turns, min_completion=min_completion, min_prefill=min_prefill)
    truncated = len(rows) >= fetch and len(turns) < last and since_seconds is not None
    moment = time.time() if now is None else now
    payload = {
        "window": {
            "last": last,
            "since_seconds": since_seconds,
            "statuses": list(statuses) if statuses is not None else "all",
            "matched": len(turns),
            "scanned": scanned,
            "truncated": truncated,
            "min_completion": min_completion,
            "min_prompt": min_prompt,
            "max_prompt": max_prompt,
            "min_prefill": min_prefill,
        },
        "summary": summary,
        "turns": turns,
    }
    if as_json:
        return json.dumps(payload, indent=2)
    return format_report(summary, turns, last=last, since_seconds=since_seconds, statuses=statuses, scanned=scanned, truncated=truncated, now=moment)
