//! Summaries of the interactions already stored by the service.
//!
//! The window is the newest stored turns, optionally cut by age, status and
//! prompt length. Decode, TTFT, prefill and acceptance are then measured
//! inside that window, each with its own sample rule.

use crate::setup::{self, SetupError};
use rusqlite::{Connection, OpenFlags};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const DEFAULT_LAST: usize = 20;
const SCAN_CAP: usize = 20_000;
const BUCKETS: &[(f64, &str)] = &[
    (4_096.0, "<4K"),
    (32_768.0, "4K–32K"),
    (65_536.0, "32K–64K"),
    (131_072.0, "64K–128K"),
    (262_144.0, "128K–256K"),
];

struct Options {
    last: usize,
    since: Option<String>,
    status: String,
    min_completion: f64,
    min_prompt: Option<f64>,
    max_prompt: Option<f64>,
    min_prefill: f64,
    json: bool,
    database: Option<PathBuf>,
}

#[derive(Clone)]
struct Turn {
    status: String,
    created: Option<f64>,
    prompt_tokens: Option<f64>,
    completion_tokens: Option<f64>,
    ttft_ms: Option<f64>,
    first_content_ms: Option<f64>,
    decode_tok_s: Option<f64>,
    prefill_tok_s: Option<f64>,
    physical_prefill_tokens: Option<f64>,
    accepted_draft_tokens: Option<f64>,
    rejected_draft_tokens: Option<f64>,
    draft_acceptance: Option<f64>,
    cache_metrics_valid: bool,
    reasoning_tokens: Option<f64>,
}

struct Spread {
    n: usize,
    median: Option<f64>,
    min: Option<f64>,
    max: Option<f64>,
}

struct Bucket {
    name: &'static str,
    n: usize,
    decode_tok_s: Option<f64>,
    ttft_ms: Option<f64>,
    acceptance: Option<f64>,
}

struct Summary {
    turns: usize,
    decode_tok_s: Spread,
    ttft_ms: Spread,
    first_content_ms: Spread,
    prefill_tok_s: Spread,
    acceptance_n: usize,
    acceptance_weighted: Option<f64>,
    acceptance_median: Option<f64>,
    accepted_tokens: f64,
    drafted_tokens: f64,
    prompt_tokens: Spread,
    completion_tokens: Spread,
    reasoning_tokens: Spread,
    by_prompt: Vec<Bucket>,
    min_completion: f64,
    min_prefill: f64,
}

pub fn execute(arguments: impl Iterator<Item = String>) -> Result<(), SetupError> {
    let options = parse(arguments)?;
    let database = match options.database.clone() {
        Some(path) => path,
        None => setup::discover_repo()?.join("state/qwasar.db"),
    };
    println!("{}", render(&database, &options, None)?);
    Ok(())
}

fn parse(arguments: impl Iterator<Item = String>) -> Result<Options, SetupError> {
    let mut options = Options {
        last: DEFAULT_LAST,
        since: None,
        status: "completed,incomplete".into(),
        min_completion: 1.0,
        min_prompt: None,
        max_prompt: None,
        min_prefill: 256.0,
        json: false,
        database: None,
    };
    let mut arguments = arguments.peekable();
    while let Some(argument) = arguments.next() {
        let mut value = || {
            arguments.next().ok_or_else(|| SetupError(format!("missing value for {argument}")))
        };
        match argument.as_str() {
            "--last" => options.last = parse_usize(&value()?)?,
            "--since" => options.since = Some(value()?),
            "--status" => options.status = value()?,
            "--min-completion" => options.min_completion = parse_nonnegative(&value()?)?,
            "--min-prompt" => options.min_prompt = Some(parse_nonnegative(&value()?)?),
            "--max-prompt" => options.max_prompt = Some(parse_nonnegative(&value()?)?),
            "--min-prefill" => options.min_prefill = parse_nonnegative(&value()?)?,
            "--json" => options.json = true,
            "--database" => options.database = Some(PathBuf::from(value()?)),
            "--help" | "-h" => {
                return Err(SetupError("see `qwarz --help` for stats options".into()));
            }
            other => return Err(SetupError(format!("unknown option {other} for stats"))),
        }
    }
    if options.last < 1 {
        return Err(SetupError("--last must be at least 1".into()));
    }
    Ok(options)
}

fn parse_usize(text: &str) -> Result<usize, SetupError> {
    text.parse::<usize>().map_err(|_| SetupError(format!("{text} is not a positive integer")))
}

fn parse_nonnegative(text: &str) -> Result<f64, SetupError> {
    let value = text.parse::<f64>().map_err(|_| SetupError(format!("{text} is not a number")))?;
    if value < 0.0 {
        return Err(SetupError("measurement floors must be zero or positive".into()));
    }
    Ok(value)
}

pub(crate) fn parse_duration(text: &str) -> Result<f64, String> {
    let value = text.trim().to_ascii_lowercase();
    if value.is_empty() {
        return Err("duration is empty".into());
    }
    let (number, scale) = match value.as_bytes().last().copied() {
        Some(b's') => (&value[..value.len() - 1], 1.0),
        Some(b'm') => (&value[..value.len() - 1], 60.0),
        Some(b'h') => (&value[..value.len() - 1], 3_600.0),
        Some(b'd') => (&value[..value.len() - 1], 86_400.0),
        _ => (value.as_str(), 1.0),
    };
    let amount = number.parse::<f64>().map_err(|_| {
        format!("duration {text:?} is not a number of seconds, minutes, hours or days")
    })?;
    if amount < 0.0 {
        return Err("duration must be zero or positive".into());
    }
    Ok(amount * scale)
}

fn parse_statuses(text: &str) -> Result<Option<Vec<String>>, String> {
    if text.trim().eq_ignore_ascii_case("all") {
        return Ok(None);
    }
    let statuses: Vec<String> = text.split(',').map(|part| part.trim().to_string()).filter(|part| !part.is_empty()).collect();
    if statuses.is_empty() {
        return Err("status list is empty".into());
    }
    let allowed = ["completed", "incomplete", "cancelled", "failed", "in_progress"];
    let unknown: Vec<_> = statuses.iter().filter(|status| !allowed.contains(&status.as_str())).cloned().collect();
    if !unknown.is_empty() {
        return Err(format!("unknown status: {}", unknown.join(", ")));
    }
    Ok(Some(statuses))
}

fn number(value: &Value) -> Option<f64> {
    value.as_f64()
}

fn object<'a>(value: &'a Value, key: &str) -> &'a Value {
    value.get(key).filter(|item| item.is_object()).unwrap_or(&Value::Null)
}

fn turn_from_row(status: &str, result: &Value) -> Turn {
    let chat = object(result, "chat");
    let metrics = object(chat, "qwasar_metrics");
    let usage = object(chat, "usage");
    let physical = number(&metrics["physical_prefill_tokens"]);
    let prefill_ms = number(&metrics["host_prefill_ms"]);
    let prefill_tok_s = match (physical, prefill_ms) {
        (Some(tokens), Some(ms)) if tokens > 0.0 && ms > 0.0 => Some(tokens / (ms / 1000.0)),
        _ => None,
    };
    Turn {
        status: status.to_string(),
        created: number(&chat["created"]),
        prompt_tokens: number(&usage["prompt_tokens"]),
        completion_tokens: number(&usage["completion_tokens"]),
        ttft_ms: number(&metrics["ttft_ms"]),
        first_content_ms: number(&metrics["first_content_ms"]),
        decode_tok_s: number(&metrics["decode_tokens_per_second"]),
        prefill_tok_s,
        physical_prefill_tokens: physical,
        accepted_draft_tokens: number(&metrics["accepted_draft_tokens"]),
        rejected_draft_tokens: number(&metrics["rejected_draft_tokens"]),
        draft_acceptance: number(&metrics["draft_acceptance"]),
        cache_metrics_valid: metrics["cache_metrics_valid"] == true,
        reasoning_tokens: number(&metrics["reasoning_tokens"]),
    }
}

fn select_turns(rows: &[(String, Value)], last: usize, since_seconds: Option<f64>, statuses: Option<&[String]>, min_prompt: Option<f64>, max_prompt: Option<f64>, now: f64) -> Result<(Vec<Turn>, usize), String> {
    if last < 1 {
        return Err("last must be at least 1".into());
    }
    let cutoff = since_seconds.map(|seconds| now - seconds);
    let mut chosen = Vec::new();
    let mut scanned = 0;
    for (status, result) in rows {
        scanned += 1;
        let turn = turn_from_row(status, result);
        if let (Some(limit), Some(created)) = (cutoff, turn.created) {
            if created < limit {
                break;
            }
        }
        if let Some(allowed) = statuses {
            if !allowed.iter().any(|item| item == &turn.status) {
                continue;
            }
        }
        if let Some(minimum) = min_prompt {
            if turn.prompt_tokens.is_none_or(|prompt| prompt < minimum) {
                continue;
            }
        }
        if let Some(maximum) = max_prompt {
            if turn.prompt_tokens.is_none_or(|prompt| prompt > maximum) {
                continue;
            }
        }
        chosen.push(turn);
        if chosen.len() >= last {
            break;
        }
    }
    Ok((chosen, scanned))
}

pub(crate) fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    let middle = ordered.len() / 2;
    if ordered.len() % 2 == 1 {
        Some(ordered[middle])
    } else {
        Some((ordered[middle - 1] + ordered[middle]) / 2.0)
    }
}

fn spread(values: &[f64]) -> Spread {
    Spread {
        n: values.len(),
        median: median(values),
        min: values.iter().copied().reduce(f64::min),
        max: values.iter().copied().reduce(f64::max),
    }
}

fn bucket_name(prompt: Option<f64>) -> &'static str {
    let Some(prompt) = prompt else {
        return "unknown";
    };
    BUCKETS.iter().find(|(limit, _)| prompt < *limit).map(|(_, name)| *name).unwrap_or("256K+")
}

fn summarize(turns: &[Turn], min_completion: f64, min_prefill: f64) -> Summary {
    let decode = turns.iter().filter(|turn| turn.decode_tok_s.is_some() && turn.completion_tokens.unwrap_or(0.0) >= min_completion).filter_map(|turn| turn.decode_tok_s).collect::<Vec<_>>();
    let ttft = turns.iter().filter_map(|turn| turn.ttft_ms).collect::<Vec<_>>();
    let first_content = turns.iter().filter_map(|turn| turn.first_content_ms).collect::<Vec<_>>();
    let prefill = turns.iter().filter(|turn| turn.prefill_tok_s.is_some() && turn.physical_prefill_tokens.unwrap_or(0.0) >= min_prefill).filter_map(|turn| turn.prefill_tok_s).collect::<Vec<_>>();
    let mut accepted = 0.0;
    let mut rejected = 0.0;
    let mut rates = Vec::new();
    for turn in turns {
        if !turn.cache_metrics_valid {
            continue;
        }
        let (Some(accepted_tokens), Some(rejected_tokens)) = (turn.accepted_draft_tokens, turn.rejected_draft_tokens) else {
            continue;
        };
        accepted += accepted_tokens;
        rejected += rejected_tokens;
        if let Some(rate) = turn.draft_acceptance {
            rates.push(rate);
        }
    }
    let drafted = accepted + rejected;
    let mut names = BUCKETS.iter().map(|(_, name)| *name).collect::<Vec<_>>();
    names.push("256K+");
    let by_prompt = names.into_iter().filter_map(|name| {
        let group = turns.iter().filter(|turn| bucket_name(turn.prompt_tokens) == name).collect::<Vec<_>>();
        if group.is_empty() {
            return None;
        }
        let group_decode = group.iter().filter(|turn| turn.decode_tok_s.is_some() && turn.completion_tokens.unwrap_or(0.0) >= min_completion).filter_map(|turn| turn.decode_tok_s).collect::<Vec<_>>();
        let group_ttft = group.iter().filter_map(|turn| turn.ttft_ms).collect::<Vec<_>>();
        let group_accepted: f64 = group.iter().filter(|turn| turn.cache_metrics_valid).map(|turn| turn.accepted_draft_tokens.unwrap_or(0.0)).sum();
        let group_rejected: f64 = group.iter().filter(|turn| turn.cache_metrics_valid && turn.accepted_draft_tokens.is_some()).map(|turn| turn.rejected_draft_tokens.unwrap_or(0.0)).sum();
        let group_drafted = group_accepted + group_rejected;
        Some(Bucket {
            name,
            n: group.len(),
            decode_tok_s: median(&group_decode),
            ttft_ms: median(&group_ttft),
            acceptance: (group_drafted > 0.0).then_some(group_accepted / group_drafted),
        })
    }).collect();
    Summary {
        turns: turns.len(),
        decode_tok_s: spread(&decode),
        ttft_ms: spread(&ttft),
        first_content_ms: spread(&first_content),
        prefill_tok_s: spread(&prefill),
        acceptance_n: rates.len(),
        acceptance_weighted: (drafted > 0.0).then_some(accepted / drafted),
        acceptance_median: median(&rates),
        accepted_tokens: accepted,
        drafted_tokens: drafted,
        prompt_tokens: spread(&turns.iter().filter_map(|turn| turn.prompt_tokens).collect::<Vec<_>>()),
        completion_tokens: spread(&turns.iter().filter_map(|turn| turn.completion_tokens).collect::<Vec<_>>()),
        reasoning_tokens: spread(&turns.iter().filter_map(|turn| turn.reasoning_tokens).collect::<Vec<_>>()),
        by_prompt,
        min_completion,
        min_prefill,
    }
}

pub(crate) fn grouped(value: f64, decimals: usize) -> String {
    let negative = value.is_sign_negative();
    let text = format!("{:.*}", decimals, value.abs());
    let (whole, fraction) = text.split_once('.').map(|(whole, fraction)| (whole.to_string(), Some(fraction.to_string()))).unwrap_or((text, None));
    let digits: Vec<char> = whole.chars().collect();
    let mut body = String::new();
    for (index, character) in digits.iter().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            body.push(',');
        }
        body.push(*character);
    }
    let rendered = match fraction {
        Some(fraction) => format!("{body}.{fraction}"),
        None => body,
    };
    if negative { format!("-{rendered}") } else { rendered }
}

fn fmt_number(value: Option<f64>) -> String {
    value.map(|item| grouped(item, 0)).unwrap_or_else(|| "—".into())
}

fn fmt_ms(value: Option<f64>) -> String {
    match value {
        None => "—".into(),
        Some(ms) if ms >= 1000.0 => format!("{:.2} s", ms / 1000.0),
        Some(ms) => format!("{:.0} ms", ms),
    }
}

fn fmt_tok_s(value: Option<f64>) -> String {
    match value {
        None => "—".into(),
        Some(rate) if rate >= 100.0 => format!("{} tok/s", grouped(rate, 0)),
        Some(rate) => format!("{rate:.1} tok/s"),
    }
}

fn fmt_ratio(value: Option<f64>) -> String {
    value.map(|item| format!("{:.0}%", item * 100.0)).unwrap_or_else(|| "—".into())
}

fn fmt_age(created: Option<f64>, now: f64) -> String {
    let Some(created) = created else {
        return "—".into();
    };
    let delta = (now - created).max(0.0) as i64;
    if delta < 60 {
        format!("{delta}s")
    } else if delta < 3_600 {
        format!("{}m", delta / 60)
    } else if delta < 86_400 {
        format!("{}h", delta / 3_600)
    } else {
        format!("{}d", delta / 86_400)
    }
}

fn line(label: &str, body: String) -> String {
    format!("{label:<14}{body}")
}

fn format_report(summary: &Summary, turns: &[Turn], last: usize, since_seconds: Option<f64>, statuses: Option<&[String]>, scanned: usize, truncated: bool, now: f64) -> String {
    let window = match since_seconds {
        None => format!("newest {last}"),
        Some(seconds) => format!("newest {last} within {}", fmt_age(Some(now - seconds), now)),
    };
    let status_text = statuses.map(|items| items.join(", ")).unwrap_or_else(|| "any status".into());
    let mut lines = vec![
        format!("Qwarz stats — {window} ({status_text})"),
        format!("{} turns in the window, {scanned} scanned{}", summary.turns, if truncated { " (scan cap reached)" } else { "" }),
        String::new(),
        line("decode", format!("{} median  n={}  min {}  max {}  (completion ≥ {})", fmt_tok_s(summary.decode_tok_s.median), summary.decode_tok_s.n, fmt_tok_s(summary.decode_tok_s.min), fmt_tok_s(summary.decode_tok_s.max), grouped(summary.min_completion, 0))),
        line("ttft", format!("{} median  n={}  min {}  max {}", fmt_ms(summary.ttft_ms.median), summary.ttft_ms.n, fmt_ms(summary.ttft_ms.min), fmt_ms(summary.ttft_ms.max))),
        line("first content", format!("{} median  n={}", fmt_ms(summary.first_content_ms.median), summary.first_content_ms.n)),
        line("prefill", format!("{} median  n={}  (new prompt tokens ≥ {})", fmt_tok_s(summary.prefill_tok_s.median), summary.prefill_tok_s.n, grouped(summary.min_prefill, 0))),
        line("acceptance", format!("{} weighted  n={}  median {}  {} accepted / {} drafted", fmt_ratio(summary.acceptance_weighted), summary.acceptance_n, fmt_ratio(summary.acceptance_median), fmt_number(Some(summary.accepted_tokens)), fmt_number(Some(summary.drafted_tokens)))),
        line("prompt", format!("{} median tokens", fmt_number(summary.prompt_tokens.median))),
        line("output", format!("{} median tokens", fmt_number(summary.completion_tokens.median))),
        line("reasoning", format!("{} median tokens", fmt_number(summary.reasoning_tokens.median))),
    ];
    if summary.by_prompt.len() > 1 {
        lines.push(String::new());
        lines.push("by prompt size".into());
        for bucket in &summary.by_prompt {
            lines.push(format!("  {:<10} n={:<4} decode {:<12} ttft {:<8} acceptance {}", bucket.name, bucket.n, fmt_tok_s(bucket.decode_tok_s), fmt_ms(bucket.ttft_ms), fmt_ratio(bucket.acceptance)));
        }
    }
    if turns.is_empty() {
        lines.push(String::new());
        lines.push("No interactions match this window.".into());
    } else {
        lines.push(String::new());
        lines.push(format!("{:<6}{:>10}{:>8}{:>12}{:>10}{:>8}  status", "age", "prompt", "out", "decode", "ttft", "accept"));
        for turn in turns {
            lines.push(format!("{:<6}{:>10}{:>8}{:>12}{:>10}{:>8}  {}", fmt_age(turn.created, now), fmt_number(turn.prompt_tokens), fmt_number(turn.completion_tokens), fmt_tok_s(turn.decode_tok_s), fmt_ms(turn.ttft_ms), fmt_ratio(turn.draft_acceptance), turn.status));
        }
    }
    lines.join("\n")
}

fn spread_json(spread: &Spread) -> Value {
    json!({"n": spread.n, "median": spread.median, "min": spread.min, "max": spread.max})
}

fn json_report(summary: &Summary, turns: &[Turn], options: &Options, since_seconds: Option<f64>, statuses: Option<&[String]>, scanned: usize, truncated: bool) -> String {
    let payload = json!({
        "window": {
            "last": options.last,
            "since_seconds": since_seconds,
            "statuses": statuses.map(|items| Value::from(items)).unwrap_or(Value::String("all".into())),
            "matched": turns.len(),
            "scanned": scanned,
            "truncated": truncated,
            "min_completion": options.min_completion,
            "min_prompt": options.min_prompt,
            "max_prompt": options.max_prompt,
            "min_prefill": options.min_prefill,
        },
        "summary": {
            "turns": summary.turns,
            "decode_tok_s": spread_json(&summary.decode_tok_s),
            "ttft_ms": spread_json(&summary.ttft_ms),
            "first_content_ms": spread_json(&summary.first_content_ms),
            "prefill_tok_s": spread_json(&summary.prefill_tok_s),
            "acceptance": {
                "n": summary.acceptance_n,
                "weighted": summary.acceptance_weighted,
                "median": summary.acceptance_median,
                "accepted_tokens": summary.accepted_tokens,
                "drafted_tokens": summary.drafted_tokens,
            },
            "prompt_tokens": spread_json(&summary.prompt_tokens),
            "completion_tokens": spread_json(&summary.completion_tokens),
            "reasoning_tokens": spread_json(&summary.reasoning_tokens),
            "by_prompt": summary.by_prompt.iter().map(|bucket| json!({
                "bucket": bucket.name,
                "n": bucket.n,
                "decode_tok_s": bucket.decode_tok_s,
                "ttft_ms": bucket.ttft_ms,
                "acceptance": bucket.acceptance,
            })).collect::<Vec<_>>(),
            "min_completion": summary.min_completion,
            "min_prefill": summary.min_prefill,
        },
        "turns": turns.iter().map(|turn| json!({
            "status": turn.status,
            "created": turn.created,
            "prompt_tokens": turn.prompt_tokens,
            "completion_tokens": turn.completion_tokens,
            "ttft_ms": turn.ttft_ms,
            "decode_tok_s": turn.decode_tok_s,
            "draft_acceptance": turn.draft_acceptance,
            "cache_metrics_valid": turn.cache_metrics_valid,
        })).collect::<Vec<_>>(),
    });
    serde_json::to_string_pretty(&payload).unwrap_or_else(|_| "{}".into())
}

fn load_results(database: &Path, limit: usize) -> Result<Vec<(String, Value)>, String> {
    if limit < 1 {
        return Err("last must be at least 1".into());
    }
    if !database.is_file() {
        return Err(format!("no interaction database at {}", database.display()));
    }
    let connection = Connection::open_with_flags(database, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .map_err(|error| error.to_string())?;
    connection.busy_timeout(std::time::Duration::from_secs(5)).map_err(|error| error.to_string())?;
    let mut statement = connection
        .prepare("SELECT status, result FROM responses ORDER BY rowid DESC LIMIT ?")
        .map_err(|error| error.to_string())?;
    let rows = statement
        .query_map([limit.min(SCAN_CAP) as i64], |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)))
        .map_err(|error| error.to_string())?;
    let mut loaded = Vec::new();
    for row in rows {
        let (status, text) = row.map_err(|error| error.to_string())?;
        if let Ok(result) = serde_json::from_str::<Value>(&text) {
            loaded.push((status, result));
        }
    }
    Ok(loaded)
}

fn render(database: &Path, options: &Options, now: Option<f64>) -> Result<String, SetupError> {
    let since_seconds = options.since.as_deref().map(parse_duration).transpose().map_err(SetupError)?;
    let statuses = parse_statuses(&options.status).map_err(SetupError)?;
    let fetch = if since_seconds.is_some() { SCAN_CAP } else { SCAN_CAP.min(options.last.saturating_mul(50).max(options.last)) };
    let rows = load_results(database, fetch).map_err(SetupError)?;
    let moment = now.unwrap_or_else(now_seconds);
    let (turns, scanned) = select_turns(&rows, options.last, since_seconds, statuses.as_deref(), options.min_prompt, options.max_prompt, moment).map_err(SetupError)?;
    let summary = summarize(&turns, options.min_completion, options.min_prefill);
    let truncated = rows.len() >= fetch && turns.len() < options.last && since_seconds.is_some();
    if options.json {
        Ok(json_report(&summary, &turns, options, since_seconds, statuses.as_deref(), scanned, truncated))
    } else {
        Ok(format_report(&summary, &turns, options.last, since_seconds, statuses.as_deref(), scanned, truncated, moment))
    }
}

fn now_seconds() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|duration| duration.as_secs_f64()).unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: f64 = 1_000_000.0;

    fn row(created: f64, prompt: f64, completion: f64, decode: f64, ttft: f64, physical: f64, prefill_ms: f64, accepted: f64, rejected: f64, status: &str, valid: bool) -> (String, Value) {
        let drafted = accepted + rejected;
        (
            status.into(),
            json!({
                "chat": {
                    "created": created,
                    "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
                    "qwasar_metrics": {
                        "ttft_ms": ttft,
                        "first_content_ms": ttft + 100.0,
                        "decode_tokens_per_second": decode,
                        "physical_prefill_tokens": physical,
                        "host_prefill_ms": prefill_ms,
                        "accepted_draft_tokens": if valid { Value::from(accepted) } else { Value::Null },
                        "rejected_draft_tokens": if valid { Value::from(rejected) } else { Value::Null },
                        "draft_acceptance": if valid && drafted > 0.0 { Value::from(accepted / drafted) } else { Value::Null },
                        "cache_metrics_valid": valid,
                        "reasoning_tokens": 10,
                    }
                }
            }),
        )
    }

    #[test]
    fn duration_units() {
        assert_eq!(parse_duration("90s").unwrap(), 90.0);
        assert_eq!(parse_duration("30m").unwrap(), 1_800.0);
        assert_eq!(parse_duration("2h").unwrap(), 7_200.0);
        assert_eq!(parse_duration("1d").unwrap(), 86_400.0);
        assert_eq!(parse_duration("15").unwrap(), 15.0);
        assert!(parse_duration("-1m").is_err());
        assert!(parse_duration("soon").is_err());
    }

    #[test]
    fn window_keeps_the_newest_turns_and_stops_at_since() {
        let rows = vec![
            row(NOW - 10.0, 1000.0, 50.0, 200.0, 400.0, 0.0, 0.0, 0.0, 0.0, "completed", true),
            row(NOW - 50.0, 8000.0, 40.0, 150.0, 800.0, 0.0, 0.0, 0.0, 0.0, "cancelled", true),
            row(NOW - 4000.0, 32000.0, 30.0, 90.0, 2000.0, 0.0, 0.0, 0.0, 0.0, "completed", true),
        ];
        let (chosen, scanned) = select_turns(&rows, 10, Some(3600.0), Some(&["completed".into(), "incomplete".into()]), None, None, NOW).unwrap();
        assert_eq!(scanned, 3);
        assert_eq!(chosen.len(), 1);
        assert_eq!(chosen[0].prompt_tokens, Some(1000.0));
    }

    #[test]
    fn decode_and_prefill_use_their_own_floors() {
        let turns = vec![
            turn_from_row("completed", &row(NOW, 20_000.0, 1.0, 100.0, 500.0, 10.0, 100.0, 3.0, 1.0, "completed", true).1),
            turn_from_row("completed", &row(NOW, 20_000.0, 40.0, 180.0, 1500.0, 8_000.0, 2000.0, 30.0, 10.0, "completed", true).1),
        ];
        let summary = summarize(&turns, 8.0, 256.0);
        assert_eq!(summary.decode_tok_s.n, 1);
        assert_eq!(summary.decode_tok_s.median, Some(180.0));
        assert_eq!(summary.ttft_ms.n, 2);
        assert_eq!(summary.prefill_tok_s.n, 1);
        assert_eq!(summary.prefill_tok_s.median, Some(4000.0));
        assert!((summary.acceptance_weighted.unwrap() - 33.0 / 44.0).abs() < 1e-9);
    }

    #[test]
    fn invalid_cache_metrics_do_not_enter_acceptance() {
        let turn = turn_from_row("completed", &row(NOW, 1000.0, 20.0, 120.0, 300.0, 0.0, 0.0, 9.0, 1.0, "completed", false).1);
        let summary = summarize(std::slice::from_ref(&turn), 1.0, 256.0);
        assert_eq!(summary.decode_tok_s.median, Some(120.0));
        assert_eq!(summary.acceptance_n, 0);
        assert!(summary.acceptance_weighted.is_none());
    }

    #[test]
    fn report_reads_the_newest_stored_interactions() {
        let path = std::env::temp_dir().join(format!("qwarz-stats-{}-{}.db", std::process::id(), NOW as u64));
        let _ = std::fs::remove_file(&path);
        let connection = Connection::open(&path).unwrap();
        connection.execute("CREATE TABLE responses(status TEXT, result TEXT)", []).unwrap();
        for (status, result) in [
            row(NOW - 900.0, 8_000.0, 2.0, 10.0, 100.0, 10.0, 10.0, 1.0, 1.0, "completed", true),
            row(NOW - 500.0, 180_000.0, 80.0, 140.0, 900.0, 2_000.0, 1000.0, 40.0, 40.0, "completed", true),
            row(NOW - 5.0, 4_000.0, 30.0, 210.0, 600.0, 3_000.0, 500.0, 20.0, 10.0, "completed", true),
        ] {
            connection.execute("INSERT INTO responses(status, result) VALUES(?1, ?2)", rusqlite::params![status, result.to_string()]).unwrap();
        }
        drop(connection);
        let options = Options {
            last: 2,
            since: None,
            status: "completed,incomplete".into(),
            min_completion: 8.0,
            min_prompt: None,
            max_prompt: None,
            min_prefill: 256.0,
            json: false,
            database: None,
        };
        let text = render(&path, &options, Some(NOW)).unwrap();
        assert!(text.contains("newest 2"), "{text}");
        assert!(text.contains("210 tok/s"), "{text}");
        assert!(text.contains("140 tok/s"), "{text}");
        assert!(!text.contains("10.0 tok/s"), "{text}");
        assert!(text.contains("<4K"), "{text}");
        assert!(text.contains("128K–256K"), "{text}");
        assert_eq!(median(&[1.0, 2.0, 3.0, 4.0]), Some(2.5));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn missing_database_is_an_error() {
        let error = render(Path::new("/tmp/qwarz-stats-missing.db"), &Options {
            last: 20,
            since: None,
            status: "completed,incomplete".into(),
            min_completion: 1.0,
            min_prompt: None,
            max_prompt: None,
            min_prefill: 256.0,
            json: false,
            database: None,
        }, Some(NOW)).unwrap_err();
        assert!(error.0.contains("no interaction database"));
    }
}
