//! Hardware readings for the machine serving Qwarz.
//!
//! One sample is a snapshot. Several samples, spaced by `--interval`, produce
//! the same readings plus a min / median / max. The GPU in the report is the
//! one running the worker unless `--gpu` says otherwise.

use crate::setup::SetupError;
use crate::stats::{self, parse_duration};
use serde_json::{Value, json};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::Path;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

const GPU_QUERY: &[&str] = &[
    "index", "uuid", "name", "utilization.gpu", "utilization.memory", "memory.used", "memory.total",
    "temperature.gpu", "power.draw", "power.limit", "clocks.sm", "clocks.mem", "pstate",
];
const CPU_FIELDS: &[&str] = &["user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal"];
const BUSY: &[&str] = &["user", "nice", "system", "irq", "softirq", "steal"];

static INTERRUPTED: AtomicBool = AtomicBool::new(false);

extern "C" fn on_sigint(_: libc::c_int) {
    INTERRUPTED.store(true, Ordering::Relaxed);
}

struct Options {
    interval: Option<f64>,
    samples: Option<usize>,
    gpu: Option<u32>,
    json: bool,
}

#[derive(Clone)]
struct Gpu {
    index: u32,
    uuid: String,
    name: String,
    utilization_gpu: Option<f64>,
    utilization_memory: Option<f64>,
    memory_used_mib: Option<f64>,
    memory_total_mib: Option<f64>,
    temperature_c: Option<f64>,
    power_w: Option<f64>,
    power_limit_w: Option<f64>,
    sm_clock_mhz: Option<f64>,
    memory_clock_mhz: Option<f64>,
    pstate: String,
}

struct App {
    uuid: String,
    pid: i64,
    memory_mib: f64,
}

struct Process {
    pid: i64,
    memory_mib: f64,
    qwarz: bool,
    name: String,
}

struct Host {
    cpu_percent: Option<f64>,
    load: [f64; 3],
    used_bytes: f64,
    total_bytes: f64,
}

struct Sample {
    gpu: Gpu,
    processes: Vec<Process>,
    qwarz_mib: f64,
    other_mib: f64,
    other_processes: usize,
    host: Host,
    others: Vec<Gpu>,
}

struct Spread {
    n: usize,
    median: Option<f64>,
    min: f64,
    max: f64,
}

pub fn execute(arguments: impl Iterator<Item = String>) -> Result<(), SetupError> {
    let options = parse(arguments)?;
    let samples = match (options.samples, options.interval) {
        (None, None) => Some(1),
        (samples, _) => samples,
    };
    let interval = match (samples, options.interval) {
        (Some(count), None) if count > 1 => Some(1.0),
        (_, interval) => interval,
    };
    if let Some(interval) = interval {
        if interval < 0.2 {
            return Err(SetupError("--interval must be at least 0.2 seconds".into()));
        }
    }
    if matches!(samples, Some(0)) {
        return Err(SetupError("--samples must be at least 1".into()));
    }
    let looping = samples.is_none() || samples.is_some_and(|count| count > 1);
    if looping {
        INTERRUPTED.store(false, Ordering::Relaxed);
        unsafe { libc::signal(libc::SIGINT, on_sigint as *const () as libc::sighandler_t); }
    }
    let mut taken = Vec::new();
    let mut health = None;
    let mut previous: Option<Vec<f64>> = None;
    loop {
        if samples.is_some_and(|count| taken.len() >= count) || INTERRUPTED.load(Ordering::Relaxed) {
            break;
        }
        let (sample, health_now, counters) = capture(options.gpu, previous.as_deref(), previous.is_none())?;
        health = health_now;
        previous = Some(counters);
        if !options.json {
            if !taken.is_empty() {
                println!();
            }
            println!("{}", format_sample(&sample, &worker_label(health.as_ref())));
        } else if samples.is_none() {
            println!("{}", serde_json::to_string(&sample_json(&sample)).unwrap_or_else(|_| "{}".into()));
        }
        taken.push(sample);
        if samples.is_some_and(|count| taken.len() >= count) || INTERRUPTED.load(Ordering::Relaxed) {
            break;
        }
        sleep_until(Duration::from_secs_f64(interval.unwrap_or(1.0)));
    }
    if taken.is_empty() {
        return Ok(());
    }
    let summary = (taken.len() > 1).then(|| summarize(&taken));
    if options.json && samples.is_some() {
        println!("{}", serde_json::to_string_pretty(&json!({
            "samples": taken.iter().map(sample_json).collect::<Vec<_>>(),
            "summary": summary.as_ref().map(summary_json),
            "worker": worker_label(health.as_ref()),
        })).unwrap_or_else(|_| "{}".into()));
    } else if let Some(summary) = summary.filter(|_| !options.json) {
        println!("{}", format_summary(&summary));
    }
    Ok(())
}

fn parse(arguments: impl Iterator<Item = String>) -> Result<Options, SetupError> {
    let mut options = Options { interval: None, samples: None, gpu: None, json: false };
    let mut arguments = arguments.peekable();
    while let Some(argument) = arguments.next() {
        let mut value = || arguments.next().ok_or_else(|| SetupError(format!("missing value for {argument}")));
        match argument.as_str() {
            "--interval" => {
                let seconds = parse_duration(&value()?).map_err(SetupError)?;
                if seconds < 0.2 {
                    return Err(SetupError("--interval must be at least 0.2 seconds".into()));
                }
                options.interval = Some(seconds);
            }
            "--samples" => {
                let count = value()?.parse::<usize>().map_err(|_| SetupError("--samples must be at least 1".into()))?;
                if count < 1 {
                    return Err(SetupError("--samples must be at least 1".into()));
                }
                options.samples = Some(count);
            }
            "--gpu" => {
                options.gpu = Some(value()?.parse::<u32>().map_err(|_| SetupError("--gpu must be a non-negative integer".into()))?);
            }
            "--json" => options.json = true,
            other => return Err(SetupError(format!("unknown option {other} for monitor"))),
        }
    }
    Ok(options)
}

fn sleep_until(total: Duration) {
    let step = Duration::from_millis(200);
    let start = std::time::Instant::now();
    while start.elapsed() < total && !INTERRUPTED.load(Ordering::Relaxed) {
        std::thread::sleep(step.min(total.saturating_sub(start.elapsed())));
    }
}

fn number(text: &str) -> Option<f64> {
    let value = text.trim();
    if value.is_empty() || value.starts_with('[') {
        return None;
    }
    value.parse().ok()
}

fn split_row(line: &str, expected: usize) -> Result<Vec<String>, String> {
    let parts: Vec<String> = line.split(',').map(|part| part.trim().to_string()).collect();
    if parts.len() < expected {
        return Err(format!("nvidia-smi row has {} fields, expected at least {expected}", parts.len()));
    }
    if parts.len() == expected {
        return Ok(parts);
    }
    let tail = expected - 3;
    let name = parts[2..parts.len() - tail].join(", ");
    let mut row = parts[..2].to_vec();
    row.push(name);
    row.extend(parts[parts.len() - tail..].iter().cloned());
    Ok(row)
}

fn parse_gpus(text: &str) -> Result<Vec<Gpu>, String> {
    let mut gpus = Vec::new();
    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        let fields = split_row(line, 13)?;
        gpus.push(Gpu {
            index: fields[0].parse::<f64>().map_err(|_| format!("bad GPU index {}", fields[0]))? as u32,
            uuid: fields[1].clone(),
            name: fields[2].clone(),
            utilization_gpu: number(&fields[3]),
            utilization_memory: number(&fields[4]),
            memory_used_mib: number(&fields[5]),
            memory_total_mib: number(&fields[6]),
            temperature_c: number(&fields[7]),
            power_w: number(&fields[8]),
            power_limit_w: number(&fields[9]),
            sm_clock_mhz: number(&fields[10]),
            memory_clock_mhz: number(&fields[11]),
            pstate: fields[12].clone(),
        });
    }
    Ok(gpus)
}

fn parse_apps(text: &str) -> Result<Vec<App>, String> {
    let mut apps = Vec::new();
    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        let fields = split_row(line, 3)?;
        apps.push(App {
            uuid: fields[0].clone(),
            pid: fields[1].parse::<f64>().map_err(|_| format!("bad pid {}", fields[1]))? as i64,
            memory_mib: number(&fields[2]).unwrap_or(0.0),
        });
    }
    Ok(apps)
}

fn parse_cpu(text: &str) -> Result<Vec<f64>, String> {
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("cpu ") {
            let values: Vec<f64> = rest.split_whitespace().take(CPU_FIELDS.len()).map(|part| part.parse().unwrap_or(0.0)).collect();
            if values.len() == CPU_FIELDS.len() {
                return Ok(values);
            }
        }
    }
    Err("cpu counters are missing from /proc/stat".into())
}

fn cpu_percent(before: &[f64], after: &[f64]) -> Option<f64> {
    let index = |name: &str| CPU_FIELDS.iter().position(|field| *field == name).unwrap();
    let busy: f64 = BUSY.iter().map(|field| after[index(field)] - before[index(field)]).sum();
    let total = busy + (after[index("idle")] - before[index("idle")]) + (after[index("iowait")] - before[index("iowait")]);
    (total > 0.0).then_some(100.0 * busy / total)
}

fn is_qwarz(pid: i64, cmdline: &str, service_pids: &[i64]) -> bool {
    service_pids.contains(&pid) || cmdline.to_ascii_lowercase().contains("qwasar") || cmdline.to_ascii_lowercase().contains("qwarz")
}

fn short_name(cmdline: &str) -> String {
    if cmdline.contains("engine.forward.worker") || cmdline.contains("qwasar_runtime.worker") {
        return "worker".into();
    }
    if cmdline.contains("qwasar-server") {
        return "server".into();
    }
    Path::new(cmdline.split_whitespace().next().unwrap_or("")).file_name().and_then(|name| name.to_str()).unwrap_or("process").to_string()
}

fn select_gpu(gpus: &[Gpu], apps: &[App], qwarz_pids: &[i64], explicit: Option<u32>) -> Result<Gpu, String> {
    if let Some(index) = explicit {
        return gpus.iter().find(|gpu| gpu.index == index).cloned().ok_or_else(|| {
            let found = gpus.iter().map(|gpu| gpu.index.to_string()).collect::<Vec<_>>().join(", ");
            format!("GPU {index} is not present (found {})", if found.is_empty() { "none" } else { &found })
        });
    }
    if let Some(gpu) = apps.iter().find(|app| qwarz_pids.contains(&app.pid)).and_then(|app| gpus.iter().find(|gpu| gpu.uuid == app.uuid)) {
        return Ok(gpu.clone());
    }
    if let Some(gpu) = gpus.iter().find(|gpu| gpu.name.contains("RTX 5090")) {
        return Ok(gpu.clone());
    }
    gpus.first().cloned().ok_or_else(|| "nvidia-smi reported no GPUs".into())
}

fn assemble(gpus: &[Gpu], apps: &[App], cmdlines: &[(i64, String)], service_pids: &[i64], host: Host, explicit: Option<u32>) -> Result<Sample, String> {
    let mut qwarz_pids = service_pids.to_vec();
    for (pid, cmdline) in cmdlines {
        if is_qwarz(*pid, cmdline, service_pids) {
            qwarz_pids.push(*pid);
        }
    }
    let gpu = select_gpu(gpus, apps, &qwarz_pids, explicit)?;
    let processes = apps.iter().filter(|app| app.uuid == gpu.uuid).map(|app| {
        let cmdline = cmdlines.iter().find(|(pid, _)| *pid == app.pid).map(|(_, text)| text.as_str()).unwrap_or("");
        Process {
            pid: app.pid,
            memory_mib: app.memory_mib,
            qwarz: is_qwarz(app.pid, cmdline, &qwarz_pids),
            name: if cmdline.is_empty() { "pid".into() } else { short_name(cmdline) },
        }
    }).collect::<Vec<_>>();
    Ok(Sample {
        qwarz_mib: processes.iter().filter(|item| item.qwarz).map(|item| item.memory_mib).sum(),
        other_mib: processes.iter().filter(|item| !item.qwarz).map(|item| item.memory_mib).sum(),
        other_processes: processes.iter().filter(|item| !item.qwarz).count(),
        others: gpus.iter().filter(|item| item.index != gpu.index).cloned().collect(),
        gpu,
        processes,
        host,
    })
}

fn fmt_mib(value: Option<f64>) -> String {
    value.map(|item| format!("{} MiB", stats::grouped(item, 0))).unwrap_or_else(|| "—".into())
}

fn fmt_percent(value: Option<f64>) -> String {
    value.map(|item| format!("{item:.0}%")).unwrap_or_else(|| "—".into())
}

fn fmt_watts(value: Option<f64>) -> String {
    match value {
        None => "—".into(),
        Some(watts) if watts >= 100.0 => format!("{watts:.0} W"),
        Some(watts) => format!("{watts:.1} W"),
    }
}

fn fmt_temp(value: Option<f64>) -> String {
    value.map(|item| format!("{item:.0} °C")).unwrap_or_else(|| "—".into())
}

fn memory_percent(used: Option<f64>, total: Option<f64>) -> Option<f64> {
    match (used, total) {
        (Some(used), Some(total)) if total > 0.0 => Some(100.0 * used / total),
        _ => None,
    }
}

fn format_sample(sample: &Sample, worker: &str) -> String {
    let gpu = &sample.gpu;
    let mut lines = vec![
        format!("Qwarz monitor — {} (GPU {})", gpu.name, gpu.index),
        String::new(),
        format!("worker         {worker}"),
        format!("utilization    {} GPU    {} memory bandwidth", fmt_percent(gpu.utilization_gpu), fmt_percent(gpu.utilization_memory)),
        format!("memory         {} / {}  ({})", fmt_mib(gpu.memory_used_mib), fmt_mib(gpu.memory_total_mib), fmt_percent(memory_percent(gpu.memory_used_mib, gpu.memory_total_mib))),
        format!("qwarz          {}", fmt_mib(Some(sample.qwarz_mib))),
    ];
    let owned = sample.processes.iter().filter(|item| item.qwarz).map(|item| format!("{} {} {}", item.name, item.pid, fmt_mib(Some(item.memory_mib)))).collect::<Vec<_>>();
    if !owned.is_empty() {
        lines.push(format!("               {}", owned.join(", ")));
    }
    let noun = if sample.other_processes == 1 { "process" } else { "processes" };
    lines.push(format!("other          {} across {} {noun}", fmt_mib(Some(sample.other_mib)), sample.other_processes));
    lines.push(format!("power          {} / {}    {}", fmt_watts(gpu.power_w), fmt_watts(gpu.power_limit_w), if gpu.pstate.is_empty() { "—" } else { &gpu.pstate }));
    let sm = gpu.sm_clock_mhz.map(|value| format!("{value:.0} MHz")).unwrap_or_else(|| "—".into());
    let memory = gpu.memory_clock_mhz.map(|value| format!("{value:.0} MHz")).unwrap_or_else(|| "—".into());
    lines.push(format!("temperature    {}", fmt_temp(gpu.temperature_c)));
    lines.push(format!("clocks         {sm} SM    {memory} memory"));
    lines.push(String::new());
    lines.push(format!("cpu            {}    load {:.2} {:.2} {:.2}", fmt_percent(sample.host.cpu_percent), sample.host.load[0], sample.host.load[1], sample.host.load[2]));
    lines.push(format!(
        "host memory    {:.1} GiB / {:.1} GiB  ({})",
        sample.host.used_bytes / 1024.0_f64.powi(3),
        sample.host.total_bytes / 1024.0_f64.powi(3),
        fmt_percent(memory_percent(Some(sample.host.used_bytes), Some(sample.host.total_bytes))),
    ));
    if !sample.others.is_empty() {
        lines.push(String::new());
        lines.push("other GPUs".into());
        for item in &sample.others {
            lines.push(format!(
                "  GPU {:<2} {}   {}   {} / {}   {}   {}",
                item.index,
                item.name,
                fmt_percent(item.utilization_gpu),
                fmt_mib(item.memory_used_mib),
                fmt_mib(item.memory_total_mib),
                fmt_temp(item.temperature_c),
                fmt_watts(item.power_w),
            ));
        }
    }
    lines.join("\n")
}

fn series(values: &[f64]) -> Spread {
    Spread {
        n: values.len(),
        median: stats::median(values),
        min: values.iter().copied().reduce(f64::min).unwrap_or(0.0),
        max: values.iter().copied().reduce(f64::max).unwrap_or(0.0),
    }
}

struct Summary {
    samples: usize,
    utilization_gpu: Spread,
    memory_used_mib: Spread,
    qwarz_mib: Spread,
    power_w: Spread,
    temperature_c: Spread,
    cpu_percent: Spread,
}

fn summarize(samples: &[Sample]) -> Summary {
    let collect = |pick: fn(&Sample) -> Option<f64>| samples.iter().filter_map(pick).collect::<Vec<_>>();
    Summary {
        samples: samples.len(),
        utilization_gpu: series(&collect(|sample| sample.gpu.utilization_gpu)),
        memory_used_mib: series(&collect(|sample| sample.gpu.memory_used_mib)),
        qwarz_mib: series(&collect(|sample| Some(sample.qwarz_mib))),
        power_w: series(&collect(|sample| sample.gpu.power_w)),
        temperature_c: series(&collect(|sample| sample.gpu.temperature_c)),
        cpu_percent: series(&collect(|sample| sample.host.cpu_percent)),
    }
}

fn span(spread: &Spread, rendered: String) -> String {
    if spread.n == 0 {
        "—".into()
    } else {
        format!("{rendered} median   min {:.0}   max {:.0}   n={}", spread.min, spread.max, spread.n)
    }
}

fn format_summary(summary: &Summary) -> String {
    [
        String::new(),
        format!("over {} samples", summary.samples),
        format!("utilization    {}", span(&summary.utilization_gpu, fmt_percent(summary.utilization_gpu.median))),
        format!("memory         {}", span(&summary.memory_used_mib, fmt_mib(summary.memory_used_mib.median))),
        format!("qwarz          {}", span(&summary.qwarz_mib, fmt_mib(summary.qwarz_mib.median))),
        format!("power          {}", span(&summary.power_w, fmt_watts(summary.power_w.median))),
        format!("temperature    {}", span(&summary.temperature_c, fmt_temp(summary.temperature_c.median))),
        format!("cpu            {}", span(&summary.cpu_percent, fmt_percent(summary.cpu_percent.median))),
    ].join("\n")
}

fn worker_label(health: Option<&Value>) -> String {
    let Some(health) = health else {
        return "not reachable".into();
    };
    let worker = health.get("worker").cloned().unwrap_or(Value::Null);
    let status = worker.get("status").and_then(Value::as_str).unwrap_or("unknown");
    match worker.get("busy").and_then(Value::as_bool) {
        Some(true) => format!("{status}, generating"),
        Some(false) => format!("{status}, idle"),
        None => status.into(),
    }
}

fn service_pids(health: Option<&Value>) -> Vec<i64> {
    let Some(pid) = health.and_then(|value| value.get("worker")).and_then(|worker| worker.get("pid")) else {
        return Vec::new();
    };
    pid.as_i64().filter(|pid| *pid > 0).or_else(|| pid.as_str().and_then(|text| text.parse().ok()).filter(|pid: &i64| *pid > 0)).into_iter().collect()
}

fn nvidia(kind: &str, fields: &[&str]) -> Result<String, String> {
    let output = Command::new("nvidia-smi")
        .arg(format!("--query-{kind}={}", fields.join(",")))
        .args(["--format=csv,noheader,nounits"])
        .output()
        .map_err(|_| "nvidia-smi is not installed".to_string())?;
    if !output.status.success() {
        let detail = String::from_utf8_lossy(&output.stderr);
        let detail = detail.trim();
        return Err(if detail.is_empty() { "nvidia-smi failed".into() } else { detail.into() });
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

fn cmdline(pid: i64) -> String {
    std::fs::read(format!("/proc/{pid}/cmdline")).map(|bytes| String::from_utf8_lossy(&bytes).replace('\0', " ")).unwrap_or_default()
}

fn health() -> Option<Value> {
    let mut stream = TcpStream::connect_timeout(&"127.0.0.1:8800".parse().ok()?, Duration::from_secs(1)).ok()?;
    stream.set_read_timeout(Some(Duration::from_secs(1))).ok()?;
    stream.write_all(b"GET /health HTTP/1.0\r\nHost: 127.0.0.1:8800\r\n\r\n").ok()?;
    let mut buffer = String::new();
    stream.read_to_string(&mut buffer).ok()?;
    let body = buffer.split_once("\r\n\r\n")?.1;
    serde_json::from_str(body).ok()
}

fn capture(gpu: Option<u32>, previous: Option<&[f64]>, initial_wait: bool) -> Result<(Sample, Option<Value>, Vec<f64>), SetupError> {
    let gpu_text = nvidia("gpu", GPU_QUERY).map_err(SetupError)?;
    let app_text = nvidia("compute-apps", &["gpu_uuid", "pid", "used_memory"]).map_err(SetupError)?;
    let before = match previous {
        Some(counters) => counters.to_vec(),
        None => {
            let counters = parse_cpu(&std::fs::read_to_string("/proc/stat").map_err(|error| SetupError(error.to_string()))?).map_err(SetupError)?;
            if initial_wait {
                std::thread::sleep(Duration::from_millis(200));
            }
            counters
        }
    };
    let current = parse_cpu(&std::fs::read_to_string("/proc/stat").map_err(|error| SetupError(error.to_string()))?).map_err(SetupError)?;
    let meminfo = std::fs::read_to_string("/proc/meminfo").map_err(|error| SetupError(error.to_string()))?;
    let mut total = None;
    let mut available = None;
    for line in meminfo.lines() {
        let Some((key, rest)) = line.split_once(':') else { continue };
        let kib = rest.split_whitespace().next().and_then(|text| text.parse::<f64>().ok());
        if key == "MemTotal" { total = kib.map(|value| value * 1024.0); }
        if key == "MemAvailable" { available = kib.map(|value| value * 1024.0); }
    }
    let (Some(total_bytes), Some(available_bytes)) = (total, available) else {
        return Err(SetupError("MemTotal or MemAvailable is missing from /proc/meminfo".into()));
    };
    let load_text = std::fs::read_to_string("/proc/loadavg").map_err(|error| SetupError(error.to_string()))?;
    let load = load_text.split_whitespace().take(3).map(|part| part.parse::<f64>().unwrap_or(0.0)).collect::<Vec<_>>();
    if load.len() < 3 {
        return Err(SetupError("load average is missing".into()));
    }
    let host = Host {
        cpu_percent: cpu_percent(&before, &current),
        load: [load[0], load[1], load[2]],
        used_bytes: total_bytes - available_bytes,
        total_bytes,
    };
    let apps = parse_apps(&app_text).map_err(SetupError)?;
    let cmdlines = apps.iter().map(|app| (app.pid, cmdline(app.pid))).collect::<Vec<_>>();
    let health = health();
    let sample = assemble(&parse_gpus(&gpu_text).map_err(SetupError)?, &apps, &cmdlines, &service_pids(health.as_ref()), host, gpu).map_err(SetupError)?;
    Ok((sample, health, current))
}

fn sample_json(sample: &Sample) -> Value {
    json!({
        "gpu": {
            "index": sample.gpu.index,
            "name": sample.gpu.name,
            "utilization_gpu": sample.gpu.utilization_gpu,
            "utilization_memory": sample.gpu.utilization_memory,
            "memory_used_mib": sample.gpu.memory_used_mib,
            "memory_total_mib": sample.gpu.memory_total_mib,
            "temperature_c": sample.gpu.temperature_c,
            "power_w": sample.gpu.power_w,
            "power_limit_w": sample.gpu.power_limit_w,
            "pstate": sample.gpu.pstate,
        },
        "qwarz_mib": sample.qwarz_mib,
        "other_mib": sample.other_mib,
        "cpu_percent": sample.host.cpu_percent,
        "load": sample.host.load,
        "processes": sample.processes.iter().map(|item| json!({"pid": item.pid, "name": item.name, "memory_mib": item.memory_mib, "qwarz": item.qwarz})).collect::<Vec<_>>(),
    })
}

fn summary_json(summary: &Summary) -> Value {
    let spread = |item: &Spread| json!({"n": item.n, "median": item.median, "min": item.min, "max": item.max});
    json!({
        "samples": summary.samples,
        "utilization_gpu": spread(&summary.utilization_gpu),
        "memory_used_mib": spread(&summary.memory_used_mib),
        "qwarz_mib": spread(&summary.qwarz_mib),
        "power_w": spread(&summary.power_w),
        "temperature_c": spread(&summary.temperature_c),
        "cpu_percent": spread(&summary.cpu_percent),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (Vec<Gpu>, Vec<App>, Vec<(i64, String)>, Host) {
        let gpus = parse_gpus("0, GPU-5090, NVIDIA GeForce RTX 5090, 37, 12, 29282, 32607, 44, 29.18, 600.00, 1800, 10001, P8\n1, GPU-3090, NVIDIA GeForce RTX 3090 Ti, 0, 0, 4, 24564, 52, 28.55, 450.00, 210, 405, P8\n").unwrap();
        let apps = parse_apps("GPU-5090, 2099, 28692\nGPU-5090, 34136, 60\n").unwrap();
        let cmdlines = vec![
            (2099, "/opt/venv/bin/python -m engine.forward.worker --prefill xqa".into()),
            (34136, "/usr/lib/chromium/chromium --type=gpu-process".into()),
        ];
        let host = Host { cpu_percent: Some(11.5), load: [0.4, 0.5, 0.6], used_bytes: 40.0 * 1024.0_f64.powi(3), total_bytes: 128.0 * 1024.0_f64.powi(3) };
        (gpus, apps, cmdlines, host)
    }

    #[test]
    fn gpu_row_survives_a_comma_inside_the_name() {
        let gpu = parse_gpus("0, GPU-1, NVIDIA, Inc. RTX 5090, 1, 2, 3, 4, 5, 6, 7, 8, 9, P0\n").unwrap().remove(0);
        assert_eq!(gpu.name, "NVIDIA, Inc. RTX 5090");
        assert_eq!(gpu.utilization_gpu, Some(1.0));
        assert_eq!(gpu.pstate, "P0");
        assert!(parse_gpus("0, GPU-1, Name, [N/A], 0, 1, 2, [N/A], 3, 4, 5, 6, P8").unwrap()[0].utilization_gpu.is_none());
    }

    #[test]
    fn qwarz_memory_is_separated_from_other_processes() {
        let (gpus, apps, cmdlines, host) = fixture();
        let sample = assemble(&gpus, &apps, &cmdlines, &[2099], host, None).unwrap();
        assert_eq!(sample.gpu.index, 0);
        assert_eq!(sample.qwarz_mib, 28692.0);
        assert_eq!(sample.other_mib, 60.0);
        assert_eq!(sample.other_processes, 1);
        let text = format_sample(&sample, "ready, idle");
        assert!(text.contains("RTX 5090"));
        assert!(text.contains("28,692 MiB"));
        assert!(text.contains("3090 Ti"));
        assert!(text.contains("ready, idle"));
    }

    #[test]
    fn explicit_gpu_overrides_where_the_worker_sits() {
        let (gpus, apps, cmdlines, host) = fixture();
        let sample = assemble(&gpus, &apps, &cmdlines, &[2099], host, Some(1)).unwrap();
        assert_eq!(sample.gpu.index, 1);
        assert_eq!(sample.qwarz_mib, 0.0);
        let (gpus, apps, cmdlines, host) = fixture();
        assert!(assemble(&gpus, &apps, &cmdlines, &[], host, Some(3)).is_err_and(|error| error.contains("GPU 3")));
    }

    #[test]
    fn cpu_percent_uses_the_gap_between_two_readings() {
        let before = vec![10.0, 0.0, 5.0, 80.0, 5.0, 0.0, 0.0, 0.0];
        let after = vec![20.0, 0.0, 10.0, 85.0, 5.0, 0.0, 0.0, 0.0];
        assert_eq!(cpu_percent(&before, &after), Some(75.0));
        assert_eq!(cpu_percent(&before, &before), None);
    }

    #[test]
    fn several_samples_summarize_min_and_max() {
        let (gpus, apps, cmdlines, host) = fixture();
        let mut first = assemble(&gpus, &apps, &cmdlines, &[2099], host, None).unwrap();
        let mut second = assemble(&gpus, &apps, &cmdlines, &[2099], Host { cpu_percent: Some(40.0), load: [0.0, 0.0, 0.0], used_bytes: 1.0, total_bytes: 2.0 }, None).unwrap();
        second.gpu.utilization_gpu = Some(80.0);
        second.gpu.power_w = Some(400.0);
        second.gpu.temperature_c = Some(70.0);
        first.gpu.utilization_gpu = Some(37.0);
        let summary = summarize(&[first, second]);
        assert_eq!(summary.utilization_gpu.median, Some(58.5));
        assert_eq!(summary.utilization_gpu.min, 37.0);
        assert_eq!(summary.utilization_gpu.max, 80.0);
        let text = format_summary(&summary);
        assert!(text.contains("over 2 samples"));
        assert!(text.contains("min 37"));
        assert!(text.contains("max 80"));
    }
}
