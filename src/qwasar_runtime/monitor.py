"""Hardware readings for the machine serving Qwarz.

One sample is a snapshot. Several samples, spaced by ``--interval``, produce
the same readings plus a min / median / max. The GPU in the report is the one
running the worker unless ``--gpu`` says otherwise.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

from .stats import median, parse_duration


GPU_QUERY = (
    "index", "uuid", "name", "utilization.gpu", "utilization.memory",
    "memory.used", "memory.total", "temperature.gpu", "power.draw",
    "power.limit", "clocks.sm", "clocks.mem", "pstate",
)
GPU_KEYS = (
    "index", "uuid", "name", "utilization_gpu", "utilization_memory",
    "memory_used_mib", "memory_total_mib", "temperature_c", "power_w",
    "power_limit_w", "sm_clock_mhz", "memory_clock_mhz", "pstate",
)
CPU_FIELDS = ("user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal")
BUSY_FIELDS = ("user", "nice", "system", "irq", "softirq", "steal")


def _number(text):
    value = text.strip()
    if not value or value.startswith("["):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _split_row(line, expected):
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < expected:
        raise ValueError(f"nvidia-smi row has {len(parts)} fields, expected at least {expected}")
    if len(parts) == expected:
        return parts
    tail = expected - 3
    name = ", ".join(parts[2:len(parts) - tail])
    return parts[:2] + [name] + parts[-tail:]


def parse_gpus(text):
    gpus = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = _split_row(line, len(GPU_KEYS))
        record = dict(zip(GPU_KEYS, fields))
        record["index"] = int(float(record["index"]))
        for key in GPU_KEYS:
            if key in ("index", "uuid", "name", "pstate"):
                continue
            record[key] = _number(record[key])
        gpus.append(record)
    return gpus


def parse_apps(text):
    apps = []
    for line in text.splitlines():
        if not line.strip():
            continue
        uuid, pid, used = _split_row(line, 3)
        apps.append({"uuid": uuid, "pid": int(float(pid)), "memory_mib": _number(used) or 0})
    return apps


def parse_cpu(text):
    for line in text.splitlines():
        if line.startswith("cpu "):
            parts = line.split()
            values = [_number(part) or 0 for part in parts[1:1 + len(CPU_FIELDS)]]
            return dict(zip(CPU_FIELDS, values))
    raise ValueError("cpu counters are missing from /proc/stat")


def cpu_percent(before, after):
    busy = sum(after[field] - before[field] for field in BUSY_FIELDS)
    total = busy + (after["idle"] - before["idle"]) + (after["iowait"] - before["iowait"])
    if total <= 0:
        return None
    return 100 * busy / total


def parse_meminfo(text):
    values = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        if key in ("MemTotal", "MemAvailable"):
            values[key] = float(rest.strip().split()[0]) * 1024
    if "MemTotal" not in values or "MemAvailable" not in values:
        raise ValueError("MemTotal or MemAvailable is missing from /proc/meminfo")
    return {
        "total_bytes": values["MemTotal"],
        "available_bytes": values["MemAvailable"],
        "used_bytes": values["MemTotal"] - values["MemAvailable"],
    }


def parse_loadavg(text):
    parts = text.split()
    if len(parts) < 3:
        raise ValueError("load average is missing")
    return [float(part) for part in parts[:3]]


def is_qwarz(pid, cmdline, service_pids):
    if pid in service_pids:
        return True
    text = cmdline.lower()
    return "qwasar" in text or "qwarz" in text


def short_name(cmdline):
    if "qwasar_runtime.worker" in cmdline:
        return "worker"
    if "qwasar-server" in cmdline:
        return "server"
    executable = cmdline.split(" ", 1)[0]
    name = Path(executable).name
    return name or "process"


def select_gpu(gpus, apps, qwarz_pids, explicit):
    if explicit is not None:
        match = next((gpu for gpu in gpus if gpu["index"] == explicit), None)
        if match is None:
            found = ", ".join(str(gpu["index"]) for gpu in gpus) or "none"
            raise ValueError(f"GPU {explicit} is not present (found {found})")
        return match
    by_uuid = {gpu["uuid"]: gpu for gpu in gpus}
    for app in apps:
        if app["pid"] in qwarz_pids and app["uuid"] in by_uuid:
            return by_uuid[app["uuid"]]
    named = next((gpu for gpu in gpus if "RTX 5090" in gpu["name"]), None)
    if named is not None:
        return named
    if not gpus:
        raise ValueError("nvidia-smi reported no GPUs")
    return gpus[0]


def assemble(gpus, apps, cmdlines, service_pids, host, explicit=None):
    qwarz_pids = set(service_pids)
    for pid, cmdline in cmdlines.items():
        if is_qwarz(pid, cmdline, service_pids):
            qwarz_pids.add(pid)
    gpu = select_gpu(gpus, apps, qwarz_pids, explicit)
    processes = []
    for app in apps:
        if app["uuid"] != gpu["uuid"]:
            continue
        cmdline = cmdlines.get(app["pid"], "")
        processes.append({
            "pid": app["pid"],
            "memory_mib": app["memory_mib"],
            "qwarz": is_qwarz(app["pid"], cmdline, qwarz_pids),
            "name": short_name(cmdline) if cmdline else "pid",
        })
    return {
        "gpu": gpu,
        "processes": processes,
        "qwarz_mib": sum(item["memory_mib"] for item in processes if item["qwarz"]),
        "other_mib": sum(item["memory_mib"] for item in processes if not item["qwarz"]),
        "other_processes": sum(1 for item in processes if not item["qwarz"]),
        "host": host,
        "others": [item for item in gpus if item["index"] != gpu["index"]],
    }


def _fmt_mib(value):
    if value is None:
        return "—"
    return f"{value:,.0f} MiB"


def _fmt_gib(value):
    if value is None:
        return "—"
    return f"{value / (1024 ** 3):.1f} GiB"


def _fmt_percent(value):
    if value is None:
        return "—"
    return f"{value:.0f}%"


def _fmt_watts(value):
    if value is None:
        return "—"
    return f"{value:.0f} W" if value >= 100 else f"{value:.1f} W"


def _fmt_temp(value):
    if value is None:
        return "—"
    return f"{value:.0f} °C"


def _memory_percent(used, total):
    if used is None or not total:
        return None
    return 100 * used / total


def format_sample(sample, worker):
    gpu = sample["gpu"]
    used = gpu["memory_used_mib"]
    total = gpu["memory_total_mib"]
    lines = [
        f"Qwarz monitor — {gpu['name']} (GPU {gpu['index']})",
        "",
        f"worker         {worker}",
        f"utilization    {_fmt_percent(gpu['utilization_gpu'])} GPU    {_fmt_percent(gpu['utilization_memory'])} memory bandwidth",
        f"memory         {_fmt_mib(used)} / {_fmt_mib(total)}  ({_fmt_percent(_memory_percent(used, total))})",
        f"qwarz          {_fmt_mib(sample['qwarz_mib'])}",
    ]
    owned = [item for item in sample["processes"] if item["qwarz"]]
    if owned:
        detail = ", ".join(f"{item['name']} {item['pid']} {_fmt_mib(item['memory_mib'])}" for item in owned)
        lines.append(f"               {detail}")
    noun = "process" if sample["other_processes"] == 1 else "processes"
    lines.append(f"other          {_fmt_mib(sample['other_mib'])} across {sample['other_processes']} {noun}")
    lines.append(f"power          {_fmt_watts(gpu['power_w'])} / {_fmt_watts(gpu['power_limit_w'])}    {gpu['pstate'] or '—'}")
    sm = "—" if gpu["sm_clock_mhz"] is None else f"{gpu['sm_clock_mhz']:.0f} MHz"
    mem = "—" if gpu["memory_clock_mhz"] is None else f"{gpu['memory_clock_mhz']:.0f} MHz"
    lines.append(f"temperature    {_fmt_temp(gpu['temperature_c'])}")
    lines.append(f"clocks         {sm} SM    {mem} memory")
    host = sample["host"]
    lines.extend([
        "",
        f"cpu            {_fmt_percent(host['cpu_percent'])}    load {' '.join(f'{item:.2f}' for item in host['load'])}",
        f"host memory    {_fmt_gib(host['used_bytes'])} / {_fmt_gib(host['total_bytes'])}  ({_fmt_percent(_memory_percent(host['used_bytes'], host['total_bytes']))})",
    ])
    if sample["others"]:
        lines.extend(["", "other GPUs"])
        for item in sample["others"]:
            lines.append(
                f"  GPU {item['index']:<2} {item['name']}   {_fmt_percent(item['utilization_gpu'])}   "
                f"{_fmt_mib(item['memory_used_mib'])} / {_fmt_mib(item['memory_total_mib'])}   "
                f"{_fmt_temp(item['temperature_c'])}   {_fmt_watts(item['power_w'])}"
            )
    return "\n".join(lines)


def _series(samples, keypath):
    values = []
    for sample in samples:
        cursor = sample
        for key in keypath:
            cursor = cursor[key]
        if cursor is not None:
            values.append(cursor)
    if not values:
        return {"n": 0, "median": None, "min": None, "max": None}
    return {"n": len(values), "median": median(values), "min": min(values), "max": max(values)}


def summarize(samples):
    return {
        "samples": len(samples),
        "utilization_gpu": _series(samples, ("gpu", "utilization_gpu")),
        "memory_used_mib": _series(samples, ("gpu", "memory_used_mib")),
        "qwarz_mib": _series(samples, ("qwarz_mib",)),
        "power_w": _series(samples, ("gpu", "power_w")),
        "temperature_c": _series(samples, ("gpu", "temperature_c")),
        "cpu_percent": _series(samples, ("host", "cpu_percent")),
    }


def _span(stats, rendered):
    if not stats["n"]:
        return "—"
    return f"{rendered} median   min {stats['min']:.0f}   max {stats['max']:.0f}   n={stats['n']}"


def format_summary(summary):
    return "\n".join([
        "",
        f"over {summary['samples']} samples",
        f"utilization    {_span(summary['utilization_gpu'], _fmt_percent(summary['utilization_gpu']['median']))}",
        f"memory         {_span(summary['memory_used_mib'], _fmt_mib(summary['memory_used_mib']['median']))}",
        f"qwarz          {_span(summary['qwarz_mib'], _fmt_mib(summary['qwarz_mib']['median']))}",
        f"power          {_span(summary['power_w'], _fmt_watts(summary['power_w']['median']))}",
        f"temperature    {_span(summary['temperature_c'], _fmt_temp(summary['temperature_c']['median']))}",
        f"cpu            {_span(summary['cpu_percent'], _fmt_percent(summary['cpu_percent']['median']))}",
    ])


def worker_label(health):
    if not health:
        return "not reachable"
    worker = health.get("worker") or {}
    status = worker.get("status") or "unknown"
    if worker.get("busy") is True:
        return f"{status}, generating"
    if worker.get("busy") is False:
        return f"{status}, idle"
    return str(status)


def service_pids(health):
    pid = ((health or {}).get("worker") or {}).get("pid")
    if isinstance(pid, int) and pid > 0:
        return {pid}
    if isinstance(pid, str) and pid.isdigit() and int(pid) > 0:
        return {int(pid)}
    return set()


def _nvidia(kind, fields):
    import subprocess
    command = ["nvidia-smi", f"--query-{kind}={','.join(fields)}", "--format=csv,noheader,nounits"]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as error:
        raise ValueError("nvidia-smi is not installed") from error
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "").strip()
        raise ValueError(detail or "nvidia-smi failed") from error
    return completed.stdout


def _cmdline(pid):
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\0", b" ").decode("utf-8", "replace")


def _health():
    import urllib.request
    try:
        with urllib.request.urlopen("http://127.0.0.1:8800/health", timeout=1) as response:
            return json.load(response)
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def capture(gpu_index=None, previous_cpu=None, cpu_wait=0.2):
    """One reading from this machine. The CPU figure is the gap since ``previous_cpu``."""
    gpu_text = _nvidia("gpu", GPU_QUERY)
    app_text = _nvidia("compute-apps", ("gpu_uuid", "pid", "used_memory"))
    if previous_cpu is None:
        previous_cpu = parse_cpu(Path("/proc/stat").read_text())
        if cpu_wait:
            time.sleep(cpu_wait)
    current_cpu = parse_cpu(Path("/proc/stat").read_text())
    host = parse_meminfo(Path("/proc/meminfo").read_text())
    host["load"] = parse_loadavg(Path("/proc/loadavg").read_text())
    host["cpu_percent"] = cpu_percent(previous_cpu, current_cpu)
    apps = parse_apps(app_text)
    health = _health()
    sample = assemble(
        parse_gpus(gpu_text),
        apps,
        {app["pid"]: _cmdline(app["pid"]) for app in apps},
        service_pids(health),
        host,
        gpu_index,
    )
    return sample, health, current_cpu


def run(samples=1, interval=None, gpu=None, as_json=False, sleep=time.sleep, take=None):
    """Print ``samples`` readings. ``samples`` None repeats until interrupted."""
    if samples is not None and samples < 1:
        raise ValueError("--samples must be at least 1")
    if interval is not None and interval < 0.2:
        raise ValueError("--interval must be at least 0.2 seconds")
    if samples is not None and samples > 1 and interval is None:
        interval = 1.0
    taken = []
    health = None
    previous = None

    def one():
        nonlocal health, previous
        if take is None:
            sample, health_now, previous_cpu = capture(gpu, previous_cpu=previous, cpu_wait=0.2)
        else:
            sample, health_now, previous_cpu = take(previous)
        health = health_now
        previous = previous_cpu
        return sample

    def emit(sample, index):
        if as_json and samples is None:
            print(json.dumps(sample), flush=True)
        elif not as_json:
            if index:
                print("", flush=True)
            print(format_sample(sample, worker_label(health)), flush=True)

    try:
        while samples is None or len(taken) < samples:
            sample = one()
            taken.append(sample)
            emit(sample, len(taken) - 1)
            if samples is not None and len(taken) >= samples:
                break
            sleep(interval or 1)
    except KeyboardInterrupt:
        if not taken:
            raise
    summary = summarize(taken) if len(taken) > 1 else None
    if as_json and samples is not None:
        print(json.dumps({"samples": taken, "summary": summary, "worker": worker_label(health)}, indent=2))
    elif summary and not as_json:
        print(format_summary(summary), flush=True)
    return taken


def parse_interval(text):
    seconds = parse_duration(text)
    if seconds < 0.2:
        raise ValueError("--interval must be at least 0.2 seconds")
    return seconds
