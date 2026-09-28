import json

import pytest

from qwasar_runtime.monitor import (
    assemble,
    cpu_percent,
    format_sample,
    format_summary,
    parse_apps,
    parse_gpus,
    parse_interval,
    run,
    summarize,
    worker_label,
)


GPUS = """\
0, GPU-5090, NVIDIA GeForce RTX 5090, 37, 12, 29282, 32607, 44, 29.18, 600.00, 1800, 10001, P8
1, GPU-3090, NVIDIA GeForce RTX 3090 Ti, 0, 0, 4, 24564, 52, 28.55, 450.00, 210, 405, P8
"""
APPS = """\
GPU-5090, 2099, 28692
GPU-5090, 34136, 60
"""
CMDLINES = {
    2099: "/opt/venv/bin/python -m qwasar_runtime.worker --prefill xqa",
    34136: "/usr/lib/chromium/chromium --type=gpu-process",
}
HOST = {
    "cpu_percent": 11.5,
    "load": [0.4, 0.5, 0.6],
    "used_bytes": 40 * 1024 ** 3,
    "total_bytes": 128 * 1024 ** 3,
    "available_bytes": 88 * 1024 ** 3,
}


def _sample():
    return assemble(parse_gpus(GPUS), parse_apps(APPS), CMDLINES, {2099}, HOST)


def test_gpu_row_survives_a_comma_inside_the_name():
    text = "0, GPU-1, NVIDIA, Inc. RTX 5090, 1, 2, 3, 4, 5, 6, 7, 8, 9, P0\n"
    gpu = parse_gpus(text)[0]
    assert gpu["name"] == "NVIDIA, Inc. RTX 5090"
    assert gpu["utilization_gpu"] == 1
    assert gpu["pstate"] == "P0"
    assert parse_gpus("0, GPU-1, Name, [N/A], 0, 1, 2, [N/A], 3, 4, 5, 6, P8")[0]["utilization_gpu"] is None


def test_qwarz_memory_is_separated_from_other_processes_on_its_gpu():
    sample = _sample()
    assert sample["gpu"]["index"] == 0
    assert sample["qwarz_mib"] == 28692
    assert sample["other_mib"] == 60
    assert sample["other_processes"] == 1
    assert [item["name"] for item in sample["processes"] if item["qwarz"]] == ["worker"]
    text = format_sample(sample, "ready, idle")
    assert "RTX 5090" in text
    assert "28,692 MiB" in text
    assert "3090 Ti" in text
    assert "ready, idle" in text


def test_explicit_gpu_overrides_where_the_worker_sits():
    sample = assemble(parse_gpus(GPUS), parse_apps(APPS), CMDLINES, {2099}, HOST, explicit=1)
    assert sample["gpu"]["index"] == 1
    assert sample["qwarz_mib"] == 0
    with pytest.raises(ValueError, match="GPU 3"):
        assemble(parse_gpus(GPUS), parse_apps(APPS), CMDLINES, set(), HOST, explicit=3)


def test_cpu_percent_uses_the_gap_between_two_readings():
    before = {"user": 10, "nice": 0, "system": 5, "idle": 80, "iowait": 5, "irq": 0, "softirq": 0, "steal": 0}
    after = {"user": 20, "nice": 0, "system": 10, "idle": 85, "iowait": 5, "irq": 0, "softirq": 0, "steal": 0}
    assert cpu_percent(before, after) == pytest.approx(75)
    assert cpu_percent(before, before) is None


def test_several_samples_summarize_min_median_max():
    first = _sample()
    second = _sample()
    second["gpu"] = dict(second["gpu"], utilization_gpu=80, power_w=400, temperature_c=70)
    second["host"] = dict(second["host"], cpu_percent=40)
    summary = summarize([first, second])
    assert summary["utilization_gpu"]["median"] == 58.5
    assert summary["utilization_gpu"]["min"] == 37
    assert summary["utilization_gpu"]["max"] == 80
    text = format_summary(summary)
    assert "over 2 samples" in text
    assert "min 37" in text
    assert "max 80" in text


def test_run_stops_after_the_requested_samples_and_can_emit_json(capsys):
    def take(previous):
        sample = _sample()
        sample["gpu"] = dict(sample["gpu"], utilization_gpu=0 if previous is None else 10)
        return sample, {"worker": {"status": "ready", "busy": False, "pid": 2099}}, {"idle": 1}

    captured = []
    taken = run(samples=2, interval=0.5, as_json=True, sleep=captured.append, take=take)
    payload = json.loads(capsys.readouterr().out)
    assert len(taken) == 2
    assert captured == [0.5]
    assert payload["summary"]["samples"] == 2
    assert payload["worker"] == "ready, idle"
    assert payload["samples"][1]["gpu"]["utilization_gpu"] == 10
    assert worker_label({"worker": {"status": "ready", "busy": True}}) == "ready, generating"
    assert worker_label(None) == "not reachable"
    with pytest.raises(ValueError):
        parse_interval("0.1s")
    assert parse_interval("1s") == 1
