#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8800"
MODEL = "qwasar-qwen38-27b"


def request(endpoint, body=None):
    data = json.dumps(body).encode() if body is not None else None
    return urllib.request.urlopen(urllib.request.Request(BASE + endpoint, data=data,
        headers={"Content-Type": "application/json"}), timeout=600)


def fetch(endpoint, body=None):
    with request(endpoint, body) as response:
        return json.load(response)


def chat(messages, **options):
    return {"model": MODEL, "messages": messages, "max_tokens": 128,
            "reasoning_effort": "off", "temperature": 0, **options}


def snapshot(response_id):
    with sqlite3.connect(f"file:{ROOT / 'state/qwasar.db'}?mode=ro", uri=True) as database:
        row = database.execute("SELECT snapshot FROM responses WHERE id=?", (response_id,)).fetchone()
        return json.loads(row[0]) if row and row[0] else None


def main():
    parser = argparse.ArgumentParser(description="Real-model service acceptance checks; requires the managed local service")
    parser.add_argument("--restart", action="store_true", help="also restart this repository's managed service and verify durable continuation")
    parser.add_argument("--output", type=Path, default=ROOT / "results/v1-service")
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    checks = {}
    def save(name, value):
        checks[name] = value
        (arguments.output / "evidence.json").write_text(json.dumps(checks, indent=2) + "\n")

    config = fetch("/config")
    assert config["status"] == "ready" and not config["config"]["fake"], config
    save("config", config)
    notes = "\n".join(f"config_{index:03d}: value_{index:03d}" for index in range(512))
    messages = [{"role": "system", "content": "Follow instructions precisely."},
                {"role": "user", "content": f"Reference notes:\n{notes}\nRespond only READY."}]
    first = fetch("/v1/chat/completions", chat(messages))
    assert first["status"] == "completed" and first["choices"][0]["message"]["content"].strip() == "READY", first
    save("first", first)
    messages += [first["choices"][0]["message"], {"role": "user", "content": "What is config_042? Reply with its value only."}]
    second = fetch("/v1/chat/completions", chat(messages))
    assert second["status"] == "completed" and "value_042" in second["choices"][0]["message"]["content"], second
    prior = snapshot(first["id"])
    current = snapshot(second["id"])
    assert current["tape"][:len(prior["tape"])] == prior["tape"]
    assert second["qwasar_metrics"]["cached_tokens"] > 0
    save("continuation", {"response": second, "exact_tape_prefix": True})
    changed = [{"role": "system", "content": "You are a precise coding assistant."}] + messages[1:]
    branch = fetch("/v1/chat/completions", chat(changed))
    assert branch["status"] == "completed"
    assert snapshot(branch["id"])["segments"][0]["tokens"] == prior["segments"][0]["tokens"]
    save("changed_header", {"response": branch, "exact_generated_segment": True})
    with request("/v1/chat/completions", chat([{"role": "user", "content": "Write a detailed 300-line Python implementation of a database with tests."}],
                max_tokens=4096, reasoning_effort="medium", stream=True)) as response:
        for line in response:
            if line.startswith(b"data: "):
                initial = json.loads(line[6:])
                break
        try:
            fetch("/v1/chat/completions", chat([{"role": "user", "content": "hello"}]))
            raise AssertionError("busy request unexpectedly admitted")
        except urllib.error.HTTPError as error:
            assert error.code == 409
        fetch(f"/v1/responses/{initial['id']}/cancel", {})
        list(response)
    cancelled = fetch(f"/v1/responses/{initial['id']}")
    assert cancelled["status"] == "cancelled" and snapshot(initial["id"]) is None
    save("cancelled", cancelled)
    recovery = fetch("/v1/chat/completions", chat([{"role": "user", "content": "Reply only RECOVERED."}]))
    assert recovery["status"] == "completed" and "RECOVERED" in recovery["choices"][0]["message"]["content"]
    save("recovery", recovery)
    try:
        fetch("/v1/chat/completions", chat([{"role": "user", "content": "x " * 270000}], max_tokens=16))
        raise AssertionError("over-budget prompt unexpectedly admitted")
    except urllib.error.HTTPError as error:
        assert error.code in (400, 422)
        save("budget_rejection", {"http_status": error.code, "error": json.load(error)})
    response = fetch("/v1/responses", {"model": MODEL, "input": "Remember the word AMBER and reply AMBER.", "max_output_tokens": 128, "reasoning_effort": "off", "temperature": 0})
    assert response["status"] == "completed"
    assert fetch("/v1/responses/" + response["id"]) == response
    save("responses", response)
    if arguments.restart:
        before = snapshot(response["id"])
        subprocess.run([sys.executable, str(ROOT / "scripts/qwasar.py"), "stop"], check=True)
        subprocess.run([sys.executable, str(ROOT / "scripts/qwasar.py"), "start"], check=True)
        assert snapshot(response["id"]) == before
        assert fetch("/v1/responses/" + response["id"]) == response
        after = fetch("/v1/responses", {"model": MODEL, "previous_response_id": response["id"], "input": "What word did I ask you to remember? Reply with that word only.", "max_output_tokens": 128, "reasoning_effort": "off", "temperature": 0})
        assert after["status"] == "completed" and "AMBER" in json.dumps(after["output"])
        save("restart", {"durable_snapshot_unchanged": True, "response": after})
    save("passed", True)
    print(json.dumps({"passed": True, "checks": list(checks), "evidence": str(arguments.output / "evidence.json")}, indent=2))


if __name__ == "__main__":
    main()
