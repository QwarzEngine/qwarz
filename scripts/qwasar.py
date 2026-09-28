#!/usr/bin/env python3
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "state"
PID_FILE = STATE / "server.json"
LOG_FILE = STATE / "server.log"
BINARY = ROOT / "target/release/qwasar-server"
URL = "http://127.0.0.1:8800"
SYSTEMD_UNIT = Path.home() / ".config/systemd/user/qwasar.service"


def systemd_command(command, profile):
    if command not in ("start", "stop", "status", "logs") or SYSTEMD_UNIT.resolve() != ROOT / "integrations/systemd/qwasar.service":
        return None
    if command == "start" and profile != "flash":
        raise ValueError("The installed systemd service pins flash; change its unit explicitly to select another profile")
    if command == "logs":
        return ["journalctl", "--user", "-u", "qwasar.service", "-n", "100", "-f"]
    return ["systemctl", "--user", command, "qwasar.service"]


def cuda_device():
    value = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    return value if value.isdigit() and int(value) >= 0 else "0"


def check_gpu():
    device = cuda_device()
    name = subprocess.check_output(["nvidia-smi", "-i", device, "--query-gpu=name", "--format=csv,noheader"], text=True).strip()
    if "RTX 5090" not in name:
        raise RuntimeError(f"GPU {device} must be RTX 5090, found {name}")
    processes = subprocess.check_output(["nvidia-smi", "-i", device, "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], text=True)
    for row in processes.splitlines():
        if not row.strip():
            continue
        process, memory = row.split(",", 1)
        if not memory.strip().isdigit() or int(memory) >= 512:
            raise RuntimeError(f"GPU {device} is occupied by PID {process.strip()} ({memory.strip()} MiB); stop its owner explicitly first")


def process_start(pid):
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]


def resolved_command(parts):
    resolved = []
    for part in parts:
        try:
            resolved.append(os.fsencode(str(Path(os.fsdecode(part)).resolve())))
        except (OSError, ValueError):
            resolved.append(part)
    return resolved


def same_process(record):
    try:
        pid = int(record["pid"])
        if process_start(pid) != record["start"]:
            return False
        executable = Path(os.readlink(f"/proc/{pid}/exe").removesuffix(" (deleted)"))
        command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        return executable == BINARY or (
            os.fsencode(Path(__file__).resolve()) in resolved_command(command) and b"serve" in command)
    except (OSError, ValueError, KeyError, IndexError):
        return False


def record():
    try:
        return json.loads(PID_FILE.read_text())
    except (OSError, ValueError):
        return {}


def health():
    try:
        with urllib.request.urlopen(URL + "/health", timeout=2) as response:
            return json.load(response)
    except (OSError, ValueError, urllib.error.URLError):
        return None


def serve(profile):
    STATE.mkdir(exist_ok=True, mode=0o700)
    lock = os.open(STATE / "server.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("Qwasar already owns the server lock") from error
    os.set_inheritable(lock, True)
    check_gpu()
    python = os.environ.get("QWASAR_EXLLAMA_PYTHON", str(ROOT.parent / "qwen38-exl3-mia/.venv/bin/python"))
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=cuda_device(), PYTHONPATH=str(ROOT / "src"))
    os.chdir(ROOT)
    os.execve(BINARY, [str(BINARY), "--python", python, "--database", str(STATE / "qwasar.db"), "--prefill", profile], environment)


def start(profile):
    existing = record()
    if same_process(existing):
        wait_ready(existing)
        return
    if health() is not None:
        raise RuntimeError("Port 8800 is already serving another process; refusing to replace it")
    check_gpu()
    subprocess.run(["cargo", "build", "--release", "--locked", "--offline"], cwd=ROOT, check=True)
    STATE.mkdir(exist_ok=True, mode=0o700)
    with LOG_FILE.open("ab") as output:
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "serve", "--prefill", profile], cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    identity = {"pid": child.pid, "start": process_start(child.pid)}
    PID_FILE.write_text(json.dumps(identity) + "\n")
    wait_ready(identity, child)


def wait_ready(identity, child=None):
    for attempt in range(540):
        if child is not None and child.poll() is not None or not same_process(identity):
            raise RuntimeError(f"Qwasar exited; inspect {LOG_FILE}")
        response = health()
        if response and response.get("worker", {}).get("status") == "ready":
            try:
                worker_pid = int(response["worker"]["pid"])
                parent_pid = int(Path(f"/proc/{worker_pid}/stat").read_text().rsplit(")", 1)[1].split()[1])
                if parent_pid == identity["pid"]:
                    print(f"Qwasar ready: {URL}/v1 (PID {identity['pid']})")
                    return
            except (OSError, ValueError, TypeError, KeyError, IndexError):
                pass
        time.sleep(1)
    raise RuntimeError(f"Startup is still pending after 540s; inspect {LOG_FILE} (PID {identity['pid']})")


def stop():
    identity = record()
    if not same_process(identity):
        print("No managed Qwasar process running; no process signalled")
        return
    os.kill(identity["pid"], signal.SIGTERM)
    for attempt in range(100):
        if not same_process(identity):
            print("Qwasar stopped")
            return
        time.sleep(0.1)
    if same_process(identity):
        os.kill(identity["pid"], signal.SIGKILL)
    print("Qwasar stopped after termination timeout")


def main():
    parser = argparse.ArgumentParser(description="Manage the dedicated local RTX 5090 Qwasar service")
    parser.add_argument("command", choices=["start", "serve", "stop", "status", "logs", "wait-ready"])
    parser.add_argument("--prefill", choices=["flash", "baseline", "xqa"], default="flash")
    parser.add_argument("--pid", type=int)
    arguments = parser.parse_args()
    if arguments.command == "wait-ready":
        if arguments.pid is None or arguments.pid <= 0:
            parser.error("wait-ready requires a positive --pid")
        wait_ready({"pid": arguments.pid, "start": process_start(arguments.pid)})
        return
    managed = systemd_command(arguments.command, arguments.prefill)
    if managed is not None:
        raise SystemExit(subprocess.run(managed).returncode)
    if arguments.command in ("start", "stop"):
        STATE.mkdir(exist_ok=True, mode=0o700)
        lifecycle = os.open(STATE / "lifecycle.lock", os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(lifecycle, fcntl.LOCK_EX)
    if arguments.command == "start":
        start(arguments.prefill)
    elif arguments.command == "serve":
        serve(arguments.prefill)
    elif arguments.command == "stop":
        stop()
    elif arguments.command == "status":
        identity = record()
        print(json.dumps({"managed": same_process(identity), "pid": identity.get("pid"), "health": health(), "log": str(LOG_FILE)}))
    else:
        subprocess.run(["tail", "-n", "100", "-f", str(LOG_FILE)], check=True)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"qwasar: {error}", file=sys.stderr)
        sys.exit(1)
