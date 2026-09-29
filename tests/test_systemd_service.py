import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[1]
UNIT = ROOT / "integrations/systemd/qwasar.service"


def launcher():
    spec = importlib.util.spec_from_file_location("launcher", ROOT / "scripts/qwasar.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unit_uses_foreground_pinned_profile_and_waits_until_ready():
    assert UNIT.exists(), "systemd unit missing"
    unit = UNIT.read_text()
    for expected in ("Type=exec", "qwasar.py serve --prefill xqa", "qwasar.py wait-ready --pid ${MAINPID}",
                     "Qwen3.8-27B-EXL3-5.0bpw", "CUDA_VISIBLE_DEVICES=0", "Restart=on-failure",
                     "KillMode=mixed", "WantedBy=default.target",
                     "WorkingDirectory=%h/Documents/llm/qwarz",
                     "ExecStart=/usr/bin/python3 %h/Documents/llm/qwarz/scripts/qwasar.py serve --prefill xqa",
                     "ExecStartPost=/usr/bin/python3 %h/Documents/llm/qwarz/scripts/qwasar.py wait-ready --pid ${MAINPID}"):
        assert expected in unit
    assert "Documents/llm/qwasar" not in unit
    assert "CUDA_VISIBLE_DEVICES=1" not in unit
    assert "cargo" not in unit


def test_readiness_rejects_another_supervisors_worker():
    module = launcher()
    response = {"worker": {"status": "ready", "pid": 1234}}
    with patch.object(module, "same_process", return_value=True), patch.object(module, "health", return_value=response), patch.object(module.Path, "read_text", return_value="1234 (python) S 9999 0"), patch.object(module.time, "sleep"):
        with pytest.raises(RuntimeError, match="pending"):
            module.wait_ready({"pid": 5678, "start": "123"})


def test_readiness_accepts_only_the_owned_worker():
    module = launcher()
    response = {"worker": {"status": "ready", "pid": 1234}}
    with patch.object(module, "same_process", return_value=True), patch.object(module, "health", return_value=response), patch.object(module.Path, "read_text", return_value="1234 (python) S 5678 0"):
        module.wait_ready({"pid": 5678, "start": "123"})


def test_managed_commands_use_systemd_after_installation(tmp_path):
    module = launcher()
    module.SYSTEMD_UNIT = tmp_path / "qwasar.service"
    assert hasattr(module, "systemd_command"), "systemd routing missing"
    assert module.systemd_command("start", "flash") is None
    module.SYSTEMD_UNIT.symlink_to(UNIT)
    assert module.systemd_command("start", "flash") == ["systemctl", "--user", "start", "qwasar.service"]
    assert module.systemd_command("stop", "flash") == ["systemctl", "--user", "stop", "qwasar.service"]
    assert module.systemd_command("logs", "flash")[:3] == ["journalctl", "--user", "-u"]
    assert module.systemd_command("serve", "flash") is None
    assert module.systemd_command("wait-ready", "flash") is None
    with pytest.raises(ValueError, match="flash"):
        module.systemd_command("start", "baseline")


def test_installed_user_unit_points_at_the_qwarz_tree():
    installed = Path.home() / ".config/systemd/user/qwasar.service"
    assert installed.exists()
    text = installed.read_text()
    assert "Documents/llm/qwarz" in text
    assert "Documents/llm/qwasar" not in text
    assert "CUDA_VISIBLE_DEVICES=0" in text
    assert "CUDA_VISIBLE_DEVICES=1" not in text
    assert "qwasar.py serve --prefill xqa" in text


def test_server_spawns_the_resident_engine_and_returns_its_token_ids(tmp_path):
    import json
    import os
    import socket
    import subprocess
    import threading
    import time
    import urllib.error
    import urllib.request

    binary = ROOT / "target/release/qwasar-server"
    assert binary.is_file(), "cargo build --release -p qwasar-server before this test"
    wrapper = tmp_path / "python"
    record = tmp_path / "argv"
    wrapper.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys, time\n"
        "open(os.environ['QWARZ_SPAWN_RECORD'], 'w').write('\\0'.join(sys.argv))\n"
        "sys.stdout.write(json.dumps({'type':'ready','protocol':1,'config':{\n"
        "    'model':'qwasar-qwen38-27b','context_size':1024,'vision':True,'streams':1,\n"
        "    'engine':'q38','linked':False,'draft_method':'mtp','draft_tokens':6,\n"
        "    'promotion_allowed':False}}) + '\\n')\n"
        "sys.stdout.flush()\n"
        "for line in sys.stdin:\n"
        "    command = json.loads(line)\n"
        "    if command.get('op') == 'shutdown':\n"
        "        break\n"
        "    if command.get('op') != 'generate':\n"
        "        continue\n"
        "    request = command.get('request') or {}\n"
        "    ids = request.get('ids') or []\n"
        "    max_new = int(request.get('max_new') or 1)\n"
        "    time.sleep(1.5)\n"
        "    tokens = [int(token) + 1 for token in ids[:max_new]]\n"
        "    sys.stdout.write(json.dumps({\n"
        "        'type':'terminal','id':command['id'],'status':'completed',\n"
        "        'message':{'role':'assistant','content':'','reasoning_content':'','tool_calls':[]},\n"
        "        'usage':{'prompt_tokens':len(ids),'completion_tokens':len(tokens),\n"
        "                 'total_tokens':len(ids)+len(tokens),\n"
        "                 'prompt_tokens_details':{'cached_tokens':0}},\n"
        "        'metrics':{},'snapshot':{'version':1,'messages':[\n"
        "            {'role':'user','content':'ids'},\n"
        "            {'role':'assistant','content':'','reasoning_content':'','tool_calls':[]}],\n"
        "            'tape':ids},'error':None,'token_ids':tokens,\n"
        "    }) + '\\n')\n"
        "    sys.stdout.flush()\n"
    )
    wrapper.chmod(0o755)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    log = (tmp_path / "server.log").open("ab")
    process = subprocess.Popen(
        [str(binary), "--port", str(port), "--database", str(tmp_path / "sessions.db"),
         "--python", str(wrapper), "--context-size", "1024", "--prefill", "xqa", "--model", "unused"],
        cwd=ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "0", "QWARZ_SPAWN_RECORD": str(record)},
        stdout=log, stderr=log,
    )
    base = f"http://127.0.0.1:{port}"

    def call(path, body=None):
        data = None if body is None else json.dumps(body).encode()
        return urllib.request.urlopen(urllib.request.Request(
            base + path, data=data, headers={"Content-Type": "application/json"}), timeout=20)

    ids = [1000, 1001, 1002]
    payload = {"model": "qwasar-qwen38-27b", "messages": [{"role": "user", "content": "ids"}],
               "ids": ids, "max_new": 2, "temperature": 0,
               "chat_template_kwargs": {"enable_thinking": False}}
    try:
        deadline = time.monotonic() + 15
        ready = False
        while time.monotonic() < deadline and process.poll() is None:
            try:
                with call("/health") as response:
                    health = json.load(response)
                ready = health.get("worker", {}).get("status") == "ready"
                if ready:
                    assert health["worker"]["config"]["streams"] == 1
                    assert health["worker"]["config"]["vision"] is True
                    assert health["worker"]["config"]["draft_tokens"] == 6
                    assert health["worker"]["config"]["promotion_allowed"] is False
                    break
            except (OSError, urllib.error.URLError, json.JSONDecodeError):
                pass
            time.sleep(0.05)
        assert ready, (tmp_path / "server.log").read_text(errors="replace")
        spawned = record.read_bytes().split(b"\0")
        assert b"engine.forward.worker" in spawned
        assert not any(b"qwasar_runtime.worker" in part for part in spawned)
        holder = {}

        def first():
            try:
                with call("/v1/chat/completions", payload) as response:
                    holder["body"] = json.load(response)
            except Exception as error:
                holder["error"] = error

        thread = threading.Thread(target=first)
        thread.start()
        busy = False
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                with call("/health") as response:
                    busy = json.load(response)["worker"]["busy"] is True
                if busy:
                    break
            except (OSError, urllib.error.URLError, json.JSONDecodeError, KeyError):
                pass
            time.sleep(0.05)
        assert busy, (tmp_path / "server.log").read_text(errors="replace")
        with pytest.raises(urllib.error.HTTPError) as caught:
            call("/v1/chat/completions", payload)
        assert caught.value.code == 409
        assert b"runtime_busy" in caught.value.read()
        print("OBSERVATION overlapping generate is runtime_busy HTTP 409")
        thread.join(timeout=20)
        assert "error" not in holder, holder.get("error")
        assert holder["body"]["token_ids"] == [1001, 1002]
        assert holder["body"]["status"] == "completed"
        assert "engine_not_linked" not in json.dumps(holder["body"])
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        log.close()
