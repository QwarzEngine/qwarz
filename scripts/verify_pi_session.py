#!/usr/bin/env python3
import json
import os
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PI = Path.home() / ".local/share/mise/installs/pi/0.84.4/pi/pi"


def main():
    output = ROOT / "results/pi-v1-session"
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="qwasar-pi-session-") as temporary:
        directory = Path(temporary)
        config = directory / "config"
        config.mkdir()
        (config / "models.json").write_bytes((ROOT / "integrations/pi/models.json").read_bytes())
        environment = dict(os.environ, PI_CODING_AGENT_DIR=str(config), PI_OFFLINE="1")
        command = [str(PI), "--provider", "qwasar", "--model", "qwasar-qwen38-27b", "--thinking", "medium",
                   "--mode", "json", "--session-dir", str(directory / "sessions"), "--no-tools",
                   "--no-extensions", "--no-skills", "--no-prompt-templates", "--no-themes", "--no-context-files", "--no-approve"]
        turns = []
        for index, prompt in enumerate(["Remember the codeword HAZEL for our next turn. Reply only HAZEL.",
                                        "What codeword did I ask you to remember? Reply with that word only."]):
            result = subprocess.run(command + (["--continue"] if index else []) + [prompt], cwd=directory,
                                    env=environment, text=True, capture_output=True, timeout=120)
            (output / f"turn-{index + 1}.jsonl").write_text(result.stdout)
            (output / f"turn-{index + 1}.stderr.log").write_text(result.stderr)
            assert result.returncode == 0, result.stderr
            events = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
            messages = [event["message"] for event in events if event.get("type") == "message_end" and event["message"].get("role") == "assistant"]
            assert messages and all(message.get("stopReason") != "error" for message in messages), messages
            text = "".join(part.get("text", "") for part in messages[-1]["content"] if part["type"] == "text")
            assert text.strip() == "HAZEL", text
            turns.append({"answer": text, "usage": messages[-1].get("usage")})
        session_files = list((directory / "sessions").glob("**/*.jsonl"))
        assert len(session_files) == 1, session_files
        (output / "session.jsonl").write_bytes(session_files[0].read_bytes())
        summary = {"passed": True, "turns": turns, "resumed_in_new_pi_process": True, "session_files": len(session_files)}
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
