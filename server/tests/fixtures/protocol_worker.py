#!/usr/bin/env python3
import json
import sys
import time


def emit(event):
    print(json.dumps(event), flush=True)


mode = sys.argv[sys.argv.index("--model") + 1]
emit({"type": "ready", "protocol": 1, "config": {"fake": True}})
if mode == "input-stall":
    while True:
        time.sleep(1)
for line in sys.stdin:
    command = json.loads(line)
    if command["op"] == "shutdown":
        break
    if command["op"] != "generate":
        continue
    response_id = command["id"]
    latest = command["request"]["messages"][-1]["content"]
    if mode != "prefill-stall" or latest != "stall":
        emit({"type": "started", "id": response_id})
    if mode in ("ignore-cancel", "prefill-stall") and latest == "stall":
        while True:
            time.sleep(1)
    count = 512 if mode == "burst" else 1
    for chunk_index in range(count):
        emit({"type": "delta", "id": response_id, "channel": "content", "text": "x"})
    message = {"role": "assistant", "content": "x" * count, "reasoning_content": "", "tool_calls": []}
    emit({"type": "terminal", "id": response_id, "status": "completed", "message": message,
          "usage": {"prompt_tokens": 1, "completion_tokens": count, "total_tokens": count + 1},
          "metrics": {}, "error": None,
          "snapshot": {"messages": [*command["request"]["messages"], message], "tape": [1, 2]}})
