from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import uuid


def save_diagnostic(directory, evidence, max_bytes=1024 * 1024, keep=20):
    if max_bytes < 1024 or keep < 1:
        raise ValueError("invalid diagnostic limits")
    directory = Path(directory)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    identifier = "tool-error-" + uuid.uuid4().hex
    temporary = identifier + ".tmp"
    try:
        os.fchmod(descriptor, 0o700)
        record = {**evidence, "version": 1, "diagnostic_id": identifier,
                  "captured_at": datetime.now(timezone.utc).isoformat()}
        payload = json.dumps(record, ensure_ascii=False, allow_nan=False).encode("utf-8")
        if len(payload) > max_bytes:
            record = {"version": 1, "diagnostic_id": identifier, "captured_at": record["captured_at"],
                      "evidence_omitted": "size_limit", "original_bytes": len(payload),
                      "original_sha256": hashlib.sha256(payload).hexdigest()}
            payload = json.dumps(record).encode("utf-8")
        file_descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
                                  0o600, dir_fd=descriptor)
        with os.fdopen(file_descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, identifier + ".json", src_dir_fd=descriptor, dst_dir_fd=descriptor)
        files = [name for name in os.listdir(descriptor) if re.fullmatch(r"tool-error-[a-f0-9]{32}\.json", name)]
        files.sort(key=lambda name: os.stat(name, dir_fd=descriptor, follow_symlinks=False).st_mtime_ns,
                   reverse=True)
        for name in files[keep:]:
            os.unlink(name, dir_fd=descriptor)
        os.fsync(descriptor)
        return {"diagnostic_id": identifier,
                "diagnostic_status": "size_limit" if "evidence_omitted" in record else "captured"}
    finally:
        try:
            os.unlink(temporary, dir_fd=descriptor)
        except FileNotFoundError:
            pass
        os.close(descriptor)


def replay_diagnostic(path):
    from .parsing import SchemaValidationError, StreamParser

    with Path(path).open('rb') as source:
        payload = source.read(1024 * 1024 + 1)
    if len(payload) > 1024 * 1024:
        raise ValueError('diagnostic exceeds replay size limit')
    record = json.loads(payload)
    if record.get('version') != 1 or record.get('evidence_omitted'):
        raise ValueError('diagnostic has no complete replayable evidence')
    tools = [{'type': 'function', 'function': {'name': name, 'parameters': schema}}
             for name, schema in record['tool_schemas'].items()]
    parser = StreamParser('off', tools, record['response_id'], record.get('tool_choice', 'auto'))
    parser.feed(record['raw_tool_calls'])
    report = {'diagnostic_id': record['diagnostic_id'], 'tool': record['tool'],
              'captured_stage': record['stage'], 'accepted': False,
              'parser_matches_capture': record['parser_sha256'] == hashlib.sha256(
                  Path(__file__).with_name('parsing.py').read_bytes()).hexdigest()}
    validation = None
    parsed_arguments = None
    try:
        parser.finish()
    except ValueError as error:
        report['error_class'] = type(error).__name__
        if isinstance(error, SchemaValidationError):
            validation = {'path': error.path, 'expected_type': error.expected_type,
                          'actual_type': error.actual_type}
        if parser.tool_diagnostic is not None:
            parsed_arguments = parser.tool_diagnostic['parsed_arguments']
    else:
        # Schema validation failures pass through since parser revision with
        # `validation_failures`: the call is delivered but still not "accepted".
        failures = parser.validation_failures
        if failures:
            match = next((failure for failure in failures
                          if failure.get('call_index') == record.get('call_index')), failures[0])
            report['error_class'] = (match.get('error') or {}).get('class')
            validation = match.get('validation')
            parsed_arguments = match.get('parsed_arguments')
        else:
            report['accepted'] = True
    report['validation'] = validation
    report['validation_matches'] = not report['accepted'] and validation == record.get('validation')
    report['parsed_arguments_match'] = (parsed_arguments is not None and
        parsed_arguments == record['parsed_arguments'])
    return report


if __name__ == '__main__':
    import argparse

    arguments = argparse.ArgumentParser(description='Replay a private tool diagnostic on CPU; prints no argument values')
    arguments.add_argument('path', type=Path)
    parsed = arguments.parse_args()
    print(json.dumps(replay_diagnostic(parsed.path), indent=2, ensure_ascii=False))
