from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from types import SimpleNamespace

from . import diagnostics
from .parsing import SchemaValidationError, StreamParser, canonical, message_key, validate_messages, validate_tools
from .rendering import Prompt, encode, render


class RequestError(ValueError):
    def __init__(self, message, code="invalid_request", http_status=400):
        super().__init__(message)
        self.code, self.http_status = code, http_status


class FatalRuntimeError(RuntimeError):
    pass


def empty_message():
    return {"role": "assistant", "content": "", "reasoning_content": "", "tool_calls": []}


def terminal_error(response_id, error):
    return {"type": "terminal", "id": response_id, "status": "failed", "message": empty_message(),
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                      "prompt_tokens_details": {"cached_tokens": 0}},
            "metrics": {}, "snapshot": None, "error": {"code": getattr(error, "code", "invalid_request"),
            "message": str(error), "http_status": getattr(error, "http_status", 400)}}


class Engine:
    def __init__(self, backend, context_size=262144, diagnostic_directory=None):
        if type(context_size) is not int or not 0 < context_size <= 262144:
            raise ValueError("context_size must be between 1 and 262144")
        self.backend, self.context_size = backend, context_size
        self.diagnostic_directory = diagnostic_directory
        self.parser_sha256 = hashlib.sha256(Path(__file__).with_name("parsing.py").read_bytes()).hexdigest()
        self.config = {"model": "qwasar-qwen38-27b", "context_size": context_size,
                       "runtime_identity": backend.identity, **backend.config,
                       "tool_diagnostics_enabled": diagnostic_directory is not None}

    def prepare(self, request, parent):
        if not isinstance(request, dict):
            raise RequestError("request must be an object")
        request = dict(request)
        tools = request.setdefault("tools", [])
        functions = validate_tools(tools)
        thinking = request.setdefault("thinking", "medium")
        if thinking not in ("off", "low", "medium", "xhigh"):
            raise RequestError("thinking must be off, low, medium, or xhigh")
        maximum = request.setdefault("max_tokens", 4096)
        if type(maximum) is not int or maximum <= 0:
            raise RequestError("max_tokens must be a positive integer")
        for name, default, low, high in (("temperature", 0.7 if thinking == "off" else 1.0, 0, 2),
                                        ("top_p", 0.8 if thinking == "off" else 0.95, 0, 1)):
            value = request.setdefault(name, default)
            if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high or name == "top_p" and value == 0:
                raise RequestError(f"invalid {name}")
        seed = request.setdefault("seed", 42)
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise RequestError("seed must be a nonnegative 63-bit integer")
        choice = request.setdefault("tool_choice", "auto")
        if isinstance(choice, dict):
            function = choice.get("function")
            if choice.get("type") != "function" or not isinstance(function, dict) or function.get("name") not in functions:
                raise RequestError("tool_choice must name a declared function")
        elif choice not in ("auto", "none", "required"):
            raise RequestError("unsupported tool_choice")
        if choice == "required" and not functions:
            raise RequestError("required tool_choice needs tools")
        messages = validate_messages(request.get("messages"), tools)
        request["messages"] = messages
        segments = []
        header = canonical({"tools": tools, "thinking": thinking, "tool_choice": choice})
        if parent is not None:
            if not isinstance(parent, dict) or parent.get("version") != 1 or parent.get("runtime_identity") != self.backend.identity:
                raise RequestError("parent snapshot belongs to a different runtime")
            segments = parent.get("segments", [])
            if not isinstance(segments, list) or any(not isinstance(segment.get("tokens"), list)
                    or not segment["tokens"] or any(type(token) is not int or token < 0 for token in segment["tokens"])
                    for segment in segments):
                raise RequestError("invalid parent token segments")
        tokens, assistant_start, retained = render(self.backend.tokenizer, messages,
            tools if choice != "none" else [], thinking, choice, segments)
        unmatched = list(retained)
        for message in messages:
            if message["role"] == "assistant":
                previous = next((segment for segment in unmatched
                    if message_key(segment["message"]) == message_key(message)), None)
                if previous is not None:
                    unmatched.remove(previous)
                    message["reasoning_content"] = previous["message"].get("reasoning_content", "")
        if parent and parent.get("header") == header:
            prior = parent.get("messages", [])
            if len(messages) > len(prior) and all(message_key(previous) == message_key(current)
                    for previous, current in zip(prior, messages)):
                tape = parent.get("tape", [])
                if tokens[:len(tape)] != tape:
                    raise RequestError("append-only reconstruction changed exact parent tape")
        reserve = max(16, self.backend.draft_tokens + 1)
        if len(tokens) + maximum + reserve > self.context_size:
            raise RequestError(f"prompt ({len(tokens)}) + max_tokens ({maximum}) + reserve ({reserve}) exceeds context_size ({self.context_size})",
                               "context_length_exceeded")
        return Prompt(tokens, assistant_start, messages, retained, request, header)

    def generate(self, response_id, request, parent, cancellation):
        started = time.perf_counter()
        try:
            prompt = self.prepare(request, parent)
        except (ValueError, TypeError, KeyError, IndexError) as error:
            yield terminal_error(response_id, error)
            return
        yield {"type": "started", "id": response_id, "prompt_tokens": len(prompt.tokens)}
        parser = StreamParser(prompt.request["thinking"], prompt.request["tools"], response_id,
                              prompt.request["tool_choice"])
        first = first_content = last_emitted = None
        first_batch = emitted = requeues = 0
        streamed_characters = {"content": 0, "reasoning": 0}
        final = None
        status = "cancelled" if cancellation.is_set() else "completed"
        sequence = []
        serial = None
        try:
            if status != "cancelled":
                serial = self.backend.start(prompt.tokens, prompt.request)
                with self.backend.tuning():
                    while self.backend.remaining():
                        if cancellation.is_set():
                            status = "cancelled"
                            break
                        events = self.backend.iterate()
                        now = time.perf_counter()
                        count = sum(len(event.get("token_ids", [])) for event in events)
                        if count and first is None:
                            first, first_batch = now, count
                        if count:
                            last_emitted = now
                        emitted += count
                        for event in events:
                            requeues += int(bool(event.get("requeue")))
                            for channel, text in parser.feed(event.get("text", "")):
                                streamed_characters[channel] += len(text)
                                if channel == "content" and text.strip() and first_content is None:
                                    first_content = now
                                yield {"type": "delta", "id": response_id, "channel": channel, "text": text}
                            if event.get("eos"):
                                final = event
                if cancellation.is_set():
                    status = "cancelled"
            if status == "cancelled":
                self.backend.cancel_reset(serial)
            else:
                if final is None:
                    raise RuntimeError("generator drained without a terminal event")
                sequence = self.backend.sequence(final)
                if sequence[:len(prompt.tokens)] != prompt.tokens:
                    raise RuntimeError("generator changed the exact prompt token tape")
                status = "incomplete" if final.get("eos_reason") == "max_new_tokens" else "completed"
                if status == "completed" and sequence[-1] != self.backend.end_token:
                    raise ValueError("generation has no actual native im_end terminator")
            message = parser.finish(complete=status == "completed")
            for channel, field in (("content", "content"), ("reasoning", "reasoning_content")):
                tail = message[field][streamed_characters[channel]:]
                if tail:
                    yield {"type": "delta", "id": response_id, "channel": channel, "text": tail}
        except Exception as error:
            if isinstance(error, FatalRuntimeError) or isinstance(error, RuntimeError) and any(
                    marker in str(error).lower() for marker in ("cuda", "device-side", "illegal memory", "cublas")):
                raise FatalRuntimeError(str(error)) from error
            try:
                self.backend.cancel_reset(serial)
            except Exception as reset_error:
                raise FatalRuntimeError(f"runtime reset failed: {reset_error}") from reset_error
            terminal = terminal_error(response_id, RequestError(str(error), "generation_failed", 500))
            if parser.tool_diagnostic is not None and self.diagnostic_directory is not None:
                evidence = {**parser.tool_diagnostic, "response_id": response_id,
                    "runtime_identity": self.backend.identity, "parser_sha256": self.parser_sha256,
                    "thinking": prompt.request["thinking"], "tool_choice": prompt.request["tool_choice"],
                    "error": {"class": type(error).__name__, "message": str(error)}}
                if isinstance(error, SchemaValidationError):
                    evidence["validation"] = {"path": error.path, "expected_type": error.expected_type,
                                              "actual_type": error.actual_type}
                try:
                    terminal["error"].update(diagnostics.save_diagnostic(self.diagnostic_directory, evidence))
                except Exception:
                    terminal["error"]["diagnostic_status"] = "unavailable"
            yield terminal
            return
        finished = time.perf_counter()
        generated = int(final.get("new_tokens", emitted)) if final else emitted
        valid = requeues == 0 and status != "cancelled"
        cached = int(final.get("cached_tokens", 0)) if final and valid else None
        accepted = int(final.get("accepted_draft_tokens", 0)) if final and valid else None
        rejected = int(final.get("rejected_draft_tokens", 0)) if final and valid else None
        metrics = {
            "ttft_ms": (first - started) * 1000 if first is not None else None,
            "first_content_ms": (first_content - started) * 1000 if first_content is not None else None,
            "elapsed_ms": (finished - started) * 1000,
            "decode_tokens_per_second": (emitted - first_batch) / (last_emitted - first)
                if first is not None and last_emitted > first and emitted > first_batch else None,
            "first_batch_tokens": first_batch, "requeue_count": requeues, "cache_metrics_valid": valid,
            "cached_tokens": cached,
            "physical_prefill_tokens": max(len(prompt.tokens) - 1 - cached, 0) if cached is not None else None,
            "accepted_draft_tokens": accepted, "rejected_draft_tokens": rejected,
            "draft_acceptance": accepted / (accepted + rejected) if accepted is not None and accepted + rejected else None,
            "host_prefill_ms": float(final.get("time_prefill", 0)) * 1000 if final and valid else None,
            "finish_reason": final.get("eos_reason") if final else "cancelled",
        }
        snapshot = None
        if status == "completed":
            segment = {"message": message, "tokens": sequence[prompt.assistant_start:]}
            snapshot = {"version": 1, "runtime_identity": self.backend.identity,
                        "header": prompt.header, "messages": prompt.messages + [message],
                        "tape": sequence, "segments": prompt.segments + [segment]}
        yield {"type": "terminal", "id": response_id, "status": status, "message": message,
               "usage": {"prompt_tokens": len(prompt.tokens), "completion_tokens": generated,
                         "total_tokens": len(prompt.tokens) + generated,
                         "prompt_tokens_details": {"cached_tokens": cached}},
               "metrics": metrics, "snapshot": snapshot, "error": None}


class FakeTokenizer:
    special = {"<|im_start|>": 1000001, "<|im_end|>": 1000002,
               "<think>": 1000003, "</think>": 1000004}

    def __init__(self):
        self.tokenizer = SimpleNamespace(encode_special_tokens=False, encode=self.literal)

    def literal(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=[ord(character) for character in text])

    def encode(self, text, encode_special_tokens=False):
        tokens = []
        for part in re.split("(" + "|".join(map(re.escape, self.special)) + ")", text):
            tokens.extend([self.special[part]] if part in self.special and encode_special_tokens
                          else [ord(character) for character in part])
        return SimpleNamespace(flatten=lambda: SimpleNamespace(tolist=lambda: tokens))

    def hf_render_chat_template(self, messages, tools, add_generation_prompt, enable_thinking,
                                reasoning_effort, preserve_thinking):
        output = ""
        if tools:
            output += "<|im_start|>system\n<tools>" + "\n".join(json.dumps(tool) for tool in tools) + "</tools><|im_end|>\n"
        for message in messages:
            output += "<|im_start|>" + message["role"] + "\n"
            if message["role"] == "assistant":
                output += "<think>\n" + message.get("reasoning_content", "") + "\n</think>\n\n"
            output += message["content"] + "<|im_end|>\n"
        return output + "<|im_start|>assistant\n<think>\n" + ("\n</think>\n\n" if not enable_thinking else "")


class FakeBackend:
    identity = "qwasar-explicit-fake-v1"
    draft_tokens = 0
    config = {"fake": True, "quantization": "fake", "prefill": "fake"}

    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.end_token = self.tokenizer.special["<|im_end|>"]
        self.enqueued = self.resets = 0
        self.active = False

    def start(self, tokens, request):
        self.enqueued += 1
        self.tape = list(tokens)
        latest = request["messages"][-1]["content"]
        self.slow = "[fake:slow]" in latest
        response = "Fake runtime response."
        tools = request["tools"]
        if tools and request["tool_choice"] != "none" and ("[fake:tool]" in latest or request["tool_choice"] == "required" or isinstance(request["tool_choice"], dict)):
            name = request["tool_choice"]["function"]["name"] if isinstance(request["tool_choice"], dict) else tools[0]["function"]["name"]
            function = next(tool["function"] for tool in tools if tool["function"]["name"] == name)
            schema = function.get("parameters", {})
            values = {key: self.fixture(schema.get("properties", {}).get(key, {})) for key in schema.get("required", [])}
            response = "<tool_call>\n<function=" + name + ">\n" + "".join(
                "<parameter=" + key + ">\n" + (value if isinstance(value, str) else canonical(value)) + "\n</parameter>\n"
                for key, value in values.items()) + "</function>\n</tool_call>"
        if request["thinking"] != "off":
            response = "Fake reasoning.</think>\n\n" + response
        self.chunks = list(response)
        self.maximum, self.generated = request["max_tokens"], 0
        self.active = True
        return self.enqueued

    @staticmethod
    def fixture(schema):
        if "enum" in schema:
            return schema["enum"][0]
        if "const" in schema:
            return schema["const"]
        kind = schema.get("type", "string")
        if kind == "object":
            return {key: FakeBackend.fixture(schema.get("properties", {}).get(key, {})) for key in schema.get("required", [])}
        if kind == "array":
            return [FakeBackend.fixture(schema.get("items", {})) for _ in range(schema.get("minItems", 0))]
        return {"string": "fake", "integer": 1, "number": 1, "boolean": True, "null": None}.get(kind, "fake")

    def remaining(self):
        return int(self.active)

    def tuning(self):
        return nullcontext()

    def iterate(self):
        if self.slow:
            time.sleep(0.03)
        if self.generated >= self.maximum:
            self.active = False
            return [{"eos": True, "eos_reason": "max_new_tokens", "new_tokens": self.generated}]
        if self.chunks:
            text = self.chunks.pop(0)
            token_ids = [ord(text)]
            self.tape.extend(token_ids)
            self.generated += 1
            return [{"text": text, "token_ids": token_ids}]
        self.tape.append(self.end_token)
        self.generated += 1
        self.active = False
        return [{"eos": True, "eos_reason": "stop_token", "new_tokens": self.generated}]

    def sequence(self, final):
        return self.tape

    def cancel_reset(self, serial):
        self.active = False
        self.resets += 1


def verify_artifact(model_path):
    from qwasar_bench.environment import sha256_path

    manifest_path = Path(__file__).resolve().parents[2] / "benchmarks/manifests/qwen38-27b-rtx5090-v1.json"
    expected = json.loads(manifest_path.read_text())["model"]["artifact_sha256"]
    actual = sha256_path(model_path)
    if actual != expected:
        raise ValueError(f"artifact hash mismatch: expected {expected}, got {actual}")
    return actual


def configure_gpu(torch_module=None):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in (None, "0"):
        raise ValueError("CUDA_VISIBLE_DEVICES must select only physical GPU 0")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    if torch_module is None:
        import torch as torch_module
    if torch_module.cuda.device_count() != 1 or "RTX 5090" not in torch_module.cuda.get_device_name(0):
        raise ValueError("v1 requires one visible RTX 5090 at physical GPU 0")


class ExLlamaBackend:
    def __init__(self, model_path, context_size, prefill="flash", gpu_split_gb=30.0):
        from qwasar_bench.exllamav3_probe import _load_exllamav3_runtime

        model_path = Path(model_path).resolve()
        artifact_sha256 = verify_artifact(model_path)
        configure_gpu()
        artifact = json.loads((model_path / "config.json").read_text())
        native = artifact.get("text_config", artifact).get("max_position_embeddings", 0)
        quantization = artifact.get("quantization_config", {})
        if quantization.get("quant_method") != "exl3" or quantization.get("bits") != 5.0:
            raise ValueError("v1 requires the pinned EXL3 5bpw artifact")
        if context_size > native:
            raise ValueError("context exceeds artifact native positions")
        identity_files = ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]
        hashes = {name: hashlib.sha256((model_path / name).read_bytes()).hexdigest() for name in identity_files}
        import exllamav3

        runtime_path = Path(exllamav3.__file__).parent
        runtime_hash = hashlib.sha256()
        for source in sorted(runtime_path.rglob("*.py")):
            runtime_hash.update(str(source.relative_to(runtime_path)).encode())
            runtime_hash.update(source.read_bytes())
        self.identity = hashlib.sha256(canonical({"model": str(model_path), "files": hashes,
            "artifact_sha256": artifact_sha256, "runtime": runtime_hash.hexdigest(),
            "renderer_abi": 2, "quantization": "5bpw-K8V4-MTP"}).encode()).hexdigest()
        self.generator, self.tokenizer, _ = _load_exllamav3_runtime(
            model_path=model_path, draft_model_path=model_path, cache_size=context_size,
            cache_quant="8,4", gpu_split_gb=gpu_split_gb, draft_method="mtp", num_draft_tokens=6)
        self.end_token = encode(self.tokenizer, "<|im_end|>")[0]
        self.draft_tokens = self.generator.num_draft_tokens
        self.prefill = prefill
        self.config = {"fake": False, "model_path": str(model_path), "quantization": "EXL3 5bpw",
            "cache_quantization": "K8/V4", "draft_method": "mtp", "draft_tokens": self.draft_tokens,
            "prefill": prefill, "native_context_size": native, "artifact_hashes": hashes,
            "artifact_sha256": artifact_sha256, "runtime_sha256": runtime_hash.hexdigest()}

    def start(self, tokens, request):
        import torch
        from exllamav3 import Job
        from exllamav3.generator.sampler.presets import ComboSampler, GreedySampler

        sampler = GreedySampler() if request["temperature"] == 0 else ComboSampler(
            temperature=request["temperature"], top_p=request["top_p"], top_k=20, min_p=0.0,
            pres_p=1.5 if request["thinking"] == "off" else 0.0)
        self.job = Job(input_ids=torch.tensor([tokens], dtype=torch.long), max_new_tokens=request["max_tokens"],
                       sampler=sampler, seed=request["seed"], stop_conditions=[self.end_token], decode_special_tokens=True)
        return self.generator.enqueue(self.job)

    def tuning(self):
        from qwasar_bench.prefill_tuning import workload_tuning_context

        settings = {"name": "flash_chunk8192", "implementation": "torch_flash", "staging": 1,
                    "minimum_query": 17, "chunk_size": 8192, "kwargs": {}} if self.prefill == "flash" else None
        return workload_tuning_context(self.generator, settings)

    def remaining(self):
        return self.generator.num_remaining_jobs()

    def iterate(self):
        events = self.generator.iterate()
        for event in events:
            token_ids = event.get("token_ids")
            if token_ids is not None:
                event["token_ids"] = token_ids.flatten().tolist()
        return events

    def sequence(self, final):
        return final["job"].sequences[0].sequence_ids.torch().flatten().tolist()

    def cancel_reset(self, serial):
        from qwasar_bench.fidelity_probe import fresh_generator

        for job in list(self.generator.active_jobs) + list(self.generator.pending_jobs):
            if serial is None or job.serial_number == serial:
                self.generator.cancel(job)
        fresh_generator(self.generator)
