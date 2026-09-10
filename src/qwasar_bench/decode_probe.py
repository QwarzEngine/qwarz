from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from qwasar_bench.exllamav3_probe import (
    DEFAULT_EXLLAMA_HOME,
    REPOSITORY_ROOT,
    _load_exllamav3_runtime,
    _token_count,
    _write_text_atomic,
)
from qwasar_bench.environment import capture_environment

TASK = (
    "Write a standalone Python module implementing a bounded LRU cache with "
    "get, put, delete, and len operations. Require positive integer capacity, "
    "support stored None values, update recency on get and put, and evict the "
    "least recently used item. Include at least six unittest cases covering "
    "updates, eviction, missing keys, deletion, None, and invalid capacity. "
    "Return the complete code. Do not copy code from the repository snapshot."
)


MTP_POLICIES = tuple(f"fixed{count}" for count in range(1, 8)) + ("adaptive4",)


def configure_mtp_policy(generator: object, policy: str) -> dict:
    """Select policy only after the generator allocated the matching draft width."""
    if policy not in MTP_POLICIES:
        raise ValueError("unknown MTP policy")
    count = int(policy[-1])
    if (not generator.mtp_draft or generator.num_draft_tokens != count
            or generator.num_remaining_jobs()):
        raise ValueError("MTP policy requires an idle MTP generator with matching preallocated width")
    names = ("dynamic_draft_alpha_up", "dynamic_draft_alpha_down",
             "dynamic_draft_skip_ema", "dynamic_draft_probe_interval")
    settings = {name: getattr(generator, name) for name in names}
    generator.dynamic_draft = policy == "adaptive4"
    generator.record_draft_stats = True
    return {"num_draft_tokens": count, "dynamic_draft": generator.dynamic_draft,
            "record_draft_stats": True, **settings}


def prompt_budget(context: int, output: int, scratch: int) -> int:
    if not 0 < context <= 262144 or output <= 0 or scratch < 0:
        raise ValueError("invalid context, output, or speculative reserve")
    budget = context - output - scratch
    if budget <= 0:
        raise ValueError("context must leave room for the prompt")
    return budget


def decode_metrics(
    *, started: float, first: float | None, finished: float,
    generated: int, first_batch_tokens: int,
) -> dict[str, float | int | None]:
    remaining = generated - first_batch_tokens
    duration = finished - first if first is not None else 0.0
    return {
        "ttft_ms": (first - started) * 1000 if first is not None else None,
        "elapsed_ms": (finished - started) * 1000,
        "decode_tokens_per_second": remaining / duration
        if duration > 0 and remaining > 0 else None,
        "first_batch_tokens": first_batch_tokens,
    }


def visible_content(text: str, thinking: str) -> str:
    if thinking == "off":
        return text.lstrip()
    return text.partition("</think>")[2].lstrip()


def serialize_event(event: dict[str, object]) -> dict[str, object]:
    return {
        key: serialize_event(value) if isinstance(value, dict)
        else value.tolist() if hasattr(value, "tolist") else value
        for key, value in event.items() if key not in ("job", "logits")
    }


def sampler_settings(thinking: str) -> dict[str, float | int]:
    return {
        "temperature": 0.7 if thinking == "off" else 1.0,
        "top_p": 0.8 if thinking == "off" else 0.95,
        "top_k": 20, "min_p": 0.0,
        "pres_p": 1.5 if thinking == "off" else 0.0,
    }


def frame_corpus(
    prefix: list[int], corpus: list[int], suffix: list[int], budget: int,
) -> list[int]:
    required = budget - len(prefix) - len(suffix)
    if required <= 0:
        raise ValueError("prompt framing exceeds budget")
    if len(corpus) < required:
        raise ValueError("corpus is too short; repetition is prohibited")
    return [*prefix, *corpus[:required], *suffix]


def encode_corpus(tokenizer: object, text: str) -> list[int]:
    backend = tokenizer.tokenizer
    previous = backend.encode_special_tokens
    try:
        backend.encode_special_tokens = True
        return backend.encode(text, add_special_tokens=False).ids
    finally:
        backend.encode_special_tokens = previous


def build_prompt(
    tokenizer: object, corpus_ids: list[int], budget: int,
    repetition: int, thinking: str,
) -> list[int]:
    placeholder = "__QWASAR_LITERAL_CORPUS_29a7__"
    messages = [
        {"role": "system", "content": "You are a coding assistant. Follow the task precisely."},
        {"role": "user", "content": "Repository snapshot:\n" + placeholder +
         f"\nEnd of snapshot. Request {repetition}.\n" + TASK},
    ]
    rendered = tokenizer.hf_render_chat_template(
        messages, add_generation_prompt=True, enable_thinking=thinking != "off",
        reasoning_effort=thinking if thinking != "off" else "medium",
        preserve_thinking=True,
    )
    if rendered.count(placeholder) != 1:
        raise ValueError("template must contain exactly one corpus placeholder")
    prefix, _, suffix = rendered.partition(placeholder)
    prefix_ids = tokenizer.encode(prefix, encode_special_tokens=True).flatten().tolist()
    suffix_ids = tokenizer.encode(suffix, encode_special_tokens=True).flatten().tolist()
    return frame_corpus(prefix_ids, corpus_ids, suffix_ids, budget)


def run_sample(
    generator: object, tokenizer: object, prompt_ids: list[int],
    args: argparse.Namespace, seed: int, event_path: Path,
    *, sequence_sink: list[int] | None = None, logits_sink: list | None = None,
) -> dict[str, object]:
    import torch
    from exllamav3 import Job
    from exllamav3.generator.sampler.presets import ComboSampler, GreedySampler

    sampler = GreedySampler() if args.sampler == "greedy" else ComboSampler(
        **sampler_settings(args.thinking),
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    job = Job(
        input_ids=torch.tensor([prompt_ids], dtype=torch.long),
        max_new_tokens=args.max_new_tokens, sampler=sampler, seed=seed,
        stop_conditions=[tokenizer.eos_token_id, "<|im_end|>"],
        return_logits=logits_sink is not None,
    )
    generator.enqueue(job)
    first = first_content = None
    first_batch_tokens = emitted = streaming_batches = 0
    requeue_count = 0
    text = ""
    final = None
    with event_path.open("x") as events:
        while generator.num_remaining_jobs():
            results = generator.iterate()
            now = time.perf_counter()
            batch_tokens = sum(
                _token_count(result["token_ids"])
                for result in results if result.get("token_ids") is not None
            )
            if first is None and batch_tokens:
                first, first_batch_tokens = now, batch_tokens
            emitted += batch_tokens
            streaming_batches += int(batch_tokens > 0)
            for result in results:
                if logits_sink is not None and not logits_sink:
                    logits = result.get("logits")
                    if logits is None:
                        logits = result.get("held", {}).get("logits")
                    if logits is not None:
                        logits_sink.append(logits[0, 0, :].detach().cpu().clone())
                requeue_count += int(bool(result.get("requeue")))
                text += result.get("text", "")
                if first_content is None and visible_content(text, args.thinking):
                    first_content = now
                payload = serialize_event(result)
                events.write(json.dumps({"elapsed_ms": (now - started) * 1000,
                                         "event": payload}) + "\n")
                if result.get("eos"):
                    final = result
            events.flush()
    finished = time.perf_counter()
    if final is None:
        raise RuntimeError("queue drained without EOS")
    if sequence_sink is not None:
        sequence = job.sequences[0].sequence_ids.torch().flatten().tolist()
        if sequence[:len(prompt_ids)] != prompt_ids:
            raise RuntimeError("runtime changed the session prefix")
        sequence_sink.extend(sequence)
    generated = int(final.get("new_tokens", emitted))
    accepted = int(getattr(job, "accepted_draft_tokens", 0))
    rejected = int(getattr(job, "rejected_draft_tokens", 0))
    cached = int(final.get("cached_tokens", 0))
    return {
        **decode_metrics(started=started, first=first, finished=finished,
                         generated=generated, first_batch_tokens=first_batch_tokens),
        "first_content_ms": (first_content - started) * 1000
        if first_content is not None else None,
        "input_tokens": len(prompt_ids), "generated_tokens": generated,
        "requeue_count": requeue_count, "cache_metrics_valid": requeue_count == 0,
        "cached_tokens": cached if not requeue_count else None,
        "physical_prefill_tokens": max(len(prompt_ids) - 1 - cached, 0) if not requeue_count else None,
        "accepted_draft_tokens": accepted if not requeue_count else None,
        "rejected_draft_tokens": rejected if not requeue_count else None,
        "draft_acceptance": accepted / (accepted + rejected)
        if accepted + rejected and not requeue_count else None,
        "draft_stats": list(job.draft_stats)
        if getattr(generator, "record_draft_stats", False) and not requeue_count else None,
        "finish_reason": final.get("eos_reason"),
        "truncated": final.get("eos_reason") == "max_new_tokens",
        "streaming_batches": streaming_batches,
        "host_prefill_ms": float(getattr(job, "time_prefill", 0)) * 1000 if not requeue_count else None,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "completion": text, "quality": "not_scored",
        "prompt_sha256": hashlib.sha256(json.dumps(prompt_ids).encode()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--draft-model", type=Path, required=True)
    parser.add_argument("--draft-method", choices=("mtp", "dflash2", "none"), required=True)
    parser.add_argument("--mtp-policy", choices=MTP_POLICIES,
                        help="Opt-in fixed MTP width or adaptive4; records per-round draft statistics")
    parser.add_argument("--cache-quant", choices=("8,4", "nvfp4", "fp8", "4"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--contexts", default="32768")
    parser.add_argument("--cache-size", type=int,
                        help="Physical pool size, independent of prompt context buckets")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--thinking", choices=("xhigh", "medium", "low", "off"), default="medium")
    parser.add_argument("--sampler", choices=("greedy", "recommended"), default="greedy")
    parser.add_argument("--workload", choices=("lru", "retrieval_session", "cache_fidelity", "turn_matrix"), default="lru")
    parser.add_argument("--tool-max-new-tokens", type=int, default=512)
    parser.add_argument("--appended-tokens", default="128,512,2048,8192")
    parser.add_argument("--profile-matrix", type=int, choices=(0, 1), default=0)
    parser.add_argument("--prefill-settings", type=Path)
    parser.add_argument("--source-run", type=Path,
                        default=REPOSITORY_ROOT / "results/20260904-session-5.0bpw-medium-long")
    parser.add_argument("--gpu-split-gb", type=float, default=30.0)
    args = parser.parse_args()
    artifact_sha256 = None
    if args.mtp_policy is not None:
        if args.draft_method != "mtp" or args.cache_quant != "8,4":
            parser.error("MTP policy experiment requires --draft-method mtp --cache-quant 8,4")
        from qwasar_runtime.engine import verify_artifact

        artifact_sha256 = verify_artifact(args.model)
    prefill_settings = None
    if args.prefill_settings is not None:
        from qwasar_bench.prefill_tuning import validate_screen_candidates

        prefill_settings = json.loads(args.prefill_settings.read_text())
        validate_screen_candidates([prefill_settings])
        if args.workload == "cache_fidelity":
            raise ValueError("prefill settings are not supported for cache_fidelity")
    contexts = [int(value) for value in args.contexts.split(",")]
    if args.repetitions <= 0 or contexts != sorted(set(contexts)):
        raise ValueError("positive repetitions and unique increasing contexts required")
    full_pool = args.workload in ("cache_fidelity", "turn_matrix")
    cache_size = args.cache_size if args.cache_size is not None else (262144 if full_pool else max(contexts))
    if not max(contexts) <= cache_size <= 262144 or (full_pool and cache_size != 262144):
        parser.error("cache pool must cover all contexts within 262144; fidelity/matrix require the full pool")
    model_config = json.loads((args.model / "config.json").read_text())
    native_limit = model_config.get("text_config", model_config)["max_position_embeddings"]
    if cache_size > native_limit:
        parser.error("cache pool exceeds the artifact native limit")
    for context in contexts:
        prompt_budget(context, args.max_new_tokens, 16)
        if args.workload == "retrieval_session":
            if args.tool_max_new_tokens <= 0:
                raise ValueError("tool output budget must be positive")
            prompt_budget(context, 2 * (args.max_new_tokens + args.tool_max_new_tokens), 1040)
        if context > native_limit:
            raise ValueError("context exceeds artifact's native limit")
    args.output.mkdir(parents=True, exist_ok=False)
    exllama_home = Path(os.environ.get("QWASAR_EXLLAMA_HOME", str(DEFAULT_EXLLAMA_HOME)))
    environment = capture_environment(
        repository_path=REPOSITORY_ROOT, backend_repository_path=exllama_home,
        selected_gpu_index=int(os.environ.get("QWASAR_CUDA_DEVICE", "0")),
    ).to_dict()
    corpus_parts = []
    for source in sorted(args.corpus_root.rglob("*")):
        if source.is_file() and source.suffix in (".py", ".cpp", ".h", ".cuh"):
            corpus_parts.append(f"\nFile: {source.relative_to(args.corpus_root)}\n" + source.read_text())
    corpus = "\n".join(corpus_parts)
    _write_text_atomic(args.output / "corpus.txt", corpus)
    generator, tokenizer, _ = _load_exllamav3_runtime(
        model_path=args.model, draft_model_path=args.draft_model,
        cache_size=cache_size, cache_quant=args.cache_quant,
        gpu_split_gb=args.gpu_split_gb, draft_method=args.draft_method,
        **({"num_draft_tokens": int(args.mtp_policy[-1])} if args.mtp_policy else {}),
    )
    mtp_settings = configure_mtp_policy(generator, args.mtp_policy) if args.mtp_policy else None
    import exllamav3
    import torch

    package_root = Path(exllamav3.__file__).parent
    digest = hashlib.sha256()
    for source in sorted(package_root.rglob("*")):
        if source.is_file() and source.suffix in (".py", ".cu", ".cuh", ".cpp", ".h", ".so"):
            digest.update(str(source.relative_to(package_root)).encode() + b"\0")
            digest.update(source.read_bytes())
    metadata = {
        "schema_version": 1, "qualification": "screening_only",
        "framing": "token_safe_v2",
        "cache_size": cache_size,
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        "environment": environment, "package_path": str(package_root),
        "torch_version": torch.__version__, "torch_cuda": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "sampling": {"type": args.sampler, "parameters":
                     sampler_settings(args.thinking) if args.sampler == "recommended" else {}},
        "model_config_sha256": hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest(),
        "runtime_code_sha256": digest.hexdigest(),
        "harness_files_sha256": {
            str(source.relative_to(REPOSITORY_ROOT)): hashlib.sha256(source.read_bytes()).hexdigest()
            for source in [Path(__file__), Path(__file__).with_name("exllamav3_probe.py"),
                           Path(__file__).with_name("resident.py"),
                           Path(__file__).with_name("session_probe.py"),
                           Path(__file__).with_name("fidelity_probe.py"),
                           Path(__file__).with_name("turn_matrix.py"),
                           Path(__file__).with_name("profile_probe.py"),
                           REPOSITORY_ROOT / "scripts/run_resident_probe.sh"]
        },
        "corpus_sha256": hashlib.sha256(corpus.encode()).hexdigest(),
        "template_sha256": hashlib.sha256((args.model / "chat_template.jinja").read_bytes()).hexdigest(),
        "num_draft_tokens": generator.num_draft_tokens,
        "mtp_settings": mtp_settings,
        "artifact_sha256": artifact_sha256,
        "prefill_settings": prefill_settings,
        "prefill_tuning_sha256": hashlib.sha256(Path(__file__).with_name("prefill_tuning.py").read_bytes()).hexdigest()
        if prefill_settings is not None else None,
        "prefill_flash_sha256": hashlib.sha256(Path(__file__).with_name("prefill_flash.py").read_bytes()).hexdigest()
        if prefill_settings is not None else None,
        "timing": "host observed; first streaming batch excluded from decode rate",
        "workload": {
            "lru": "fixed LRU task over source snapshot; warm branches, not live append",
            "retrieval_session": "two retrieval/tool cycles; exact append-only token history",
            "cache_fidelity": "same exact archived tool-result prompt; greedy cached vs cold replay",
            "turn_matrix": "controlled warm branches with exact new-input sizes; strict configuration retrieval",
        }[args.workload],
    }
    _write_text_atomic(args.output / "run.json", json.dumps(metadata, indent=2) + "\n")
    corpus_ids = encode_corpus(tokenizer, corpus)
    from qwasar_bench.prefill_tuning import workload_tuning_context

    with workload_tuning_context(generator, prefill_settings,
                                 counters_path=args.output / "prefill-counters.json"):
        if args.workload == "cache_fidelity":
            from qwasar_bench.fidelity_probe import run_fidelity

            run_fidelity(generator, tokenizer, corpus_ids, args)
        elif args.workload == "turn_matrix":
            from qwasar_bench.turn_matrix import run_matrix

            run_matrix(generator, tokenizer, corpus_ids, args)
        elif args.workload == "retrieval_session":
            from qwasar_bench.session_probe import run_session

            for context in contexts:
                for repetition in range(args.repetitions):
                    run_session(generator, tokenizer, corpus_ids, args, context, repetition)
        else:
            for context in contexts:
                for repetition in range(args.repetitions + 1):
                    preparation_started = time.perf_counter()
                    scratch = max(16, generator.num_draft_tokens + 1)
                    prompt_ids = build_prompt(
                        tokenizer, corpus_ids, prompt_budget(context, args.max_new_tokens, scratch),
                        repetition, args.thinking,
                    )
                    preparation_ms = (time.perf_counter() - preparation_started) * 1000
                    label = f"{context}-{repetition}"
                    print(json.dumps({"starting": label, "input_tokens": len(prompt_ids)}), flush=True)
                    sample = run_sample(generator, tokenizer, prompt_ids, args, 42 + repetition,
                                        args.output / f"events-{label}.jsonl")
                    sample.update(context_budget=context, repetition=repetition,
                                  phase="seed" if repetition == 0 else "warm_branch",
                                  preparation_ms=preparation_ms)
                    _write_text_atomic(args.output / f"sample-{label}.json", json.dumps(sample, indent=2) + "\n")
                    print(json.dumps({key: value for key, value in sample.items() if key != "completion"}), flush=True)
    _write_text_atomic(args.output / "completed.json", '{"completed": true}\n')


if __name__ == "__main__":
    main()
