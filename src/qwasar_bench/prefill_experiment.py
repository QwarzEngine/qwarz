import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import random
import time

from qwasar_bench.decode_probe import encode_corpus, run_sample, visible_content
from qwasar_bench.environment import capture_environment
from qwasar_bench.exllamav3_probe import _load_exllamav3_runtime, _write_text_atomic, REPOSITORY_ROOT
from qwasar_bench.fidelity_probe import verify_prompt
from qwasar_bench.prefill_tuning import capture_context, tuning_context, validate_screen_candidates
from qwasar_bench.session_probe import grade_answer
from qwasar_bench.turn_matrix import build_turn, summarize, warm_qualified


def write_json(path, value):
    _write_text_atomic(path, json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("capture", "screen"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, default=REPOSITORY_ROOT / "results/20260904-turn-matrix-5bpw-mtp")
    parser.add_argument("--model", type=Path, default=Path("/home/rekeyea/models/Qwen3.8-27B-EXL3-5.0bpw"))
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--appended-tokens", default="128,512,2048,8192")
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    candidates = None
    if args.mode == "screen":
        if args.candidates is None:
            raise ValueError("screen requires candidates")
        candidates = json.loads(args.candidates.read_text())
        validate_screen_candidates(candidates)
    if args.repetitions <= 0 or any(int(value) <= 0 for value in args.appended_tokens.split(",")):
        raise ValueError("positive repetitions and deltas required")
    args.max_new_tokens, args.thinking, args.sampler = 1536, "medium", "recommended"
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from exllamav3.modules.attention_fn import triton_paged as backend

    if "RTX 5090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("prefill experiment requires selected RTX 5090")
    source = args.source_run
    archived = json.loads((source / "turn-262144-8192-0-sample.json").read_text())
    history = json.loads((source / archived["seed_history_file"]).read_text())
    prompt = history + json.loads((source / archived["delta_file"]).read_text())
    verify_prompt(prompt, archived)
    metadata = {
        "mode": args.mode, "model": str(args.model), "source_run": str(source),
        "prompt_sha256": archived["prompt_sha256"], "cache_quant": "8,4", "draft_method": "mtp",
        "environment": capture_environment(repository_path=REPOSITORY_ROOT,
            backend_repository_path=Path("/home/rekeyea/Documents/llm/qwen38-exl3-mia"),
            selected_gpu_index=0).to_dict(),
        "source_hashes": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (Path(__file__), Path(backend.__file__), Path(__file__).with_name("prefill_tuning.py"),
                                       Path(__file__).with_name("prefill_flash.py"))},
        "model_config_sha256": hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest(),
    }
    write_json(args.output / "run.json", metadata)
    generator, tokenizer, _ = _load_exllamav3_runtime(model_path=args.model, draft_model_path=args.model,
        cache_size=262144, cache_quant="8,4", gpu_split_gb=30.0, draft_method="mtp")
    prime_args = argparse.Namespace(**vars(args))
    prime_args.max_new_tokens, prime_args.sampler = 1, "greedy"
    print(json.dumps({"priming": len(history)}), flush=True)
    prime = run_sample(generator, tokenizer, history, prime_args, 42, args.output / "prime-events.jsonl")
    write_json(args.output / "prime.json", prime)
    if not prime["cache_metrics_valid"] or prime["cached_tokens"] != 0:
        raise RuntimeError("priming must be physically cold")
    if args.mode == "capture":
        def cpu(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().clone()
            if isinstance(value, tuple):
                return tuple(cpu(item) for item in value)
            return value

        def save(kwargs):
            bundle = {"kwargs": {key: cpu(value) for key, value in kwargs.items()}, "metadata": metadata}
            torch.save(bundle, args.output / "attention.pt")
            write_json(args.output / "capture.json", {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                       if isinstance(value, torch.Tensor) else str(value) if key == "qc" else value
                       for key, value in kwargs.items()})

        with capture_context(backend, save):
            sample = run_sample(generator, tokenizer, prompt, args, 42, args.output / "capture-events.jsonl")
        sample["quality_pass"] = grade_answer(visible_content(sample["completion"], args.thinking),
                                             archived["expected"], sample["truncated"])
        sample["warm_valid"] = warm_qualified(sample, len(history), generator.recurrent_checkpoint_interval)
        write_json(args.output / "capture-sample.json", sample)
        if not (args.output / "attention.pt").exists() or not sample["quality_pass"] or not sample["warm_valid"]:
            raise RuntimeError("capture or quality failed")
    else:
        write_json(args.output / "candidates.json", candidates)
        corpus_bytes = (source / "corpus.txt").read_bytes()
        source_metadata = json.loads((source / "run.json").read_text())
        if hashlib.sha256(corpus_bytes).hexdigest() != source_metadata["corpus_sha256"]:
            raise RuntimeError("archived corpus hash mismatch")
        corpus = encode_corpus(tokenizer, corpus_bytes.decode())[252400:]
        fixture = json.loads((source / "matrix-fixture.json").read_text())
        rows = []
        warmups = [(delta, -1, index) for delta in map(int, args.appended_tokens.split(","))
                   for index in range(len(candidates))]
        trials = [(delta, repetition, index) for delta in map(int, args.appended_tokens.split(","))
                  for repetition in range(args.repetitions) for index in range(len(candidates))]
        random.Random(20260904).shuffle(trials)
        for ordinal, (delta, repetition, index) in enumerate(warmups + trials):
            candidate = candidates[index]
            case = fixture["cases"][repetition % len(fixture["cases"])]
            payload = json.loads(fixture["files"][case["path"]]) | {"service": case["service"], "input_value": case["value"]}
            label = f"pf-{ordinal}-{index}-{delta}-{repetition}"
            started = time.perf_counter()
            turn = build_turn(tokenizer, history, corpus, delta, "Request: " + json.dumps(payload) + "\nReturn JSON.",
                              args.thinking, label)
            if len(turn) + args.max_new_tokens + max(16, generator.num_draft_tokens + 1) > 262144:
                raise ValueError("turn exceeds native context reserve")
            write_json(args.output / f"{label}-delta.json", turn[len(history):])
            chunk = generator.max_chunk_size
            try:
                generator.max_chunk_size = candidate.get("chunk_size", chunk)
                context = tuning_context(backend, candidate) if candidate.get("name") != "baseline" else nullcontext({})
                with context as counts:
                    sample = run_sample(generator, tokenizer, turn, args, 42 + repetition,
                                        args.output / f"{label}-events.jsonl")
            finally:
                generator.max_chunk_size = chunk
            sample.update(label=label, candidate=candidate, counters=dict(counts), nominal_context=262144,
                          phase="warmup" if repetition == -1 else "measured",
                          appended_tokens=delta, repetition=repetition, history_tokens=len(history),
                          user_wait_ms=(time.perf_counter() - started) * 1000,
                          quality_pass=grade_answer(visible_content(sample["completion"], args.thinking),
                                                    case["expected"], sample["truncated"]),
                          warm_valid=warm_qualified(sample, len(history), generator.recurrent_checkpoint_interval),
                          expected=case["expected"])
            rows.append(sample)
            write_json(args.output / f"{label}-sample.json", sample)
            print(json.dumps({key: value for key, value in sample.items() if key != "completion"}), flush=True)
        write_json(args.output / "screen-summary.json", {
            candidate["name"]: summarize([row for row in rows if row["candidate"]["name"] == candidate["name"]
                                         and row["phase"] == "measured"])
            for candidate in candidates})
        write_json(args.output / "qualification.json", {
            "all_rows_quality_pass": all(row["quality_pass"] for row in rows),
            "all_rows_warm_valid": all(row["warm_valid"] for row in rows),
            "warmup_rows": sum(row["phase"] == "warmup" for row in rows),
            "measured_rows": sum(row["phase"] == "measured" for row in rows)})
    write_json(args.output / "completed.json", {"completed": True})


if __name__ == "__main__":
    main()
