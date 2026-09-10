from __future__ import annotations

import json
import os
import subprocess
import time
from argparse import ArgumentParser
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from qwasar_bench.environment import capture_environment
from qwasar_bench.resident import (
    JobObservation,
    ResidentProbeConfig,
    evaluate_resident_slo,
    summarize_observations,
)

PAGE_SIZE = 256
CODING_TOKEN_SOURCE = (
    "Repository task: inspect the parser, preserve public behavior, add a focused "
    "regression test, run the smallest relevant test target, and report exact file "
    "paths and failures. Do not change unrelated code."
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXLLAMA_HOME = Path("/home/rekeyea/Documents/llm/qwen38-exl3-mia")


def run_probe(
    *,
    model_path: Path,
    draft_model_path: Path,
    output_directory: Path,
    context_tokens: tuple[int, ...],
    appended_tokens: int,
    max_new_tokens: int,
    repetitions: int,
    cache_size: int,
    cache_quant: str,
    gpu_split_gb: float,
) -> dict[str, object]:
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(model_path / "config.json")
    if not (draft_model_path / "config.json").is_file():
        raise FileNotFoundError(draft_model_path / "config.json")
    if (output_directory / "summary.json").exists():
        raise FileExistsError(output_directory / "summary.json")

    config = ResidentProbeConfig(
        context_tokens=context_tokens,
        appended_tokens=appended_tokens,
        max_new_tokens=max_new_tokens,
        repetitions=repetitions,
    )
    selected_gpu_index = int(os.environ.get("QWASAR_CUDA_DEVICE", "0"))
    exllama_home = Path(
        os.environ.get("QWASAR_EXLLAMA_HOME", str(DEFAULT_EXLLAMA_HOME))
    )
    environment_before = capture_environment(
        repository_path=REPOSITORY_ROOT,
        backend_repository_path=exllama_home,
        selected_gpu_index=selected_gpu_index,
    ).to_dict()
    generator, tokenizer, job_factory = _load_exllamav3_runtime(
        model_path=model_path,
        draft_model_path=draft_model_path,
        cache_size=cache_size,
        cache_quant=cache_quant,
        gpu_split_gb=gpu_split_gb,
    )
    observations, summary = run_resident_matrix(
        generator, tokenizer, config, job_factory
    )
    environment_after = capture_environment(
        repository_path=REPOSITORY_ROOT,
        backend_repository_path=exllama_home,
        selected_gpu_index=selected_gpu_index,
    ).to_dict()
    environment = {
        "before": environment_before,
        "after": environment_after,
        "backend_dirty": _git_dirty(exllama_home),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "process_id": os.getpid(),
    }
    summary_payload = write_probe_artifacts(
        output_directory,
        config=config,
        observations=observations,
        summary=summary,
        environment=environment,
        model_path=str(model_path),
        draft_model_path=str(draft_model_path),
        cache_size=cache_size,
        cache_quant=cache_quant,
        gpu_split_gb=gpu_split_gb,
    )
    return {
        "completed": True,
        "output_directory": str(output_directory),
        **summary_payload,
    }


def _model_init_argv(
    *,
    model_path: str,
    draft_model_path: str,
    cache_size: int,
    cache_quant: str,
    gpu_split_gb: float,
    draft_method: str = "dflash2",
) -> list[str]:
    arguments = [
        "--model_dir",
        model_path,
        "--gpu_split",
        f"{gpu_split_gb:g}",
    ]
    if draft_method == "dflash2":
        arguments.extend(("--draft_model_dir", draft_model_path))
    elif draft_method == "mtp":
        arguments.append("--mtp")
    elif draft_method != "none":
        raise ValueError("draft_method must be dflash2, mtp, or none")
    arguments.extend(("--cache_size", str(cache_size)))
    if cache_quant != "none":
        arguments.extend(("--cache_quant", cache_quant))
    return arguments


def _load_exllamav3_runtime(
    *,
    model_path: Path,
    draft_model_path: Path,
    cache_size: int,
    cache_quant: str,
    gpu_split_gb: float,
    draft_method: str = "dflash2",
    num_draft_tokens: int | None = None,
) -> tuple[object, object, Callable[..., object]]:
    if num_draft_tokens is not None and (
        draft_method != "mtp" or type(num_draft_tokens) is not int or not 1 <= num_draft_tokens <= 7
    ):
        raise ValueError("explicit draft width requires MTP and an integer from 1 to 7")
    from exllamav3 import Generator, Job, model_init
    from exllamav3.generator.sampler.presets import GreedySampler

    parser = ArgumentParser(add_help=False)
    model_init.add_args(parser, add_draft_model_args=True)
    arguments = parser.parse_args(
        _model_init_argv(
            model_path=str(model_path),
            draft_model_path=str(draft_model_path),
            cache_size=cache_size,
            cache_quant=cache_quant,
            gpu_split_gb=gpu_split_gb,
            draft_method=draft_method,
        ) + (["--num_draft_tokens", str(num_draft_tokens)] if num_draft_tokens is not None else [])
    )
    model, _, cache, tokenizer, draft_model, _, draft_cache = model_init.init(
        arguments, progress=True
    )
    generator = Generator(
        model,
        cache,
        tokenizer,
        draft_model=draft_model,
        draft_cache=draft_cache,
        **({"num_draft_tokens": num_draft_tokens} if num_draft_tokens is not None else {}),
    )

    def job_factory(
        *, input_ids: object, max_new_tokens: int, seed: int
    ) -> object:
        return Job(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            sampler=GreedySampler(),
            seed=seed,
        )

    return generator, tokenizer, job_factory


def write_probe_artifacts(
    output_directory: Path,
    *,
    config: ResidentProbeConfig,
    observations: Sequence[JobObservation],
    summary: dict[str, object],
    environment: dict[str, object],
    model_path: str,
    draft_model_path: str,
    cache_size: int,
    cache_quant: str,
    gpu_split_gb: float,
) -> dict[str, object]:
    output_directory.mkdir(parents=True, exist_ok=True)
    config_payload = {
        "schema_version": 1,
        "context_semantics": "total_sequence_budget",
        **asdict(config),
        "model_path": model_path,
        "draft_model_path": draft_model_path,
        "cache_size": cache_size,
        "cache_quant": cache_quant,
        "gpu_split_gb": gpu_split_gb,
    }
    environment_payload = {"schema_version": 1, **environment}
    summary_payload = {
        "schema_version": 1,
        "qualification": "bringup_only",
        "model_path": model_path,
        "draft_model_path": draft_model_path,
        "cache_size": cache_size,
        "cache_quant": cache_quant,
        "gpu_split_gb": gpu_split_gb,
        **summary,
    }
    observation_lines = "".join(
        json.dumps(asdict(observation), sort_keys=True) + "\n"
        for observation in observations
    )
    _write_text_atomic(
        output_directory / "config.json",
        json.dumps(config_payload, indent=2, sort_keys=True) + "\n",
    )
    _write_text_atomic(
        output_directory / "environment.json",
        json.dumps(environment_payload, indent=2, sort_keys=True) + "\n",
    )
    _write_text_atomic(
        output_directory / "observations.jsonl", observation_lines
    )
    _write_text_atomic(
        output_directory / "summary.json",
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n",
    )
    return summary_payload


def build_branch_token_ids(
    seed_ids: Sequence[int], suffix_ids: Sequence[int], bucket_tokens: int
) -> list[int]:
    if bucket_tokens <= 0:
        raise ValueError("bucket_tokens must be positive")
    if len(seed_ids) >= bucket_tokens:
        raise ValueError("seed must be shorter than bucket_tokens")
    required_suffix = bucket_tokens - len(seed_ids)
    if len(suffix_ids) < required_suffix:
        raise ValueError("suffix does not contain enough tokens for the bucket")
    return [*seed_ids, *suffix_ids[:required_suffix]]


def run_generator_job(
    generator: object,
    job_factory: Callable[..., object],
    input_ids: object,
    *,
    max_new_tokens: int,
    seed: int,
    bucket_tokens: int,
    phase: Literal["seed", "turn"],
    repetition: int,
) -> tuple[JobObservation, object]:
    input_tokens = _token_count(input_ids)
    page_metrics_before = _page_metrics(generator)
    job = job_factory(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        seed=seed,
    )
    started_ns = time.perf_counter_ns()
    generator.enqueue(job)

    first_token_ns: int | None = None
    generated_tokens = 0
    saw_eos = False
    while generator.num_remaining_jobs():
        for result in generator.iterate():
            if result.get("stage") != "streaming":
                continue
            if first_token_ns is None and _result_has_token(result):
                first_token_ns = time.perf_counter_ns()
            result_new_tokens = result.get("new_tokens")
            if isinstance(result_new_tokens, int):
                generated_tokens = max(generated_tokens, result_new_tokens)
            else:
                generated_tokens += _token_count_or_zero(result.get("token_ids"))
            if result.get("eos"):
                saw_eos = True

    if first_token_ns is None:
        raise RuntimeError("generator queue drained without a streaming token")
    if not saw_eos:
        raise RuntimeError("generator queue drained without EOS")

    cached_pages = int(getattr(job, "cached_pages", 0))
    cached_tail_tokens = int(getattr(job, "cached_tokens", 0))
    cached_tokens = min(
        cached_pages * PAGE_SIZE + cached_tail_tokens,
        max(input_tokens - 1, 0),
    )
    page_metrics_after = _page_metrics(generator)
    page_metrics = {
        key: max(page_metrics_after.get(key, 0) - page_metrics_before.get(key, 0), 0)
        for key in sorted(page_metrics_before.keys() | page_metrics_after.keys())
    }
    if generated_tokens == 0:
        generated_tokens = int(getattr(job, "new_tokens", 0))

    observation = JobObservation(
        bucket_tokens=bucket_tokens,
        phase=phase,
        repetition=repetition,
        input_tokens=input_tokens,
        cached_tokens=cached_tokens,
        prefill_tokens=max(input_tokens - 1 - cached_tokens, 0),
        ttft_ms=(first_token_ns - started_ns) / 1_000_000,
        prefill_ms=float(getattr(job, "time_prefill", 0.0)) * 1000,
        generated_tokens=generated_tokens,
        page_metrics=page_metrics,
    )
    return observation, job


def run_resident_matrix(
    generator: object,
    tokenizer: object,
    config: ResidentProbeConfig,
    job_factory: Callable[..., object],
) -> tuple[list[JobObservation], dict[str, object]]:
    source_ids = _encode_token_ids(tokenizer, CODING_TOKEN_SOURCE)
    if not source_ids:
        raise RuntimeError("tokenizer produced an empty coding token source")

    observations: list[JobObservation] = []
    for bucket_tokens in config.context_tokens:
        input_budget = bucket_tokens - config.max_new_tokens
        seed_input_tokens = (
            input_budget - config.appended_tokens - config.max_new_tokens
        )
        seed_ids = _tile_token_ids(source_ids, seed_input_tokens)
        seed_observation, seed_job = run_generator_job(
            generator,
            job_factory,
            _to_input_tensor(seed_ids),
            max_new_tokens=config.max_new_tokens,
            seed=bucket_tokens,
            bucket_tokens=bucket_tokens,
            phase="seed",
            repetition=0,
        )
        observations.append(seed_observation)
        resident_seed_ids = _job_sequence_ids(seed_job)

        for repetition in range(config.repetitions):
            suffix_source = _encode_token_ids(
                tokenizer,
                f" Resident branch {repetition}: inspect, patch, test, and report.",
            )
            suffix_ids = _tile_token_ids(
                suffix_source,
                input_budget - len(resident_seed_ids),
            )
            branch_ids = build_branch_token_ids(
                resident_seed_ids, suffix_ids, input_budget
            )
            observation, _ = run_generator_job(
                generator,
                job_factory,
                _to_input_tensor(branch_ids),
                max_new_tokens=config.max_new_tokens,
                seed=bucket_tokens + repetition + 1,
                bucket_tokens=bucket_tokens,
                phase="turn",
                repetition=repetition,
            )
            observations.append(observation)

    aggregate = summarize_observations(observations)
    summary = {**aggregate, "slo": evaluate_resident_slo(aggregate)}
    return observations, summary


def _token_count(value: object) -> int:
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) == 0:
        raise TypeError("input_ids must expose a non-empty shape")
    return int(shape[-1])


def _token_count_or_zero(value: object) -> int:
    if value is None:
        return 0
    return _token_count(value)


def _result_has_token(result: dict[str, object]) -> bool:
    if _token_count_or_zero(result.get("token_ids")) > 0:
        return True
    text = result.get("text")
    return isinstance(text, str) and bool(text)


def _page_metrics(generator: object) -> dict[str, int]:
    pagetable = getattr(generator, "pagetable", None)
    metrics = getattr(pagetable, "metrics", {})
    return {str(key): int(value) for key, value in dict(metrics).items()}


def _encode_token_ids(tokenizer: object, text: str) -> list[int]:
    encoded = tokenizer.encode(text, add_bos=False)
    if hasattr(encoded, "flatten"):
        encoded = encoded.flatten()
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    return [int(token_id) for token_id in encoded]


def _tile_token_ids(source_ids: Sequence[int], count: int) -> list[int]:
    if count < 0:
        raise ValueError("token count must be non-negative")
    if count == 0:
        return []
    if not source_ids:
        raise ValueError("cannot tile an empty token sequence")
    repetitions = (count + len(source_ids) - 1) // len(source_ids)
    return (list(source_ids) * repetitions)[:count]


def _to_input_tensor(token_ids: Sequence[int]) -> object:
    import torch

    return torch.tensor([token_ids], dtype=torch.long)


def _job_sequence_ids(job: object) -> list[int]:
    sequences = getattr(job, "sequences", None)
    if not sequences:
        raise RuntimeError("completed job does not expose a sequence")
    sequence_ids = sequences[0].sequence_ids.torch()
    return [int(token_id) for token_id in sequence_ids.flatten().tolist()]


def _write_text_atomic(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _git_dirty(repository: Path) -> bool | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), "status", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return bool(completed.stdout.strip())
