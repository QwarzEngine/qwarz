"""Capture, train and evaluate an opt-in MTP residual adapter on one RTX 5090."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys

from .mtp_adapter import PairCollector, ResidualAdapter, file_hash, install
from .mtp_tasks import manifest
from .nvfp4_study import MODEL, ROOT, setup, write


def backend(context):
    from qwasar_runtime.engine import ExLlamaBackend
    from .decode_probe import configure_mtp_policy
    result = ExLlamaBackend(MODEL, context, prefill="xqa")
    cfg = result.config
    if not (cfg["vision"] and cfg["mtp_head"]["installed"] and cfg["rendezvous"]["gpu_embedding"]):
        raise RuntimeError("incomplete production-stack load")
    configure_mtp_policy(result.generator, "fixed6")
    return result


def prompt_ids(tokenizer, prompt):
    text = tokenizer.hf_render_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True,
        enable_thinking=True, reasoning_effort="medium", preserve_thinking=True)
    return tokenizer.encode(text, encode_special_tokens=True).flatten().tolist()


def warmup_tokens(cache):
    count = min(8200, cache.max_num_tokens - 1024)
    if count <= 0:
        raise ValueError("cache too small for warmup")
    return count


def warmup(b, torch):
    from exllamav3 import Job
    from exllamav3.generator.sampler.presets import GreedySampler
    from .fidelity_probe import fresh_generator
    count = warmup_tokens(b.generator.cache)
    ids = torch.randint(10000, 50000, (1, count))
    with b.tuning():
        b.generator.enqueue(Job(input_ids=ids, max_new_tokens=48, sampler=GreedySampler(),
                                seed=923, stop_conditions=[]))
        while b.remaining():
            b.iterate()
    fresh_generator(b.generator)


def export_head(b, out, torch):
    from exllamav3.ext import exllamav3_ext as ext
    from safetensors.torch import save_file
    head = b.generator.draft_model.mtp_sub_lm_head
    weight = torch.empty((head.in_features, head.out_features), dtype=torch.float16, device="cpu")
    for start in range(0, head.out_features, 4096):
        block = torch.empty((head.in_features, 4096), dtype=torch.float16, device="cuda")
        ext.reconstruct_had_slice(block, head.trellis, head.suh, head.svh[start:],
                                  head.K, head.mcg, head.mul1, start)
        weight[:, start:start + 4096] = block.cpu()
    torch.cuda.synchronize()
    dense = weight.cuda()
    x = torch.randn(32, head.in_features, device="cuda", dtype=torch.float16)
    reference = head.forward(x, {})
    actual = x @ dense
    error = ((actual.float() - reference.float()).square().mean()
             / reference.float().square().mean()).sqrt().item()
    agreement = (actual.argmax(-1) == reference.argmax(-1)).float().mean().item()
    if not error < .03 or agreement < .9:
        raise RuntimeError(f"head reconstruction mismatch: {error}, {agreement}")
    save_file({"weight": weight, "token_ids": b.generator.draft_model.mtp_hot_id_map.cpu()},
              str(out / "head.safetensors"))
    write(out / "head-validation.json", {"relative_rms": error, "argmax_agreement": agreement,
                                       "rows": 32, "sha256": file_hash(out / "head.safetensors")})


def capture(out, limit, tokens):
    torch = setup()
    from safetensors.torch import save_file
    from .decode_probe import run_sample
    from .fidelity_probe import fresh_generator
    b = backend(8192)
    data = manifest()
    write(out / "tasks.json", data)
    write(out / "configuration.json", b.config)
    rows = []
    try:
        warmup(b, torch)
        export_head(b, out, torch)
        with PairCollector(b.generator) as collector, b.tuning():
            for task in data["tasks"][:limit]:
                fresh_generator(b.generator)
                collector.reset()
                ids = prompt_ids(b.tokenizer, task["prompt"])
                options = argparse.Namespace(max_new_tokens=tokens, thinking="medium", sampler="recommended")
                sample = run_sample(b.generator, b.tokenizer, ids, options, task["seed"],
                                    out / f"{task['id']}-events.jsonl")
                tensors = collector.tensors()
                path = out / f"{task['id']}.safetensors"
                save_file(tensors, str(path))
                record = {**task, "pairs": tensors["draft"].shape[0], "windows": collector.windows,
                          "path": path.name, "sha256": file_hash(path),
                          "generated_tokens": sample["generated_tokens"]}
                rows.append(record)
                write(out / f"{task['id']}-sample.json", sample)
                print(json.dumps({k: record[k] for k in ("id", "pairs", "windows", "generated_tokens")}), flush=True)
        write(out / "dataset.json", {"documents": rows, "manifest_sha256": data["sha256"],
              "pairing": "D[t] predicts after input[t]; teacher H[t] has the identical speculative prefix",
              "includes_rejected_speculative_prefixes": True, "completed": True})
    finally:
        b._lifetime.close()


def load_split(root, split, max_per_doc=1024):
    import torch
    from safetensors.torch import load_file
    data = json.loads((root / "dataset.json").read_text())
    rows = []
    for doc in data["documents"]:
        if doc["split"] != split:
            continue
        path = root / doc["path"]
        if file_hash(path) != doc["sha256"]:
            raise ValueError("captured document hash changed")
        tensors = load_file(str(path))
        # Evenly retain all depths/positions, with equal per-document caps.
        idx = torch.linspace(0, tensors["draft"].shape[0] - 1,
                             min(max_per_doc, tensors["draft"].shape[0])).long()
        rows.append({key: value[idx] for key, value in tensors.items()})
    if not rows:
        raise ValueError(f"empty {split} split")
    return {key: torch.cat([r[key] for r in rows]).cuda() for key in rows[0]}


def teacher_targets(data, weight, token_ids, torch):
    targets, top = [], []
    with torch.no_grad():
        for start in range(0, len(data["teacher"]), 128):
            logits = data["teacher"][start:start + 128] @ weight
            top.append(logits.argmax(-1))
            scores, indices = logits.float().topk(64, dim=-1)
            # Sparse teacher over the fixed hot vocabulary; tail mass is recorded.
            logz = logits.float().logsumexp(-1, keepdim=True)
            targets.append((indices, scores.softmax(-1), (scores.logsumexp(-1, keepdim=True) - logz).exp()))
    data["teacher_top"] = torch.cat(top)
    data["top_indices"] = torch.cat([t[0] for t in targets])
    data["top_probs"] = torch.cat([t[1] for t in targets])
    data["retained_mass"] = torch.cat([t[2] for t in targets])
    data["full_top_in_hot"] = torch.isin(data["full_top"], token_ids)


def evaluate(adapter, data, weight, torch):
    total, nll, agrees, delta_norm = 0, 0., 0., 0.
    depth_counts = {}
    with torch.no_grad():
        for start in range(0, len(data["draft"]), 128):
            x = data["draft"][start:start + 128]
            z = adapter(x) if adapter else x
            logits = (z @ weight).float()
            lp = logits.log_softmax(-1)
            idx, probs = data["top_indices"][start:start + 128], data["top_probs"][start:start + 128]
            loss = -(lp.gather(1, idx) * probs).sum(-1)
            correct = logits.argmax(-1) == data["teacher_top"][start:start + 128]
            total += len(x)
            nll += loss.sum().item()
            agrees += correct.sum().item()
            delta_norm += ((z.float() - x.float()).square().mean(-1) / x.float().square().mean(-1)).sum().item()
            for depth in range(1, 7):
                mask = data["depth"][start:start + 128] == depth
                entry = depth_counts.setdefault(depth, {"n": 0, "agree": 0})
                entry["n"] += mask.sum().item()
                entry["agree"] += correct[mask].sum().item()
    return {"n": total, "teacher_cross_entropy": nll / total,
            "hot_teacher_argmax_agreement": agrees / total,
            "relative_correction_rms": (delta_norm / total) ** .5,
            "by_depth": depth_counts,
            "teacher_top64_mass_mean": data["retained_mass"].mean().item(),
            "full_teacher_top_in_hot": data["full_top_in_hot"].float().mean().item()}


def train(out, source, steps):
    torch = setup()
    from safetensors.torch import load_file, save_file
    import torch.nn.functional as F
    head = load_file(str(source / "head.safetensors"), device="cuda")
    weight, token_ids = head["weight"], head["token_ids"]
    train_data = load_split(source, "train")
    validation = load_split(source, "validation")
    teacher_targets(train_data, weight, token_ids, torch)
    teacher_targets(validation, weight, token_ids, torch)
    write(out / "baseline.json", {"train": evaluate(None, train_data, weight, torch),
                                 "validation": evaluate(None, validation, weight, torch)})
    best_score, best = float("inf"), None
    histories = []
    for rank in (16, 32):
        adapter = ResidualAdapter(weight.shape[0], rank, device="cuda")
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=.01)
        rng = torch.Generator(device="cuda").manual_seed(923)
        for step in range(1, steps + 1):
            idx = torch.randint(len(train_data["draft"]), (128,), generator=rng, device="cuda")
            x = train_data["draft"][idx]
            z = adapter(x)
            logits = (z @ weight).float()
            loss = -(F.log_softmax(logits, dim=-1).gather(1, train_data["top_indices"][idx])
                     * train_data["top_probs"][idx]).sum(-1).mean()
            regularizer = ((z.float() - x.float()).square().mean()
                           / x.float().square().mean().clamp_min(1e-8))
            objective = loss + .1 * regularizer
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.)
            optimizer.step()
            if step % 100 == 0 or step == steps:
                metrics = evaluate(adapter, validation, weight, torch)
                record = {"rank": rank, "step": step, "train_loss": loss.item(), "validation": metrics}
                histories.append(record)
                print(json.dumps(record), flush=True)
                save_file(adapter.state_dict(), str(out / f"r{rank}-s{step}.safetensors"))
                if metrics["teacher_cross_entropy"] < best_score:
                    best_score, best = metrics["teacher_cross_entropy"], (rank, step)
        del optimizer, adapter
    if best is None:
        raise RuntimeError("no checkpoint selected")
    selected = out / f"r{best[0]}-s{best[1]}.safetensors"
    write(out / "selection.json", {"path": str(selected), "sha256": file_hash(selected),
          "rank": best[0], "step": best[1], "selection_split": "validation",
          "test_used_for_selection": False, "head_sha256": file_hash(source / "head.safetensors"),
          "dataset_manifest_sha256": json.loads((source / "dataset.json").read_text())["manifest_sha256"]})
    write(out / "history.json", histories)
    # The held-out family split is first opened after checkpoint selection is fixed.
    heldout = load_split(source, "test")
    teacher_targets(heldout, weight, token_ids, torch)
    adapter = ResidualAdapter.load(selected)
    write(out / "heldout.json", {"control": evaluate(None, heldout, weight, torch),
                                "candidate": evaluate(adapter, heldout, weight, torch)})


def online(out, adapter_path, contexts, repeats, tokens):
    torch = setup()
    from .decode_probe import run_sample
    from .fidelity_probe import fresh_generator
    b = backend(262144)
    adapter = ResidualAdapter.load(adapter_path)
    records = []
    try:
        write(out / "configuration.json", {"production": b.config, "adapter": str(adapter_path),
              "adapter_sha256": file_hash(adapter_path), "rank": adapter.a.shape[1]})
        warmup(b, torch)
        data = [row for row in manifest()["tasks"] if row["split"] == "test"
                and row["id"].endswith("-0")]
        # Existing long corpus is used only for held-out online measurement, not training.
        cells = json.loads((ROOT / "results/20260910-quality-gate/prompts-phase2.json").read_text())
        corpus = max(cells, key=lambda cell: len(cell["prompt_ids"]))["prompt_ids"]
        with b.tuning():
            for context in contexts:
                for rep in range(repeats):
                    row = data[rep % len(data)]
                    # Long filler is enclosed within the user turn of the new task.
                    marker = "__MTP_STUDY_CORPUS__"
                    text = b.tokenizer.hf_render_chat_template(
                        [{"role": "user", "content": "Reference text:\n" + marker +
                          "\nEnd reference.\n" + row["prompt"]}], add_generation_prompt=True,
                        enable_thinking=True, reasoning_effort="medium", preserve_thinking=True)
                    prefix, suffix = text.split(marker)
                    begin = b.tokenizer.encode(prefix, encode_special_tokens=True).flatten().tolist()
                    end = b.tokenizer.encode(suffix, encode_special_tokens=True).flatten().tolist()
                    # Strip the archived framing and retain literal ordinary token IDs only.
                    filler = [t for t in corpus if t < 248044]
                    ids = begin + filler[:context - len(begin) - len(end)] + end
                    if len(ids) != context or context + tokens + 16 > 262144:
                        raise ValueError("invalid online context budget")
                    for arm in (("control", "candidate", "candidate", "control") if rep % 2 == 0
                                else ("candidate", "control", "control", "candidate")):
                        label = f"c{context}-r{rep}-{len(records):03}-{arm}"
                        fresh_generator(b.generator)
                        manager = install(b.generator, adapter) if arm == "candidate" else nullcontext()
                        with manager:
                            options = argparse.Namespace(max_new_tokens=tokens, thinking="medium", sampler="recommended")
                            sample = run_sample(b.generator, b.tokenizer, ids, options, 827 + rep,
                                                out / f"{label}-events.jsonl")
                        verifies = len(sample["draft_stats"])
                        record = {"label": label, "arm": arm, "context": context, "rep": rep,
                                  **{k: sample[k] for k in ("ttft_ms", "elapsed_ms", "generated_tokens",
                                     "decode_tokens_per_second", "draft_acceptance", "truncated", "prompt_sha256")},
                                  "verifies": verifies,
                                  "decode_ms_per_verify_estimate":
                                      (sample["elapsed_ms"] - sample["ttft_ms"]) / verifies if verifies else None}
                        write(out / f"{label}-sample.json", {**sample, **record})
                        records.append(record)
                        print(json.dumps(record), flush=True)
        write(out / "summary.json", {"samples": records, "completed": True})
    finally:
        b._lifetime.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("capture", "train", "online"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--contexts", type=int, nargs="+", default=[4096, 32768])
    parser.add_argument("--repeats", type=int, default=4)
    args = parser.parse_args()
    if args.mode == "train" and args.source is None or args.mode == "online" and args.adapter is None:
        parser.error("train requires --source; online requires --adapter")
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "source.json", {"command": sys.argv, "source_sha256": file_hash(Path(__file__)),
          "adapter_code_sha256": file_hash(Path(__file__).with_name("mtp_adapter.py"))})
    if args.mode == "capture":
        capture(args.output, args.limit, args.tokens)
    elif args.mode == "train":
        train(args.output, args.source, args.steps)
    else:
        online(args.output, args.adapter, args.contexts, args.repeats, args.tokens)


if __name__ == "__main__":
    main()
