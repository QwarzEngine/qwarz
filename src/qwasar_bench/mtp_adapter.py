"""Experimental learned MTP output correction; target weights are never touched."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import math


class ResidualAdapter:
    def __init__(self, hidden, rank, device="cpu", seed=923):
        import torch
        if not 0 < rank <= hidden:
            raise ValueError("invalid adapter rank")
        generator = torch.Generator(device=device).manual_seed(seed)
        self.a = torch.nn.Parameter(torch.randn(hidden, rank, generator=generator, device=device) / math.sqrt(hidden))
        self.b = torch.nn.Parameter(torch.zeros(rank, hidden, device=device))

    def parameters(self):
        return (self.a, self.b)

    def __call__(self, x):
        delta = (x.float() @ self.a) @ self.b
        return (x.float() + delta).to(x.dtype)

    def state_dict(self):
        return {"a": self.a.detach().contiguous(), "b": self.b.detach().contiguous()}

    @classmethod
    def load(cls, path, device="cuda"):
        import torch
        from safetensors.torch import load_file
        tensors = load_file(str(path), device=device)
        if set(tensors) != {"a", "b"}:
            raise ValueError("unexpected adapter tensors")
        a, b = tensors["a"], tensors["b"]
        if a.ndim != 2 or b.shape != (a.shape[1], a.shape[0]) or not 0 < a.shape[1] <= a.shape[0]:
            raise ValueError("invalid adapter geometry")
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError("nonfinite adapter")
        result = cls(a.shape[0], a.shape[1], device=device)
        result.a = torch.nn.Parameter(a.float(), requires_grad=False)
        result.b = torch.nn.Parameter(b.float(), requires_grad=False)
        return result


def validate_alignment(draft_ids, positions, verify_ids, start):
    """All rows predict after the SAME input token and position, including depth 1."""
    if not draft_ids or len(draft_ids) != len(positions) or len(draft_ids) > 6:
        raise ValueError("invalid draft window")
    count = min(len(draft_ids), len(verify_ids))
    if draft_ids[:count] != verify_ids[:count]:
        raise ValueError("draft/verifier input token mismatch")
    if positions != list(range(start, start + len(positions))):
        raise ValueError("draft/verifier position mismatch")
    return count


@contextmanager
def install(generator, adapter):
    draft, target = generator.draft_model, generator.model
    if not generator.mtp_draft or generator.num_draft_tokens != 6:
        raise ValueError("requires MTP6")
    norm = draft.final_norm
    if norm is target.modules[target.logit_layer_idx - 1]:
        raise ValueError("MTP norm aliases target")
    if adapter.a.shape[0] != draft.config.hidden_size:
        raise ValueError("adapter hidden size mismatch")
    original = norm.forward
    target_forward, target_head = target.forward, target.modules[target.logit_layer_idx]

    def forward(*args, **kwargs):
        return adapter(original(*args, **kwargs))

    norm.forward = forward
    try:
        yield
    finally:
        norm.forward = original
        if target.forward != target_forward or target.modules[target.logit_layer_idx] is not target_head:
            raise RuntimeError("target changed during adapter experiment")


class PairCollector:
    """Instrumentation only. Captures all six speculative prefixes, not just accepted rows."""
    def __init__(self, generator):
        self.generator = generator
        self.pending = []
        self.rows = []
        self.windows = 0
        self.original_draft = generator.draft_model.forward
        self.original_target = generator.model.forward

    def __enter__(self):
        import torch

        def draft_forward(input_ids, params=None):
            # IDs and CPU cache lengths are mutated by the rendezvous loop.
            ids = input_ids.detach().cpu().flatten().tolist()
            positions = params["cache_seqlens"].detach().cpu().flatten().tolist()
            output = self.original_draft(input_ids, params)
            if len(ids) != 1 or len(positions) != 1:
                raise ValueError("collector supports one serial session")
            self.pending.append((ids[0], positions[0], output.detach().cpu().half().clone()))
            return output

        def target_forward(input_ids, params=None):
            is_verify = bool(params.get("pinned_staging"))
            if is_verify:
                ids = input_ids.detach().cpu().flatten().tolist()
                start = params["cache_seqlens"].item()
            elif self.pending:
                raise ValueError("unconsumed draft window before prefill")
            output = self.original_target(input_ids, params)
            if is_verify and self.pending:
                count = validate_alignment([p[0] for p in self.pending],
                                           [p[1] for p in self.pending], ids, start)
                hidden = params["export_states"][-1]
                if hidden.shape[:2] != input_ids.shape:
                    raise ValueError("missing full target hidden export")
                draft = torch.cat([p[2] for p in self.pending[:count]], dim=1).squeeze(0)
                teacher = hidden[:, :count].detach().cpu().half().squeeze(0).clone()
                top = output[:, :count].argmax(-1).detach().cpu().flatten()
                self.rows.append({"draft": draft, "teacher": teacher, "full_top": top,
                                  "depth": torch.arange(1, count + 1),
                                  "position": torch.arange(start, start + count),
                                  "input_id": torch.tensor(ids[:count])})
                self.windows += 1
                self.pending.clear()
            return output

        self.generator.draft_model.forward = draft_forward
        self.generator.model.forward = target_forward
        return self

    def reset(self):
        if self.pending:
            raise ValueError("unconsumed draft states at document boundary")
        self.rows.clear()
        self.windows = 0

    def tensors(self):
        import torch
        if self.pending or not self.rows:
            raise ValueError("incomplete or empty captured document")
        return {key: torch.cat([row[key] for row in self.rows]).contiguous() for key in self.rows[0]}

    def __exit__(self, *args):
        self.generator.draft_model.forward = self.original_draft
        self.generator.model.forward = self.original_target


def file_hash(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
