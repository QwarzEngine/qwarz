"""Device-side runner for one resident session.

Weights reload per layer. The Gated DeltaNet state and the NVFP4 pages stay
allocated across turns, so a suffix forward starts at the committed cursor.
"""
from __future__ import annotations

from pathlib import Path

import torch

from engine.forward.embed import gather, load_weight
from engine.forward.mlp import _runtime
from engine.forward.token import LAYERS, apply_layer, load_spec, logits, rms
from engine.forward.vision import MM_TOKEN_BASE


class ModelRunner:
    def __init__(self, pages, model_dir=None, retain=False):
        self.pages = pages
        self.model_dir = Path(model_dir) if model_dir else Path.home() / "models/Qwen3.8-27B-EXL3-5.0bpw"
        self.retain = retain
        self.states = {}
        self.caches = {}
        self.specs = {}
        self.calls = []
        self._model = None
        self._donor = None
        self._table = None
        self.last = None
        self.hidden = None

    def _ensure(self):
        if self._model is not None:
            return
        _runtime()
        from qwasar_runtime.hybrid import donor_dir, prepare_environment
        from engine.forward.token import Catalog

        prepare_environment()
        self._model = Catalog(self.model_dir)
        self._donor = Catalog(donor_dir())
        self._table = load_weight(torch, self.model_dir)

    def forward(self, token_ids, cache_len, embedded=None, inv_freq=None):
        self._ensure()
        ids = list(token_ids)
        if any(token >= MM_TOKEN_BASE for token in ids) and embedded is None:
            raise RuntimeError("image ids need vision rows")
        if embedded is None:
            residual = gather(
                self._table, torch.tensor([ids], dtype=torch.long), torch.float32,
            ).cuda().contiguous()
        else:
            residual = embedded if embedded.is_cuda else embedded.cuda()
            if residual.dtype != torch.float32:
                residual = residual.float()
            if residual.dim() == 2:
                residual = residual.unsqueeze(0)
            # The norm fuses the residual in place. A later span still reads this table.
            residual = residual.contiguous().clone()
        for index in range(LAYERS):
            spec = self.specs.get(index)
            if spec is None:
                spec = load_spec(self._model, self._donor, index)
                if self.retain:
                    self.specs[index] = spec
            residual, _normed, _sublayer, _produced = apply_layer(
                residual, index, spec, self.states, self.caches, cache_len, self.pages, inv_freq,
            )
            if not self.retain:
                del spec
        self.calls.append((cache_len, list(token_ids)))
        self.last = residual
        self.hidden = rms(self._model.tensor("model.language_model.norm.weight"), residual)
        return residual

    def predict(self):
        if self.last is None:
            raise RuntimeError("no residual to score")
        return int(logits(self._model, self.last[:, -1:, :]).reshape(-1).argmax())

    def capture(self):
        return {
            index: (state.conv.clone(), state.recurrent.clone(), state.ready)
            for index, state in self.states.items()
        }

    def restore(self, snapshot):
        for index, (conv, recurrent, ready) in snapshot.items():
            state = self.states[index]
            state.conv.copy_(conv)
            state.recurrent.copy_(recurrent)
            state.ready = ready

    def reset(self):
        self.states.clear()
        self.caches.clear()
        self.last = None
        self.hidden = None
