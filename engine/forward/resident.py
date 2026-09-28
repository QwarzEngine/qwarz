"""Device-side runner for one resident session.

Weights reload per layer. The Gated DeltaNet state and the NVFP4 pages stay
allocated across turns, so a suffix forward starts at the committed cursor.
"""
from __future__ import annotations

from pathlib import Path

import torch

from engine.forward.embed import gather, load_weight
from engine.forward.mlp import _runtime
from engine.forward.token import LAYERS, apply_layer, load_spec, logits


class ModelRunner:
    def __init__(self, pages, model_dir=None):
        self.pages = pages
        self.model_dir = Path(model_dir) if model_dir else Path.home() / "models/Qwen3.8-27B-EXL3-5.0bpw"
        self.states = {}
        self.caches = {}
        self.calls = []
        self._model = None
        self._donor = None
        self._table = None
        self.last = None

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

    def forward(self, token_ids, cache_len):
        self._ensure()
        ids = torch.tensor([list(token_ids)], dtype=torch.long)
        residual = gather(self._table, ids, torch.float32).cuda().contiguous()
        for index in range(LAYERS):
            spec = load_spec(self._model, self._donor, index)
            residual, _normed, _sublayer, _produced = apply_layer(
                residual, index, spec, self.states, self.caches, cache_len, self.pages,
            )
            del spec
        self.calls.append((cache_len, list(token_ids)))
        self.last = residual
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
