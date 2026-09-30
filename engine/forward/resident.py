"""Device-side runner for one resident session.

Weights reload per layer. The Gated DeltaNet state and the NVFP4 pages stay
allocated across turns, so a suffix forward starts at the committed cursor.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

from engine.forward.embed import gather, load_weight
from engine.forward.mlp import _runtime
from engine.forward.mtp import DRAFT_TOKENS
from engine.forward.schedule import PAGE
from engine.forward.token import LAYERS, _proposer_head, apply_layer, load_spec, logits, rms
from engine.forward.vision import MM_TOKEN_BASE


REWIND_GRAPHS = os.environ.get("QWARZ_REWIND_GRAPH", "1") != "0"


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
        self._decode_graphs = {}
        self._prefill_graphs = {}
        self._rewind_graphs = {}
        self._graph_pool = None
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
        # The whole table stays on the device. Gathering rows on the host
        # and copying them in was a sync on every prefill chunk and every draft token.
        self._table = load_weight(torch, self.model_dir).cuda()
        # Reloading this row from the host is an unpinned copy. A CUDA graph
        # of the verify forward cannot do that, and it would sync every token.
        self._final_norm = self._model.tensor("model.language_model.norm.weight")
        # The proposer slice is resident before any prefill graph is captured,
        # so the graph pool does not fragment the allocation that holds it.
        _proposer_head(self._model)
        self._decode_graphs = {}
        self._prefill_graphs = {}
        self._rewind_graphs = {}
        self._graph_pool = torch.cuda.graph_pool_handle()

    def _decode_bucket(self, cache_len, width=DRAFT_TOKENS + 1):
        """Power-of-two page count that covers this verify, capped at the allocation."""
        used = (cache_len + width + PAGE - 1) // PAGE
        bucket = 1
        while bucket < used:
            bucket <<= 1
        if bucket > self.pages:
            return self.pages
        return bucket

    def _decode_ready(self, token_ids, cache_len, embedded, inv_freq):
        return (
            embedded is None
            and inv_freq is None
            and isinstance(cache_len, int)
            and len(token_ids) == DRAFT_TOKENS + 1
            and bool(self.states)
            and all(state.ready for state in self.states.values())
        )

    def _cold_prefill(self, token_ids, cache_len):
        # The 255-id graph replays the chunk kernel from a zero recurrent state.
        # A later 255-id span (the tail of a 4096 prompt) already has a past.
        if not isinstance(cache_len, int) or cache_len != 0 or len(token_ids) != 255:
            return False
        return not self.states or not any(state.ready for state in self.states.values())

    def forward(self, token_ids, cache_len, embedded=None, inv_freq=None, ids_tensor=None, cache_tensor=None):
        self._ensure()
        plain = ids_tensor is None and cache_tensor is None and embedded is None and inv_freq is None
        if plain:
            replayed = self._replay_prefill(token_ids, cache_len)
            if replayed is not None:
                return replayed
            replayed = self._replay_decode(token_ids, cache_len)
            if replayed is not None:
                return replayed
            if self._cold_prefill(token_ids, cache_len):
                slot = self._prefill_graphs.get(255)
                if slot is not None and slot.get("warm", 0) >= 2 and not slot.get("ready"):
                    return self._capture_prefill(token_ids)
            if self._decode_ready(token_ids, cache_len, embedded, inv_freq):
                bucket = self._decode_bucket(cache_len)
                slot = self._decode_graphs.get(bucket)
                if slot is not None and slot.get("warm", 0) >= 2 and not slot.get("ready"):
                    return self._capture_decode(token_ids, cache_len, bucket)
        prefill_cold = plain and self._cold_prefill(token_ids, cache_len)
        decode_bucket = None
        if plain and self._decode_ready(token_ids, cache_len, embedded, inv_freq):
            decode_bucket = self._decode_bucket(cache_len)
        if decode_bucket is not None:
            from engine.forward.attention import bind_table_pages

            bind_table_pages(decode_bucket)
        try:
            residual = self._forward_eager(
                token_ids, cache_len, embedded, inv_freq, ids_tensor, cache_tensor,
            )
        finally:
            if decode_bucket is not None:
                from engine.forward.attention import bind_table_pages

                bind_table_pages(None)
        if prefill_cold:
            slot = self._prefill_graphs.setdefault(255, {"warm": 0, "ready": False})
            if not slot.get("ready"):
                slot["warm"] = slot.get("warm", 0) + 1
        if decode_bucket is not None:
            slot = self._decode_graphs.setdefault(
                decode_bucket, {"warm": 0, "ready": False, "pages": decode_bucket},
            )
            if not slot.get("ready"):
                slot["warm"] = slot.get("warm", 0) + 1
        return residual

    def _forward_eager(self, token_ids, cache_len, embedded, inv_freq, ids_tensor, cache_tensor):
        ids = list(token_ids)
        if ids_tensor is None and any(token >= MM_TOKEN_BASE for token in ids) and embedded is None:
            raise RuntimeError("image ids need vision rows")
        if embedded is None:
            if ids_tensor is None:
                ids_tensor = torch.tensor([ids], dtype=torch.long, device=self._table.device)
            residual = gather(self._table, ids_tensor, torch.float32).contiguous()
        else:
            residual = embedded if embedded.is_cuda else embedded.cuda()
            if residual.dtype != torch.float32:
                residual = residual.float()
            if residual.dim() == 2:
                residual = residual.unsqueeze(0)
            # The norm fuses the residual in place. A later span still reads this table.
            residual = residual.contiguous().clone()
        length = cache_tensor if cache_tensor is not None else cache_len
        for index in range(LAYERS):
            spec = self.specs.get(index)
            if spec is None:
                spec = load_spec(self._model, self._donor, index)
                if self.retain:
                    self.specs[index] = spec
            residual, _normed, _sublayer, _produced = apply_layer(
                residual, index, spec, self.states, self.caches, length, self.pages, inv_freq,
            )
            if not self.retain:
                del spec
        recorded = cache_len if isinstance(cache_len, int) else int(cache_tensor.detach().cpu())
        self.calls.append((recorded, ids if ids else ids_tensor.view(-1).tolist()))
        self.last = residual
        self.hidden = rms(self._final_norm, residual)
        return residual

    def _replay_prefill(self, token_ids, cache_len):
        if not self._cold_prefill(token_ids, cache_len) or not self.states:
            return None
        slot = self._prefill_graphs.get(255)
        if slot is None or not slot.get("ready"):
            return None
        slot["ids"].copy_(torch.tensor([list(token_ids)], dtype=torch.long, device=slot["ids"].device))
        slot["graph"].replay()
        self.last = slot["residual"]
        self.hidden = slot["hidden"]
        self.calls.append((0, list(token_ids)))
        # Replay does not run the Python that marks the chunk state current.
        for state in self.states.values():
            state.ready = True
        return slot["residual"]

    def _capture_prefill(self, token_ids):
        """Record the 255-id cold prefill. The chunk kernel does not read the old state."""
        ids = torch.tensor([list(token_ids)], dtype=torch.long, device=self._table.device)
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.no_grad():
            with torch.cuda.graph(graph, pool=self._graph_pool):
                residual = self._forward_eager(token_ids, 0, None, None, ids, None)
        self._prefill_graphs[255] = {
            "ids": ids,
            "graph": graph,
            "residual": residual,
            "hidden": self.hidden,
            "ready": True,
            "warm": 2,
        }
        # Capture records the launches and does not run them.
        graph.replay()
        self.last = residual
        for state in self.states.values():
            state.ready = True
        return residual

    def _replay_decode(self, token_ids, cache_len):
        # A 7-token prefill span is not a verify. Replay only when every GDN
        # state is already on the recurrent path the graph recorded.
        if not self._decode_ready(token_ids, cache_len, None, None):
            return None
        bucket = self._decode_bucket(cache_len)
        slot = self._decode_graphs.get(bucket)
        if slot is None or not slot.get("ready"):
            return None
        if slot.get("pages", 0) * PAGE < cache_len + len(token_ids):
            return None
        slot["ids"].copy_(torch.tensor([list(token_ids)], dtype=torch.long, device=slot["ids"].device))
        slot["cache"].fill_(cache_len)
        slot["graph"].replay()
        self.last = slot["residual"]
        self.hidden = slot["hidden"]
        self.calls.append((cache_len, list(token_ids)))
        # The stash kernels ran again. The Python length they pair with does not.
        width = len(token_ids)
        for state in self.states.values():
            state._rewind_len = width
        return slot["residual"]

    def _capture_decode(self, token_ids, cache_len, bucket):
        """Record the verify forward while running it, so this window stays real."""
        from engine.forward.attention import bind_table_pages

        ids = torch.tensor([list(token_ids)], dtype=torch.long, device=self._table.device)
        cache = torch.tensor([cache_len], dtype=torch.int32, device=self._table.device)
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        bind_table_pages(bucket)
        try:
            with torch.cuda.graph(graph, pool=self._graph_pool):
                residual = self._forward_eager(token_ids, cache_len, None, None, ids, cache)
        finally:
            bind_table_pages(None)
        self._decode_graphs[bucket] = {
            "ids": ids,
            "cache": cache,
            "graph": graph,
            "residual": residual,
            "hidden": self.hidden,
            "ready": True,
            "warm": 2,
            "pages": bucket,
        }
        # Stream capture records the launches and does not run them, so the
        # static residual is still empty. Replay once for this window.
        graph.replay()
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

    def capture_host(self, into=None):
        """Copy every GDN state into pinned host memory on the current stream.

        Attention pages are append-only, so a prefix needs only this recurrent
        state (~151 MB). ``into`` reuses an evicted mark's buffers; the copy is
        stream-ordered, so neither the capture nor a later restore syncs.
        """
        if into is None:
            into = {
                index: (
                    torch.empty(state.conv.shape, dtype=state.conv.dtype, pin_memory=True),
                    torch.empty(state.recurrent.shape, dtype=state.recurrent.dtype, pin_memory=True),
                    True,
                )
                for index, state in self.states.items()
            }
        marked = {}
        for index, state in self.states.items():
            conv, recurrent, _ready = into[index]
            conv.copy_(state.conv, non_blocking=True)
            recurrent.copy_(state.recurrent, non_blocking=True)
            marked[index] = (conv, recurrent, state.ready)
        return marked

    def restore(self, snapshot):
        for index, (conv, recurrent, ready) in snapshot.items():
            state = self.states[index]
            state.conv.copy_(conv, non_blocking=True)
            state.recurrent.copy_(recurrent, non_blocking=True)
            state.ready = ready
            state._rewind_len = None

    def rewind_recurrent(self, keep):
        """Cut every GDN state to the accepted prefix of the last short chunk.

        After a 7-token verify the cut is one graph per ``keep``: 48 layers of
        restore and replay were ~1.5 ms of Python launches for ~0.3 ms of work.
        Any other window length stays eager.
        """
        from engine.forward.projections import rewind_gdn_state

        if not self.states:
            raise RuntimeError("no GDN state to rewind")
        states = list(self.states.values())
        width = states[0]._rewind_len
        if (
            REWIND_GRAPHS
            and self._graph_pool is not None
            and width == DRAFT_TOKENS + 1
            and 0 <= keep < width
            and all(state._rewind_len == width for state in states)
        ):
            slot = self._rewind_graphs.setdefault(keep, {"warm": 0, "graph": None})
            if slot["graph"] is not None:
                slot["graph"].replay()
                for state in states:
                    state._rewind_len = None
                return
            if slot["warm"] >= 2:
                graph = torch.cuda.CUDAGraph()
                torch.cuda.synchronize()
                with torch.cuda.graph(graph, pool=self._graph_pool):
                    for state in states:
                        rewind_gdn_state(state, keep)
                slot["graph"] = graph
                # Capture records the launches and does not run them.
                graph.replay()
                for state in states:
                    state._rewind_len = None
                return
            slot["warm"] += 1
        for state in states:
            rewind_gdn_state(state, keep)

    def synchronize(self):
        torch.cuda.synchronize()

    def rezero(self):
        """Zero recurrent state and KV pages without freeing them."""
        for state in self.states.values():
            state.conv.zero_()
            state.recurrent.zero_()
            state.ready = False
            state._rewind_len = None
        for cache in self.caches.values():
            for tensor in cache:
                tensor.zero_()
        self.last = None
        self.hidden = None
        self.calls.clear()

    def reset(self):
        from engine.forward.attention import release_attention_plans

        release_attention_plans()
        self._decode_graphs = {}
        self._prefill_graphs = {}
        self._rewind_graphs = {}
        # Dropping every graph releases its private pool. Capturing into the
        # old handle afterwards trips the caching allocator's use_count assert.
        if self._graph_pool is not None:
            self._graph_pool = torch.cuda.graph_pool_handle()
        self.states.clear()
        self.caches.clear()
        self.last = None
        self.hidden = None
