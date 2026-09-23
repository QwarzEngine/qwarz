"""Draft-loop CUDA graph (promoted 2026-09-23).

Captures the six-step MTP draft walk into one CUDA graph replay per verify,
replacing the per-step host dispatch of the rendezvous R1a walk. Verbatim port
of the campaign class in ``results/20260922-draft-graph/draft_graph.py``; the
semantics below were certified in the 2026-09-22/23 campaigns:

  * Cycle effect: -0.9..-1.1 ms per verify at 32K, +5.6% tok/s at 256K, memory
    delta 0 (results/20260923-graph-band, all seven gates passed).
  * Fidelity: with the GDN determinism patch the stack is bit-reproducible;
    E-E and G-G runs are bit-identical, and the graph's in-process eager
    validation showed zero mismatches. E-G draft ids can diverge at near-tie
    states at 32K+ (a deterministic graph-vs-eager numeric difference in the
    draft walk), but greedy speculative verification makes the completion the
    target's own greedy sequence regardless of the proposals: 6/6 measured
    E-G pairs produced identical completions (results/20260923-gdn-followup).

Requirements at install time (all checked by DraftGraph.__init__): MTP draft
with a fixed six-token window and the promoted hot64k proposer head. The eager
fallback is whatever walk is installed at construction; on the production
stack that must be the rendezvous R1a GPU-chained walk (the captured body
feeds GPU ids to the draft forward), so the engine only installs this after
rendezvous and hot_head both installed.

BC bypass inside the capture: the draft's attention and MLP may route through
BC_Attention/BC_GatedMLP, which end in cudaGraphLaunch and cannot be recorded
by stream capture. The bypass routes them to their dispatch paths (the same
kernels BC replays) and is applied only around capture and validation.

Kill switch: QWASAR_DRAFT_GRAPH=0. QWASAR_DRAFT_GRAPH_VALIDATE=N eagerly
cross-checks the first N graph verifies against the eager walk (costs one
extra eager walk per validated verify; diagnostics only).
"""
from __future__ import annotations

import contextlib
import math
import os
import time

import torch

from exllamav3.constants import PAGE_SIZE

REVISION = "draft-graph-20260923-promoted"
BUCKET_PAGES = 32  # 8192 tokens per bucket at PAGE_SIZE 256


def enabled(env=os.environ) -> bool:
    return env.get("QWASAR_DRAFT_GRAPH", "1") != "0"


def install(generator, env=os.environ):
    """Install the captured draft walk on the generator and return the
    DraftGraph instance (its .stats are JSON-safe counters). Raises on any
    precondition failure; the engine treats that as "not installed"."""
    validate = int(env.get("QWASAR_DRAFT_GRAPH_VALIDATE", "0"))
    return DraftGraph(generator, min_eager_verifies=2, validate_verifies=validate)


class DraftGraph:
    def __init__(self, generator, *, min_eager_verifies: int = 2, validate_verifies: int = 0):
        assert generator.mtp_draft and generator.num_draft_tokens == 6 and not generator.dynamic_draft
        assert getattr(generator.draft_model, "mtp_sub_lm_head", None) is not None, \
            "install the promoted hot64k head first"
        self.g = generator
        self.draft = generator.draft_model
        self.target = generator.model
        # The eager fallback is whatever walk is installed at construction:
        # on the promoted stack that is the rendezvous R1a GPU-chained walk.
        self.eager = generator.iterate_draftmodel_mtp_gen
        self.min_eager = min_eager_verifies
        self.validate_left = validate_verifies
        self.enabled = True  # False: hand every verify to the eager walk (in-process A/B)
        self.log_ids = False  # True: record (kv_position, 6 ids) per verify, ALL arms
        self.ids_log = []
        self.graph = None
        self.pool = None
        self.bucket = None
        self.static = None
        self.host_block = None
        self.host_block_prev = None
        self.stats = {"eager_verifies": 0, "graph_verifies": 0, "fallbacks": 0, "captures": [],
                      "block_table_uploads": 0,
                      "validation": {"compared": 0, "mismatches": 0, "rows": []}}
        from exllamav3.modules import Attention, GatedMLP, MLP

        def walk(module):
            yield module
            for child in getattr(module, "modules", []):
                yield from walk(child)
        everything = [m for top in self.draft.modules for m in walk(top)]
        self.attention_modules = [m for m in everything if isinstance(m, Attention)]
        self.mlp_modules = [m for m in everything if isinstance(m, (GatedMLP, MLP))]
        assert len(self.attention_modules) == 1, [m.key for m in self.attention_modules]
        assert len(self.mlp_modules) == 1, [m.key for m in self.mlp_modules]
        generator.iterate_draftmodel_mtp_gen = self.run

    @contextlib.contextmanager
    def bc_bypassed(self):
        """Public wrapper: the eager reference arm of A/B harnesses runs the
        WHOLE eager walk with the draft's two modules on their dispatch paths
        (the would-be promoted fallback semantics), so its numerics match the
        captured graph's. Same context used internally around captures and
        validation references."""
        with self._bc_bypassed():
            yield

    @contextlib.contextmanager
    def _bc_bypassed(self):
        """Route the draft attention and MLP through their eager paths for the
        duration: BC_Attention::run and BC_GatedMLP::run_bszN replay internal
        CUDA graphs (cudaGraphLaunch), which stream capture cannot record.
        ``bc_attn_step`` returning None makes decode_flash_attn fall through
        to the regular dispatch path; ``mlp.bc = None`` does the same for the
        MLP."""
        saved_attn, saved_mlp = [], []
        for m in self.attention_modules:
            saved_attn.append((m, m.__dict__.get("bc_attn_step")))
            m.bc_attn_step = lambda *args, **kwargs: None
        for m in self.mlp_modules:
            saved_mlp.append((m, m.__dict__.get("bc")))
            m.bc = None
        try:
            yield
        finally:
            for m, previous in saved_attn:
                if previous is None:
                    del m.bc_attn_step
                else:
                    m.bc_attn_step = previous
            for m, previous in saved_mlp:
                if previous is None:
                    del m.bc
                else:
                    m.bc = previous

    # ----- per-verify prologue (mirrors the rendezvous walk's batch shape) --

    def _prologue(self):
        g = self.g
        jobs = [job for job in g.active_jobs if job.is_prefill_done()]
        if len(jobs) != 1:
            return None
        job = jobs[0]
        if len(job.sequences) != 1 or job.mtp_last_hidden is None:
            return None
        if g.draft_window() != 6:
            return None
        seq = job.sequences[0]
        max_seq_len = job.get_max_seq_len() + g.num_draft_tokens + 1
        max_pages = (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        return job, seq, max_pages

    def run(self, results: list):
        if not self.enabled:
            out = self.eager(results)
            if self.log_ids:
                self._log(out)
            return out
        pro = self._prologue()
        if pro is None:
            self.stats["fallbacks"] += 1
            out = self.eager(results)
            if self.log_ids:
                self._log(out)
            return out
        if self.stats["eager_verifies"] < self.min_eager:
            self.stats["eager_verifies"] += 1
            if self.stats["eager_verifies"] == self.min_eager:
                # Last warmup verify: JIT/autotune the dispatch attention
                # kernels eagerly (BC bypassed) BEFORE any capture.
                with self._bc_bypassed():
                    return self.eager(results)
            return self.eager(results)
        job, seq, max_pages = pro
        if job.time_first_token is None:
            from exllamav3.util import cuda_sync_active
            cuda_sync_active()
            job.time_first_token = time.time()

        bucket = math.ceil(max_pages / BUCKET_PAGES) * BUCKET_PAGES
        hidden = job.mtp_last_hidden
        if self.graph is None or bucket != self.bucket:
            self._capture(bucket, hidden)

        reference = None
        if self.validate_left > 0:
            # Eager first (dispatch attention, the same kernels the capture
            # records), then the graph on identical inputs. Both append the
            # same K/V at the same positions, so the cache ends in the same
            # state either way and the ids must be bit-identical.
            with self._bc_bypassed():
                reference = self.eager(results)[:1, :6].clone()

        st = self.static
        ids_cpu = job.get_input_ids_list()[0]  # (1, 1) long on CPU
        block = seq.block_index_tensor[:, :max_pages]
        self.host_block.zero_()
        self.host_block[:, :block.shape[1]].copy_(block)
        if self.host_block_prev is None or not torch.equal(self.host_block, self.host_block_prev):
            st["block"].copy_(self.host_block, non_blocking=True)
            self.host_block_prev = self.host_block.clone()
            self.stats["block_table_uploads"] += 1
        st["seqlens"].fill_(int(seq.kv_position))
        st["hidden"].copy_(hidden)
        st["ids"].copy_(ids_cpu.reshape(st["ids"].shape), non_blocking=True)
        self.graph.replay()
        self.g.draft_ids_pinned[:1, :6].copy_(st["out"], non_blocking=True)
        torch.cuda.synchronize()  # the one host sync per verify (same as the R1a walk)
        self.stats["graph_verifies"] += 1

        if reference is not None:
            self.validate_left -= 1
            got = self.g.draft_ids_pinned[:1, :6]
            same = torch.equal(reference, got)
            v = self.stats["validation"]
            v["compared"] += 1
            v["mismatches"] += int(not same)
            if len(v["rows"]) < 64:
                v["rows"].append({"kv_position": int(seq.kv_position), "eager": reference.flatten().tolist(),
                                  "graph": got.flatten().tolist(), "equal": same})
        if self.log_ids:
            self._log(self.g.draft_ids_pinned[:, :6])
        return self.g.draft_ids_pinned[:, :6]

    def _log(self, out):
        """Record the verify's draft ids (CPU pinned) for offline arm
        comparison: first divergence between arms is auditable."""
        row = out[:1, :6].tolist() if out is not None else None
        self.ids_log.append(row)

    def replay_floor_ms(self, repeats: int = 20):
        """Back-to-back replays of the current graph with its current inputs
        (idempotent: the same K/V land at the same positions). Kernel floor of
        one six-step draft."""
        if self.graph is None:
            return None
        torch.cuda.synchronize()
        self.graph.replay()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            self.graph.replay()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / repeats

    def profile_replay(self, repeats: int = 3):
        """Kernel table of the captured graph (CUPTI records the kernels a
        replay launches). Smoke-run diagnostic only."""
        if self.graph is None:
            return None
        from torch.profiler import ProfilerActivity, profile
        torch.cuda.synchronize()
        self.graph.replay()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(repeats):
                self.graph.replay()
            torch.cuda.synchronize()
        table = {}
        order = []
        for evt in prof.events():
            if evt.device_type.name != "CUDA":
                continue
            row = table.setdefault(evt.name, {"count": 0, "total_us": 0.0})
            if row["count"] == 0:
                order.append(evt.name)
            row["count"] += 1
            row["total_us"] += evt.device_time
        for name in order:
            row = table[name]
            row["count"] = row["count"] // repeats
            row["total_us"] = row["total_us"] / repeats
            row["mean_us"] = row["total_us"] / max(row["count"], 1)
        kernels = sum(r["count"] for r in table.values())
        total = sum(r["total_us"] for r in table.values())
        return {"repeats": repeats, "kernels_per_replay": kernels, "kernel_us_per_replay": total,
                "first_seen_order": order, "table": table}

    # ----- capture ---------------------------------------------------------

    def _capture(self, bucket, hidden):
        t0 = time.perf_counter()
        dev = hidden.device
        if self.static is None:
            self.static = {
                "hidden": torch.zeros_like(hidden),
                "ids": torch.zeros((1, 1), dtype=torch.long, device=dev),
                "seqlens": torch.zeros((1,), dtype=torch.int32, device=dev),
                "work_seqlens": torch.zeros((1,), dtype=torch.int32, device=dev),
                "out": torch.zeros((1, 6), dtype=torch.long, device=dev),
            }
        self.static["block"] = torch.zeros((1, bucket), dtype=torch.int32, device=dev)
        self.host_block = torch.zeros((1, bucket), dtype=torch.int32, pin_memory=True)
        self.host_block_prev = None
        if self.graph is not None:
            del self.graph
            self.graph = None
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        kwargs = {"pool": self.pool} if self.pool is not None else {}
        with self._bc_bypassed(), torch.cuda.graph(graph, **kwargs):
            self._body()
        self.pool = graph.pool()
        torch.cuda.synchronize()
        self.graph, self.bucket = graph, bucket
        self.stats["captures"].append({"bucket_pages": bucket,
                                       "capture_ms": (time.perf_counter() - t0) * 1000})

    def _body(self):
        """The six draft steps, mirroring the rendezvous R1a walk verbatim
        with static device tensors. Params are exactly the walk's five keys;
        positions derive from cache_seqlens inside the attention path, as in
        the walk."""
        st = self.static
        st["work_seqlens"].copy_(st["seqlens"])
        lm_head = self.target.modules[self.target.logit_layer_idx]
        hidden = st["hidden"]
        ids = st["ids"]
        for idx in range(6):
            params = {
                "target_hidden": hidden,
                "attn_mode": "flash_attn",
                "block_table": st["block"],
                "cache": self.g.draft_cache,
                "cache_seqlens": st["work_seqlens"],
            }
            state = self.draft.forward(ids, params)
            state = lm_head.prepare_for_device(state, params)
            new_ids = self.draft.sample_from_state(state, params)  # hot64k: argmax 64K + map
            st["out"][:, idx:idx + 1].copy_(new_ids)
            ids = new_ids
            hidden = state
            st["work_seqlens"] += 1
