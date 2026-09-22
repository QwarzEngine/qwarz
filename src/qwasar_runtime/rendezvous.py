"""Process-local rendezvous elimination for MTP generation (R1).

Two host rendezvous structures dominate the per-cycle host/GPU chatter of
the stock MTP loop (qf3-prof5: ~6.5 ms/cycle of exposed latency):

1. The DRAFT walk: ``iterate_draftmodel_mtp_gen`` copies each drafted
   token to TWO pageable CPU buffers per step — the output buffer
   ``draft_ids_pinned`` and the input buffer ``draft_input_ids_pinned``
   — so width 6 pays ~12 blocking D2H reads per cycle with the host
   inside the data-dependent loop. The chain is entirely on-device
   (step k+1 consumes step k's sampled id): this module's patched walk
   keeps ids on GPU (D2D per step) and reads the window back once —
   ONE sync per cycle instead of twelve.

2. The VERIFY walk: each position's sampled token round-trips to the
   host (``Job.receive_sample`` does ``next_token.cpu()`` + ``.item()``),
   so an MTP6 verify pays up to 7 blocking D2H reads per cycle. For jobs
   whose sampling does not depend on earlier positions within the
   window (no ACTIVE presence/frequency penalties, no active logit-mask
   filters, no token healing, no probability reporting), every
   position's sample is data-independent: the whole window is sampled
   in ONE ``sampler.forward`` call, read back in ONE pinned copy and
   ONE synchronize, and the acceptance walk then runs entirely on host
   arithmetic (the draft ids are already pinned). This covers greedy
   (``ArgmaxSampler``) and the plain temperature/top-k/top-p stack
   (``ComboSampler`` with neutral penalties); one batched RNG draw per
   window replaces the per-position draws (statistically equivalent,
   documented nondeterminism band). Jobs with active penalties
   (thinking=off chat carries ``pres_p``) fall back to the stock serial
   walk.

Mechanics (no ``iterate_gen`` copy):

- ``install(generator)`` wraps ``generator.model.forward``: after a
  forward whose params carry ``pinned_staging`` (the verify batch's
  signature; the MTP draft walk's forwards and the draft cache prefill
  go through the same wrapped forward with minimal params) whose logits
  look like a speculative verify window (``2 <= width <=
  num_draft_tokens + 1``) and whose active jobs all qualify, it runs
  each job's ``sampler.forward`` once over the job's whole ``[rows,
  width, vocab]`` slice (the exact call ``Job.receive_logits`` makes
  per position, including the vocab clamp), copies the result to a
  pinned buffer non-blocking and synchronizes once. The result is
  stashed against the logits tensor's address range: position slices
  resolve by POINTER GEOMETRY (``data_ptr`` within the stash span at a
  vocab boundary), because views of inference-mode tensors do not carry
  ``_base`` and identity checks cannot match.
- ``Job.receive_logits`` is patched process-locally: when the incoming
  slice is a view of the stashed logits, it returns the position's CPU
  token directly instead of launching the sampler. Everything else
  (``receive_sample`` bookkeeping, rejection, rewind, checkpoint and
  budget handling) is untouched; ``receive_sample``'s ``.cpu()`` is a
  no-op on the already-CPU token.

Gated per batch, all-or-nothing: any non-qualifying job, a capture in
progress, or any error in the fast path falls back to the stock serial
walk for that batch. ``QWASAR_RDZ=0`` disables the path entirely.

Projected effect (qf3-prof5 decomposition): removes the per-position
round trips (~1.5-3 ms/cycle) and shrinks the post-verify bubble, for
both the production stack (control) and the Fase-3 candidate.
"""
from __future__ import annotations

import os
import time
from typing import Any

import torch

from exllamav3.generator.job import Job

ENABLED = os.environ.get("QWASAR_RDZ", "1") != "0"

_counters: dict[str, int] = {
    "windows": 0,
    "positions_served": 0,
    "fallbacks": 0,
    "errors": 0,
    # fallback reasons (debug)
    "skip_width": 0,
    "skip_not_contiguous": 0,
    "skip_capturing": 0,
    "skip_input_width": 0,
    "skip_not_eligible": 0,
    "skip_rows_mismatch": 0,
    "resolve_base_mismatch": 0,
    "resolve_shape_mismatch": 0,
    "fwd_calls": 0,
    # GPU-chained draft walk
    "draft_windows": 0,
    "draft_steps": 0,
    "draft_syncs_saved": 0,
    # draft embedding relocation (R1b)
    "emb_moved": 0,
    "emb_mb": 0,
    "emb_move_failed": 0,
}

_fwd_width_hist: dict[int, int] = {}
_stash_width_hist: dict[int, int] = {}

_debug_first: dict | None = None

_pinned_buffers: dict[tuple[int, int], torch.Tensor] = {}


class _Stash:
    """CPU window tokens for one forward's verify batch.

    Position slices resolve by POINTER GEOMETRY, not by tensor identity:
    the model's forward runs under ``torch.inference_mode()``, and views of
    inference tensors do not carry ``_base``, so identity checks cannot
    match. The stash pins the logits tensor's address range; a slice that
    starts inside it, at a vocab boundary, resolves to its (row, position).
    """

    __slots__ = ("base_ptr", "span_bytes", "esize", "width", "vocab",
                 "cpu_tokens", "rows")

    def __init__(self, base_ptr, span_bytes, esize, width, vocab, cpu_tokens, rows):
        self.base_ptr = base_ptr
        self.span_bytes = span_bytes
        self.esize = esize
        self.width = width
        self.vocab = vocab
        self.cpu_tokens = cpu_tokens
        self.rows = rows

    def resolve(self, logits: torch.Tensor) -> torch.Tensor | None:
        """CPU ``[rows, 1]`` token tensor for a ``[rows, 1, vocab]`` slice."""
        if logits.dim() != 3 or logits.shape[1] != 1 or logits.shape[-1] != self.vocab:
            _counters["resolve_shape_mismatch"] += 1
            return None
        rel = logits.data_ptr() - self.base_ptr
        if rel < 0 or rel >= self.span_bytes or rel % self.esize:
            _counters["resolve_base_mismatch"] += 1
            return None
        off = rel // self.esize
        if off % self.vocab:
            _counters["resolve_base_mismatch"] += 1
            return None
        pos = (off // self.vocab) % self.width
        rem = off // (self.width * self.vocab)
        if rem + logits.shape[0] > self.rows:
            _counters["resolve_base_mismatch"] += 1
            return None
        rows = [[int(self.cpu_tokens[rem + r, pos])] for r in range(logits.shape[0])]
        return torch.tensor(rows, dtype=torch.long)


_stash: _Stash | None = None
_walk_input_gpu: bool = False
_orig_receive_logits = Job.receive_logits
_orig_mtp_walk: Any = None
_orig_embedding_forward: Any = None
_installed_generator: Any = None


def move_draft_embedding_to_gpu(generator) -> bool:
    """R1b: relocate the shared embedding table to GPU.

    The MTP head borrows the TARGET's embed_tokens and lm_head
    (``attach_to``), and the loader keeps that embedding on CPU (a ~2 GB
    saving under the gpu_split budget), which forces the draft walk to
    round-trip every sampled id to the host for the gather. With the
    table on GPU the whole walk chains on-device. CPU-id callers (the
    verify batch staging, the draft cache prefill, the target's own
    prefill) are covered by the device-agnostic ``Embedding.forward``
    patch.
    """
    global _walk_input_gpu
    from exllamav3.modules.embedding import Embedding
    target = generator.model
    emb_mod = next((m for m in target.modules if isinstance(m, Embedding)), None)
    if emb_mod is None or getattr(emb_mod, "embedding", None) is None:
        _counters["emb_move_failed"] += 1
        return False
    try:
        dev = generator.model.device \
            if getattr(generator.model, "device", None) is not None \
            else torch.device("cuda")
        emb_mod.embedding = emb_mod.embedding.to(dev)
        emb_mod.device = dev
        _counters["emb_moved"] += 1
        _counters["emb_mb"] += int(
            emb_mod.embedding.weight.numel() * emb_mod.embedding.weight.element_size()
        ) // (1024 * 1024)
        _walk_input_gpu = True
        return True
    except Exception:
        _counters["emb_move_failed"] += 1
        return False


def _patched_embedding_forward(self, x: torch.Tensor, params: dict,
                               out_dtype: torch.Tensor.dtype | None = None):
    """Device-agnostic ``Embedding.forward``: ids auto-move to the table's device.

    With the draft embedding relocated to GPU, the draft walk feeds GPU
    ids directly (no host round trip), while the draft cache prefill
    still passes CPU ids from the verify batch staging; both work.
    """
    if self.device is not None and x.device != self.device:
        x = x.to(self.device, non_blocking=True)
    return _orig_embedding_forward(self, x, params, out_dtype)


def _patched_iterate_draftmodel_mtp_gen(self, results: list):
    """GPU-chained MTP draft walk (drop-in for ``Generator.iterate_draftmodel_mtp_gen``).

    The stock walk makes TWO blocking D2H copies per drafted token — the
    pageable output buffer ``draft_ids_pinned`` and the pageable input
    buffer ``draft_input_ids_pinned`` — so at width 6 every cycle pays
    ~12 host round trips with the queue drained and the host inside the
    data-dependent loop. The dependency chain itself is entirely
    on-device: step k+1 consumes step k's sampled id. This patch keeps
    ids on GPU (D2D per step) and reads the whole window back once at
    the end: one sync per cycle instead of twelve.
    """
    from exllamav3.constants import PAGE_SIZE
    from exllamav3.util import cuda_sync_active

    # Batch shape (verbatim from stock)
    batch_size = 0
    max_seq_len = 0
    for job in self.active_jobs:
        if not job.is_prefill_done(): continue
        max_seq_len = max(max_seq_len, job.get_max_seq_len() + self.num_draft_tokens + 1)
        batch_size += 1
    if batch_size == 0:
        return None

    # Block index table (verbatim from stock)
    max_pages_batch = (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    block_index = torch.zeros((batch_size, max_pages_batch), dtype=torch.int32)
    cache_seqlens = torch.zeros((batch_size,), dtype=torch.int32)
    batch = 0
    for job in self.active_jobs:
        if not job.is_prefill_done(): continue
        for seq in job.sequences:
            seq_block_index = seq.block_index_tensor[:, :max_pages_batch]
            block_index[batch:batch+1, :seq_block_index.shape[-1]].copy_(seq_block_index)
            cache_seqlens[batch] = seq.kv_position
            batch += 1

    # Input collection (verbatim from stock)
    input_ids_list = []
    mtp_hidden_list = []
    for job in self.active_jobs:
        if not job.is_prefill_done(): continue
        assert len(job.sequences) == 1, \
            "Qwen3.5 MTP drafting does not currently support CFG/multi-sequence jobs"
        if job.mtp_last_hidden is None:
            # A one-token prompt has no token to prefill before the
            # generation input; run one normal target step first
            return None
        if job.time_first_token is None:
            cuda_sync_active()
            job.time_first_token = time.time()
        job_ids = job.get_input_ids_list()
        input_ids_list += job_ids
        mtp_hidden_list.append(job.mtp_last_hidden)
    temp_hidden = torch.cat(mtp_hidden_list, dim=0)

    window = self.draft_window()
    if window == 0:
        return None

    # GPU-side output chain (D2D per step, one D2H readback at the end).
    # The input side depends on the draft embedding's placement: on GPU
    # (R1b) the input chain is also on-device and the walk never touches
    # the host; still on CPU, the per-step input copy stays a blocking
    # D2H (the CPU embedding lookup needs the id value), which is
    # unavoidable without relocating the table.
    dev = temp_hidden.device
    gpu_ids = torch.empty((batch_size, window), dtype=torch.long, device=dev)
    if _walk_input_gpu:
        batch_ids = torch.empty((batch_size, 1), dtype=torch.long, device=dev)
    else:
        batch_ids = self.draft_input_ids_pinned[:batch_size, :]
    batch_ids.copy_(torch.cat(input_ids_list, dim=0))

    for idx in range(window):
        params = {
            "target_hidden": temp_hidden,
            "attn_mode": "flash_attn",
            "block_table": block_index,
            "cache": self.draft_cache,
            "cache_seqlens": cache_seqlens,
        }
        batch_state = self.draft_model.forward(batch_ids, params)
        lm_head = self.model.modules[self.model.logit_layer_idx]
        batch_state = lm_head.prepare_for_device(batch_state, params)
        new_ids = self.draft_model.sample_from_state(batch_state, params)
        gpu_ids[:, idx:idx+1].copy_(new_ids)
        batch_ids.copy_(new_ids)
        cache_seqlens += 1
        temp_hidden = batch_state

    self.draft_ids_pinned[:batch_size, :window].copy_(gpu_ids, non_blocking=True)
    torch.cuda.synchronize()
    _counters["draft_windows"] += 1
    _counters["draft_steps"] += window
    _counters["draft_syncs_saved"] += max(0, 2 * window - 1)
    return self.draft_ids_pinned[:, :window]


def diagnostics() -> dict:
    return {
        "counters": dict(_counters),
        "pinned_buffers": len(_pinned_buffers),
        "installed": _installed_generator is not None,
        "debug": _debug_first,
        "fwd_width_hist": dict(_fwd_width_hist),
        "stash_width_hist": dict(_stash_width_hist),
    }


def _eligible_jobs(generator):
    """Prefill-done jobs of a batched-eligible verify batch, or None.

    All-or-nothing: any job needing per-position state (penalties,
    advancing masks) or any rejection-sampling mode disqualifies the
    whole batch, which then takes the stock serial walk.
    """
    if not generator.draft_model or not generator.num_draft_tokens:
        return None
    if getattr(generator, "spec_rs", False):
        # Rejection-sampling verify needs each position's full proposal
        return None
    jobs = [job for job in generator.active_jobs if job.is_prefill_done()]
    if not jobs or not all(_qualify(job) for job in jobs):
        return None
    return jobs


def _qualify(job) -> bool:
    # Active penalties make position n+1's distribution depend on position
    # n's result (past ids grow per accepted position). The conservative
    # ``sampler.reqs_past_ids`` flag is set even for NEUTRAL penalties
    # (ComboSampler always stacks SS_RepP/SS_PresFreqP and ors the flags
    # before no-op simplification drops them), so eligibility is decided on
    # the simplified ``steps`` where neutral penalty steps are gone.
    steps = getattr(job.sampler, "steps", None)
    if steps is not None:
        if any(step.reqs_past_ids() for step in steps):
            return False
    elif getattr(job.sampler, "reqs_past_ids", False):
        return False
    # Probability/top-k reporting gathers per-position logit slices on GPU
    if job.return_probs or job.return_top_tokens:
        return False
    # A token-dependent logit mask (constrained decoding) invalidates the
    # batched sample; healing rewrites the first position
    if any(f.is_active for f in job.filters):
        return False
    if job.prefix_token is not None and job.new_tokens == -1:
        return False
    return True


def _pinned(rows: int, width: int) -> torch.Tensor:
    key = (rows, width)
    buf = _pinned_buffers.get(key)
    if buf is None:
        buf = torch.empty((rows, width), dtype=torch.long, pin_memory=True)
        _pinned_buffers[key] = buf
    return buf


def _maybe_stash(generator, batch_logits: torch.Tensor, params, input_ids) -> None:
    """Batched window sampling + single readback, when the batch qualifies."""
    global _stash
    _stash = None
    if not ENABLED or batch_logits is None or batch_logits.dim() != 3:
        return
    # The verify batch's params dict carries pinned_staging=True; the MTP
    # draft walk's forwards and the draft cache prefill go through the same
    # wrapped Model.forward with minimal params and must NOT stash (their
    # logits belong to a different batch, and a stash there would add a
    # sampler call plus a sync per draft step).
    if not isinstance(params, dict) or not params.get("pinned_staging"):
        return
    rows, width, vocab_padded = batch_logits.shape
    if not 2 <= width <= generator.num_draft_tokens + 1:
        _counters["skip_width"] += 1
        return
    if not batch_logits.is_contiguous():
        _counters["skip_not_contiguous"] += 1
        return
    if torch.cuda.is_current_stream_capturing():
        # A capture (BCAttn warmup, rewind check) cannot absorb a sync
        _counters["skip_capturing"] += 1
        return
    if input_ids is not None and input_ids.dim() == 2 and input_ids.shape[-1] != width:
        # The verify window width must match the input width; anything else
        # (small prefill tails) is left to the stock path
        _counters["skip_input_width"] += 1
        return
    jobs = _eligible_jobs(generator)
    if jobs is None:
        _counters["fallbacks"] += 1
        _counters["skip_not_eligible"] += 1
        return
    if sum(len(job.sequences) for job in jobs) != rows:
        # The logits rows belong to some other batch (e.g. a short prefill
        # tail for a job that is still ingesting); the eligible jobs' rows
        # cannot be this tensor. Never sample one job's window from
        # another's logits.
        _counters["fallbacks"] += 1
        _counters["skip_rows_mismatch"] += 1
        return
    # Mirror Job.receive_logits per job, once over the whole window: the
    # same sampler stack (vocab clamp, masks), the same RNG source, one
    # [rows, width] result per job.
    device = batch_logits.device
    window_gpu = torch.empty((rows, width), dtype=torch.long, device=device)
    row = 0
    with torch.no_grad():
        for job in jobs:
            n = len(job.sequences)
            tok = job.sampler.forward(
                batch_logits[row:row + n],
                job.current_device_ids,
                job.rng.randint(0, (1 << 32) - 1),
                generator.tokenizer,
                logit_mask=job.device_logit_mask,
            )
            window_gpu[row:row + n] = tok
            row += n
    pinned = _pinned(rows, width)
    pinned.copy_(window_gpu, non_blocking=True)
    torch.cuda.synchronize(batch_logits.device)
    _stash_width_hist[width] = _stash_width_hist.get(width, 0) + 1
    _stash = _Stash(
        base_ptr=batch_logits.data_ptr(),
        span_bytes=batch_logits.numel() * batch_logits.element_size(),
        esize=batch_logits.element_size(),
        width=width,
        vocab=vocab_padded,
        cpu_tokens=pinned,
        rows=rows,
    )
    _counters["windows"] += 1


def _receive_logits(self: Job, logits: torch.Tensor):
    stash = _stash
    if stash is not None:
        try:
            token = stash.resolve(logits)
        except Exception:
            token = None
        if token is not None:
            _counters["positions_served"] += 1
            return token, None, None, None
    return _orig_receive_logits(self, logits)


def install(generator, gpu_embedding: bool = False,
            walk: bool = True, verify: bool = True) -> None:
    """Installs the rendezvous fast paths on ``generator``.

    - ``verify`` (R1a): batched window sampling + single readback for the
      verify walk (``model.forward`` wrapper + ``Job.receive_logits``).
    - ``walk`` (R1a): GPU-chained MTP draft walk (one sync per cycle
      instead of two blocking D2H copies per drafted token).
    - ``gpu_embedding`` (R1b): relocate the shared embedding table to GPU
      (~2.4 GB) and make ``Embedding.forward`` device-agnostic, so the
      draft walk chains entirely on-device.

    One generator per process: the class-level ``receive_logits`` patch
    resolves against the single installed generator's stash. The draft
    walk and embedding patches are class-level too and only change
    dispatch, not state.
    """
    global _installed_generator, _stash, _orig_mtp_walk, _orig_embedding_forward
    if _installed_generator is generator:
        return
    assert _installed_generator is None, "rendezvous is installed for another generator"
    _installed_generator = generator
    _stash = None
    model = generator.model
    orig_forward = model.forward

    from exllamav3.generator.generator import Generator
    if walk and getattr(generator, "mtp_draft", False) and _orig_mtp_walk is None:
        _orig_mtp_walk = Generator.iterate_draftmodel_mtp_gen
        Generator.iterate_draftmodel_mtp_gen = _patched_iterate_draftmodel_mtp_gen
    if gpu_embedding:
        from exllamav3.modules.embedding import Embedding
        if move_draft_embedding_to_gpu(generator) and _orig_embedding_forward is None:
            _orig_embedding_forward = Embedding.forward
            Embedding.forward = _patched_embedding_forward
    if not verify:
        # A forward wrapper that never stashes disables the batched verify
        # while keeping the other patches (fidelity isolation mode)
        def _forward(*args, **kwargs):
            return orig_forward(*args, **kwargs)
        model.forward = _forward
        return

    def _forward(*args, **kwargs):
        batch_logits = orig_forward(*args, **kwargs)
        _counters["fwd_calls"] += 1
        try:
            if batch_logits is not None and batch_logits.dim() == 3:
                w = batch_logits.shape[1]
                _fwd_width_hist[w] = _fwd_width_hist.get(w, 0) + 1
            params = kwargs.get("params")
            if params is None and len(args) > 1:
                params = args[1]
            _maybe_stash(generator, batch_logits, params, kwargs.get("input_ids"))
        except Exception:
            # The fast path must never break generation; fall back wholesale
            global _stash
            _stash = None
            _counters["errors"] += 1
        return batch_logits

    model.forward = _forward
    Job.receive_logits = _receive_logits


def uninstall() -> None:
    """Restores the stock forward, walk, embedding and ``receive_logits`` (test hook)."""
    global _installed_generator, _stash, _orig_mtp_walk, _orig_embedding_forward
    global _walk_input_gpu
    if _installed_generator is None:
        return
    try:
        # The wrapper was installed as an instance attribute shadowing the
        # class method; deleting it restores the stock bound method
        del _installed_generator.model.forward
    except AttributeError:
        pass
    if _orig_mtp_walk is not None:
        from exllamav3.generator.generator import Generator
        Generator.iterate_draftmodel_mtp_gen = _orig_mtp_walk
        _orig_mtp_walk = None
    if _orig_embedding_forward is not None:
        from exllamav3.modules.embedding import Embedding
        Embedding.forward = _orig_embedding_forward
        _orig_embedding_forward = None
    Job.receive_logits = _orig_receive_logits
    _installed_generator = None
    _stash = None
    _walk_input_gpu = False
