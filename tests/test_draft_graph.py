"""CPU tests for the promoted draft-graph module (gating and fallbacks only;
capture/replay needs the GPU and was certified by the campaigns)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("exllamav3")

from qwasar_runtime import draft_graph


class _FakeJob:
    def is_prefill_done(self):
        return True


def _fake_generator(*, draft_tokens=6, dynamic=False, sub_head=True):
    """Minimal generator double satisfying DraftGraph.__init__'s checks."""
    from exllamav3.modules import Attention, GatedMLP

    class DraftModel:
        mtp_sub_lm_head = object() if sub_head else None
        modules = [object.__new__(Attention), object.__new__(GatedMLP)]

    class Generator:
        mtp_draft = True
        num_draft_tokens = draft_tokens
        dynamic_draft = dynamic
        draft_model = DraftModel()
        model = object()  # target model; only stored by DraftGraph.__init__
        active_jobs = []
        sentinels = object()

        def iterate_draftmodel_mtp_gen(self, results):
            return self.sentinels

    gen = Generator()
    gen.eager_walk = gen.iterate_draftmodel_mtp_gen
    return gen


def test_enabled_defaults_true_and_kill_switch():
    assert draft_graph.enabled({}) is True
    assert draft_graph.enabled({"QWASAR_DRAFT_GRAPH": "1"}) is True
    assert draft_graph.enabled({"QWASAR_DRAFT_GRAPH": "0"}) is False


def test_install_validate_env(monkeypatch):
    gen = _fake_generator()
    graph = draft_graph.install(gen, env={"QWASAR_DRAFT_GRAPH_VALIDATE": "7"})
    assert graph.validate_left == 7
    assert gen.iterate_draftmodel_mtp_gen == graph.run  # bound method of the graph


def test_init_rejects_wrong_stack():
    with pytest.raises(AssertionError):
        draft_graph.DraftGraph(_fake_generator(draft_tokens=8))
    with pytest.raises(AssertionError):
        draft_graph.DraftGraph(_fake_generator(dynamic=True))
    with pytest.raises(AssertionError):
        draft_graph.DraftGraph(_fake_generator(sub_head=False))


def test_run_falls_back_to_eager_without_active_job():
    gen = _fake_generator()
    graph = draft_graph.install(gen, env={})
    assert graph.run([]) is gen.sentinels
    assert graph.stats["fallbacks"] == 1
    assert graph.stats["graph_verifies"] == 0


def test_disabled_graph_delegates_every_verify():
    gen = _fake_generator()
    gen.active_jobs = [_FakeJob()]  # a plausible job, but the graph is off
    graph = draft_graph.install(gen, env={})
    graph.enabled = False
    assert graph.run([]) is gen.sentinels
    assert graph.stats["graph_verifies"] == 0
    assert graph.stats["fallbacks"] == 0  # disabled delegation is not a fallback
