from types import SimpleNamespace

import pytest

from qwasar_bench import decode_probe


def generator(**overrides):
    return SimpleNamespace(**({
        "mtp_draft": True, "num_draft_tokens": 4, "dynamic_draft": False,
        "record_draft_stats": False, "num_remaining_jobs": lambda: 0,
        "dynamic_draft_alpha_up": 1.30, "dynamic_draft_alpha_down": .65,
        "dynamic_draft_skip_ema": .3, "dynamic_draft_probe_interval": 16,
    } | overrides))


def test_adaptive_policy_changes_only_adaptation_and_enables_equal_instrumentation():
    fixed, adaptive = generator(), generator()
    a = decode_probe.configure_mtp_policy(fixed, "fixed4")
    b = decode_probe.configure_mtp_policy(adaptive, "adaptive4")
    assert fixed.num_draft_tokens == adaptive.num_draft_tokens == 4
    assert fixed.dynamic_draft is False and adaptive.dynamic_draft is True
    assert fixed.record_draft_stats is adaptive.record_draft_stats is True
    assert {k: v for k, v in a.items() if k != "dynamic_draft"} == {
        k: v for k, v in b.items() if k != "dynamic_draft"}
    assert b["dynamic_draft_skip_ema"] == .3


@pytest.mark.parametrize("overrides", [
    {"mtp_draft": False}, {"num_draft_tokens": 7},
    {"num_remaining_jobs": lambda: 1},
])
def test_policy_refuses_to_change_an_incompatible_or_active_generator(overrides):
    model = generator(**overrides)
    before = vars(model).copy()
    with pytest.raises(ValueError):
        decode_probe.configure_mtp_policy(model, "adaptive4")
    assert vars(model) == before


def test_invalid_policy_is_rejected_before_mutation():
    model = generator()
    with pytest.raises(ValueError):
        decode_probe.configure_mtp_policy(model, "typo")
    assert model.dynamic_draft is False


@pytest.mark.parametrize("count", range(1, 8))
def test_fixed_policy_requires_matching_preallocated_width(count):
    model = generator(num_draft_tokens=count, dynamic_draft=True)
    settings = decode_probe.configure_mtp_policy(model, f"fixed{count}")
    assert settings["num_draft_tokens"] == count
    assert model.dynamic_draft is False
    assert model.num_draft_tokens == count
    wrong = generator(num_draft_tokens=count % 7 + 1)
    with pytest.raises(ValueError):
        decode_probe.configure_mtp_policy(wrong, f"fixed{count}")
