import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine" / "oracle"))

from taps import capture_plan, select_taps


def test_selects_the_first_layer_of_each_kind_in_model_order():
    modules = [
        ("model.embed", "Embedding"),
        ("model.layers.0", "GatedDeltaNet"),
        ("model.layers.1", "GatedDeltaNet"),
        ("model.layers.3", "Attention"),
        ("model.layers.3.mlp", "GatedMLP"),
        ("model.layers.4.mlp", "GatedMLP"),
        ("lm_head", "LMHead"),
    ]
    chosen = select_taps(modules)
    assert chosen == {
        "gdn": "model.layers.0",
        "attention": "model.layers.3",
        "mlp": "model.layers.3.mlp",
        "embedding": "model.embed",
        "head": "lm_head",
    }
    plan = capture_plan(modules)
    assert plan["contexts"] == [64, 32768, 261888]
    assert plan["serves_traffic"] is False
    assert plan["full_tensors"] == "short context only"


def test_a_missing_kind_is_an_error():
    try:
        select_taps([("model.layers.0", "GatedDeltaNet")])
    except ValueError as error:
        assert "Attention" in str(error)
        assert "GatedMLP" in str(error)
    else:
        raise AssertionError("expected a missing-tap error")
