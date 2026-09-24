import pytest

torch = pytest.importorskip("torch")

from qwasar_bench.nondet_bisect import Recorder, compare_runs, fingerprint


def fp_record(seq, call, value, inp=1.0, prefill=False):
    in_fp = fingerprint(torch, torch.full((1, 2, 4), inp, dtype=torch.float16))
    out_fp = fingerprint(torch, torch.full((1, 2, 4), value, dtype=torch.float16))
    return {"seq": seq, "call": call, "qlen": 2, "prefill": prefill,
            "in": in_fp, "out": out_fp}


def test_fingerprint_stable_and_sensitive():
    x = torch.randn(1, 8, 16, dtype=torch.float16)
    assert fingerprint(torch, x) == fingerprint(torch, x.clone())
    flipped = x.clone()
    flipped[0, 0, 0] += 1
    assert fingerprint(torch, x) != fingerprint(torch, flipped)
    # Integer tensors hash their bytes: different ids, same shape, must differ.
    ids = torch.zeros(1, 4, dtype=torch.int64)
    other = torch.zeros(1, 4, dtype=torch.int64)
    other[0, 0] = 7
    fp = fingerprint(torch, ids)
    assert fp["dtype"] == "torch.int64" and fp["shape"] == [1, 4] and "sha" in fp
    assert fp != fingerprint(torch, other)
    assert fp == fingerprint(torch, ids.clone())
    assert fingerprint(torch, None) is None


def test_compare_identical_runs():
    calls = {"t[000]emb": [fp_record(0, 0, 1.0)], "t[001]blk": [fp_record(1, 0, 2.0)]}
    verdict = compare_runs(calls, {k: list(v) for k, v in calls.items()})
    assert verdict["stable"] and not verdict["sources"] and not verdict["propagated"]


def test_compare_finds_source_not_propagation():
    # Layer 1 output flips with matching input (source); layer 2 only sees the
    # flipped input (propagation).
    a = {"t[001]blk": [fp_record(0, 0, 1.0)],
         "t[002]blk": [fp_record(1, 0, 3.0, inp=1.0)]}
    b = {"t[001]blk": [fp_record(0, 0, 2.0)],
         "t[002]blk": [fp_record(1, 0, 4.0, inp=2.0)]}
    verdict = compare_runs(a, b)
    assert [s["module"] for s in verdict["sources"]] == ["t[001]blk"]
    assert [p["module"] for p in verdict["propagated"]] == ["t[002]blk"]
    assert not verdict["stable"]


def test_compare_length_mismatch_and_module_sets():
    a = {"m": [fp_record(0, 0, 1.0), fp_record(1, 1, 1.0)]}
    b = {"m": [fp_record(0, 0, 1.0)]}
    verdict = compare_runs(a, b)
    assert verdict["length_mismatches"] == [{"module": "m", "calls_a": 2, "calls_b": 1}]
    with pytest.raises(ValueError):
        compare_runs(a, {"other": []})


class DummyModule:
    def __init__(self, key, gain):
        self.key = key
        self.gain = gain

    def forward(self, x, params):
        return x * self.gain


class DummyModel:
    def __init__(self, gains):
        self.modules = [DummyModule(f"m{i}", g) for i, g in enumerate(gains)]


def test_recorder_wraps_and_restores():
    model = DummyModel([2.0, 3.0])
    originals = [m.forward for m in model.modules]
    recorder = Recorder(torch)
    recorder.attach(model, "t")
    x = torch.ones(1, 2, 4, dtype=torch.float16)
    out = model.modules[0].forward(x, {})
    out = model.modules[1].forward(out, {})
    assert float(out.double().sum()) == pytest.approx(2 * 3 * 8)
    assert len(recorder.calls["t[000]m0"]) == 1
    assert recorder.calls["t[001]m1"][0]["seq"] == 1
    recorder.close()
    assert [m.forward for m in model.modules] == originals
    # In-place mutation by the module must not corrupt the input fingerprint.
    class Mutating(DummyModule):
        def forward(self, x, params):
            x += 1
            return x
    mut = Mutating("mut", 1.0)
    model2 = DummyModel([])
    model2.modules = [mut]
    rec2 = Recorder(torch)
    rec2.attach(model2, "t")
    y = torch.zeros(1, 2, 4, dtype=torch.float16)
    mut.forward(y, {})
    rec2.close()
    assert rec2.calls["t[000]mut"][0]["in"] == fingerprint(
        torch, torch.zeros(1, 2, 4, dtype=torch.float16))


def test_compare_reports_first_call_only():
    a = {"m": [fp_record(i, i, 1.0) for i in range(4)]}
    b = {"m": [fp_record(0, 0, 1.0)] + [fp_record(i, i, 9.0) for i in range(1, 4)]}
    verdict = compare_runs(a, b)
    assert len(verdict["sources"]) == 1
    assert verdict["sources"][0]["call"] == 1
