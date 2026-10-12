"""The copy-on-write state layers behave exactly like plain dicts, never leak a
child's change into a parent, and refuse to modify a state that has children."""

import random

import numpy as np
import pytest

from green.state import Layer, State


def test_layers_match_a_plain_dict_under_random_operations():
    pick = random.Random(1)
    for trial in range(30):
        base = Layer()
        model = {}
        for k in range(20):
            base[k] = model[k] = pick.randint(0, 9)
        stack = [(base, dict(model))]
        for depth in range(12):
            top, expected = stack[-1]
            child = Layer(top)
            expected = dict(expected)
            for _ in range(15):
                k = pick.randint(0, 30)
                op = pick.random()
                if op < 0.5:
                    child[k] = expected[k] = pick.randint(0, 99)
                elif op < 0.7 and k in expected:
                    del child[k]
                    del expected[k]
                elif op < 0.8:
                    assert child.pop(k, None) == expected.pop(k, None)
                else:
                    assert child.setdefault(k, 7) == expected.setdefault(k, 7)
            stack.append((child, expected))
        for layer, expected in stack:            # every level still sees exactly its own view
            assert dict(layer.items()) == expected
            assert len(layer) == len(expected)
            assert sorted(layer) == sorted(expected)
            for k in range(31):
                assert (k in layer) == (k in expected)
                assert layer.get(k) == expected.get(k)


def test_a_child_never_changes_its_parent():
    parent = State()
    parent.balances["a"] = 5
    parent.jobs["j"] = {"done": {}, "reserved": {}, "delivered": {}, "batches": {}, "escrow": 10}
    parent.job_events[7] = [["expire", "j", "b", None]]
    before = parent.fingerprint()
    child = parent.copy()
    child.balances["a"] = 6
    child.jobs["j"]["escrow"] = 0                    # nested mutation goes to a private copy
    child.jobs["j"]["done"]["0,0,0"] = "b"
    child.job_events.setdefault(7, []).append(["settle", "j", "b", None])
    child.job_events.pop(7)
    del child.balances["a"]
    assert parent.fingerprint() == before
    assert parent.balances["a"] == 5 and parent.jobs["j"]["escrow"] == 10 and parent.job_events[7] == [["expire", "j", "b", None]]
    assert "a" not in child.balances and child.jobs["j"]["escrow"] == 0


def test_a_state_with_children_is_frozen():
    parent = State()
    parent.balances["a"] = 1
    parent.copy()
    with pytest.raises(RuntimeError, match="frozen"):
        parent.balances["a"] = 2
    with pytest.raises(RuntimeError, match="frozen"):
        parent.credit("b", 1)
    with pytest.raises(RuntimeError, match="frozen"):
        parent.job_events.setdefault(1, [])


def test_absorb_and_rebase_keep_every_view():
    base = State()
    base.balances.update({"a": 1, "b": 2})
    s1 = base.copy()
    s1.balances["a"] = 10
    del s1.balances["b"]
    s1.open_disputes = 3
    s2 = s1.copy()
    s2.balances["c"] = 30
    view2 = s2.fingerprint()
    # Move the base up to s1: merge s1 into the base, then hang s2 on the base.
    base.balances.frozen = False                    # the chain owns its base and may absorb into it
    base.absorb(s1)
    s2.rebase(base)
    assert s2.fingerprint() == view2
    assert dict(base.balances.items()) == {"a": 10} and base.open_disputes == 3


def test_diff_round_trip():
    base = State()
    base.balances.update({"a": 1, "b": 2})
    child = base.copy()
    child.balances["a"] = 5
    del child.balances["b"]
    child.jobs["j"] = {"done": {}, "reserved": {}, "delivered": {}, "batches": {}}
    rebuilt = State.from_diff(base, child.diff())
    assert rebuilt.fingerprint() == child.fingerprint()
    assert child.flattened().fingerprint() == child.fingerprint()
    assert child.flattened().depth() == 1


def test_state_root_is_incremental_and_order_independent():
    """The lattice hash after random layered changes equals the hash recomputed from
    scratch, and two states with the same content have the same root however they
    were built."""
    pick = random.Random(5)
    for trial in range(20):
        s = State()
        s.commit()
        model = {}
        for depth in range(8):
            s = s.copy()
            for _ in range(10):
                k = f"k{pick.randint(0, 15)}"
                if pick.random() < 0.7:
                    s.balances[k] = model[k] = pick.randint(0, 99)
                elif k in model:
                    del s.balances[k]
                    del model[k]
            if pick.random() < 0.3:
                s.open_disputes = pick.randint(0, 3)
            s.jobs.get("never-there")                     # reads change nothing
            s.commit()
            assert np.array_equal(s.lattice, s.recompute_lattice())
        fresh = State()                                   # same content, written in another order
        for k in sorted(model, reverse=True):
            fresh.balances[k] = model[k]
        fresh.open_disputes = s.open_disputes
        fresh.commit()
        assert fresh.root() == s.root()


def test_state_root_sees_every_change():
    base = State()
    base.balances["a"] = 1
    base.jobs["j"] = {"done": {}, "reserved": {}, "delivered": {}, "batches": {}, "escrow": 10}
    base.commit()
    roots = {base.root()}
    for change in (lambda s: s.credit("a", 1), lambda s: s.credit("b", 3),
                   lambda s: s.jobs["j"]["done"].update({"0,0,0": "x"}),
                   lambda s: s.balances.pop("a"), lambda s: setattr(s, "open_disputes", 1)):
        child = base.copy()
        change(child)
        roots.add(child.commit())
    assert len(roots) == 6                                # every change moved the root
    child = base.copy()
    child.jobs["j"]                                       # copied on access, unchanged
    child.credit("nobody", 0)                             # a zero credit leaves no entry behind
    assert "nobody" not in child.balances
    assert child.commit() == base.root()
    with pytest.raises(RuntimeError, match="committed"):
        child.commit()
