# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""A frame's objective total is a fold, not a second computation (#1060).

`MultiCobordism::objectiveOf` is a static function of `ObjectiveTerms`, and
`MultiCobordism::objective()` is `objectiveOf(objectiveTermsFor(spacetime_))`.
A frame that has already read the terms therefore holds everything the total
is made of, and calling `objective()` recomputes those terms and throws all
but the scalar away.

That is not a small waste. The terms carry the whole-complex harmonic reading,
which is one Riesz contour band -- 1.7 s on a 90-edge collar and 80 s on the
426-edge four-torus complex. Profiling a reported four-torus run showed the
duplicate at 40% of a 240-second sample in which the run had not yet finished
its first frame.

These tests hold the two facts the fix rests on: the fold returns the same
double as the recomputation, and the frame records the folded one.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))),
    "examples", "cobordism"))

import emergence_animation as ea  # noqa: E402
from tessera._tessera.cobordism import MultiCobordism as MC  # noqa: E402

TAU_A, TAU_B, GRID = 0.3 + 1.1j, -0.2 + 0.8j, 3


@pytest.fixture(scope="module")
def node():
    held = {}
    config = ea.build_config(steps=0, inputs=ea.InputMode.QUBIT, grid=GRID,
                             tau_a=TAU_A, tau_b=TAU_B,
                             input_weight=100.0, regge=False, pin_boundary=True,
                             readout="whole", tori=2, output_state="0.1+1.3j")
    ea.drive(config, progress=False, on_node=lambda n: held.update(node=n))
    return held["node"]


def test_the_fold_and_the_recomputation_are_the_same_double(node):
    """Bit for bit, not merely close: one is defined as the other."""
    terms = node.objective_terms()
    assert MC.objective_of(terms) == node.objective()


def test_the_frame_records_the_folded_total(node):
    """What `_read_objective` writes is what the terms fold to."""
    block = ea.EmergenceFrame._read_objective(node)
    assert block["total"] == MC.objective_of(node.objective_terms())


def test_the_frame_total_still_agrees_with_the_engine(node):
    """The recorded total is unchanged by the fix: a run document written
    before it and one written after it carry the same number."""
    block = ea.EmergenceFrame._read_objective(node)
    assert block["total"] == node.objective()


def test_reading_the_objective_costs_one_terms_computation(node):
    """The point of the fix, stated as a count rather than a duration.

    A timing assertion would be flaky on a shared box; the invariant is that
    `_read_objective` asks the engine for the terms exactly once and never
    asks for the total, which is what makes the second band read disappear.

    The engine object is a pybind11 binding whose attributes are read-only, so
    the count is taken through a proxy that forwards everything else to it.
    """
    class Counting:
        def __init__(self, inner):
            self.inner = inner
            self.calls = {"terms": 0, "objective": 0}

        def objective_terms(self):
            self.calls["terms"] += 1
            return self.inner.objective_terms()

        def objective(self):
            self.calls["objective"] += 1
            return self.inner.objective()

        def __getattr__(self, name):
            return getattr(self.inner, name)

    counting = Counting(node)
    ea.EmergenceFrame._read_objective(counting)
    assert counting.calls == {"terms": 1, "objective": 0}
