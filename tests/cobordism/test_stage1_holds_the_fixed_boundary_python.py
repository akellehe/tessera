# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Stage 1 may not rewrite the fixed boundary's geometry (#1022).

A cobordism's boundary is the input state. ``dW = M0 + M1`` is given, and the
search is over the bulk between them. Stage 1 honoured half of that: the cone
kinds were refused when they handed the boundary a face it did not have, which
is the facet SET changing. Nothing watched the boundary's GEOMETRY.

A disposition flip carries one edge across the light cone by negating its
squared length. On an edge of a boundary facet that leaves the facet set
identical and changes the metric of dW underneath it, so the torus is still the
block's surface while the state it represents is a different state. Measured on
the 3x3 collar before this fix, one stage-1 update at a time:

    start            F 123.123551902   blocks 2.199394e-30 1.293515e-30  cells 54
    unit 1 (depth 1) F 112.641218536   blocks 6.747766e-02 1.293515e-30  cells 54
    unit 2 (depth 1) F 104.709114694   blocks 1.413030e-01 1.293515e-30  cells 54
    unit 3 (depth 1) F  99.430475762   blocks 1.294618e-01 1.293515e-30  cells 54

The cell count never moves, so every committed move was a disposition flip, and
diffing the edges names it: edge (6,7) of input torus 0, spacelike 0.380058475
to timelike -0.380058475i. F fell because the Einstein-Hilbert gradient term
dropped by more than the residual term rose. Nothing was wrong with the
arithmetic; the move simply was not a member of the configuration space, and
saying so is the gate's job rather than the objective's.

The move space is still walked in full -- ``enumerate_move_specifications``
offers every candidate it always did. The refusal is in the gate, where the
manifold and boundary checks already live, so there is one place that decides
what a valid trial is.

Stage 2 is the other half, and it is a DRIVER default rather than an engine
rule. It cannot flip a disposition -- it cannot cross the singular l^2 = 0 --
but it does reshape a boundary, and far enough that the modulus moves. Holding
it is not always what a caller wants, though: relaxing a boundary toward its
state is exactly what the input residual weight is for. So the engine still
allows it and ``qubit_animation.py`` pins by default, with
``--no-pin-boundary`` to ask for the old behaviour.
"""
import os
import sys

import pytest

from tessera import cobordism as cob

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))),
    "tests", "cobordism"))

import test_block_surface_residual_python as B  # noqa: E402

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))),
    "examples", "cobordism"))

import qubit_animation as qa  # noqa: E402

MC = cob.MultiCobordism
HL = cob.HodgeLaplacian

#: The residual floor of a flat torus's own state fiber: the zero mode carries
#: the holomorphic form to rounding (observed near 1e-30).
FLOOR = 1e-24


@pytest.fixture
def whitney_default():
    previous = HL.defaultMetricSource()
    HL.setDefaultMetricSource(cob.HodgeMetricSource.WhitneyPencil)
    yield
    HL.setDefaultMetricSource(previous)


def boundary_geometry(node):
    """Every boundary edge's stored length and causal character.

    Read from the complex the node is currently driving, never from the host
    it was constructed with: stage 1 REPLACES the complex when it commits.
    """
    spacetime = node.spacetime()
    facets = [tuple(int(v) for v in f) for f in spacetime.getBoundary()]
    wanted = set()
    for facet in facets:
        for first in range(len(facet)):
            for second in range(first + 1, len(facet)):
                wanted.add((min(facet[first], facet[second]),
                            max(facet[first], facet[second])))
    out = {}
    for edge in spacetime.getEdgeList().toVector():
        key = (min(edge.getSource().getId(), edge.getTarget().getId()),
               max(edge.getSource().getId(), edge.getTarget().getId()))
        if key in wanted:
            out[key] = (edge.getLength(), edge.isTimelike())
    return out


def test_a_committed_pass_leaves_every_boundary_edge_alone(whitney_default):
    """The whole claim, read off the edges rather than off a residual."""
    _qa, _qb, _seed, node = B.collar(3, seed_value=2, einstein_hilbert=True)
    before = boundary_geometry(node)
    assert before, "the collar has a boundary to hold"
    node.run_stage1(max_steps=3, n_candidate_moves=8)
    assert boundary_geometry(node) == before


def test_the_input_states_survive_a_committed_pass(whitney_default):
    """The same claim in the currency the run is scored in.

    Redundant with the edge check by construction and kept anyway: the edges
    are what the gate compares, the residuals are what the physics cares
    about, and a change that satisfied one while breaking the other would be
    the interesting failure.
    """
    _qa, _qb, _seed, node = B.collar(3, seed_value=2, einstein_hilbert=True)
    before = B.residuals(node)
    assert all(r < FLOOR for r in before), before
    node.run_stage1(max_steps=3, n_candidate_moves=8)
    assert all(r < FLOOR for r in B.residuals(node)), B.residuals(node)


def test_an_exhaustive_pass_leaves_it_alone_too(whitney_default):
    """The sampled draw hid this; the walk is where it has to hold.

    ``n_candidate_moves=0`` prices EVERY candidate, so if any offered move
    could change the boundary this pass would find it.
    """
    _qa, _qb, _seed, node = B.collar(3, seed_value=2, einstein_hilbert=True)
    before = boundary_geometry(node)
    node.run_stage1(max_steps=2, n_candidate_moves=0)
    assert boundary_geometry(node) == before
    assert all(r < FLOOR for r in B.residuals(node)), B.residuals(node)


def test_the_walk_still_offers_the_moves_the_gate_refuses(whitney_default):
    """Enumeration stays complete; refusing is the gate's job, not its job.

    Two places deciding what a valid trial is would eventually disagree, so
    the move space is walked in full and one gate settles membership.
    """
    _qa, _qb, _seed, node = B.collar(3, seed_value=2, einstein_hilbert=True)
    specifications = MC.enumerate_move_specifications(node.spacetime(), True)
    boundary_edges = set(boundary_geometry(node))
    offered = [site for kind, site in specifications
               if kind == "flip_disposition"
               and (min(site), max(site)) in boundary_edges]
    assert offered, "the walk must still price boundary dispositions"


def test_a_node_without_a_fixed_boundary_is_unaffected():
    """The gate hangs off a DECLARED fixed boundary and nothing else.

    An emergent host grown from a seed simplex declares none, so its
    dispositions stay free and every free build behaves as it did.
    """
    node = MC(MC.seed_simplex(3), [], [], degrees=[1], seed=0, precone=10,
              einstein_hilbert=True)
    assert not node.has_fixed_boundary
    before = boundary_geometry(node)
    node.run_stage1(max_steps=6, n_candidate_moves=0)
    assert node.last_stage1_lookahead >= 0
    assert boundary_geometry(node) != before or node.last_stage1_lookahead == 0


def test_the_drive_holds_the_input_tori_by_default():
    """The other half of the same rule, and why it is a default not a gate.

    Stage 1's gate refuses a move that changes a boundary edge. Stage 2 cannot
    flip a disposition -- it cannot cross the singular l^2 = 0 -- but it does
    reshape a boundary, and once stage 1 could walk its whole move space it
    reshaped it far enough that the modulus moved. Measured on the two-unit
    qubit drive, the same run twice:

        unheld  residuals 0 0 -> 2.787845e-03 5.718170e-06  cells 54 -> 54
        held    residuals 0 0 -> 0.000000e+00 0.000000e+00  cells 54 -> 56

    The unheld run spends its whole descent reshaping the boundary and commits
    no cell; the held one puts that descent into the bulk.

    A DEFAULT rather than an engine rule, because relaxing a boundary toward
    its state is a real capability -- it is what the input residual weight is
    for, and what `own_state_residual` measures. A run that wants it asks with
    --no-pin-boundary.
    """
    assert qa.DECLARED_PIN_BOUNDARY is True
    assert qa.build_config()["pin_boundary"] is True
    assert qa.build_config(pin_boundary=False)["pin_boundary"] is False


def test_the_engine_still_lets_stage_two_move_a_boundary(whitney_default):
    """The capability the default merely declines to use.

    Holding is the driver's choice. `MultiCobordism` gives a fixed boundary no
    special treatment in stage 2, so a caller that declares no pinned region
    can still relax a boundary toward its state.
    """
    _qa, _qb, _seed, node = B.collar(3, seed_value=2, einstein_hilbert=True)
    assert node.has_fixed_boundary
    before = boundary_geometry(node)
    node.run_stage2(max_iters=20, tolerance=1e-15)
    assert boundary_geometry(node) != before
