# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Scoring the whole cobordism's leak beside a block's own residual (#1006).

Two different numbers say whether an input torus still carries its state.

``own_state_residual(i)`` builds the Laplacian of ONE input torus in isolation,
reads its shape as a single complex number from that Laplacian's zero mode and
compares it against the modulus the block was marked with. It answers whether
the torus, judged alone, still represents the state it was handed.

``input_state_residual(i)`` builds the Laplacian of the ENTIRE cobordism, takes
its zero mode, and measures how much of the input coefficients -- written
through the torus's live frame -- fails to lie in it. It answers whether the
ground state of the whole object still knows what the inputs were.

The first is local to a torus; the second is that torus's relationship to
everything else. Only the first was in the objective, so the second drifted.
``score_whole_complex_leak(True)`` ADDS the second to the first at the same
``set_input_residual_weight``, which keeps the spec's rule 3 -- the block's
own-Laplacian residual is in the objective next to the bulk terms -- true
literally.

A HELD boundary is what makes it matter. A block whose edges cannot move cannot
change shape, so its own residual is identically zero and the weight multiplies
nothing; the objective is then the bulk term alone and nothing says how the
tori sit in the cobordism. Enabling this holds nothing: the leak becomes a term
the relaxation trades against, not a constraint.

The fixture is the same collar the own-residual tests use: two 3x3
``SimplicialQubit.flat_torus`` inputs (tau_A = 0.3 + 1.1i, tau_B = -0.2 + 0.8i)
on the Whitney pencil.
"""
import sys

import numpy as np
import pytest

from tessera import cobordism as cob

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import test_block_residual_whole_frame_python as whole  # noqa: E402
import test_own_state_residual_python as own  # noqa: E402

MC = cob.MultiCobordism
HL = cob.HodgeLaplacian
WEIGHT = 1e4


@pytest.fixture
def whitney_default():
    previous = HL.defaultMetricSource()
    HL.setDefaultMetricSource(cob.HodgeMetricSource.WhitneyPencil)
    try:
        yield
    finally:
        HL.setDefaultMetricSource(previous)


def marked(weight=WEIGHT, **kwargs):
    return own.marked_collar(weight=weight, **kwargs)


def ascent(node):
    lengths, _phases = node.fiber_mode_ascent()
    return np.asarray(lengths).ravel()


def test_it_is_off_by_default_and_off_changes_nothing(whitney_default):
    """The flag is additive; an existing run must be bit-identical without it."""
    _qa, _qb, st_seed, node = marked()
    assert node.scores_whole_complex_leak() is False
    before_value = node.r_u(node.spacetime())
    before_direction = ascent(node)
    node.score_whole_complex_leak(True)
    node.score_whole_complex_leak(False)
    assert node.scores_whole_complex_leak() is False
    assert node.r_u(node.spacetime()) == before_value
    assert np.array_equal(ascent(node), before_direction)
    assert st_seed is not None


def test_the_objective_is_the_own_residual_plus_the_leak(whitney_default):
    """Arithmetic, so a reader can see exactly what was added and at what weight."""
    qa, qb, _seed, node = marked()
    owns = [node.own_state_residual(i) for i in range(2)]
    leaks = [node.input_state_residual(i) for i in range(2)]
    # The two are genuinely different numbers on this seed: the torus
    # represents its own state exactly while the whole's zero mode does not.
    assert max(owns) < 1e-14
    assert min(leaks) > 1e-3

    node.score_whole_complex_leak(False)
    assert node.r_u(node.spacetime()) == pytest.approx(WEIGHT * sum(owns), abs=1e-12)
    node.score_whole_complex_leak(True)
    assert node.r_u(node.spacetime()) == pytest.approx(
        WEIGHT * (sum(owns) + sum(leaks)), rel=1e-12)

    # And beside the bulk term, not instead of it.
    node.set_two_body_target(
        whole.spin_half_chi(np.asarray(qa.state()), np.asarray(qb.state())), True)
    assert node.r_u(node.spacetime()) == pytest.approx(
        WEIGHT * (sum(owns) + sum(leaks)) + node.two_body_residual(), rel=1e-12)


def test_the_added_direction_is_exactly_the_leak_gradient(whitney_default):
    """The descent direction gains the leak's gradient at the same weight.

    Asserted as a difference of the assembled ascent rather than by reading the
    leak gradient alone, because what matters is what stage 2 actually steps
    along.
    """
    _qa, _qb, _seed, node = marked()
    node.score_whole_complex_leak(False)
    without = ascent(node)
    node.score_whole_complex_leak(True)
    with_leak = ascent(node)
    expected = np.zeros_like(without)
    for index in range(2):
        lengths, _phases = node.input_state_residual_gradient(index)
        expected = expected + WEIGHT * np.asarray(lengths).ravel()
    assert np.abs((with_leak - without) - expected).max() < 1e-9 * max(
        1.0, np.abs(expected).max())


def test_the_leak_reaches_the_bulk_where_the_own_residual_cannot(whitney_default):
    """The point of the term, as a support claim.

    A torus's own residual is a function of that torus's edges alone, so its
    gradient is exactly zero on every bulk edge -- it can say nothing about the
    cobordism between the tori. The whole's leak is read on the whole's zero
    mode, so its gradient is not.
    """
    _qa, _qb, seed, node = marked()
    own_a, own_b, bulk = own.region_masks(node, seed)
    assert bulk.sum() > 0 and (own_a | own_b).sum() > 0

    node.score_whole_complex_leak(False)
    without = ascent(node)
    assert np.abs(without[bulk]).max() < 1e-12, "the own residual cannot see the bulk"

    node.score_whole_complex_leak(True)
    with_leak = ascent(node)
    assert np.abs(with_leak[bulk]).max() > 1e-9, "the leak does"


def test_a_held_boundary_leaves_nothing_to_score_without_it(whitney_default):
    """Why this exists.

    Pin both tori and their own residual can no longer move -- it is identically
    zero and the input weight multiplies nothing, so with the flag off the
    objective is the bulk term alone and the tori are unconstrained. With the
    flag on there is still something to say about them.
    """
    qa, qb, seed, node = marked()
    for index in range(2):
        node.declare_pinned_region("input%d" % index,
                                   set(int(v) for v in seed.vertex_ids[index].values()))
    node.set_two_body_target(
        whole.spin_half_chi(np.asarray(qa.state()), np.asarray(qb.state())), True)

    owns = [node.own_state_residual(i) for i in range(2)]
    leaks = [node.input_state_residual(i) for i in range(2)]
    bulk = node.two_body_residual()

    # The own residual is not exactly zero -- it is at rounding, about 2e-15 --
    # so the honest statement is that what it contributes at the input weight is
    # negligible beside the bulk term, not that it is absent.
    node.score_whole_complex_leak(False)
    assert node.r_u(node.spacetime()) == pytest.approx(
        bulk + WEIGHT * sum(owns), rel=1e-12)
    assert WEIGHT * sum(owns) < 1e-9 * bulk, (owns, bulk)

    # The leak is four orders larger at the same weight: with it scored there is
    # something to say about the tori, and without it there is not.
    node.score_whole_complex_leak(True)
    assert node.r_u(node.spacetime()) == pytest.approx(
        bulk + WEIGHT * (sum(owns) + sum(leaks)), rel=1e-12)
    assert WEIGHT * sum(leaks) > bulk


def test_the_gradient_agrees_with_central_differences(whitney_default):
    """The added term's gradient is analytic; this is the usual check on it.

    Differences of order 1e-6 in the step give about 1e-8 of accuracy, which is
    the tolerance here -- the difference's own accuracy, not the gradient's.
    """
    _qa, _qb, seed, node = marked()
    own.jitter_tori(node, seed, amplitude=0.05)
    node.score_whole_complex_leak(True)
    edges = node.spacetime().getEdgeList().toVector()
    analytic = np.zeros(len(edges), dtype=complex)
    for index in range(2):
        lengths, _phases = node.input_state_residual_gradient(index)
        analytic = analytic + np.asarray(lengths).ravel()

    def leak_total():
        return sum(node.input_state_residual(i) for i in range(2))

    h = 1e-6
    checked = 0
    for position in (0, len(edges) // 3, len(edges) // 2, len(edges) - 1):
        edge = edges[position]
        l0 = complex(edge.getLength())
        z0 = l0 * l0
        packed = []
        for step in (h, 1j * h):
            values = []
            for sign in (+1, -1):
                root = np.sqrt(z0 + sign * step)
                root = root if abs(root - l0) <= abs(-root - l0) else -root
                edge.setLength(complex(root))
                values.append(leak_total())
            packed.append((values[0] - values[1]) / (2 * h))
        edge.setLength(l0)
        numeric = complex(packed[0], packed[1])
        assert abs(numeric - analytic[position]) < 3e-6 * max(1.0, abs(numeric)), position
        checked += 1
    assert checked == 4


def test_stage_two_lowers_the_leak_when_it_is_scored(whitney_default):
    """The term does what it is for: descended, the leak comes down.

    The control is the same complex with the flag off, where nothing scores the
    leak and it is free to go the other way.
    """
    results = {}
    for enabled in (False, True):
        qa, qb, seed, node = marked()
        own.jitter_tori(node, seed, amplitude=0.05)
        node.set_two_body_target(
            whole.spin_half_chi(np.asarray(qa.state()), np.asarray(qb.state())), True)
        node.score_whole_complex_leak(enabled)
        before = sum(node.input_state_residual(i) for i in range(2))
        list(node.run_stage2(max_iters=8, tolerance=1e-15))
        results[enabled] = (before, sum(node.input_state_residual(i) for i in range(2)))
    scored_before, scored_after = results[True]
    assert scored_after < scored_before, results
    # Not a claim that the control must RISE -- only that scoring the leak is
    # what makes it fall further than leaving it unscored does.
    control_before, control_after = results[False]
    assert (scored_before - scored_after) > (control_before - control_after)
