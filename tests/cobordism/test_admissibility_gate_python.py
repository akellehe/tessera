# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The Kontsevich-Segal admissibility gate of `MultiCobordism` is declared.

The Kontsevich-Segal margin of a geometry is the least, over its top
simplices T, of pi - sum_i |arg lambda_i(g_T)|, with lambda_i the eigenvalues
of the simplex's metric from its complex squared lengths
(`HodgeLaplacian.kontsevichSegalMargin`): positive is allowable, zero is the
real Lorentzian boundary, negative is not allowable.

The engine's gate treats a geometry whose margin is below minus the
admissibility tolerance as outside the configuration space: a candidate move
that leads to one is rejected before it is scored, and a trial of the line
search that lands on one is not scored and the step is halved. The gate is
off unless `MultiCobordism.admissibility_gate` is set. Off, every candidate
that leaves a manifold and every trial is scored, and the margin is a number
a caller reads.
"""
import cmath
import math

import numpy as np
import pytest

import tessera as T
from tessera import cobordism as cob
from tessera.drivers import baryon_poles as bp
from tessera.drivers import cell_solve as cs
from tessera.drivers import recursion as R

from tests.drivers import _recursion_levels_2026_10_01 as LEVELS

MC = cob.MultiCobordism
HL = cob.HodgeLaplacian


def _tetrahedron(squared):
    spacetime = T.Spacetime.fromVertexTuples(3, [[0, 1, 2, 3]], 1.0, 0.0)
    for edge in spacetime.getEdgeList().toVector():
        edge.setLength(cmath.sqrt(complex(squared)))
    return spacetime


def test_the_gate_is_off_unless_declared():
    """A node applies no admissibility gate until one is declared, whatever
    its metric source; the admissibility tolerance is declared beside it."""
    for source in (HL.MetricSource.WhitneyPencil,
                   HL.MetricSource.DiagonalWeights):
        node = MC(_tetrahedron(8.0), [], [], [1], metric_source=source)
        assert node.admissibility_gate is False
        assert node.admissibility_tolerance == 1e-15
        node.admissibility_gate = True
        assert node.admissibility_gate is True
        node.admissibility_gate = False
        assert node.admissibility_gate is False


def test_admissibility_is_a_measurement_whatever_the_gate():
    """`geometryAdmissible` says whether a geometry is in the closure of the
    allowable domain with the gate off as with it on. A tetrahedron with
    every squared length -1 has the metric of minus a Euclidean one, three
    eigenvalues of argument pi, margin -2 pi."""
    inside, outside = _tetrahedron(8.0), _tetrahedron(-1.0)
    assert HL.kontsevichSegalMargin(inside) == pytest.approx(math.pi)
    assert HL.kontsevichSegalMargin(outside) == pytest.approx(-2 * math.pi)
    node = MC(inside, [], [], [1],
              metric_source=HL.MetricSource.WhitneyPencil)
    for gate in (False, True):
        node.admissibility_gate = gate
        assert node.geometryAdmissible(inside)
        assert not node.geometryAdmissible(outside)


# ------------------------------------------------- the search over the moves


class _Counted(cs.StationarityObjective):
    """The stationarity objective, keeping the cells of every complex it is
    asked to score."""

    def __init__(self, system):
        super().__init__(system)
        self.cells = []

    def terms(self, context):
        self.cells.append(len(cs.top_cells(context.spacetime)))
        return super().terms(context)


def _moves_scored(gate):
    """One update of the search over the Pachner moves of the level grown at
    tick 0 of the run of 2026-10-01 (five tetrahedra, squared lengths 4.4 to
    8.6), every candidate: the numbers of cells of the complexes scored."""
    level, count = R.build_level(LEVELS.TICK1_CELLS,
                                 dict(LEVELS.GROWN_SQUARED_LENGTHS),
                                 dict(LEVELS.GROWN_LINKS))
    base = R.sheet_base(level, count)
    system, _ = R.level_system(base, R.default_config(), [], R.SHEETS)
    system.begin(base)
    objective = _Counted(system)
    objective.begin()
    node = cs.cell_node(base, objective)
    if gate is not None:
        node.admissibility_gate = gate
    node.run_stage1(max_steps=1, n_candidate_moves=0, grow_boundaries=False,
                    max_lookahead=1, combinatorial_breadth=0)
    return objective.cells


def test_without_the_gate_every_move_that_leaves_a_manifold_is_scored():
    """The level has five 1-4 moves that leave a manifold, each giving its
    four new edges the squared length one. Without the gate, which is the
    default, all five are scored. With it, four of them are outside the
    allowable domain and only one is scored."""
    default, off, on = (_moves_scored(None), _moves_scored(False),
                        _moves_scored(True))
    assert default == off
    assert sorted(off).count(8) == 5
    assert sorted(on).count(8) == 1
    # the base itself, scored before and after the candidates
    assert off.count(5) == on.count(5) == 2


# ------------------------------------------------------------ the line search


class _TowardMinusOne(cob.CobordismObjective):
    """An objective whose scalar is sum_e |z_e + 1|^2 and whose direction
    takes every squared length to -1 at scale one: a tetrahedron with every
    squared length -1 is outside the allowable domain (margin -2 pi)."""

    def __init__(self):
        super().__init__()
        self.scored = []

    def name(self):
        return "toward_minus_one"

    def term_names(self):
        return [cob.ObjectiveTermName.JOINT_ACTION_STATIONARITY]

    def is_target_conditioned(self):
        return False

    @staticmethod
    def _squared(spacetime):
        return np.array([length * length
                         for _, _, length, _ in cs.edge_fields(spacetime)])

    def terms(self, context):
        squared = self._squared(context.spacetime)
        self.scored.append(float(squared[0].real))
        terms = MC.ObjectiveTerms()
        terms.joint_action_stationarity = float(np.sum(abs(squared + 1.0)
                                                       ** 2))
        return terms

    def direction(self, context):
        squared = self._squared(context.scalar.spacetime)
        out = cob.ObjectiveDirection()
        out.ascent = squared + 1.0
        out.phase_ascent = np.zeros(len(squared), dtype=complex)
        out.baseline = float(np.sum(abs(squared + 1.0) ** 2))
        out.baseline_computed = True
        return out


def _one_update(gate):
    spacetime = _tetrahedron(1.0)
    objective = _TowardMinusOne()
    node = cs.cell_node(spacetime, objective)
    node.admissibility_gate = gate
    node.run_stage2(beta=1.0, max_iters=1, alpha0=1.0, tolerance=1e-15)
    final = objective._squared(node.spacetime())
    return objective.scored, final, HL.kontsevichSegalMargin(node.spacetime())


def test_without_the_gate_a_trial_outside_the_domain_is_scored():
    """From squared lengths 1 the full step lands on squared lengths -1.
    Without the gate that trial is scored and accepted, since it lowers the
    scalar from 24 to 0, and the geometry reached has margin -2 pi. With the
    gate the trial is not scored: the step is halved until the trial is
    inside the domain, and the geometry reached has margin pi."""
    scored, final, margin = _one_update(False)
    assert -1.0 in scored
    np.testing.assert_allclose(final, -1.0, atol=1e-14)
    assert margin == pytest.approx(-2 * math.pi)

    scored, final, margin = _one_update(True)
    assert min(scored) > 0.0
    assert np.all(final.real > 0.0) and np.all(final.real < 1.0)
    assert margin == pytest.approx(math.pi)


# ----------------------------------------------------------------- the drivers


def test_the_drivers_declare_the_gate_off_and_offer_it():
    """``--admissibility-gate`` is an option of the solves of both drivers,
    off by default, and reaches the node of every drive and the growth
    step."""
    assert dict((key, value) for key, value, _ in bp.SOLVE_OPTIONS)[
        "admissibility_gate"] is False
    for driver in (bp, R):
        args = driver.build_parser().parse_args(["run"])
        assert args.admissibility_gate is False
        args = driver.build_parser().parse_args(
            ["run", "--admissibility-gate"])
        assert args.admissibility_gate is True
    for declared in (False, True):
        config = R.default_config(solve={"admissibility_gate": declared})
        assert config["admissibility_gate"] is declared
        node = MC(_tetrahedron(8.0), [], [], [1])
        node.admissibility_gate = not declared
        bp.node_configuration(config)(node)
        assert node.admissibility_gate is declared
    assert R.default_config()["admissibility_gate"] is False


def test_the_growth_step_and_a_level_record_the_margin_and_the_gate():
    """The growth step's record and a level relaxation's record say whether
    the gate was applied and carry the Kontsevich-Segal margin of the base
    they end on."""
    cells = LEVELS.TICK1_CELLS
    squared = dict(LEVELS.GROWN_SQUARED_LENGTHS)
    links = dict(LEVELS.GROWN_LINKS)
    for declared in (False, True):
        config = R.default_config(solve={"admissibility_gate": declared})
        record = R.pachner_stage(cells, squared, links, config)[3]
        assert record["admissibility_gate"] is declared
        assert math.isfinite(record["kontsevich_segal_margin"])
    config = R.default_config(solve={"pachner_moves": False},
                              limits={"iteration_limit": 1})
    level, count = R.build_level(cells, squared, links)
    relaxed = R.relax_level(level, config, [], count=count)
    assert relaxed["admissibility_gate"] is False
    assert math.isfinite(relaxed["kontsevich_segal_margin"])
