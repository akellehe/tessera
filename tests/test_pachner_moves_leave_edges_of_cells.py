# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The edges a pre-geometric Pachner move leaves: every edge of the edge
list is an edge of a top cell after the 2-3, 3-2, 1-4 and 4-1 moves in
three dimensions, and a rolled-back 3-2 move restores every edge with its
squared length and its link.

Terms used below:

* the *edges of the cells* are the vertex pairs of the top cells; the
  *listed* edges are those of `Spacetime.getEdgeList`, which the cell solve
  reads (`cell_solve.edge_fields`); the operator's edges are those of the
  chain complex of the top cells (`ChainComplex.fromSpacetime`);
* the *fan* is three tetrahedra around the edge (0, 1) and a fourth on the
  face (0, 2, 3), on which the 3-2 move collapses (0, 1).
"""
import cmath
import itertools

import numpy as np
import pytest

import tessera as T
from tessera import cobordism as cob
from tessera.drivers import baryon_poles as bp
from tessera.drivers import cell_solve as cs

PRE = T.PachnerMode.PreGeometric
FAN = [[0, 1, 2, 3], [0, 1, 3, 4], [0, 1, 2, 4], [0, 2, 3, 5]]


def _complex(cells, seed=7):
    """The cells with a seeded squared length near 8 and a seeded phase on
    every edge."""
    spacetime = T.Spacetime.fromVertexTuples(3, cells, 1.0, 0.0)
    rng = np.random.default_rng(seed)
    for edge in spacetime.getEdgeList().toVector():
        edge.setLength(cmath.sqrt(8.0 * (1.0 + 0.05 * rng.normal())))
        edge.setPhase(0.1 * rng.normal())
    return spacetime


def _apply(spacetime, move_class, *arguments):
    """The first move of the class, by seed, that applies; returned applied."""
    for seed in itertools.count():
        move = move_class(spacetime, seed, *arguments)
        if move.propose() and move.apply():
            return move


def _edges(spacetime):
    """The listed edges, the edges of the cells and the operator's edges."""
    listed = sorted(tuple(sorted(edge[:2])) for edge in
                    cs.edge_fields(spacetime))
    of_cells = sorted({pair for cell in cs.top_cells(spacetime)
                       for pair in itertools.combinations(cell, 2)})
    operator = sorted(tuple(int(v) for v in pair) for pair in
                      cob.ChainComplex.fromSpacetime(
                          spacetime).kSimplexVertices(1))
    return listed, of_cells, operator


@pytest.mark.parametrize("cells, moves", [
    ([[0, 1, 2, 3], [1, 2, 3, 4]], [(T.FlipMove, PRE, False)]),
    ([[0, 1, 2, 3]], [(T.AddMove, False, PRE, False)]),
    ([[0, 1, 2, 3]], [(T.AddMove, False, PRE, False),
                      (T.RemoveMove, PRE, False)]),
], ids=["2-3", "1-4", "1-4 then 4-1"])
def test_a_move_leaves_only_edges_of_cells(cells, moves):
    """After the 2-3 move, the 1-4 move and the 4-1 move that follows it,
    the listed edges are the edges of the cells and the operator's."""
    spacetime = _complex(cells)
    for move_class, *arguments in moves:
        _apply(spacetime, move_class, *arguments)
    listed, of_cells, operator = _edges(spacetime)
    assert listed == of_cells == operator


@pytest.mark.parametrize("cells, moves", [
    (FAN, [(T.IFlipMove, PRE, False)]),
    ([[0, 1, 2, 3], [1, 2, 3, 4]], [(T.FlipMove, PRE, False),
                                    (T.IFlipMove, PRE, False)]),
], ids=["3-2", "2-3 then 3-2"])
def test_a_3_2_move_leaves_only_edges_of_cells(cells, moves):
    """After the 3-2 move, alone on the fan and after a 2-3 move on two
    tetrahedra, the listed edges are the edges of the cells: the collapsed
    edge is not listed."""
    spacetime = _complex(cells)
    for move_class, *arguments in moves:
        _apply(spacetime, move_class, *arguments)
    listed, of_cells, operator = _edges(spacetime)
    assert of_cells == operator
    assert listed == of_cells


def test_a_rolled_back_3_2_move_restores_every_edge():
    """The 3-2 move on the fan rolled back: the cells, and every edge with
    its stored orientation, its squared length and its link, are those
    before the move, exactly."""
    spacetime = _complex(FAN)
    before = sorted(cs.edge_fields(spacetime))
    cells = sorted(cs.top_cells(spacetime))
    move = _apply(spacetime, T.IFlipMove, PRE, False)
    assert sorted(cs.top_cells(spacetime)) != cells
    move.rollback()
    assert sorted(cs.top_cells(spacetime)) == cells
    assert sorted(cs.edge_fields(spacetime)) == before


def test_the_complex_a_3_2_move_makes_has_a_residual():
    """The cell solve scores the complex the 3-2 move makes on the fan: its
    sheeted support carries every listed edge, and the residual norm of the
    joint action's stationarity there is finite."""
    spacetime = _complex(FAN)
    _apply(spacetime, T.IFlipMove, PRE, False)
    config = bp.default_config(kappas=[1.0], betas=[1.0])
    config["held_sectors"] = []
    host = cs.sheeted_support(spacetime, bp.SHEETS)

    def declare(complex_):
        return bp.action_declaration(complex_, 1.0, 1.0,
                                     config["regge_hinges"])

    system = cs.GeometricSystem(
        declare, lambda support: bp.support_geometry(config, support, host),
        bp.SHEETS)
    objective = cs.StationarityObjective(system)
    context = cob.ObjectiveContext()
    context.spacetime = spacetime
    score = objective.terms(context).joint_action_stationarity
    assert objective.undefined == []
    assert np.isfinite(score)
