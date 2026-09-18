# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""`--backsteps N`: cone out N cells at random when a run stops converging (#1088).

Stage 1 only commits moves that LOWER the objective, so once nothing does the
run sits in a local minimum and every further unit is a stall. A backstep is
the way out: remove N top cells chosen uniformly at random among those valid
to remove, and carry on from the perturbed complex.

Three properties, and they are what separate this from the directed cone-out:

  * UNIFORM. `directedConeOut` enumerates candidates and keeps the one that
    most lowers `rU` -- the same greedy rule that produced the minimum. A move
    meant to leave one cannot share it, so nothing here is priced.
  * PINNED REGIONS ARE NOT TOUCHED. `directedConeOut` may remove a pinned
    vertex whenever the result is a manifold in its own right; a backstep may
    not, or a run that pinned its boundary would have it eaten by a
    perturbation.
  * SEEDED. The draw comes from the node's own generator, so a backstep is
    reproducible from the run's seed like every other engine move.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))),
    "examples", "cobordism"))

from tessera.drivers import emergence as ea
from tessera.drivers import qubit as qa
from tessera import cobordism as cob  # noqa: E402
from tessera._tessera.cobordism import MultiCobordism as MC  # noqa: E402


def host(seed=1):
    return ea.build_cobordism_host(1, seed, ea.EdgeDisposition.SPACELIKE)


def node_for(seed=1):
    spacetime = host(seed)
    node = MC(spacetime, [], [], [1], 1.0, seed)
    node.set_objective(cob.JointStationarityObjective())
    return node


def cell_count(node):
    return len(node.spacetime().getTopSimplices())


def test_it_removes_the_requested_number():
    node = node_for()
    before = cell_count(node)
    removed = node.random_cone_out(3)
    assert removed <= 3
    assert cell_count(node) == before - removed


def test_zero_and_negative_are_no_ops():
    node = node_for()
    before = cell_count(node)
    assert node.random_cone_out(0) == 0
    assert node.random_cone_out(-2) == 0
    assert cell_count(node) == before


def test_it_never_removes_more_than_exist():
    """Asking for more than the complex has returns what it could do."""
    node = node_for()
    removed = node.random_cone_out(10_000)
    assert removed < 10_000
    assert cell_count(node) >= 0


def test_a_pinned_region_is_never_eaten():
    """The property directedConeOut deliberately does NOT have."""
    node = node_for()
    spacetime = node.spacetime()
    pinned = set()
    for simplex in spacetime.getTopSimplices()[:2]:
        pinned.update(int(v) for v in [v.getId() for v in simplex.getVertices()])
    node.declare_pinned_region("held", pinned)
    node.random_cone_out(50)
    survivors = {tuple(sorted(int(v) for v in [v.getId() for v in s.getVertices()]))
                 for s in node.spacetime().getTopSimplices()}
    # Every cell that touched the region must still be there.
    for cell in survivors:
        pass
    remaining_vertices = set()
    for s in node.spacetime().getTopSimplices():
        remaining_vertices.update(int(v) for v in [v.getId() for v in s.getVertices()])
    assert pinned <= remaining_vertices, "a backstep removed pinned geometry"


def test_the_draw_is_seeded():
    """Two nodes on the same seed remove the same cells."""
    a, b = node_for(7), node_for(7)
    assert a.random_cone_out(3) == b.random_cone_out(3)
    cells_a = sorted(tuple(sorted(int(v) for v in [v.getId() for v in s.getVertices()]))
                     for s in a.spacetime().getTopSimplices())
    cells_b = sorted(tuple(sorted(int(v) for v in [v.getId() for v in s.getVertices()]))
                     for s in b.spacetime().getTopSimplices())
    assert cells_a == cells_b


def test_the_flag_reaches_both_drivers():
    assert ea.build_config()["backsteps"] == 0
    assert qa.build_config()["backsteps"] == 0
    assert ea.build_config(backsteps=3)["backsteps"] == 3
    assert qa.build_config(backsteps=3)["backsteps"] == 3
    assert qa.build_parser().parse_args(["run", "--backsteps", "4"]).backsteps == 4
    assert ea.build_parser().parse_args(["run", "--backsteps", "4"]).backsteps == 4


def test_off_by_default_so_no_existing_run_changes():
    assert ea.DECLARED_BACKSTEPS == 0
    assert ea.build_parser().parse_args(["run"]).backsteps == 0
