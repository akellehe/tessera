# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The final geometry a run can be rebuilt from (#995).

``examples/cobordism/emergence_animation.py --geometry PATH`` writes the
complex the drive ended on, in the campaign worker's schema 1: the dimension
read off the top cells, the cells in their intrinsic vertex order, every edge
as ``[source, target, Re l^2, Im l^2]``, the per-vertex times, the edge
phases when any is nonzero, and -- in the qubit mode -- each input block's
vertex set, marking and input coefficients. It is the only output a run can
be rebuilt from: the run document records measurements OF a geometry, never
the geometry.

Measured here: a two-unit qubit drive's dump rebuilds through
``Spacetime.fromCells`` to a complex with the same cells and the same edge
squared lengths to a few units in the last place, and a node rebuilt on
it reports the same objective and the same block residuals as the node that
was driven. The last place is where the schema puts the limit: an ``Edge``
stores the complex length, the dump records its square, and the rebuild
takes a square root, so a round trip is exact in the length and correct to
rounding in its square.
"""
import json
import os
import sys
import warnings

import numpy as np
import pytest

import tessera
from tessera import cobordism as cob

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))),
    "examples", "cobordism"))

import emergence_animation as ea  # noqa: E402

MC = cob.MultiCobordism
TAU_A = complex(0.3, 1.1)
TAU_B = complex(-0.2, 0.8)


@pytest.fixture(scope="module")
def driven():
    """One short qubit drive, with the node it drove and its dump."""
    config = ea.build_config(steps=2, stage1_iters=1, stage2_iters=2,
                             tolerance=1e-30, inputs=ea.InputMode.QUBIT,
                             tau_a=TAU_A, tau_b=TAU_B, grid=3)
    held = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ea.drive(config, progress=False,
                          on_node=lambda node: held.update(node=node))
        document = ea.geometry_document(held["node"], result.inputs)
    return held["node"], result, json.loads(json.dumps(document))


def squared_lengths(spacetime):
    out = {}
    for edge in spacetime.getEdgeList().toVector():
        a, b = edge.getSource().getId(), edge.getTarget().getId()
        out[(min(a, b), max(a, b))] = complex(edge.getLength()) ** 2
    return out


def rebuild(document):
    """The dump's own rebuild path (`tests/cobordism/_causal_specimen`)."""
    spacetime = tessera.spacetime.Spacetime.fromCells(document["dimensions"],
                                                      document["cells"])
    vertices = spacetime.getVertexList()
    for vid, t in document["vertex_times"]:
        vertices.get(int(vid)).setTime(float(t))
    by_pair = {}
    for edge in spacetime.getEdgeList().toVector():
        a, b = edge.getSource().getId(), edge.getTarget().getId()
        by_pair[(min(a, b), max(a, b))] = edge
    for u, v, re_l2, im_l2 in document["edges"]:
        key = (min(int(u), int(v)), max(int(u), int(v)))
        by_pair[key].setLength(np.sqrt(complex(re_l2, im_l2)))
    for entry in document.get("edge_phases", []):
        u, v, re_p, im_p = entry
        by_pair[(min(int(u), int(v)), max(int(u), int(v)))].setPhase(
            complex(re_p, im_p))
    spacetime.materializeFacets()
    return spacetime


def test_the_dump_is_schema_1_and_describes_the_driven_complex(driven):
    node, result, document = driven
    spacetime = node.spacetime()
    assert document["schema"] == 1
    assert document["dimensions"] == 3
    cells = {tuple(sorted(c)) for c in document["cells"]}
    live = {tuple(sorted(int(v.getId()) for v in cell.getVertices()))
            for cell in spacetime.getTopSimplices()}
    assert cells == live and len(document["cells"]) == len(live)
    assert len(document["edges"]) == len(spacetime.getEdgeList().toVector())
    assert len(document["vertex_times"]) == len(spacetime.getVertexList().toVector())
    # the qubit blocks: what a Spacetime alone cannot say
    assert [b["label"] for b in document["blocks"]] == ["A", "B"]
    for index, tau_in in enumerate((TAU_A, TAU_B)):
        block = document["blocks"][index]
        assert len(block["vertices"]) == 9
        assert len(block["marking"]) == 2 and all(len(c) > 0 for c in block["marking"])
        assert [complex(*z) for z in block["coefficients"]] == [1.0 + 0j, tau_in]
        assert complex(*block["tau_in"]) == tau_in
        # each torus's own surface, in the same shape as the whole
        surface = block["surface"]
        assert surface["dimensions"] == 2
        assert len(surface["cells"]) == 18 and len(surface["edges"]) == 27
        assert all(len(cell) == 3 for cell in surface["cells"])
        inside = set(block["vertices"])
        assert all(set(cell) <= inside for cell in surface["cells"])
        # the torus's edges ARE the whole's on those vertices, value for value
        whole = {(min(u, v), max(u, v)): (re, im)
                 for u, v, re, im in document["edges"]}
        for u, v, re, im in surface["edges"]:
            assert whole[(min(u, v), max(u, v))] == (re, im)
    # the tori are relaxed off the real locus, which the dump carries
    assert any(abs(edge[3]) > 0 for edge in document["edges"])


def test_the_dump_rebuilds_the_complex_to_rounding(driven):
    node, _result, document = driven
    rebuilt = rebuild(document)
    before, after = squared_lengths(node.spacetime()), squared_lengths(rebuilt)
    assert set(before) == set(after)
    for key, value in before.items():
        # The schema records l^2 and the rebuild takes its square root, so
        # the squared length comes back one rounding away, not bit-identical.
        assert after[key] == pytest.approx(value, rel=1e-15, abs=0.0), (key, value, after[key])


def test_each_torus_loads_on_its_own_as_the_qubit_it_was(driven):
    """A block's surface plus its marking is a SimplicialQubit: the point of
    writing it separately is that a torus can be picked up and fiddled with
    without carrying the bulk."""
    node, _result, document = driven
    from tessera import observables as obs

    for index, block in enumerate(document["blocks"]):
        surface = rebuild(block["surface"])
        ids = sorted(int(v.getId()) for v in surface.getVertexList().toVector())
        position = {vid: n for n, vid in enumerate(ids)}
        pairs = sorted((min(position[int(e.getSource().getId())],
                            position[int(e.getTarget().getId())]),
                        max(position[int(e.getSource().getId())],
                            position[int(e.getTarget().getId())]))
                       for e in surface.getEdgeList().toVector())
        index_of = {pair: n for n, pair in enumerate(pairs)}
        cycles = []
        for cycle in block["marking"]:
            steps = []
            for u, v in cycle:
                a, b = position[int(u)], position[int(v)]
                steps.append((index_of[(min(a, b), max(a, b))], 1 if a < b else -1))
            cycles.append(steps)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loaded = obs.SimplicialQubit(surface, cycles[0], cycles[1],
                                         reversed=False)
            if loaded.intersection_number() < 0:
                loaded = obs.SimplicialQubit(surface, cycles[0], cycles[1],
                                             reversed=True)
        assert abs(loaded.intersection_number() - 1.0) < 1e-12
        assert loaded.tau() == pytest.approx(complex(node.block_qubit(index).tau()),
                                             rel=1e-12)


def test_a_node_rebuilt_on_the_dump_reads_what_the_driven_node_read(driven):
    node, _result, document = driven
    previous = cob.HodgeLaplacian.defaultMetricSource()
    cob.HodgeLaplacian.setDefaultMetricSource(cob.HodgeMetricSource.WhitneyPencil)
    try:
        rebuilt = MC(rebuild(document), [[1.0 + 0j], [1.0 + 0j]], [], degrees=[1],
                     seed=0, einstein_hilbert=True, real_squared_lengths_only=False,
                     metric_source=cob.HodgeMetricSource.WhitneyPencil)
        rebuilt.seed_inputs([block["vertices"] for block in document["blocks"]])
        rebuilt.use_fiber_residuals(True)
        rebuilt.set_input_residual_weight(ea.DECLARED_INPUT_WEIGHT)
        for index, block in enumerate(document["blocks"]):
            fiber = node.inputs[index].fiber
            rebuilt.attach_input_fiber(index, fiber, [list(c) for c in fiber.cells])
            rebuilt.set_input_marking(
                index, [[tuple(step) for step in cycle] for cycle in block["marking"]],
                [complex(*z) for z in block["coefficients"]])
        rebuilt.set_two_body_target(np.asarray(node.two_body_target().chi), True)
        assert rebuilt.two_body_residual() == pytest.approx(node.two_body_residual(), rel=1e-12)
        for index in range(2):
            assert rebuilt.input_state_residual(index) == pytest.approx(
                node.input_state_residual(index), rel=1e-9)
        assert rebuilt.r_u(rebuilt.spacetime()) == pytest.approx(
            node.r_u(node.spacetime()), rel=1e-9)
    finally:
        cob.HodgeLaplacian.setDefaultMetricSource(previous)
