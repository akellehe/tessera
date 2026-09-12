# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""A twisted collar, so the cobordism is not the identity propagator (#1064).

`seedCollar` identifies the two surfaces' vertices by ascending id: the
PRODUCT collar, whose two boundary tori are homologous, so every harmonic form
has the same periods on both ends and the monodromy `M = P_B P_A^-1` is the
identity. That cobordism carries its input straight through.

It is not a matter of tuning. `M` is the map induced on `H^1`, an integer
matrix fixed by the topology, and it does not move: measured at
`||M - I|| = 5.4e-15` under a jitter of every squared length by 75%. No
relaxation reaches a different propagator; only the gluing does.

A twist relabels the far surface's base indices by a mapping class before the
identification. It must be a SIMPLICIAL automorphism of the shared
triangulation, which the collar's existing face-set comparison enforces. On
the standard grid torus that admits the swap `[[0,1],[1,0]]` -- and not a Dehn
twist, which sends the diagonal to a step that is no edge.

What this does NOT buy, stated so nobody expects it: `M` is an integer matrix
either way, so the whole-complex harmonic reading realises `GL(2, Z)` and no
more. A continuous gate lives in the transfer, which the same jitter moves by
two orders of magnitude.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))),
    "examples", "cobordism"))

import qubit_animation as qa  # noqa: E402
from tessera import cobordism as cob  # noqa: E402
from tessera import observables as obs  # noqa: E402
from tessera._tessera.cobordism import MultiCobordism as MC, PencilLayer  # noqa: E402

GRID, TAU_A, TAU_B = 3, 0.3 + 1.1j, -0.2 + 0.8j
SWAP = [(k % GRID) * GRID + (k // GRID) for k in range(GRID * GRID)]
#: (i, j) -> (i + j, j): a Dehn twist, and not simplicial on this triangulation.
DEHN = [((k // GRID + k % GRID) % GRID) * GRID + (k % GRID) for k in range(GRID * GRID)]


def collar(twist):
    a = obs.SimplicialQubit.flat_torus(TAU_A, GRID, GRID)
    b = obs.SimplicialQubit.flat_torus(TAU_B, GRID, GRID)
    seed = MC.seed_collar(a.spacetime(), b.spacetime(), 3, twist)
    ids = [{int(k): int(v) for k, v in m.items()} for m in seed.vertex_ids]
    node = MC(seed.host, [[1.0 + 0j]] * 2, [], degrees=[1], seed=1,
              einstein_hilbert=False, real_squared_lengths_only=False,
              metric_source=cob.HodgeMetricSource.WhitneyPencil)
    node.seed_inputs([sorted(m.values()) for m in ids])
    markings = [qa._host_marking(torus, ids[index])
                for index, torus in enumerate((a, b))]
    return node, markings


def monodromy(node, markings):
    assembled = PencilLayer.assemble([node.spacetime()])
    connection = assembled.op.connection()
    band = assembled.op.harmonicBand(1)
    images = np.asarray(band.images)
    blocks = [np.array([[connection.transportedPeriod(images[:, a], walk)
                         for a in range(band.rank())] for walk in marking])
              for marking in markings]
    return blocks[1] @ np.linalg.inv(blocks[0])


@pytest.fixture(scope="module")
def product():
    node, markings = collar([])
    return node, markings, monodromy(node, markings)


@pytest.fixture(scope="module")
def twisted():
    node, markings = collar(SWAP)
    return node, markings, monodromy(node, markings)


def test_the_product_collar_is_the_identity_propagator(product):
    _, _, M = product
    assert np.linalg.norm(M - np.eye(2)) < 1e-10


def test_the_twisted_collar_is_not(twisted):
    """M = [[0, 1], [1, 0]]: the two cycles exchanged, tau -> 1/tau."""
    _, _, M = twisted
    assert np.linalg.norm(M - np.array([[0.0, 1.0], [1.0, 0.0]])) < 1e-10
    assert np.linalg.norm(M - np.eye(2)) > 1.0


def test_the_twist_reverses_orientation(twisted):
    """det -1, which is what makes it an I-bundle and not a product."""
    _, _, M = twisted
    assert abs(np.linalg.det(M) + 1.0) < 1e-10


def test_the_twist_leaves_the_topology_count_alone(product, twisted):
    for node, _, _ in (product, twisted):
        assert node.betti(node.spacetime()) == [1, 2, 1, 0]


@pytest.mark.parametrize("name", ["product", "twisted"])
def test_the_monodromy_is_topological(request, name):
    """The fact that makes this a gluing question and not a tuning one.

    Every squared length is moved by an independent complex jitter. The
    monodromy does not follow, because it is the induced map on `H^1`.
    """
    node, markings, base = request.getfixturevalue(name)
    spacetime = node.spacetime()
    edges = list(spacetime.getEdgeList().toVector())
    original = [edge.getLength() for edge in edges]
    try:
        for amplitude in (0.05, 0.25, 0.75):
            generator = np.random.default_rng(7)
            for edge, length in zip(edges, original):
                step = generator.standard_normal() + 1j * generator.standard_normal()
                edge.setLength(length * (1.0 + amplitude * step))
            assert np.linalg.norm(monodromy(node, markings) - base) < 1e-8
    finally:
        for edge, length in zip(edges, original):
            edge.setLength(length)


def test_a_dehn_twist_is_refused_by_name():
    """Not every mapping class is simplicial. The collar's own face-set
    comparison is what catches it -- no separate gate was added."""
    a = obs.SimplicialQubit.flat_torus(TAU_A, GRID, GRID)
    b = obs.SimplicialQubit.flat_torus(TAU_B, GRID, GRID)
    with pytest.raises(ValueError, match="differ in combinatorics"):
        MC.seed_collar(a.spacetime(), b.spacetime(), 3, DEHN)


def test_a_twist_that_is_not_a_permutation_is_refused():
    a = obs.SimplicialQubit.flat_torus(TAU_A, GRID, GRID)
    b = obs.SimplicialQubit.flat_torus(TAU_B, GRID, GRID)
    with pytest.raises(ValueError, match="not a permutation"):
        MC.seed_collar(a.spacetime(), b.spacetime(), 3, [0] * (GRID * GRID))
    with pytest.raises(ValueError, match="entries for"):
        MC.seed_collar(a.spacetime(), b.spacetime(), 3, [0, 1])


def test_the_driver_flag_builds_the_permutation():
    assert qa._collar_twist({"collar_twist": "none"}, GRID) == []
    assert qa._collar_twist({"collar_twist": "swap"}, GRID) == SWAP
    with pytest.raises(ValueError, match="unknown --collar-twist"):
        qa._collar_twist({"collar_twist": "dehn"}, GRID)
