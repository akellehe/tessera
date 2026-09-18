# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The deficit angle and its gradient agree whatever order a cell stores.

`Simplex::dihedralAngle` evaluates in the canonical sorted-by-id frame, so that
a cell a Pachner move stored in causal order yields the same deficit as the same
geometry built sorted; without that the action would depend on build history.
Its derivatives evaluate in the raw stored order instead.

The two frames are related by a simultaneous permutation of the rows and columns
of the Cayley-Menger matrix, under which the cofactor ratio
-C_ij / (sqrt(C_ii) sqrt(C_jj)) is invariant, so the value and the derivatives
should agree. Nothing in the code enforces that, and the three copies of the
kernel share no implementation, so it is asserted here: the gradient must match
a finite difference of the value on a cell whose stored order is NOT sorted.
"""

import cmath

import pytest

try:
    import tessera
    _IMPORT_OK = True
except Exception:  # pragma: no cover
    _IMPORT_OK = False

pytestmark = pytest.mark.skipif(not _IMPORT_OK, reason="tessera not built")

#: Central-difference step in l^2, and the agreement it can support. The
#: truncation error is O(h^2) and the roundoff O(eps/h), so a few times 1e-10 is
#: the floor; a frame disagreement would show at order one.
_STEP = 1e-6
_TOLERANCE = 1e-7


def _spacetime(dim):
    metric = tessera.Metric(
        True, tessera.Signature(dim, tessera.SignatureType.Lorentzian))
    return tessera.Spacetime(metric, tessera.SpacetimeType.CDT, 1.0, 1.0,
                             tessera.Foliation.PREFERRED,
                             tessera.SolidSimplex(dim))


def _cell_in_order(order):
    """A (4,1)-type Lorentzian 4-simplex whose stored vertex order is `order`.

    The base tetrahedron is spacelike with l^2 = +1 and every apex edge is
    timelike with l^2 = -1.
    """
    st = _spacetime(4)
    st.build()
    vertices = {v.getId(): v for v in st.getVertexList().toVector()}
    for simplex in list(st.getSimplices()):
        if len(simplex.getVertices()) == 5:
            st.removeSimplex(simplex)
    st.createSimplex([vertices[i] for i in order])
    for edge in st.getEdgeList().toVector():
        endpoints = (edge.getSource().getId(), edge.getTarget().getId())
        edge.setLength(cmath.sqrt(complex(-1.0 if 4 in endpoints else 1.0)))
    # The ReggeSolver constructor materializes the facet and hinge skeleton.
    tessera.ReggeSolver(st, tessera.MatterConfiguration())
    return st


def _edges_by_pair(st):
    out = {}
    for edge in st.getEdgeList().toVector():
        a, b = edge.getSource().getId(), edge.getTarget().getId()
        out[(min(a, b), max(a, b))] = edge
    return out


def _finite_difference(st, hinge, pair):
    """Central difference of the deficit angle in the l^2 of one edge."""
    edge = _edges_by_pair(st)[pair]
    original = edge.getLength() * edge.getLength()
    edge.setLength(cmath.sqrt(complex(original + _STEP)))
    plus = complex(hinge.deficitAngle())
    edge.setLength(cmath.sqrt(complex(original - _STEP)))
    minus = complex(hinge.deficitAngle())
    edge.setLength(cmath.sqrt(complex(original)))
    return (plus - minus) / (2.0 * _STEP)


@pytest.mark.parametrize("order", [
    (0, 1, 2, 3, 4),   # sorted: the frames coincide
    (3, 0, 2, 1, 4),
    (4, 2, 0, 3, 1),
], ids=["sorted", "permuted", "reversed_apex_first"])
def test_gradient_matches_the_value_whatever_order_the_cell_stores(order):
    st = _cell_in_order(order)
    cell = next(s for s in st.getSimplices() if len(s.getVertices()) == 5)
    assert [v.getId() for v in cell.getVertices()] == list(order), (
        "createSimplex is expected to keep the given vertex order")

    hinges = [s for s in st.getSimplices() if len(s.getVertices()) == 3]
    assert hinges, "the Regge solver should materialize the 2-simplex hinges"
    hinge = hinges[0]

    gradient = {tuple(k): v for k, v in hinge.deficitAngleGradient().items()}
    assert gradient, "a hinge of a 4-simplex has a gradient in every edge"

    for pair, analytic in gradient.items():
        numeric = _finite_difference(st, hinge, pair)
        assert abs(complex(analytic) - numeric) < _TOLERANCE, (
            f"edge {pair}: analytic {analytic} against finite difference "
            f"{numeric}; the derivative's frame disagrees with the value's")
