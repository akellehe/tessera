# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The complex builders' geometry handling, and SpacetimeComposition.

Three things are asserted here that nothing else covered:

  1. ``getTopSimplexCount`` counts cells whatever their causal type. It used to
     return ``N41 + N32``, which is zero for a complex carrying no CDT causal
     typing at all -- every all-spacelike complex, for instance.
  2. ``fromSimplices`` carries the cells' edge geometry, including the U(1)
     phase's orientation dependence: an edge stored target-to-source carries the
     inverse of the phase on the source-to-target orientation, so copying it
     verbatim flips the link.
  3. ``SpacetimeComposition`` assembles the three compositions at the dimensions
     and values the definitions require.
"""

import cmath
import math

import numpy as np
import pytest

try:
    import tessera
    import tessera._tessera.cobordism as cob
    _IMPORT_OK = True
except Exception:  # pragma: no cover
    _IMPORT_OK = False

pytestmark = pytest.mark.skipif(not _IMPORT_OK, reason="tessera not built")

_TRIANGLE = [[0, 1, 2]]


def _spacetime(dim, topology):
    metric = tessera.Metric(
        True, tessera.Signature(dim, tessera.SignatureType.Lorentzian))
    return tessera.Spacetime(metric, tessera.SpacetimeType.CDT, 1.0, 1.0,
                             tessera.Foliation.PREFERRED, topology)


def _edges_by_pair(st):
    return {(min(e.getSource().getId(), e.getTarget().getId()),
             max(e.getSource().getId(), e.getTarget().getId())): e
            for e in st.getEdgeList().toVector()}


# --------------------------------------------------------------------------- #
# (a) the top-cell count
# --------------------------------------------------------------------------- #
def test_top_simplex_count_holds_without_a_cdt_causal_type():
    st = _spacetime(4, tessera.SolidSimplex(4))
    st.build()
    for edge in st.getEdgeList().toVector():
        edge.setLength(cmath.sqrt(complex(1.0)))   # all spacelike
    assert st.getN41() == 0 and st.getN32() == 0, (
        "an all-spacelike cell carries no CDT causal type")
    assert st.getTopSimplexCount() == len(st.getTopSimplices()) == 1, (
        "the count must come from the live top cells, not the causal counters")


# --------------------------------------------------------------------------- #
# (b) fromSimplices carries geometry
# --------------------------------------------------------------------------- #
def test_from_simplices_carries_lengths_and_phases():
    source = tessera.Spacetime.fromVertexTuples(2, _TRIANGLE)
    lengths = {(0, 1): 2.0 + 0.0j, (0, 2): 3.0 + 0.0j, (1, 2): 5.0 + 0.0j}
    phases = {(0, 1): 0.25 + 0.0j, (0, 2): -0.5 + 0.0j, (1, 2): 0.75 + 0.0j}
    for pair, edge in _edges_by_pair(source).items():
        edge.setLength(cmath.sqrt(lengths[pair]))
        edge.setPhase(phases[pair])

    rebuilt = tessera.Spacetime.fromSimplices(2, list(source.getTopSimplices()))
    got = _edges_by_pair(rebuilt)
    assert set(got) == set(lengths)
    for pair, edge in got.items():
        assert edge.getLength() * edge.getLength() == pytest.approx(
            lengths[pair], rel=1e-12), f"squared length of {pair}"
        assert edge.getPhase() == pytest.approx(phases[pair], rel=1e-12), (
            f"phase of {pair}")


def test_from_simplices_inverts_the_phase_of_a_reversed_edge():
    """The U(1) phase is orientation-dependent.

    An edge stored target-to-source carries the inverse of the phase on the
    source-to-target orientation. fromVertexTuples wires edges over sorted
    vertex tuples, so a rebuilt edge always runs low id to high; a source cell
    may store the same edge the other way round. Copying the phase verbatim
    would flip the link, and no other test constructs a reversed edge.
    """
    # Build the cell so that its edges are stored descending: create the
    # vertices in descending id order and wire the edges before the simplex, so
    # the edge list adopts that orientation rather than the sorted one.
    st = _spacetime(2, tessera.SolidSimplex(2))
    vertices = [st.createVertex(i) for i in (2, 1, 0)]
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            st.createEdge(vertices[i], vertices[j])   # 2->1, 2->0, 1->0
    st.createSimplex(vertices)

    descending = [e for e in st.getEdgeList().toVector()
                  if e.getSource().getId() > e.getTarget().getId()]
    assert descending, "this fixture exists to hold descending edges"

    phase = 0.4 + 0.0j
    for edge in st.getEdgeList().toVector():
        edge.setPhase(phase if edge.getSource().getId() > edge.getTarget().getId()
                      else 0.0 + 0.0j)

    rebuilt = tessera.Spacetime.fromSimplices(2, list(st.getTopSimplices()))
    for edge in rebuilt.getEdgeList().toVector():
        assert edge.getSource().getId() < edge.getTarget().getId(), (
            "the builder wires low id to high")
        # Each rebuilt edge runs opposite to its descending source, so it must
        # carry the inverted phase. Verbatim copying would give +0.4 here.
        assert edge.getPhase() == pytest.approx(-phase, rel=1e-12), (
            "a reversed edge's phase must be inverted, not copied")


def test_from_simplices_refuses_cells_that_disagree_on_a_shared_edge():
    source = tessera.Spacetime.fromVertexTuples(2, [[0, 1, 2], [1, 2, 3]])
    cells = list(source.getTopSimplices())
    assert len(cells) == 2
    # The shared edge (1, 2) belongs to both cells, so it is one edge and they
    # cannot disagree; set it through one cell and both see it.
    shared = _edges_by_pair(source)[(1, 2)]
    shared.setLength(cmath.sqrt(complex(7.0)))
    rebuilt = tessera.Spacetime.fromSimplices(2, cells)
    got = _edges_by_pair(rebuilt)[(1, 2)]
    assert got.getLength() * got.getLength() == pytest.approx(7.0, rel=1e-12)


def test_from_vertex_tuples_rejects_a_mismatched_per_edge_vector():
    with pytest.raises(Exception):
        tessera.Spacetime.fromVertexTuples(
            2, _TRIANGLE, 1.0, 0.0, None, [1.0 + 0j, 2.0 + 0j])  # 2 for 3 edges


def test_from_vertex_tuples_applies_per_edge_geometry_in_order():
    weights = [1.0 + 0j, 4.0 + 0j, 9.0 + 0j]
    st = tessera.Spacetime.fromVertexTuples(2, _TRIANGLE, 1.0, 0.0, None, weights)
    got = [e.getLength() * e.getLength() for e in st.getEdgeList().toVector()]
    assert got == pytest.approx(weights, rel=1e-12), (
        "per-edge weights follow getEdgeList().toVector() order")


# --------------------------------------------------------------------------- #
# (c) SpacetimeComposition
# --------------------------------------------------------------------------- #
def _square(flat):
    n = math.isqrt(len(flat))
    assert n * n == len(flat)
    return np.array(flat, dtype=complex).reshape(n, n)


def test_matrix_level_compositions_match_numpy():
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=complex)
    b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=complex)
    flat_a, flat_b = list(a.ravel()), list(b.ravel())

    ksum = _square(cob.SpacetimeComposition.kroneckerSum(flat_a, 2, flat_b, 2))
    expected_sum = np.kron(a, np.eye(2)) + np.kron(np.eye(2), b)
    assert np.allclose(ksum, expected_sum)

    kprod = _square(cob.SpacetimeComposition.kroneckerProduct(flat_a, 2, flat_b, 2))
    assert np.allclose(kprod, np.kron(a, b))

    dsum = _square(cob.SpacetimeComposition.directSum(flat_a, 2, flat_b, 2))
    expected_direct = np.zeros((4, 4), dtype=complex)
    expected_direct[:2, :2] = a
    expected_direct[2:, 2:] = b
    assert np.allclose(dsum, expected_direct)


def test_space_level_compositions_use_the_factors_laplacians():
    factor = tessera.Spacetime.fromVertexTuples(2, _TRIANGLE)
    lap = _square(cob.HodgeLaplacian(factor).laplacian(0, True))
    n = lap.shape[0]

    cart = _square(cob.SpacetimeComposition.cartesianProduct(factor, factor, 0))
    assert cart.shape == (n * n, n * n)
    assert np.allclose(cart, np.kron(lap, np.eye(n)) + np.kron(np.eye(n), lap))

    tens = _square(cob.SpacetimeComposition.tensorProduct(factor, factor, 0))
    assert np.allclose(tens, np.kron(lap, lap))

    dsum = _square(cob.SpacetimeComposition.directSum(factor, factor, 0))
    assert dsum.shape == (2 * n, 2 * n)
    assert np.allclose(dsum[:n, :n], lap) and np.allclose(dsum[n:, n:], lap)
    assert np.allclose(dsum[:n, n:], 0.0), "the off-diagonal blocks are zero"


def test_cartesian_spectrum_is_the_pairwise_sums():
    """The Kronecker sum's terms commute, which is what makes the shortcut in
    KuennethProduct.pairwiseSpectrum exact."""
    factor = tessera.Spacetime.fromVertexTuples(2, _TRIANGLE)
    lap = _square(cob.HodgeLaplacian(factor).laplacian(0, True))
    own = np.linalg.eigvals(lap)

    cart = _square(cob.SpacetimeComposition.cartesianProduct(factor, factor, 0))
    assembled = np.sort_complex(np.linalg.eigvals(cart))
    pairwise = np.sort_complex(np.array([x + y for x in own for y in own]))
    assert np.allclose(assembled, pairwise, atol=1e-9)
