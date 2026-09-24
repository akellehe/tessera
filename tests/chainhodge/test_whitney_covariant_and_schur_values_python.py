# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The chain-level Hodge core the recursion driver reaches, held to the
formulas of the Whitney integration specification and to exact linear
algebra.

`recursion.base_operator` builds `ChainHodge` and `CovariantChainHodge` over
the level's `ChainComplex` and reads h_1(z, U) (`covariantOperator(1)`) and
its pencil; `recursion.grow` reads `WhitneyMass.topSimplexBlocks`,
`GrownCellRule.invertVertexPairing`, `gaugeInvariantPairing` and
`transportConnection`; `baryon_poles` reads the same operator through
`JointAction.carrier_operator` and eliminates with
`PencilSchur.feshbach`'s Drazin inverse. Expected values:

* the Whitney mass blocks of one tetrahedron from the specification
  (``rsf_whitney_integration_spec.md``, "The inverse chain metric M_k"):
  (M_0)_ab = |T| (1 + delta_ab) / ((d + 1)(d + 2)) and
  (M_1)_[ij][kl] = |T| / ((d + 1)(d + 2)) [(1 + d_ik) G_jl - (1 + d_il) G_jk
  - (1 + d_jk) G_il + (1 + d_jl) G_ik], G the barycentric gradient Gram,
  computed here independently from the edge Gram matrix; M_2 = 1/|t| in two
  dimensions;
* the two operator paths of the program are one operator: the carrier of
  `JointAction` is h_1 in chain variables and `covariantOperator` its
  similarity transform in geometric-image variables, with the same spectrum,
  and at trivial connection both are the Hodge operator;
* exact spectral invariance of h_1(z, U) under a pure gauge;
* a Riesz band is idempotent, of the enclosed rank, and equal to the
  eigenprojector;
* the Drazin inverse of a Jordan block of index two, which is not the
  Moore-Penrose inverse, with A^D A = A A^D = I - Pi_0, A^D Pi_0 = 0,
  A^{k+1} A^D = A^k and similarity covariance;
* the grown-cell rule's N5 normalization det((Z^vee)^T Z) = 3 and its phase
  rule U = det M, which returns U_e for rank-one transports.
"""
import cmath
import math

import numpy as np
import pytest

from tessera import chainhodge as ch
from tessera import cobordism as cob
from tessera.drivers import baryon_poles as bp
from tessera.drivers import recursion as R

TETRAHEDRON = cob.ChainComplex.fromTopCells([[0, 1, 2, 3]])
TRIANGLE = cob.ChainComplex.fromTopCells([[0, 1, 2]])


def _squared(points):
    n = len(points)
    return [complex(np.sum((points[j] - points[i]) ** 2))
            for i in range(n) for j in range(i + 1, n)]


def _gradient_gram(s, d):
    """|T| and the barycentric gradient Gram Gamma of a d-simplex from its
    squared lengths in lexicographic edge order, following the
    specification: g_ij = (s_0i + s_0j - s_ij) / 2, Gamma_ij = (g^-1)_ij for
    i, j >= 1, Gamma_0j = -sum_i Gamma_ij, Gamma_00 = sum_ij Gamma_ij."""
    pairs = [(i, j) for i in range(d + 1) for j in range(i + 1, d + 1)]
    z = {p: s[k] for k, p in enumerate(pairs)}

    def sq(i, j):
        return 0.0 if i == j else z[(min(i, j), max(i, j))]

    g = np.array([[0.5 * (sq(0, i) + sq(0, j) - sq(i, j))
                   for j in range(1, d + 1)] for i in range(1, d + 1)],
                 dtype=complex)
    volume = cmath.sqrt(np.linalg.det(g)) / math.factorial(d)
    inner = np.linalg.inv(g)
    gamma = np.zeros((d + 1, d + 1), dtype=complex)
    gamma[1:, 1:] = inner
    gamma[0, 1:] = gamma[1:, 0] = -inner.sum(axis=0)
    gamma[0, 0] = inner.sum()
    return volume, gamma, pairs


def _tetrahedra():
    rng = np.random.default_rng(29)
    points = rng.normal(size=(4, 3))
    euclidean = _squared(points)
    yield "euclidean", euclidean
    yield "complex", [z * (1 + 0.04j * (k + 1))
                      for k, z in enumerate(euclidean)]


@pytest.mark.parametrize("name, s", list(_tetrahedra()))
def test_the_vertex_block_is_the_barycentric_integral(name, s):
    """(M_0)_ab = |T| (1 + delta_ab) / 20 on a tetrahedron, with |T| =
    sqrt(det g) / 3!."""
    blocks = ch.WhitneyMass.topSimplexBlocks(TETRAHEDRON, s, 0)
    volume, _, _ = _gradient_gram(s, 3)
    expected = volume * (np.eye(4) + np.ones((4, 4))) / 20.0
    assert len(blocks) == 1
    np.testing.assert_allclose(np.asarray(blocks[0].block), expected,
                               rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("name, s", list(_tetrahedra()))
def test_the_edge_block_is_the_gradient_gram_formula(name, s):
    """The degree-1 block of the specification's M_1 formula, entry by
    entry, with Gamma computed independently from the edge Gram matrix."""
    blocks = ch.WhitneyMass.topSimplexBlocks(TETRAHEDRON, s, 1)
    volume, gamma, pairs = _gradient_gram(s, 3)
    c = volume / 20.0

    def delta(a, b):
        return 1.0 if a == b else 0.0

    expected = np.zeros((6, 6), dtype=complex)
    for r, (i, j) in enumerate(pairs):
        for q, (k, l) in enumerate(pairs):
            expected[r, q] = c * ((1 + delta(i, k)) * gamma[j, l]
                                  - (1 + delta(i, l)) * gamma[j, k]
                                  - (1 + delta(j, k)) * gamma[i, l]
                                  + (1 + delta(j, l)) * gamma[i, k])
    block = np.asarray(blocks[0].block)
    assert list(blocks[0].edgeIndices) == list(range(6))
    np.testing.assert_allclose(block, expected, rtol=1e-11, atol=1e-14)
    forward = np.asarray(ch.GrownCellRule.whitneyBlock(c * gamma))
    np.testing.assert_allclose(forward, block, rtol=1e-11, atol=1e-14)


def test_the_top_block_of_a_triangle_is_its_inverse_area():
    """M_2 = 1/|t| for d = 2: the 3-4-5 right triangle has area 6."""
    s = [9.0 + 0j, 16.0 + 0j, 25.0 + 0j]
    blocks = ch.WhitneyMass.topSimplexBlocks(TRIANGLE, s, 2)
    assert complex(np.asarray(blocks[0].block).reshape(-1)[0]) == \
        pytest.approx(1.0 / 6.0, rel=1e-14)


def test_the_cdt_torus_top_block_is_minus_two_root_two_i():
    """Specification T6: on the CDT-like 3 x 3 torus (slice 1, transverse
    -1/2, diagonal 1/2) every triangle has det g_t = -1/2, so
    M_2 = -2 sqrt 2 i I_18 exactly."""
    from tests.chainhodge._fixtures import torus33
    complex_, s = torus33()
    blocks = ch.WhitneyMass.topSimplexBlocks(complex_, s, 2)
    assert len(blocks) == 18
    for block in blocks:
        assert complex(np.asarray(block.block).reshape(-1)[0]) == \
            pytest.approx(-2.0 * math.sqrt(2.0) * 1j, rel=1e-14)


# ----------------------------------------------- the covariant operator


def _monopole_tetrahedron(links=None, squared=8.0):
    support = bp.monopole_support()
    edges = [tuple(e) for e in TETRAHEDRON.kSimplexVertices(1)]
    values = links or [support.transport(x, y) for x, y in edges]
    hodge = ch.ChainHodge(TETRAHEDRON, [complex(squared)] * 6)
    return hodge, ch.CovariantChainHodge(hodge, ch.Connection(TETRAHEDRON,
                                                              values))


def test_the_two_operator_paths_are_one_operator():
    """On the unit-monopole tetrahedron of squared length 8 (the run's host
    cell), the carrier of `JointAction` (h_1 in chain variables, the
    per-cell read's operator) and `CovariantChainHodge.covariantOperator(1)`
    (the level operator of `recursion.base_operator`) have the same six
    eigenvalues to 1e-12, and h_1 = Minv^-1 op Minv exactly, which is the
    pencil's B^-1 A; the sheets of the host carry no coupling."""
    host = bp.build_host()
    carrier = bp.matrix(cob.JointAction(
        host, bp.action_declaration(host, 1.0, 1.0)).carrier_operator())
    assert np.max(np.abs(carrier[:6, 6:])) == 0.0
    h = carrier[:6, :6]
    _, covariant = _monopole_tetrahedron()
    op = np.asarray(covariant.covariantOperator(1))
    np.testing.assert_allclose(np.sort_complex(np.linalg.eigvals(op)),
                               np.sort_complex(np.linalg.eigvals(h)),
                               atol=1e-12)
    minv = covariant.Minv(1)
    minv = np.asarray(minv.toarray() if hasattr(minv, "toarray") else minv)
    assert np.max(np.abs(np.linalg.solve(minv, op @ minv) - h)) < 1e-13
    pencil = covariant.pencil(1)
    a = np.asarray(pencil.A.toarray() if hasattr(pencil.A, "toarray")
                   else pencil.A)
    b = np.asarray(pencil.B.toarray() if hasattr(pencil.B, "toarray")
                   else pencil.B)
    assert np.max(np.abs(np.linalg.solve(b, a) - h)) < 1e-13


def test_at_trivial_connection_both_paths_are_the_hodge_operator():
    hodge, covariant = _monopole_tetrahedron(links=[1.0 + 0j] * 6)
    host = bp.build_host(cell={"squared_lengths": [8.0] * 6,
                               "links": [1.0] * 6})
    carrier = bp.matrix(cob.JointAction(
        host, bp.action_declaration(host, 1.0, 1.0)).carrier_operator())[:6, :6]
    op = np.asarray(covariant.covariantOperator(1))
    hodge_op = hodge.hodgeOperator(1)
    hodge_op = np.asarray(hodge_op.toarray() if hasattr(hodge_op, "toarray")
                          else hodge_op)
    assert np.max(np.abs(op - carrier)) < 1e-13
    assert np.max(np.abs(hodge_op - carrier)) < 1e-13
    # the Whitney spectrum of the regular tetrahedron (WP v17 line 263)
    np.testing.assert_allclose(np.sort(np.linalg.eigvals(op).real),
                               [5, 5, 5, 10, 10, 10], atol=1e-12)


@pytest.mark.parametrize("unitary", [True, False])
def test_the_operator_is_exactly_isospectral_under_pure_gauge(unitary):
    """A vertex gauge g_v (unit modulus, or complex) multiplies U_xy by
    g_x U_xy / g_y; h_1(z, U) moves by a similarity and its spectrum does
    not move, to 1e-12 relative."""
    support = bp.monopole_support()
    edges = [tuple(e) for e in TETRAHEDRON.kSimplexVertices(1)]
    links = [support.transport(x, y) for x, y in edges]
    phases = [0.0, 0.7, -1.3, 2.1]
    moduli = [1.0, 1.0, 1.0, 1.0] if unitary else [1.0, 1.4, 0.6, 2.2]
    g = [m * cmath.exp(1j * p) for m, p in zip(moduli, phases)]
    gauged = [g[x] * u / g[y] for (x, y), u in zip(edges, links)]
    _, before = _monopole_tetrahedron(links)
    _, after = _monopole_tetrahedron(gauged)
    a = np.sort_complex(np.linalg.eigvals(np.asarray(
        before.covariantOperator(1))))
    b = np.sort_complex(np.linalg.eigvals(np.asarray(
        after.covariantOperator(1))))
    assert np.max(np.abs(a - b)) < 1e-12 * np.max(np.abs(a))


def test_a_riesz_band_is_the_eigenprojector():
    """On the unit-monopole tetrahedron (six simple eigenvalues), the Riesz
    band of `covariantOperator(1)` on the circle about its lowest eigenvalue
    with radius half the gap to the next (64 nodes) is idempotent to 1e-15,
    of rank one, and equal to the eigenprojector v w^T built from the matched
    right and left eigenvectors (w^T v = 1) to 1e-14."""
    _, covariant = _monopole_tetrahedron()
    op = np.asarray(covariant.covariantOperator(1))
    values, right = np.linalg.eig(op)
    order = np.argsort(values.real)
    lowest, second = values[order[0]], values[order[1]]
    contour = ch.Contour.circle(complex(lowest), 0.5 * abs(second - lowest),
                                64)
    band = covariant.band(1, contour)
    projector = np.asarray(band.projector)
    assert band.rank() == 1
    assert np.max(np.abs(projector @ projector - projector)) < 1e-15
    left = np.linalg.inv(right)
    eigenprojector = np.outer(right[:, order[0]], left[order[0], :])
    assert np.max(np.abs(projector - eigenprojector)) < 1e-14


# ------------------------------------------------------------ Drazin


JORDAN = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 2]], dtype=complex)


def _drazin(a, radius=1e-10):
    read = ch.PencilSchur.feshbach(a, np.zeros_like(a), 0j, [], 1e-12,
                                   radius)
    return read, np.asarray(read.interiorInverse), np.asarray(
        read.nullProjector)


def test_the_drazin_inverse_of_an_index_two_jordan_block():
    """A = N (+) 2 with N the 2 x 2 nilpotent Jordan block (index 2): the
    Drazin inverse is 0 (+) 1/2 and the Riesz projector onto the generalized
    null space is I_2 (+) 0; the Moore-Penrose inverse [[0, 0, 0], [1, 0, 0],
    [0, 0, 1/2]] differs from it, and no group inverse exists at index 2."""
    read, drazin, null = _drazin(JORDAN)
    assert read.interiorSingular and read.interiorRank == 1
    np.testing.assert_allclose(drazin, np.diag([0, 0, 0.5]), atol=1e-14)
    np.testing.assert_allclose(null, np.diag([1, 1, 0]), atol=1e-14)
    assert np.max(np.abs(np.linalg.pinv(JORDAN) - drazin)) == \
        pytest.approx(1.0, abs=1e-14)


def test_the_drazin_identities():
    """A^D A = A A^D = I - Pi_0, A^D Pi_0 = 0, A^D A A^D = A^D, and
    A^{k+1} A^D = A^k at the index k = 2."""
    _, drazin, null = _drazin(JORDAN)
    identity = np.eye(3)
    assert np.max(np.abs(drazin @ JORDAN - (identity - null))) < 1e-14
    assert np.max(np.abs(JORDAN @ drazin - (identity - null))) < 1e-14
    assert np.max(np.abs(drazin @ null)) < 1e-14
    assert np.max(np.abs(drazin @ JORDAN @ drazin - drazin)) < 1e-14
    a2 = JORDAN @ JORDAN
    assert np.max(np.abs(a2 @ JORDAN @ drazin - a2)) < 1e-14


def test_the_drazin_inverse_is_similarity_covariant():
    """(S A S^-1)^D = S A^D S^-1 for a non-unitary S. Rounding splits the
    transformed defective zero eigenvalue by about sqrt(machine epsilon), so
    the resonance disc is declared at 1e-6 of the spectral radius here, not
    the 1e-10 that suffices for the exact block."""
    s = np.array([[1.0, 2.0, 0.5], [0.0, 1.0, -1.0], [0.3, 0.0, 2.0]],
                 dtype=complex)
    _, drazin, _ = _drazin(JORDAN)
    read, similar, _ = _drazin(s @ JORDAN @ np.linalg.inv(s), 1e-6)
    assert read.interiorSingular and read.interiorRank == 1
    np.testing.assert_allclose(similar, s @ drazin @ np.linalg.inv(s),
                               atol=1e-7)


def test_the_feshbach_determinant_factorization():
    """det(A - lambda M) = det(A_II - lambda M_II) det F_B(lambda) with the
    interface {0, 2} of a generic complex 4 x 4 pencil at lambda = 0.3 + 0.2i,
    checked against numpy's determinants."""
    rng = np.random.default_rng(3)
    a = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    m = np.eye(4) + 0.1 * rng.normal(size=(4, 4))
    lam = 0.3 + 0.2j
    read = ch.PencilSchur.feshbach(a, m.astype(complex), lam, [0, 2])
    pencil = a - lam * m
    interior = [1, 3]
    direct = np.linalg.det(pencil)
    assert complex(read.pencilDeterminant) == pytest.approx(direct,
                                                            rel=1e-12)
    ii = pencil[np.ix_(interior, interior)]
    bb = pencil[np.ix_([0, 2], [0, 2])]
    bi = pencil[np.ix_([0, 2], interior)]
    ib = pencil[np.ix_(interior, [0, 2])]
    schur = bb - bi @ np.linalg.solve(ii, ib)
    assert np.linalg.det(ii) * np.linalg.det(schur) == pytest.approx(
        direct, rel=1e-12)
    assert read.determinantResidual < 1e-12


# ------------------------------------------------------ grown-cell rule


def test_the_n5_normalization_makes_the_image_pairing_three():
    """`recursion.normalize_image_pairing` divides the first column of the
    dual image by det((Z^vee)^T Z) / 3, so the pairing is exactly 3
    afterwards (N5), and the divisor is the original pairing over 3."""
    rng = np.random.default_rng(13)
    image = rng.normal(size=(6, 2)) + 1j * rng.normal(size=(6, 2))
    dual = rng.normal(size=(6, 2)) + 1j * rng.normal(size=(6, 2))
    before = np.linalg.det(dual.T @ image)
    normalized, scale = R.normalize_image_pairing(image, dual)
    assert np.linalg.det(normalized.T @ image) == pytest.approx(3.0,
                                                                abs=1e-13)
    assert scale == pytest.approx(before / 3.0, rel=1e-14)
    with pytest.raises(ValueError, match="pair singularly"):
        R.normalize_image_pairing(image, np.zeros_like(dual))


def test_the_phase_rule_is_the_transport_determinant():
    """U_vw = det M_vw: a rank-one transport [[U_e]] returns U_e; a 2 x 2
    transport returns its determinant; a non-square block is refused."""
    u = cmath.exp(0.4j) * 1.3
    assert complex(ch.GrownCellRule.transportConnection(
        np.array([[u]]))) == u
    block = np.array([[1.0, 2.0j], [0.5, 3.0]])
    assert complex(ch.GrownCellRule.transportConnection(block)) == \
        pytest.approx(3.0 - 1.0j, abs=1e-15)
    with pytest.raises(ValueError):
        ch.GrownCellRule.transportConnection(np.ones((2, 3)))


def test_the_three_dimensional_scale_and_the_two_dimensional_caveat():
    """From |T| Gamma of a regular tetrahedron of squared length 8, the
    vertex-pairing inversion returns every squared length 8 with
    C = 14400 / det(g/C) and |T| = 20 C = 8/3 (the content 8^{3/2} V_3); in
    two dimensions the scale is undetermined (C and the lengths unread)."""
    s = [8.0 + 0j] * 6
    volume, gamma, _ = _gradient_gram(s, 3)
    inversion = ch.GrownCellRule.invertVertexPairing(volume * gamma)
    assert inversion.scaleDetermined
    np.testing.assert_allclose(np.asarray(inversion.squaredLengths), s,
                               rtol=1e-12)
    assert complex(inversion.volume) == pytest.approx(8.0 / 3.0, rel=1e-12)
    assert complex(inversion.scale) == pytest.approx(
        14400.0 / np.linalg.det(np.asarray(inversion.scaledMetric)),
        rel=1e-12)
    assert inversion.rowSumDefect < 1e-14
    t = [1.0 + 0j, 1.0 + 0j, 1.0 + 0j]
    area, gamma2, _ = _gradient_gram(t, 2)
    flat = ch.GrownCellRule.invertVertexPairing(area * gamma2)
    assert not flat.scaleDetermined
    assert math.isnan(complex(flat.scale).real)
    assert list(flat.squaredLengths) == []
