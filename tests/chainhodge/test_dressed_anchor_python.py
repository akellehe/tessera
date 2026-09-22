# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The anchor by the dressed coordinate and its transitions
(:class:`tessera.chainhodge.DressedAnchor`), ticket #1198 and whitepaper
section 10.

The whitepaper is the specification these tests hold the code to.

* ``res_{tau->p}(U)`` restricts a one-chain to an oriented triangle's three
  ordered boundary edges and parallel-transports those three coefficients to
  one chosen base vertex ``p`` along the declared walks. Its covariance law is
  ``res(U^g) rho_1(g) = g_p^{-1} res(U)`` with the repository's
  ``rho_1(g) = diag(g_{b(sigma)}^{-1})``; the whitepaper writes the same law
  with ``rho_1(g)^{-1}`` because it takes the opposite sign convention for
  ``rho_1``.
* ``Delta_tau = det(res_{tau->p}(U) Phi_Q)`` transforms by the common factor
  ``g_p^{-3} det g_Q``, so the profile ``[Delta_Q] = [Delta_tau]`` is
  frame-independent as a point of a projective space wherever it is nonzero.
* For a band of rank ``r < 3`` the same construction uses
  ``Lambda^r res_{tau->p}(U) Phi_Q``, the maximal minors of a ``3 x r``
  matrix, with ``det g_Q`` replaced by the induced action on ``Lambda^r``,
  which on a maximal exterior power is again multiplication by ``det g_Q``.
* The determinant-line transitions on overlaps are the ratios of two faces'
  coordinates. They are invariant under both gauges and satisfy the cocycle
  identity. Changing the base vertex is the other chart change, and when the
  walks factor through the old base vertex the ratio is the scalar
  ``T_{p'}(p)^r`` on every face.
* No modulus, square root, free face weight or real-valued score enters the
  physical definition; the anchor refuses rather than reporting a
  gauge-dependent raw restriction.
* The anchoring theorem: the restriction to tau's three ordered boundary edges
  of the twisted coboundary of a vertex potential has determinant
  ``F_tau - 1``; an exact band therefore anchors only where the face holonomy
  is nontrivial and nowhere at all at flat connection; a coexact band anchors
  on every face of the tetrahedron, equal in modulus; and the certificate
  selects the coexact band uniquely.

Closed forms verified here rather than fitted:

* for ``tau = (x < y < z)`` and the path rule's transports ``T``,
  ``Delta_tau = T(x)^2 T(y) U_xz (F_tau - 1) det(Phi0|_{x,y,z})`` for the exact
  band ``Phi = delta_0^U Phi0``, which is the whitepaper's
  ``Delta_tau = (F_tau - 1) det res_{x,y,z} Phi0`` carrying this convention's
  nonvanishing path factor;
* on the regular tetrahedron with trivial connection the coexact band spanned
  by three triangle boundaries has ``Delta_tau = 1`` exactly on all four faces.
"""
import numpy as np
import pytest

from tessera import chainhodge as ch
from tessera import cobordism as cob

DA = ch.DressedAnchor

TETRAHEDRON = [[0, 1, 2, 3]]
TWO_TETRAHEDRA = [[0, 1, 2, 3], [1, 2, 3, 4]]
TWO_TRIANGLES = [[0, 1, 2], [3, 4, 5]]

MACHINE = 1e-10


def _complex_of(cells):
    return cob.ChainComplex.fromTopCells(cells)


def _edges(K):
    return [tuple(int(v) for v in e) for e in K.kSimplexVertices(1)]


def _triangles(K):
    return [tuple(int(v) for v in t) for t in K.kSimplexVertices(2)]


def _trivial(K):
    return ch.Connection(K, [1.0 + 0j] * K.numSimplices(1))


def _connection(K, table):
    """A connection declaring `table[edge]` on the named edges and 1 elsewhere."""
    links = []
    for edge in _edges(K):
        links.append(complex(table.get(edge, 1.0)))
    return ch.Connection(K, links)


def _random_links(K, rng):
    return ch.Connection(
        K, [complex(rng.normal(), rng.normal()) + 1.5 for _ in range(K.numSimplices(1))]
    )


def _random_gauge(K, rng):
    return {
        int(v[0]): complex(rng.normal(), rng.normal()) + 1.5 for v in K.kSimplexVertices(0)
    }


def _rho1(K, gauge):
    """rho_1(g) = diag(g_{b(e)}^{-1}) on the canonical edges, b(e) = min e."""
    return np.array([1.0 / gauge[e[0]] for e in _edges(K)], dtype=complex)


def _boundary(K, k):
    rows = K.numSimplices(k - 1)
    cols = K.numSimplices(k)
    return np.asarray(K.boundaryMatrix(k), dtype=float).reshape(rows, cols).astype(complex)


def _exact_band(K, U, potentials):
    """Phi = delta_0^U Phi0 with (delta_0^U f)_{xy} = U_xy f_y - f_x."""
    rows = []
    for (x, y) in _edges(K):
        rows.append(U.link(x, y) * potentials[y, :] - potentials[x, :])
    return np.asarray(rows, dtype=complex)


def _coexact_band(K, columns):
    """A band inside im d_2: the named triangle boundaries."""
    return _boundary(K, 2)[:, list(columns)]


def _face_holonomy(U, triangle):
    """F_tau = U_xy U_yz U_zx for tau = (x -> y -> z -> x)."""
    x, y, z = triangle
    return U.link(x, y) * U.link(y, z) * U.link(z, x)


# --------------------------------------------------------------------------- #
# The restriction and its covariance
# --------------------------------------------------------------------------- #
class TestRestriction:
    def test_the_ordered_boundary_edges_are_the_cyclic_ones(self):
        K = _complex_of(TETRAHEDRON)
        paths = ch.DeclaredPaths.breadthFirst(K, 0)
        edges = _edges(K)
        for index, (v0, v1, v2) in enumerate(_triangles(K)):
            face = DA.faceRestriction(K, _trivial(K), paths, index)
            assert [edges[j] for j in face.edgeIndices] == [(v0, v1), (v1, v2), (v0, v2)]
            assert list(face.incidenceSigns) == [+1, +1, -1]
            assert [int(b) for b in face.basePoints] == [v0, v1, v0]

    def test_the_restriction_matrix_is_the_three_scaled_rows(self):
        K = _complex_of(TETRAHEDRON)
        rng = np.random.default_rng(3)
        U = _random_links(K, rng)
        paths = ch.DeclaredPaths.breadthFirst(K, 0)
        R = np.asarray(DA.restriction(K, U, paths, 0))
        assert R.shape == (3, K.numSimplices(1))
        assert np.count_nonzero(R) == 3
        face = DA.faceRestriction(K, U, paths, 0)
        for i in range(3):
            assert abs(R[i, face.edgeIndices[i]] - face.factors[i]) < MACHINE

    def test_the_restricted_frame_is_the_restriction_times_the_frame(self):
        K = _complex_of(TETRAHEDRON)
        rng = np.random.default_rng(4)
        U = _random_links(K, rng)
        paths = ch.DeclaredPaths.breadthFirst(K, 0)
        Phi = rng.normal(size=(K.numSimplices(1), 3)) + 1j * rng.normal(
            size=(K.numSimplices(1), 3)
        )
        direct = np.asarray(DA.restrictedFrame(K, U, paths, 2, Phi))
        viaMatrix = np.asarray(DA.restriction(K, U, paths, 2)) @ Phi
        assert np.allclose(direct, viaMatrix, atol=MACHINE)

    def test_the_transport_is_the_ordered_product_along_the_declared_walk(self):
        K = _complex_of(TETRAHEDRON)
        rng = np.random.default_rng(5)
        U = _random_links(K, rng)
        paths = ch.DeclaredPaths.breadthFirst(K, 0)
        walks = paths.walks()
        for vertex, walk in walks.items():
            expected = 1.0 + 0j
            for a, b in zip(walk[:-1], walk[1:]):
                expected *= U.link(a, b)
            assert abs(paths.transport(U, vertex) - expected) < MACHINE

    def test_rho_one_is_the_repositorys_own_degree_one_representation(self):
        """The base-point convention b(e) = min e that makes the covariance law
        close is the same one CovariantChainHodge uses."""
        K = _complex_of(TETRAHEDRON)
        rng = np.random.default_rng(6)
        gauge = _random_gauge(K, rng)
        hodge = ch.ChainHodge(K, [1.0 + 0j] * K.numSimplices(1), ch.Preset.L2)
        cov = ch.CovariantChainHodge(hodge, _trivial(K))
        assert np.allclose(np.asarray(cov.rho(1, gauge)).ravel(), _rho1(K, gauge), atol=MACHINE)

    def test_the_covariance_law_holds_on_the_full_restriction_matrices(self):
        K = _complex_of(TETRAHEDRON)
        rng = np.random.default_rng(7)
        U = _random_links(K, rng)
        gauge = _random_gauge(K, rng)
        paths = ch.DeclaredPaths.breadthFirst(K, 0)
        rho = _rho1(K, gauge)
        for index in range(K.numSimplices(2)):
            left = np.asarray(DA.restriction(K, U.gauge(gauge), paths, index)) * rho[None, :]
            right = np.asarray(DA.restriction(K, U, paths, index)) / gauge[paths.basePoint()]
            assert np.allclose(left, right, atol=MACHINE)

    def test_the_reported_covariance_residual_is_at_machine_precision(self):
        K = _complex_of(TETRAHEDRON)
        rng = np.random.default_rng(8)
        U = _random_links(K, rng)
        faces = list(range(K.numSimplices(2)))
        paths = ch.DeclaredPaths.breadthFirst(K, 0)
        gauge = DA.verificationGauge(K, 7)
        assert DA.covarianceResidual(K, U, paths, faces, gauge) < 1e-12

    def test_the_verification_gauge_is_reproducible_and_never_zero(self):
        K = _complex_of(TETRAHEDRON)
        first = DA.verificationGauge(K, 11)
        again = DA.verificationGauge(K, 11)
        assert first == again
        assert all(abs(value) > 0.0 for value in first.values())
        assert DA.verificationGauge(K, 12) != first


# --------------------------------------------------------------------------- #
# The dressed coordinate and its transformation law
# --------------------------------------------------------------------------- #
class TestDressedCoordinate:
    def _setup(self, seed=13):
        K = _complex_of(TETRAHEDRON)
        rng = np.random.default_rng(seed)
        U = _random_links(K, rng)
        paths = ch.DeclaredPaths.breadthFirst(K, 0)
        n1 = K.numSimplices(1)
        Phi = rng.normal(size=(n1, 3)) + 1j * rng.normal(size=(n1, 3))
        return K, rng, U, paths, Phi

    def test_the_coordinate_is_the_determinant_of_the_restricted_frame(self):
        K, _, U, paths, Phi = self._setup()
        for index in range(K.numSimplices(2)):
            value = DA.dressedCoordinate(K, U, paths, index, Phi)
            expected = np.linalg.det(np.asarray(DA.restrictedFrame(K, U, paths, index, Phi)))
            assert abs(value - expected) < MACHINE * max(1.0, abs(expected))

    def test_a_frame_change_multiplies_every_face_by_the_same_determinant(self):
        K, rng, U, paths, Phi = self._setup()
        g = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        before = np.array(
            [DA.dressedCoordinate(K, U, paths, i, Phi) for i in range(K.numSimplices(2))]
        )
        after = np.array(
            [DA.dressedCoordinate(K, U, paths, i, Phi @ g) for i in range(K.numSimplices(2))]
        )
        assert np.allclose(after, before * np.linalg.det(g), rtol=1e-9)

    def test_a_vertex_gauge_multiplies_every_face_by_the_cube_of_the_base_factor(self):
        K, rng, U, paths, Phi = self._setup()
        gauge = _random_gauge(K, rng)
        rho = _rho1(K, gauge)
        before = np.array(
            [DA.dressedCoordinate(K, U, paths, i, Phi) for i in range(K.numSimplices(2))]
        )
        after = np.array(
            [
                DA.dressedCoordinate(K, U.gauge(gauge), paths, i, rho[:, None] * Phi)
                for i in range(K.numSimplices(2))
            ]
        )
        assert np.allclose(after, before / gauge[paths.basePoint()] ** 3, rtol=1e-9)

    def test_the_projective_profile_is_invariant_under_both_gauges(self):
        K, rng, U, paths, Phi = self._setup()
        gauge = _random_gauge(K, rng)
        rho = _rho1(K, gauge)
        g = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        faces = list(range(K.numSimplices(2)))
        plain = DA.profile(K, U, paths, faces, Phi)
        moved = DA.profile(K, U.gauge(gauge), paths, faces, (rho[:, None] * Phi) @ g)
        assert plain.anchored and moved.anchored
        assert DA.projectiveDistance(plain, moved) < 1e-9
        for slot in range(1, len(faces)):
            assert abs(
                DA.faceTransition(plain, slot, 0) - DA.faceTransition(moved, slot, 0)
            ) < 1e-9


# --------------------------------------------------------------------------- #
# Lambda^r for a band of rank below three
# --------------------------------------------------------------------------- #
class TestExteriorPower:
    def test_rank_three_is_the_determinant(self):
        rng = np.random.default_rng(21)
        A = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        values = DA.exteriorPower(A)
        assert len(values) == 1
        assert abs(values[0] - np.linalg.det(A)) < MACHINE

    def test_rank_two_is_the_three_maximal_minors_in_lexicographic_order(self):
        rng = np.random.default_rng(22)
        A = rng.normal(size=(3, 2)) + 1j * rng.normal(size=(3, 2))
        values = DA.exteriorPower(A)
        expected = [
            np.linalg.det(A[[0, 1], :]),
            np.linalg.det(A[[0, 2], :]),
            np.linalg.det(A[[1, 2], :]),
        ]
        assert np.allclose(values, expected, atol=MACHINE)

    def test_rank_one_is_the_three_entries(self):
        rng = np.random.default_rng(23)
        A = rng.normal(size=(3, 1)) + 1j * rng.normal(size=(3, 1))
        assert np.allclose(DA.exteriorPower(A), A.ravel(), atol=MACHINE)

    def test_the_maximal_exterior_power_scales_by_the_frame_determinant(self):
        rng = np.random.default_rng(24)
        A = rng.normal(size=(3, 2)) + 1j * rng.normal(size=(3, 2))
        g = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        assert np.allclose(
            DA.exteriorPower(A @ g),
            np.asarray(DA.exteriorPower(A)) * np.linalg.det(g),
            rtol=1e-9,
        )

    def test_a_rank_two_band_anchors_by_three_coordinates_per_face(self):
        K = _complex_of(TETRAHEDRON)
        rng = np.random.default_rng(25)
        U = _random_links(K, rng)
        paths = ch.DeclaredPaths.breadthFirst(K, 0)
        Phi = _coexact_band(K, (0, 1))
        faces = list(range(K.numSimplices(2)))
        read = DA.profile(K, U, paths, faces, Phi)
        assert read.bandRank == 2
        assert all(len(face) == 3 for face in read.coordinates)
        g = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        rotated = DA.profile(K, U, paths, faces, Phi @ g)
        for before, after in zip(read.coordinates, rotated.coordinates):
            assert np.allclose(
                np.asarray(after), np.asarray(before) * np.linalg.det(g), rtol=1e-9
            )

    def test_a_rank_four_band_is_refused(self):
        K = _complex_of(TETRAHEDRON)
        paths = ch.DeclaredPaths.breadthFirst(K, 0)
        Phi = np.ones((K.numSimplices(1), 4), dtype=complex)
        with pytest.raises(ValueError):
            DA.profile(K, _trivial(K), paths, [0], Phi)


# --------------------------------------------------------------------------- #
# The anchoring theorem
# --------------------------------------------------------------------------- #
class TestAnchoringTheorem:
    def test_the_twisted_coboundary_block_has_determinant_the_curvature_minus_one(self):
        K = _complex_of(TETRAHEDRON)
        rng = np.random.default_rng(31)
        U = _random_links(K, rng)
        for index, triangle in enumerate(_triangles(K)):
            block = np.asarray(DA.twistedCoboundaryBlock(K, U, index))
            holonomy = _face_holonomy(U, triangle)
            assert abs(np.linalg.det(block) - (holonomy - 1.0)) < MACHINE * max(
                1.0, abs(holonomy)
            )
            # The repository's Connection.curvature walks the reverse cycle, so
            # it reports the inverse of the same face holonomy.
            assert abs(U.curvature(*triangle) - 1.0 / holonomy) < MACHINE

    def test_an_exact_band_anchors_exactly_where_the_face_holonomy_is_nontrivial(self):
        K = _complex_of(TETRAHEDRON)
        rng = np.random.default_rng(32)
        # One non-unit link: the two faces containing edge (0, 1) carry
        # curvature and the other two do not.
        U = _connection(K, {(0, 1): 2.0})
        paths = ch.DeclaredPaths.breadthFirst(K, 0)
        potentials = rng.normal(size=(4, 3)) + 1j * rng.normal(size=(4, 3))
        Phi = _exact_band(K, U, potentials)
        assert np.linalg.matrix_rank(Phi) == 3
        for index, triangle in enumerate(_triangles(K)):
            x, y, z = triangle
            holonomy = _face_holonomy(U, triangle)
            value = DA.dressedCoordinate(K, U, paths, index, Phi)
            predicted = (
                paths.transport(U, x) ** 2
                * paths.transport(U, y)
                * U.link(x, z)
                * (holonomy - 1.0)
                * np.linalg.det(potentials[[x, y, z], :])
            )
            assert abs(value - predicted) < 1e-9 * max(1.0, abs(predicted))
            if abs(holonomy - 1.0) < MACHINE:
                assert abs(value) < 1e-9
            else:
                assert abs(value) > 1e-3

    def test_no_exact_band_anchors_anywhere_at_flat_connection(self):
        K = _complex_of(TETRAHEDRON)
        rng = np.random.default_rng(33)
        U = _trivial(K)
        paths = ch.DeclaredPaths.breadthFirst(K, 0)
        potentials = rng.normal(size=(4, 3)) + 1j * rng.normal(size=(4, 3))
        Phi = _exact_band(K, U, potentials)
        faces = list(range(K.numSimplices(2)))
        read = DA.profile(K, U, paths, faces, Phi)
        assert read.anchoringFaces == 0
        assert not read.anchored
        assert "anchor-profile-identically-zero" in list(read.failedCertificates)
        with pytest.raises(RuntimeError):
            DA.projectiveDistance(read, read)

    def test_the_coexact_band_anchors_on_every_face_of_the_tetrahedron(self):
        K = _complex_of(TETRAHEDRON)
        U = _trivial(K)
        paths = ch.DeclaredPaths.breadthFirst(K, 0)
        Phi = _coexact_band(K, (0, 1, 2))
        faces = list(range(K.numSimplices(2)))
        read = DA.profile(K, U, paths, faces, Phi)
        assert read.anchored
        assert read.anchoringFaces == 4
        values = np.array([face[0] for face in read.coordinates])
        # The closed form: Delta_tau = 1 exactly on all four faces, so in
        # particular equal in modulus, which is the theorem's statement.
        assert np.allclose(values, np.ones(4), atol=MACHINE)
        assert np.allclose(np.abs(values), np.abs(values[0]), rtol=1e-12)

    def test_the_certificate_selects_the_coexact_band_uniquely(self):
        K = _complex_of(TETRAHEDRON)
        rng = np.random.default_rng(34)
        U = _trivial(K)
        paths = ch.DeclaredPaths.breadthFirst(K, 0)
        faces = list(range(K.numSimplices(2)))
        potentials = rng.normal(size=(4, 3)) + 1j * rng.normal(size=(4, 3))
        exact = DA.profile(K, U, paths, faces, _exact_band(K, U, potentials))
        coexact = DA.profile(K, U, paths, faces, _coexact_band(K, (0, 1, 2)))
        assert not exact.anchored
        assert coexact.anchored

    def test_a_coexact_band_anchors_on_a_larger_contractible_cluster(self):
        """Two tetrahedra glued on a face: a contractible three-ball, with no
        cycle for the band to be supported on."""
        K = _complex_of(TWO_TETRAHEDRA)
        assert list(K.bettiNumbers()) == [1, 0, 0, 0]
        U = _trivial(K)
        paths = ch.DeclaredPaths.breadthFirst(K, 0)
        triangles = _triangles(K)
        target = triangles.index((0, 1, 2))
        neighbours = [
            i
            for i, t in enumerate(triangles)
            if i != target and len(set(t) & set(triangles[target])) == 2
        ][:2]
        Phi = _coexact_band(K, [target] + neighbours)
        faces = list(range(K.numSimplices(2)))
        read = DA.profile(K, U, paths, faces, Phi)
        assert read.anchored
        assert abs(read.coordinates[target][0]) > 1e-6
        assert read.anchoringFaces >= 3

    def test_an_exact_band_refuses_on_the_larger_cluster_too(self):
        K = _complex_of(TWO_TETRAHEDRA)
        rng = np.random.default_rng(35)
        U = _trivial(K)
        paths = ch.DeclaredPaths.breadthFirst(K, 0)
        potentials = rng.normal(size=(5, 3)) + 1j * rng.normal(size=(5, 3))
        read = DA.profile(
            K, U, paths, list(range(K.numSimplices(2))), _exact_band(K, U, potentials)
        )
        assert not read.anchored
        assert "anchor-profile-identically-zero" in list(read.failedCertificates)


# --------------------------------------------------------------------------- #
# The determinant-line transitions
# --------------------------------------------------------------------------- #
class TestTransitions:
    def _read(self, seed=41):
        K = _complex_of(TETRAHEDRON)
        rng = np.random.default_rng(seed)
        U = _random_links(K, rng)
        paths = ch.DeclaredPaths.breadthFirst(K, 0)
        n1 = K.numSimplices(1)
        Phi = rng.normal(size=(n1, 3)) + 1j * rng.normal(size=(n1, 3))
        faces = list(range(K.numSimplices(2)))
        return K, U, paths, Phi, faces, DA.profile(K, U, paths, faces, Phi)

    def test_a_transition_is_the_ratio_of_the_two_coordinates(self):
        _, _, _, _, _, read = self._read()
        for a in range(len(read.faceIndices)):
            for b in range(len(read.faceIndices)):
                expected = read.coordinates[a][0] / read.coordinates[b][0]
                assert abs(DA.faceTransition(read, a, b) - expected) < 1e-9 * max(
                    1.0, abs(expected)
                )

    def test_the_transitions_satisfy_the_cocycle_identity(self):
        _, _, _, _, _, read = self._read()
        assert read.transitionCocycleResidual < 1e-9
        for a in range(4):
            for b in range(4):
                for c in range(4):
                    left = DA.faceTransition(read, a, b) * DA.faceTransition(read, b, c)
                    right = DA.faceTransition(read, a, c)
                    assert abs(left - right) < 1e-9 * max(1.0, abs(right))

    def test_an_empty_chart_refuses_the_transition(self):
        K = _complex_of(TETRAHEDRON)
        rng = np.random.default_rng(42)
        U = _connection(K, {(0, 1): 3.0})
        paths = ch.DeclaredPaths.breadthFirst(K, 0)
        potentials = rng.normal(size=(4, 3)) + 1j * rng.normal(size=(4, 3))
        read = DA.profile(
            K, U, paths, list(range(K.numSimplices(2))), _exact_band(K, U, potentials)
        )
        triangles = _triangles(K)
        flat = [i for i, t in enumerate(triangles) if abs(_face_holonomy(U, t) - 1.0) < MACHINE]
        curved = [i for i in range(len(triangles)) if i not in flat]
        with pytest.raises(RuntimeError):
            DA.faceTransition(read, curved[0], flat[0])

    def test_a_base_vertex_whose_walks_factor_through_the_old_one_gives_a_scalar(self):
        K = _complex_of(TETRAHEDRON)
        rng = np.random.default_rng(43)
        U = _random_links(K, rng)
        first = ch.DeclaredPaths.breadthFirst(K, 0)
        # Every walk from 1 is the step 1 -> 0 followed by the walk from 0.
        factored = {v: [1] + list(w) for v, w in first.walks().items()}
        second = ch.DeclaredPaths.fromWalks(K, 1, factored)
        n1 = K.numSimplices(1)
        Phi = rng.normal(size=(n1, 3)) + 1j * rng.normal(size=(n1, 3))
        faces = list(range(K.numSimplices(2)))
        readOne = DA.profile(K, U, first, faces, Phi)
        readTwo = DA.profile(K, U, second, faces, Phi)
        scalar = U.link(1, 0) ** 3
        for slot in range(len(faces)):
            ratio = DA.basePointTransition(readTwo, readOne, slot)
            assert abs(ratio - scalar) < 1e-9 * max(1.0, abs(scalar))

    def test_an_independently_declared_base_vertex_reports_face_holonomy(self):
        """Walks that do not factor through the old base vertex make the base
        point transition vary from face to face; the dependence is reported,
        not erased."""
        K = _complex_of(TETRAHEDRON)
        rng = np.random.default_rng(44)
        U = _random_links(K, rng)
        first = ch.DeclaredPaths.breadthFirst(K, 0)
        second = ch.DeclaredPaths.breadthFirst(K, 1)
        n1 = K.numSimplices(1)
        Phi = rng.normal(size=(n1, 3)) + 1j * rng.normal(size=(n1, 3))
        faces = list(range(K.numSimplices(2)))
        readOne = DA.profile(K, U, first, faces, Phi)
        readTwo = DA.profile(K, U, second, faces, Phi)
        ratios = [DA.basePointTransition(readTwo, readOne, slot) for slot in faces]
        assert max(abs(r - ratios[0]) for r in ratios) > 1e-6


# --------------------------------------------------------------------------- #
# The second coordinate and the refusals
# --------------------------------------------------------------------------- #
class TestCertificateAndRefusals:
    def test_the_invariant_coordinates_come_from_the_whitney_face_blocks(self):
        K = _complex_of(TETRAHEDRON)
        U = _trivial(K)
        paths = ch.DeclaredPaths.breadthFirst(K, 0)
        squared = [1.0 + 0j] * K.numSimplices(1)
        cov = ch.CovariantChainHodge(ch.ChainHodge(K, squared, ch.Preset.L2), U)
        Phi = _coexact_band(K, (0, 1, 2))
        Z = np.asarray(cov.applyG(1, Phi))
        Zdual = np.asarray(cov.dual().applyG(1, Phi))
        faces = list(range(K.numSimplices(2)))
        read = DA.withInvariantCoordinates(
            DA.profile(K, U, paths, faces, Phi), cov, Zdual, Z
        )
        expected = np.asarray(ch.FaceAnchor.anchorCoordinates(cov, Zdual, Z))
        assert np.allclose(np.asarray(read.invariantCoordinates), expected, atol=MACHINE)

    def test_a_transport_declared_as_a_bare_number_is_refused(self):
        K = _complex_of(TETRAHEDRON)
        rng = np.random.default_rng(51)
        U = _random_links(K, rng)
        walked = ch.DeclaredPaths.breadthFirst(K, 0)
        raw = ch.DeclaredPaths.declaredTransports(
            0, {int(v[0]): walked.transport(U, int(v[0])) for v in K.kSimplexVertices(0)}
        )
        assert not raw.derivedFromWalks()
        n1 = K.numSimplices(1)
        Phi = rng.normal(size=(n1, 3)) + 1j * rng.normal(size=(n1, 3))
        faces = list(range(K.numSimplices(2)))
        read = DA.profile(K, U, raw, faces, Phi)
        assert not read.anchored
        assert "connection-dressed-covariance" in list(read.failedCertificates)
        assert read.covarianceResidual > 1e-9
        # The same numbers, declared as the walks they came from, are accepted.
        assert DA.profile(K, U, walked, faces, Phi).anchored

    def test_an_empty_atlas_is_refused_by_name(self):
        K = _complex_of(TETRAHEDRON)
        paths = ch.DeclaredPaths.breadthFirst(K, 0)
        Phi = _coexact_band(K, (0, 1, 2))
        read = DA.profile(K, _trivial(K), paths, [], Phi)
        assert not read.anchored
        assert "empty-anchor-atlas" in list(read.failedCertificates)

    def test_an_unreachable_base_vertex_is_refused(self):
        K = _complex_of(TWO_TRIANGLES)
        paths = ch.DeclaredPaths.breadthFirst(K, 0)
        anchorable = list(DA.anchorableFaces(K, paths))
        triangles = _triangles(K)
        assert [triangles[i] for i in anchorable] == [(0, 1, 2)]
        unreachable = triangles.index((3, 4, 5))
        Phi = np.ones((K.numSimplices(1), 3), dtype=complex)
        with pytest.raises(ValueError):
            DA.faceRestriction(K, _trivial(K), paths, unreachable)
        with pytest.raises(ValueError):
            DA.profile(K, _trivial(K), paths, [unreachable], Phi)

    def test_a_frame_of_the_wrong_height_is_refused(self):
        K = _complex_of(TETRAHEDRON)
        paths = ch.DeclaredPaths.breadthFirst(K, 0)
        with pytest.raises(ValueError):
            DA.profile(K, _trivial(K), paths, [0], np.ones((2, 3), dtype=complex))

    def test_a_declared_walk_that_is_not_a_walk_is_refused(self):
        K = _complex_of(TWO_TRIANGLES)
        with pytest.raises(ValueError):
            ch.DeclaredPaths.fromWalks(K, 0, {3: [0, 3]})
        with pytest.raises(ValueError):
            ch.DeclaredPaths.fromWalks(K, 0, {1: [2, 1]})
        with pytest.raises(ValueError):
            ch.DeclaredPaths.fromWalks(K, 0, {1: [0, 2]})

    def test_a_support_that_excludes_the_base_vertex_is_refused(self):
        K = _complex_of(TETRAHEDRON)
        with pytest.raises(ValueError):
            ch.DeclaredPaths.breadthFirst(K, 0, [1, 2, 3])

    def test_a_declared_support_confines_the_walks(self):
        K = _complex_of(TWO_TETRAHEDRA)
        paths = ch.DeclaredPaths.breadthFirst(K, 0, [0, 1, 2, 3])
        assert all(paths.reaches(v) for v in (0, 1, 2, 3))
        assert not paths.reaches(4)
        with pytest.raises(ValueError):
            paths.transport(_trivial(K), 4)
        # A face is anchorable when the rule reaches the base vertices of its
        # three edges, which for a triangle (v0 < v1 < v2) are v0, v1 and v0.
        # Vertex 4 is the largest id of every triangle that contains it, so it
        # is never a base vertex and every face stays anchorable.
        assert list(DA.anchorableFaces(K, paths)) == list(range(K.numSimplices(2)))
