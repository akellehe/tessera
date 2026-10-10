"""Cross-validation of the lazy Fock engine (#771) against the quasi-free
covariance layer (#780) on shared quasi-free fixtures.

Both tickets deferred this direct comparison to post-merge integration:
each side was validated against dense #766 references independently; this
suite closes the loop by comparing the two implementations to each other
on identical Slater states, with no dense construction in the loop.

Conventions used here:
  - ``LazyFockEngine.covariance_matrix`` returns Gamma_ef = <a_f^+ a_e>,
    the same convention as ``CovarianceState.gamma()``.
  - ``CovarianceState.wick_normal_ordered(creators, annihilators)`` uses
    paired slot order, so equal distinct lists give joint occupations.
  - ``LazyFockEngine.inner_product(a, b)`` is <a|b>, antilinear in ``a``.
"""

import numpy as np
import pytest

from tessera.quantum import CovarianceState, LazyFockEngine

M = 6  # shared mode-universe size
RANK = 3  # occupied Slater rank
TOL = 1e-12


def _seeded_frame(m: int, r: int, seed: int = 20260822) -> np.ndarray:
    """Deterministic orthonormal complex frame: r occupied columns."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(m, r)) + 1j * rng.normal(size=(m, r))
    q, _ = np.linalg.qr(a)
    return np.ascontiguousarray(q[:, :r])


@pytest.fixture(scope="module")
def frame():
    return _seeded_frame(M, RANK)


@pytest.fixture(scope="module")
def projector(frame):
    return frame @ frame.conj().T


@pytest.fixture(scope="module")
def cov(projector):
    return CovarianceState.from_band_projector(projector)


@pytest.fixture(scope="module")
def engine():
    return LazyFockEngine(M)


@pytest.fixture(scope="module")
def slater(engine, projector):
    ref = engine.slater_from_projector(list(range(M)), projector, TOL)
    assert ref.certificate.holds()
    return ref.state


def _lazy_expectation(engine, state, one_particle_factors):
    """<d_gamma(A_1) ... d_gamma(A_n)> on a normalized lazy state."""
    modes = list(range(M))
    ket = state
    for a in reversed(one_particle_factors):
        ket = engine.apply_d_gamma(ket, modes, np.ascontiguousarray(a))
    value = engine.inner_product(state, ket).value
    norm = engine.norm_squared(state).value
    return value / norm


class TestCovarianceAgreement:
    def test_gamma_matrices_agree(self, engine, slater, cov):
        lazy_gamma = engine.covariance_matrix(slater).matrix
        assert np.max(np.abs(lazy_gamma - cov.gamma())) <= TOL

    def test_occupations_agree(self, engine, slater, cov):
        lazy_gamma = engine.covariance_matrix(slater).matrix
        for mode in range(M):
            wick = cov.wick_occupation(mode)
            assert wick.certificate.holds()
            assert abs(lazy_gamma[mode, mode].real - wick.value.real) <= TOL
            assert abs(wick.value.imag) <= TOL

    def test_particle_number_agrees(self, engine, slater, cov):
        lazy_n = np.trace(engine.covariance_matrix(slater).matrix)
        wick_n = cov.wick_total_number().value
        assert abs(lazy_n - wick_n) <= TOL
        assert abs(wick_n - RANK) <= TOL

    def test_parity_is_analytic(self, cov):
        # (-1)^N on a rank-RANK Slater state; det(I - 2 Gamma) on the
        # covariance side.  The lazy side fixes N = RANK by construction,
        # so the analytic value anchors both.
        parity = cov.wick_parity().value
        assert abs(parity - (-1.0) ** RANK) <= TOL


class TestMomentAgreement:
    def _hermitian(self, seed):
        rng = np.random.default_rng(seed)
        a = rng.normal(size=(M, M)) + 1j * rng.normal(size=(M, M))
        return np.ascontiguousarray((a + a.conj().T) / 2.0)

    def test_quadratic_moment_agrees(self, engine, slater, cov):
        h = self._hermitian(7)
        wick = cov.wick_bilinear_moment([h])
        lazy = _lazy_expectation(engine, slater, [h])
        assert wick.certificate.holds()
        assert abs(wick.value - lazy) <= 1e-11

    def test_quartic_ordered_moment_agrees(self, engine, slater, cov):
        a = self._hermitian(11)
        b = self._hermitian(13)
        wick = cov.wick_bilinear_moment([a, b])
        lazy = _lazy_expectation(engine, slater, [a, b])
        assert wick.certificate.holds()
        assert abs(wick.value - lazy) <= 1e-10

    def test_joint_occupation_agrees(self, engine, slater, cov):
        # <n_0 n_1> both ways: Wick determinant vs nested bit-level dGamma.
        wick = cov.wick_normal_ordered([0, 1], [0, 1])
        d0 = np.zeros((M, M), dtype=complex)
        d0[0, 0] = 1.0
        d1 = np.zeros((M, M), dtype=complex)
        d1[1, 1] = 1.0
        lazy = _lazy_expectation(engine, slater, [d0, d1])
        assert abs(wick.value - lazy) <= TOL


class TestSpinAgreement:
    """Spin-1/2 fixture: 3 orbitals x 2 spin components (M = 6 modes,
    mode 2*o + s), J_alpha = sum_orbitals sigma_alpha / 2."""

    def _spin_matrices(self):
        sx = np.array([[0, 1], [1, 0]], dtype=complex) / 2.0
        sy = np.array([[0, -1j], [1j, 0]], dtype=complex) / 2.0
        sz = np.array([[1, 0], [0, -1]], dtype=complex) / 2.0
        eye3 = np.eye(3, dtype=complex)
        return tuple(
            np.ascontiguousarray(np.kron(eye3, s)) for s in (sx, sy, sz)
        )

    def test_single_particle_spin_half_both_sides(self, engine):
        jx, jy, jz = self._spin_matrices()
        v = np.zeros((M, 1), dtype=complex)
        v[0, 0] = 1.0  # orbital 0, spin up
        cov1 = CovarianceState.from_slater_frame(v)
        expect = cov1.wick_spin_squared_expectation(jx, jy, jz)
        var = cov1.wick_spin_squared_variance(jx, jy, jz)
        assert abs(expect.value - 0.75) <= TOL
        assert abs(var.value) <= TOL
        ref = engine.slater_from_projector(
            list(range(M)), v @ v.conj().T, TOL
        )
        lazy_j2 = sum(
            _lazy_expectation(engine, ref.state, [j, j])
            for j in (jx, jy, jz)
        )
        assert abs(lazy_j2 - 0.75) <= TOL

    def test_generic_slater_j2_agrees_with_lazy(self, engine, frame, cov):
        jx, jy, jz = self._spin_matrices()
        expect = cov.wick_spin_squared_expectation(jx, jy, jz)
        ref = engine.slater_from_projector(
            list(range(M)), frame @ frame.conj().T, TOL
        )
        lazy_j2 = sum(
            _lazy_expectation(engine, ref.state, [j, j])
            for j in (jx, jy, jz)
        )
        assert abs(expect.value - lazy_j2) <= 1e-10
