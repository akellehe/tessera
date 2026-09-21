# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The random-phase approximation on the modes of a pencil: the independent-particle
polarizability, the screened interaction, the correlation energy, the one-shot
quasiparticle correction, and the second-order response of the occupied energy
to the connection (the gauge quanta).

Everything is finite linear algebra on the eigenpairs of a pencil. With occupied
modes i, j and empty modes a, b of a closed-shell mean field (two sheets), the
direct random-phase equations in the basis of particle-hole pairs are

    A_{ia,jb} = (e_a - e_i) delta_ij delta_ab + 2 (ia|jb),     B_{ia,jb} = 2 (ia|jb),

with (ia|jb) the Coulomb integral of the two pair densities and the factor 2
from the two sheets. For real modes (A - B) is diagonal and the excitation
energies are the square roots of the eigenvalues of
(A - B)^{1/2} (A + B) (A - B)^{1/2}. The screened interaction is then known
analytically at every frequency, and with it the correlation self-energy

    Sigma^c_nn(w) = sum_s [ sum_i |w^s_ni|^2 / (w - e_i + W_s)
                          + sum_a |w^s_na|^2 / (w - e_a - W_s) ],
    w^s_nm = sqrt(2) sum_jb (nm|jb) (X + Y)^s_jb ,

with no frequency grid and no plasmon-pole model.

Reference: Hybertsen & Louie, "Electron correlation in semiconductors and
insulators: band gaps and quasiparticle energies", Physical Review B 34, 5390
(1986), for the quasiparticle equation.
"""
import numpy as np
import scipy.linalg


def real_modes(A, M, vectors):
    """Real M-orthonormal eigenvectors spanning the same space as `vectors`,
    for a real symmetric pencil (the zone centre with a real potential). The
    solver returns complex vectors with arbitrary phases and arbitrary mixing
    inside a degenerate level; the real and imaginary parts span the same
    space, and a Rayleigh-Ritz step on them with the real matrices returns
    real eigenvectors. Returns (energies, modes)."""
    A = A.real if hasattr(A, "real") else A
    M = M.real if hasattr(M, "real") else M
    stacked = np.hstack([vectors.real, vectors.imag])
    gram = stacked.T @ (M @ stacked)
    values, basis = np.linalg.eigh(0.5 * (gram + gram.T))
    keep = values > 1e-10 * values.max()
    span = stacked @ (basis[:, keep] / np.sqrt(values[keep]))
    if span.shape[1] != vectors.shape[1]:
        raise ValueError("the modes do not span a real space of their own dimension; is the pencil real?")
    projected = span.T @ (A @ span)
    energies, rotation = np.linalg.eigh(0.5 * (projected + projected.T))
    return energies, span @ rotation


class RandomPhase:
    """The direct random-phase approximation of a closed-shell state.

    `energies` are the mean-field levels of real modes, `W` their Coulomb
    integrals `W[m, n, p, q] = (mn|pq)` (`ModeInteraction.W`), and the lowest
    `occupied` modes are doubly filled.
    """

    def __init__(self, energies, W, occupied):
        self.energies = np.asarray(energies, dtype=float)
        W = np.asarray(W)
        if np.abs(W.imag).max() > 1e-9 * max(1.0, np.abs(W.real).max()):
            raise ValueError("RandomPhase needs real modes (see real_modes)")
        self.W = W.real
        self.occupied = int(occupied)
        count = len(self.energies)
        self.pairs = [(i, a) for i in range(self.occupied) for a in range(self.occupied, count)]
        self.gaps = np.array([self.energies[a] - self.energies[i] for i, a in self.pairs])
        rows = np.array([i for i, _ in self.pairs])
        cols = np.array([a for _, a in self.pairs])
        # (ia|jb) over particle-hole pairs.
        self.coupling = self.W[rows[:, None], cols[:, None], rows[None, :], cols[None, :]]
        root = np.sqrt(self.gaps)
        casida = np.diag(self.gaps ** 2) + 4.0 * root[:, None] * self.coupling * root[None, :]
        squared, Z = np.linalg.eigh(0.5 * (casida + casida.T))
        if squared.min() <= 0.0:
            raise ValueError("the mean field is unstable in the random-phase approximation")
        self.excitations = np.sqrt(squared)
        self.x_plus_y = (root[:, None] * Z) / np.sqrt(self.excitations)[None, :]
        # w^s_mn = sqrt(2) sum_jb (mn|jb) (X + Y)^s_jb
        ph = self.W[:, :, rows, cols]
        self.transition = np.sqrt(2.0) * np.einsum("mnp,ps->mns", ph, self.x_plus_y)

    # -- energies

    def correlation_energy(self):
        """E_c = 1/2 (sum_s W_s - tr A), the plasmon formula."""
        trace_a = self.gaps.sum() + 2.0 * np.trace(self.coupling)
        return 0.5 * (self.excitations.sum() - trace_a)

    def correlation_energy_by_frequency(self, points=200):
        """The same energy from the adiabatic-connection integral over imaginary
        frequency, (1/2 pi) int_0^inf du tr[ln(1 - v chi0(iu)) + v chi0(iu)],
        in the basis of particle-hole pairs, where v chi0(iu) has the matrix
        -4 (ia|jb) gap_jb / (gap_jb^2 + u^2). An independent route to
        `correlation_energy`."""
        nodes, weights = np.polynomial.legendre.leggauss(points)
        scale = float(np.median(self.gaps))
        u = scale * (1.0 + nodes) / (1.0 - nodes)
        du = weights * 2.0 * scale / (1.0 - nodes) ** 2
        total = 0.0
        identity = np.eye(len(self.gaps))
        for frequency, weight in zip(u, du):
            kernel = -4.0 * self.coupling * (self.gaps / (self.gaps ** 2 + frequency ** 2))[None, :]
            sign, logdet = np.linalg.slogdet(identity - kernel)
            total += weight * (logdet + np.trace(kernel))
        return total / (2.0 * np.pi)

    # -- self-energy

    def exchange(self, n):
        """Sigma^x_nn = -sum_i (ni|in)."""
        return -sum(self.W[n, i, i, n] for i in range(self.occupied))

    def correlation(self, n, frequency):
        """Sigma^c_nn at a real frequency (and its derivative)."""
        value = derivative = 0.0
        for m, level in enumerate(self.energies):
            weights = self.transition[n, m, :] ** 2
            poles = level - self.excitations if m < self.occupied else level + self.excitations
            value += np.sum(weights / (frequency - poles))
            derivative -= np.sum(weights / (frequency - poles) ** 2)
        return value, derivative

    def second_order_correlation(self, n, frequency):
        """The direct second-order self-energy, the weak-coupling limit of
        `correlation`: 2 sum_ia [sum_j (nj|ia)^2 / (w - e_j + gap_ia)
        + sum_b (nb|ia)^2 / (w - e_b - gap_ia)]."""
        rows = np.array([i for i, _ in self.pairs])
        cols = np.array([a for _, a in self.pairs])
        value = 0.0
        for m, level in enumerate(self.energies):
            weights = 2.0 * self.W[n, m, rows, cols] ** 2
            poles = level - self.gaps if m < self.occupied else level + self.gaps
            value += np.sum(weights / (frequency - poles))
        return value

    def quasiparticle(self, n, iterations=50, tolerance=1e-10):
        """Solve E = e_n + Sigma^c_nn(E) on a Hartree-Fock starting point (where
        the exchange self-energy already sits in e_n) by Newton's method from
        e_n. Returns (E, renormalization factor Z at the solution)."""
        level = self.energies[n]
        energy = level
        for _ in range(iterations):
            value, derivative = self.correlation(n, energy)
            step = (level + value - energy) / (1.0 - derivative)
            energy += step
            if abs(step) < tolerance:
                break
        _, derivative = self.correlation(n, energy)
        return energy, 1.0 / (1.0 - derivative)


def static_polarizability(energies, T, occupied, probe):
    """chi0(q, q; w = 0) of a closed-shell state for the probe with nodal values
    `probe` (a plane wave e^{i q . x} on the vertices):
    -4 sum_ia |<a| probe |i>|^2 / (e_a - e_i), divided by nothing: the caller
    divides by the volume. `T` are the pair densities of the modes."""
    matrix = np.einsum("v,vmn->mn", np.conj(probe), T)       # <m| e^{-iq.x} |n>
    total = 0.0
    for i in range(occupied):
        for a in range(occupied, len(energies)):
            total -= 4.0 * abs(matrix[a, i]) ** 2 / (energies[a] - energies[i])
    return total


class GaugeResponse:
    """The second-order response of the occupied energy of the covariant
    operator h_k(s, U) to the link phases: the stiffness the fermions induce on
    the connection,

        D_ab = sum_m <m| d^2 h / dphi_a dphi_b |m>,
        Pi_ab(w) = sum_{m occ, n emp} 2 gap_nm (O_a)_mn (O_b)_nm / (gap_nm^2 - w^2),

    with O_a = dh/dphi_a. D - Pi(0) is the Hessian of the occupied energy and
    vanishes on every pure-gauge direction (the Ward identity), which the
    paramagnetic term alone would violate.

    The operator is self-adjoint in the chain metric G = (M^U)^{-1}, which moves
    with the connection, so dh/dphi_a is not self-adjoint in the metric at the
    base point and (O_a)_mn (O_b)_nm is not symmetric in a and b. Brackets are
    taken between the left and right eigenvectors at the base point,
    <m| = z_m^dagger G, and the paramagnetic term is symmetrized, which is what
    second-order perturbation theory of a mixed second derivative gives.
    """

    def __init__(self, cov, k, occupied):
        pencil = cov.pencil(k)
        self.metric = np.linalg.inv(pencil.B)                 # G_k = M_k^{-1}
        h = cov.covariantOperator(k)
        symmetric = self.metric @ h
        values, vectors = scipy.linalg.eigh(0.5 * (symmetric + symmetric.conj().T),
                                            0.5 * (self.metric + self.metric.conj().T))
        self.levels, self.modes = values, vectors
        self.occupied = list(occupied)
        self.empty = [n for n in range(len(values)) if n not in self.occupied]
        edges = cov.base().complex().numSimplices(1)
        self.edges = edges
        bra = self.modes.conj().T @ self.metric
        self.currents = np.array([bra @ cov.covariantOperatorPhaseDerivative(k, a) @ self.modes
                                  for a in range(edges)])
        self.diamagnetic = np.zeros((edges, edges), dtype=complex)
        for a in range(edges):
            for b in range(a, edges):
                second = bra @ cov.covariantOperatorPhaseHessian(k, a, b) @ self.modes
                value = sum(second[m, m] for m in self.occupied)
                self.diamagnetic[a, b] = self.diamagnetic[b, a] = value

    def paramagnetic(self, frequency=0.0):
        out = np.zeros((self.edges, self.edges), dtype=complex)
        for m in self.occupied:
            for n in self.empty:
                gap = self.levels[n] - self.levels[m]
                forward = np.outer(self.currents[:, m, n], self.currents[:, n, m])
                out += gap / (gap ** 2 - frequency ** 2) * (forward + forward.T)
        return out

    def induced_stiffness(self, frequency=0.0):
        """D - Pi(w)."""
        return self.diamagnetic - self.paramagnetic(frequency)
