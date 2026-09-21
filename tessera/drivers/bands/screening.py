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
    # The conjugate of an eigenvector of a real pencil is an eigenvector of the
    # same level, so the real span is invariant. It is larger than the mode
    # count only when the last level was cut inside a degenerate multiplet, and
    # then holds further copies of that level; the lowest ones are kept.
    projected = span.T @ (A @ span)
    energies, rotation = np.linalg.eigh(0.5 * (projected + projected.T))
    count = vectors.shape[1]
    return energies[:count], (span @ rotation)[:, :count]


class RandomPhase:
    """The direct random-phase approximation of a closed-shell state.

    `energies` are the mean-field levels of real modes, `W` their Coulomb
    integrals `W[m, n, p, q] = (mn|pq)` (`ModeInteraction.W`), and the lowest
    `occupied` modes are doubly filled.
    """

    def __init__(self, energies, W, occupied):
        W = np.asarray(W)
        if np.abs(W.imag).max() > 1e-9 * max(1.0, np.abs(W.real).max()):
            raise ValueError("RandomPhase needs real modes (see real_modes)")
        self.W = W.real
        count = len(energies)
        rows = np.array([i for i in range(occupied) for _ in range(occupied, count)])
        cols = np.array([a for _ in range(occupied) for a in range(occupied, count)])
        coupling = self.W[rows[:, None], cols[:, None], rows[None, :], cols[None, :]]
        self._solve(energies, occupied, coupling)
        # w^s_mn = sqrt(2) sum_jb (mn|jb) (X + Y)^s_jb, for every pair of modes.
        ph = self.W[:, :, rows, cols]
        self.transition = np.sqrt(2.0) * np.einsum("mnp,ps->mns", ph, self.x_plus_y)

    @classmethod
    def from_pieces(cls, energies, occupied, coupling, integrals):
        """Build from the Coulomb integrals that are actually needed, for a mode
        count at which the full four-index tensor cannot be held: `coupling` is
        (ia|jb) over the particle-hole pairs (filled index slow), and `integrals`
        maps a mode n to the array (nm|jb) over every mode m (rows) and every
        pair (columns). The self-energy is then available for those n."""
        self = cls.__new__(cls)
        self.W = None
        self._solve(energies, occupied, np.asarray(coupling, dtype=float))
        self.transition = {n: np.sqrt(2.0) * np.asarray(block, dtype=float) @ self.x_plus_y
                           for n, block in integrals.items()}
        return self

    def set_head(self, constant, momenta, shifts=None):
        """Restore the zero-momentum term of the screened interaction, which a
        Coulomb kernel of zero mean leaves out.

        Each entry of `momenta` describes the particle-hole pairs of one
        momentum transfer q: their level differences `gaps`, their Hermitian
        coupling `coupling` without the G = 0 entry of the kernel, their
        `charges` (the G = 0 components of the pair densities, of order q), and
        `entry`, the energy u of a normalized charge in the G = 0 entry (of
        order 1 / q^2). `MeshCrystal.vanishing_momentum_pairs` supplies the limit
        q -> 0 in closed form (charges per unit momentum, and u q^2), along each
        Cartesian axis; `MeshCrystal.momentum_pairs` a small finite momentum.
        `shifts`, one number per mode, moves the level differences when the
        levels have been updated.

        With the modes (W_s, X_s) of the random-phase problem of those pairs
        and the residues a_s = 2 u |sum_ia charge_ia (X + Y)^s_ia|^2, the
        macroscopic dielectric constant along q is 1 + S, S = sum_s 2 a_s / W_s.
        The inverse dielectric function has other poles: those of the problem
        with the G = 0 entry u conj(charge_ia) charge_jb added to the coupling,
        a rank-one change. With its modes (W~_t, a~_t),

            1 / eps(w) - 1 = sum_t 2 W~_t a~_t / (w^2 - W~_t^2) ,

        and the missing term of the self-energy of a state is the intraband
        one, `constant` * sum_t a~_t / (w - e_n -+ W~_t), averaged over the
        momenta given; `constant` is the integral of the Coulomb kernel over
        the cell of momentum space that the sampling leaves out
        (`GridCoulombKernel.zero_momentum_constant`). On the energy shell it is
        +-constant (1 - 1/eps) / 2. Returns the macroscopic dielectric
        constant."""
        poles, weights, inverse, static, independent = [], [], [], [], []
        for momentum in momenta:
            gaps = np.asarray(momentum["gaps"], dtype=float)
            if shifts is not None:
                shifts = np.asarray(shifts, dtype=float)
                gaps = gaps + np.array([shifts[a] - shifts[i] for i, a in momentum["pairs"]])
            charges, entry = np.asarray(momentum["charges"]), float(momentum["entry"])
            root = np.sqrt(gaps)
            body = np.diag(gaps ** 2) + 4.0 * root[:, None] * np.asarray(momentum["coupling"]) * root[None, :]
            squared, Z = np.linalg.eigh(0.5 * (body + body.conj().T))
            omega = np.sqrt(squared)
            amplitudes = ((root[:, None] * Z) / np.sqrt(omega)[None, :]).T @ charges
            static.append(np.sum(2.0 * entry * 2.0 * np.abs(amplitudes) ** 2 / omega))
            loaded = root * charges.conj()
            long_range = body + 4.0 * entry * np.outer(loaded, loaded.conj())
            squared, Z = np.linalg.eigh(0.5 * (long_range + long_range.conj().T))
            omega = np.sqrt(squared)
            amplitudes = ((root[:, None] * Z) / np.sqrt(omega)[None, :]).T @ charges
            residues = entry * 2.0 * np.abs(amplitudes) ** 2
            poles.append(omega)
            weights.append(constant * residues / len(momenta))
            inverse.append(1.0 - np.sum(2.0 * residues / omega))
            independent.append(entry * 4.0 * np.sum(np.abs(charges) ** 2 / gaps))
        self.head_poles, self.head = np.concatenate(poles), np.concatenate(weights)
        self.dielectric_constant = 1.0 + float(np.mean(static))
        # The two routes to the static inverse must agree: 1 / (1 + S) = 1 - sum_t 2 a~_t / W~_t.
        self.head_defect = float(max(abs(value - 1.0 / (1.0 + s)) for value, s in zip(inverse, static)))
        self.independent_particle_dielectric_constant = 1.0 + float(np.mean(independent))
        return self.dielectric_constant

    def _solve(self, energies, occupied, coupling):
        self.head = None
        self.propagator = None        # the levels of G when they differ from those W was built from
        self.energies = np.asarray(energies, dtype=float)
        self.occupied = int(occupied)
        count = len(self.energies)
        self.pairs = [(i, a) for i in range(self.occupied) for a in range(self.occupied, count)]
        self.gaps = np.array([self.energies[a] - self.energies[i] for i, a in self.pairs])
        self.coupling = coupling
        root = np.sqrt(self.gaps)
        casida = np.diag(self.gaps ** 2) + 4.0 * root[:, None] * self.coupling * root[None, :]
        squared, Z = np.linalg.eigh(0.5 * (casida + casida.T))
        if squared.min() <= 0.0:
            raise ValueError("the mean field is unstable in the random-phase approximation")
        self.excitations = np.sqrt(squared)
        self.x_plus_y = (root[:, None] * Z) / np.sqrt(self.excitations)[None, :]

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
        if self.W is None:
            raise ValueError("the exchange self-energy needs the full tensor of Coulomb integrals")
        return -sum(self.W[n, i, i, n] for i in range(self.occupied))

    def correlation(self, n, frequency):
        """Sigma^c_nn at a real frequency (and its derivative)."""
        value = derivative = 0.0
        levels = self.energies if self.propagator is None else self.propagator
        for m, level in enumerate(levels):
            weights = self.transition[n][m, :] ** 2
            poles = level - self.excitations if m < self.occupied else level + self.excitations
            value += np.sum(weights / (frequency - poles))
            derivative -= np.sum(weights / (frequency - poles) ** 2)
            if m == n and self.head is not None:
                poles = level - self.head_poles if m < self.occupied else level + self.head_poles
                value += np.sum(self.head / (frequency - poles))
                derivative -= np.sum(self.head / (frequency - poles) ** 2)
        return value, derivative

    def second_order_correlation(self, n, frequency):
        """The direct second-order self-energy, the weak-coupling limit of
        `correlation`: 2 sum_ia [sum_j (nj|ia)^2 / (w - e_j + gap_ia)
        + sum_b (nb|ia)^2 / (w - e_b - gap_ia)]."""
        if self.W is None:
            raise ValueError("the second-order self-energy needs the full tensor of Coulomb integrals")
        rows = np.array([i for i, _ in self.pairs])
        cols = np.array([a for _, a in self.pairs])
        value = 0.0
        for m, level in enumerate(self.energies):
            weights = 2.0 * self.W[n, m, rows, cols] ** 2
            poles = level - self.gaps if m < self.occupied else level + self.gaps
            value += np.sum(weights / (frequency - poles))
        return value

    def quasiparticle(self, n, iterations=50, tolerance=1e-10, reference=None, start=None):
        """Solve E = e_n + Sigma^c_nn(E) on a Hartree-Fock starting point (where
        the exchange self-energy already sits in e_n) by Newton's method.
        `reference` supplies the Hartree-Fock levels when the levels this object
        was built from are already quasiparticle levels, and `start` the first
        iterate. Returns (E, renormalization factor Z at the solution)."""
        level = self.energies[n] if reference is None else reference[n]
        energy = level if start is None else start
        for _ in range(iterations):
            value, derivative = self.correlation(n, energy)
            step = (level + value - energy) / (1.0 - derivative)
            energy += step
            if abs(step) < tolerance:
                break
        _, derivative = self.correlation(n, energy)
        return energy, 1.0 / (1.0 - derivative)


def self_consistent_quasiparticles(mean_field, occupied, coupling, integrals, head=None, update_screening=True,
                                    tolerance=1e-6, max_iterations=60, damping=0.7):
    """Eigenvalue self-consistency on a Hartree-Fock starting point: the
    orbitals and their Coulomb integrals stay fixed, and the quasiparticle
    levels are fed back into the propagator (`update_screening=False`, the
    scheme usually written GW0: the screened interaction keeps the Hartree-Fock
    levels) or into the propagator and the screened interaction both
    (`update_screening=True`). A Hartree-Fock gap is several times too large, so
    the interaction screened with it is too weak and the one-shot gap too large;
    feeding the levels back into the screening is what closes it.

    `integrals` must cover every mode. `head` is the pair (constant, momenta)
    of `RandomPhase.set_head`; the level differences at the small momenta move
    with the zone-centre levels of the same modes. Returns (levels, history of the largest change, the
    last RandomPhase)."""
    mean_field = np.asarray(mean_field, dtype=float)
    energies = mean_field.copy()
    history, rpa = [], None
    for iteration in range(max_iterations):
        if rpa is None or update_screening:
            rpa = RandomPhase.from_pieces(energies, occupied, coupling, integrals)
            if head is not None:
                rpa.set_head(head[0], head[1], energies - mean_field)
        rpa.propagator = energies
        produced = np.array([rpa.quasiparticle(n, reference=mean_field, start=energies[n])[0]
                             for n in range(len(energies))])
        change = float(np.abs(produced - energies).max())
        history.append(change)
        energies = (1.0 - damping) * energies + damping * produced
        if change < tolerance:
            break
    return energies, history, rpa


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


def one_shot_gap(cell, potential, electrons, modes, strength, tolerance=1e-10):
    """Hartree-Fock and the one-shot quasiparticle gap of a closed-shell cell
    sampled at the zone centre, end to end on the mesh.

    The lowest `modes` levels of the pencil with `potential` are made real,
    their pair densities and Coulomb integrals are formed with the
    finite-element kernel of the cell (`strength` is 4 pi e^2 in the units of
    the run), the Roothaan loop is run for `electrons` electrons on two sheets,
    and the random-phase self-energy is evaluated on the Hartree-Fock levels.
    The mode count is the basis of the screened interaction, the finite-element
    analogue of a plane-wave cutoff for the dielectric matrix; the result
    converges with it and the caller is expected to vary it.

    Returns a dict with the one-particle, Hartree-Fock and quasiparticle gaps,
    the renormalization factors, and the residuals every step was held to.
    """
    from tessera.drivers.bands import coulomb
    if electrons % 2:
        raise ValueError("one_shot_gap is closed shell: an even number of electrons")
    occupied = electrons // 2
    # A degenerate level cut in two does not span a real space, so the mode set
    # is closed at the first gap at or above the requested count.
    margin = 12
    read = cell.solve((0.0, 0.0, 0.0), modes + margin, potential, tolerance=tolerance)
    scale = max(1.0, np.abs(read.energies).max())
    gaps = np.diff(read.energies) > 1e-6 * scale
    closed = [c for c in range(modes, modes + margin) if gaps[c - 1]]
    if not closed:
        raise ValueError("no gap within the margin above the requested mode count")
    modes = closed[0]
    A, M = cell.pencil((0.0, 0.0, 0.0), potential)
    levels, real = real_modes(A, M, read.vectors[:, :modes])
    kernel = coulomb.CoulombKernel.of_cell(cell, strength)
    T = coulomb.pair_densities(cell.complex, cell.squared_lengths, real)
    both = coulomb.ModeInteraction(kernel, levels, T, sheets=2)
    gamma, energy, iterations, residual = both.roothaan(electrons)
    fock_levels, rotation = np.linalg.eigh(both.fock(gamma)[:modes, :modes].real)
    orbitals = coulomb.ModeInteraction(kernel, fock_levels,
                                       coulomb.pair_densities(cell.complex, cell.squared_lengths, real @ rotation))
    rpa = RandomPhase(fock_levels, orbitals.W, occupied)
    (top, weight_top), (bottom, weight_bottom) = rpa.quasiparticle(occupied - 1), rpa.quasiparticle(occupied)
    return {"one_particle_gap": levels[occupied] - levels[occupied - 1],
            "hartree_fock_gap": fock_levels[occupied] - fock_levels[occupied - 1],
            "quasiparticle_gap": bottom - top,
            "renormalization": (weight_top, weight_bottom),
            "hartree_fock_energy": energy, "correlation_energy": rpa.correlation_energy(),
            "modes": modes, "roothaan_residual": residual, "roothaan_iterations": iterations,
            "solver_residual": read.residual, "certified": read.certified() and residual < 1e-8}
