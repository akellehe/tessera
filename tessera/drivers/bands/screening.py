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
import os

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


def long_range_problem(gaps, coupling, charges, entry):
    """The random-phase problem of particle-hole pairs with the G = 0 entry of
    the kernel added to their coupling, entry * conj(charge_ia) charge_jb, a
    rank-one change: (excitations W~_t, modes (X + Y)^t over the pairs, residues
    a~_t = 2 entry |sum_ia charge_ia (X + Y)^t_ia|^2). See `RandomPhase.set_head`."""
    gaps, charges = np.asarray(gaps, dtype=float), np.asarray(charges)
    root = np.sqrt(gaps)
    body = np.diag(gaps ** 2) + 4.0 * root[:, None] * np.asarray(coupling) * root[None, :]
    loaded = root * charges.conj()
    long_range = body + 4.0 * entry * np.outer(loaded, loaded.conj())
    squared, Z = np.linalg.eigh(0.5 * (long_range + long_range.conj().T))
    if squared.min() <= 0.0:
        raise ValueError("the mean field is unstable in the random-phase approximation")
    omega = np.sqrt(squared)
    modes = (root[:, None] * Z) / np.sqrt(omega)[None, :]
    return omega, modes, entry * 2.0 * np.abs(modes.T @ charges) ** 2


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
        self._blocks = {n: ph[n] for n in range(count)}
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
        self._blocks = integrals
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
        momenta given. The same modes (W~_t, with the G = 0 entry in the
        coupling) are the modes of the whole screened interaction at that
        momentum, so the transitions of the other terms are rebuilt on them and
        averaged over the momenta as well: the G = 0 entry screens the rest of
        the interaction through the mixed entries of the dielectric matrix, and
        at a single sampled momentum that carries full weight. The terms mixed
        between G = 0 and the rest are odd in the momentum and average to zero.
        `constant` is `constant` is the integral of the Coulomb kernel over
        the cell of momentum space that the sampling leaves out
        (`GridCoulombKernel.zero_momentum_constant`). On the energy shell it is
        +-constant (1 - 1/eps) / 2. Returns the macroscopic dielectric
        constant."""
        poles, weights, inverse, static, independent, long_range_modes = [], [], [], [], [], []
        solved = None                                  # the problem without the G = 0 entry, shared by the directions of one limit
        for momentum in momenta:
            gaps = np.asarray(momentum["gaps"], dtype=float)
            if shifts is not None:
                shifts = np.asarray(shifts, dtype=float)
                gaps = gaps + np.array([shifts[a] - shifts[i] for i, a in momentum["pairs"]])
            charges, entry = np.asarray(momentum["charges"]), float(momentum["entry"])
            coupling = np.asarray(momentum["coupling"])
            # A common phase of the charges drops out; real modes have charges i r, and the problem is then real.
            charges = charges * np.exp(-1j * np.angle(charges[np.argmax(np.abs(charges))]))
            if np.isrealobj(coupling) and np.abs(charges.imag).max() < 1e-12 * np.abs(charges).max():
                charges = charges.real
            root = np.sqrt(gaps)
            body = np.diag(gaps ** 2) + 4.0 * root[:, None] * coupling * root[None, :]
            if solved is None or solved[0] is not momentum["coupling"] or not np.array_equal(solved[1], gaps):
                solved = (momentum["coupling"], gaps) + tuple(np.linalg.eigh(0.5 * (body + body.conj().T)))
            squared, Z = solved[2], solved[3]
            omega = np.sqrt(squared)
            amplitudes = ((root[:, None] * Z) / np.sqrt(omega)[None, :]).T @ charges
            static.append(np.sum(2.0 * entry * 2.0 * np.abs(amplitudes) ** 2 / omega))
            omega, modes, residues = long_range_problem(gaps, coupling, charges, entry)
            poles.append(omega)
            if momentum.get("limit", False):           # the pairs of a finite momentum are not those of the integrals
                long_range_modes.append((omega, modes))
            weights.append(constant * residues / len(momenta))
            inverse.append(1.0 - np.sum(2.0 * residues / omega))
            independent.append(entry * 4.0 * np.sum(np.abs(charges) ** 2 / gaps))
        self.head_poles, self.head, self.head_constant = np.concatenate(poles), np.concatenate(weights), float(constant)
        self.long_range_modes, self._long_range_transition = long_range_modes, {}
        self.dielectric_constant = 1.0 + float(np.mean(static))
        # The two routes to the static inverse must agree: 1 / (1 + S) = 1 - sum_t 2 a~_t / W~_t.
        self.head_defect = float(max(abs(value - 1.0 / (1.0 + s)) for value, s in zip(inverse, static)))
        self.independent_particle_dielectric_constant = 1.0 + float(np.mean(independent))
        return self.dielectric_constant

    def _solve(self, energies, occupied, coupling):
        self.head, self.long_range_modes, self._long_range_transition = None, [], {}
        self.momentum_terms, self.head_constant = [], None
        self.vertex_order, self.vertex_bands, self._vertex_engines, self.vertex_states = 1, [], None, set()
        self.screening_shifts = self.propagator_shifts = None
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

    _generation = 0                                              # of the tables of transitions on disk

    def set_momentum_terms(self, terms, weights, average, scratch=None):
        """Average the self-energy integrand S_n(q; w) over the momentum
        transfers that sampling the zone centre leaves out, instead of taking
        its closed form at vanishing momentum. With A_0 the coefficient of the
        zero-momentum term (from `set_head`) and F(q) the auxiliary function of
        the zero-momentum constant (`GridCoulombKernel.auxiliary_function`),

            Sigma_c = A_0 <F> + sum_i weight_i [ S_n(q_i; w) - A_0 F(q_i) ] :

        the singular part is averaged analytically (`average` is <F>, the
        zero-momentum constant plus F at the zone centre), and the remainder,
        which is bounded and periodic, by the grid of `terms`
        (`MeshCrystal.momentum_term`, `Approximations.momentum_nodes`) with the
        `weights`. At every node the whole kernel of that momentum enters, so
        nothing is split into a zero-momentum entry and a rest there.

        The transition amplitudes of every mode at every node are kept for as
        long as the screened interaction is (`_transition_table`), under
        `scratch` when it is given, because they take bands x modes numbers per
        mode and node; that changes where they live and not what they are."""
        self.momentum_terms, self.momentum_weights, self.zone_average = list(terms), list(weights), float(average)
        self._momentum_modes, self._transition_scratch, self.prefetch = {}, scratch, False

    def _integrand(self, index, n, frequency):
        """S_n(q; w) and its derivative at the momentum of `momentum_terms[index]`.
        `screening_shifts` and `propagator_shifts` (one number per mode) move the
        levels there with those of the zone centre when they have been updated."""
        term = self.momentum_terms[index]
        if index not in self._momentum_modes:
            gaps = np.asarray(term["gaps"], dtype=float)
            if self.screening_shifts is not None:
                gaps = gaps + np.array([self.screening_shifts[a] - self.screening_shifts[i] for i, a in term["pairs"]])
            if gaps.min() <= 0.0:
                raise ValueError(f"a level difference at the momentum {term['kappa']} is not positive: the levels "
                                 "fed back have closed a gap there")
            root = np.sqrt(gaps)
            casida = np.diag(gaps ** 2) + 4.0 * root[:, None] * np.asarray(term["coupling"]) * root[None, :]
            squared, Z = np.linalg.eigh(0.5 * (casida + casida.conj().T))
            omega = np.sqrt(squared)
            self._momentum_modes[index] = (omega, (root[:, None] * Z) / np.sqrt(omega)[None, :], None)
        omega, modes, _ = self._momentum_modes[index]
        weights = self._transitions_at(index, n)                  # |sqrt(2) (nm|s)|^2, bands x modes
        levels = np.asarray(term["levels"], dtype=float)
        if self.propagator_shifts is not None:
            levels = levels + np.asarray(self.propagator_shifts)[:len(levels)]
        filled = (np.arange(len(levels)) < self.occupied)[:, None]
        poles = np.where(filled, levels[:, None] - omega[None, :], levels[:, None] + omega[None, :])
        inverse = 1.0 / (frequency - poles)
        return float(np.sum(weights * inverse)), float(-np.sum(weights * inverse ** 2))

    def _transitions_at(self, index, n):
        """|sqrt(2) (n m | s)|^2 of the mode n with every section m at the node
        `index` and every mode s of the screened interaction there. They stay
        fixed while the screened interaction does, so they are computed once:
        one mode at a time, or, with `prefetch` (a loop that asks for every
        mode), every mode of the node in batched matrix products."""
        omega, modes, table = self._momentum_modes[index]
        term = self.momentum_terms[index]
        if table is None:
            states = sorted(term["blocks"])
            first = np.asarray(term["blocks"][states[0]])
            shape = (len(states), first.shape[0], modes.shape[1])
            if self._transition_scratch is None:
                store = np.empty(shape)
            else:
                directory = os.path.join(self._transition_scratch, "transitions")
                os.makedirs(directory, exist_ok=True)
                if getattr(self, "_table_generation", None) is None:
                    RandomPhase._generation += 1                  # a new screened interaction: the old tables go
                    self._table_generation = RandomPhase._generation
                    for name in os.listdir(directory):
                        if not name.startswith(f"g{self._table_generation}_"):
                            os.remove(os.path.join(directory, name))
                store = np.lib.format.open_memmap(os.path.join(directory, f"g{self._table_generation}_{index}.npy"),
                                                  mode="w+", dtype=float, shape=shape)
            table = (store, {state: row for row, state in enumerate(states)}, np.zeros(len(states), dtype=bool))
            self._momentum_modes[index] = (omega, modes, table)
        store, rows, done = table
        row = rows[n]
        if not done[row]:
            pending = [r for r in range(len(done)) if not done[r]] if self.prefetch else [row]
            states = sorted(rows, key=rows.get)
            bands = store.shape[1]
            batch = max(1, (1 << 24) // max(1, bands * modes.shape[0]))
            for start in range(0, len(pending), batch):
                chunk = pending[start:start + batch]
                stacked = np.concatenate([np.asarray(term["blocks"][states[r]]) for r in chunk])
                store[chunk] = (2.0 * np.abs(stacked @ modes) ** 2).reshape(len(chunk), bands, -1)
                done[chunk] = True
        return store[row]

    def set_vertex(self, order, bands, interaction, poles, states=None):
        """Add the skeleton diagrams of the orders 2 .. `order` in the screened
        interaction (`diagrams.SkeletonSelfEnergy`) to `correlation`, for the
        modes in `bands`, which are also the modes on the internal lines.
        `interaction[p, q, r, s]` are the Coulomb integrals (pq|rs) over those
        modes, the instantaneous part of W; the retarded part is the `poles`
        random-phase modes that couple to them most strongly (every set of
        modes that `correlation` averages over is used, and averaged). `states`
        limits the modes that receive the diagrams (all of `bands` by default):
        a self-consistent loop solves every mode at every iteration, and the
        diagrams are what that costs."""
        self.vertex_order, self.vertex_bands = int(order), [int(b) for b in bands]
        self.vertex_states = set(self.vertex_bands if states is None else [int(n) for n in states])
        self.vertex_interaction, self.vertex_poles = np.asarray(interaction), int(poles)
        self._vertex_engines = None

    def _vertex(self, n, frequency, levels):
        """The diagrams beyond the first order for mode n: (value, derivative)."""
        from tessera.drivers.bands.diagrams import SkeletonSelfEnergy
        if self.vertex_order < 2 or n not in self.vertex_bands or n not in self.vertex_states:
            return 0.0, 0.0
        bands = self.vertex_bands
        key = tuple(np.round(levels[bands], 12))
        if self._vertex_engines is None or self._vertex_engines[0] != key:
            filled = sum(1 for b in bands if b < self.occupied)
            chemical_potential = 0.5 * (levels[self.occupied - 1] + levels[self.occupied])
            engines = []
            for excitations, _ in self._transitions(bands[0]):
                index = len(engines)
                g = np.stack([self._transitions(p)[index][1][bands, :] for p in bands])          # [p, q, s]
                keep = np.argsort(-(np.abs(g) ** 2).sum(axis=(0, 1)))[:self.vertex_poles]
                engines.append(SkeletonSelfEnergy(levels[bands] - chemical_potential, filled, excitations[keep],
                                                  np.moveaxis(g[:, :, keep], 2, 0), self.vertex_interaction))
            self._vertex_engines = (key, chemical_potential, engines)
        _, chemical_potential, engines = self._vertex_engines
        local = bands.index(n)
        value = derivative = 0.0
        for engine in engines:
            for order in range(2, self.vertex_order + 1):
                v, d = engine.value_and_derivative(local, frequency - chemical_potential, order)
                value, derivative = value + v / len(engines), derivative + d / len(engines)
        return value, derivative

    def _transitions(self, n):
        """(excitations, transition amplitudes of mode n with every mode m) per
        set of modes: those of the problem without the G = 0 entry, or, once
        `set_head` has run, those with it along each momentum given."""
        if not self.long_range_modes:
            return [(self.excitations, self.transition[n])]
        if n not in self._long_range_transition:
            block = np.asarray(self._blocks[n])
            self._long_range_transition[n] = [(omega, np.sqrt(2.0) * (block @ modes))
                                              for omega, modes in self.long_range_modes]
        return self._long_range_transition[n]

    def correlation(self, n, frequency):
        """Sigma^c_nn at a real frequency (and its derivative)."""
        value = derivative = 0.0
        levels = self.energies if self.propagator is None else self.propagator
        sets = self._transitions(n)
        filled = (np.arange(len(levels)) < self.occupied)[:, None]
        for excitations, transition in sets:
            weights = np.abs(transition[:len(levels), :]) ** 2 / len(sets)                     # levels x modes
            poles = np.where(filled, levels[:, None] - excitations[None, :], levels[:, None] + excitations[None, :])
            inverse = 1.0 / (frequency - poles)
            value += np.sum(weights * inverse)
            derivative -= np.sum(weights * inverse ** 2)
        head_value = head_derivative = 0.0
        if self.head is not None:
            level = levels[n]
            poles = level - self.head_poles if n < self.occupied else level + self.head_poles
            head_value = np.sum(self.head / (frequency - poles))
            head_derivative = -np.sum(self.head / (frequency - poles) ** 2)
        if self.momentum_terms:
            coefficient = np.array([head_value, head_derivative]) / self.head_constant            # A_0 and its derivative
            total = coefficient * self.zone_average
            for index, (term, weight) in enumerate(zip(self.momentum_terms, self.momentum_weights)):
                total = total + weight * (np.array(self._integrand(index, n, frequency)) - coefficient * term["auxiliary"])
            vertex = self._vertex(n, frequency, levels)
            return float(total[0]) + vertex[0], float(total[1]) + vertex[1]
        vertex = self._vertex(n, frequency, levels)
        value, derivative = value + vertex[0], derivative + vertex[1]
        return value + head_value, derivative + head_derivative

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


class KineticBasisScreening:
    """The screened interaction as a matrix in the low-lying eigenbasis of the
    kinetic pencil, and the correlation self-energy from it by contour
    deformation: the second route to `RandomPhase`, and the one that scales to
    momentum sets, because its size is the basis and not the number of
    particle-hole pairs.

    In the eigenbasis B of the kinetic pencil (A B = M B diag(lambda)) the
    Coulomb kernel is diagonal, v_mu = strength / lambda_mu, so with the
    coefficients C_p = B^T load_p of the pair densities

        chi0(i u) = - sum_p C_p C_p^T 4 gap_p / (gap_p^2 + u^2) ,
        W_c(i u) = v^(1/2) [ (1 - v^(1/2) chi0 v^(1/2))^-1 - 1 ] v^(1/2) .

    `head` lists, per direction of vanishing momentum, the charges of the pairs
    per unit momentum and the entry of the kernel (`vanishing_momentum_pairs`):
    they enter as one more basis function, the G = 0 one. The rest of the matrix
    is then screened through the mixed entries, the intraband term is
    `constant` [(1 / eps)_00 - 1], and both are averaged over the directions.

    The self-energy of mode n with the coefficients G_m = B^T load_nm is

        Sigma_c(w) = sum_m { -(1 / pi) int_0^inf F_m(u) a_m / (a_m^2 + u^2) du }
                     - sum_{m filled, e_m > w} F_m[e_m - w] + sum_{m empty, e_m < w} F_m[w - e_m] ,

    a_m = w - e_m, F_m(u) = G_m^T W_c(i u) G_m and F_m[x] the same at the real
    frequency x. F_m is analytic and even in u and falls as 1 / u^2, so it is
    represented by its Chebyshev interpolant of degree `nodes` - 1 in the
    variable t = (u - s) / (u + s), s the smallest level difference, a series
    that converges geometrically; nothing is linearized. With u = |a_m| tan(theta)
    the Lorentzian becomes the measure d(theta), and the integral is taken by
    Gauss-Legendre quadrature on two panels split where u = 3 s, so a small and
    a large a_m cost no accuracy."""

    def __init__(self, energies, occupied, pair_coefficients, kernel, head=None, constant=0.0, nodes=64):
        self.energies, self.occupied = np.asarray(energies, dtype=float), int(occupied)
        count = len(self.energies)
        self.gaps = np.array([self.energies[a] - self.energies[i] for i in range(self.occupied)
                              for a in range(self.occupied, count)])
        self.C = np.asarray(pair_coefficients, dtype=float)
        self.v = np.asarray(kernel, dtype=float)
        self.head, self.constant = head or [], float(constant)
        self.scale = float(self.gaps.min())
        j = np.arange(nodes)
        self._t = np.cos(np.pi * (2 * j + 1) / (2 * nodes))                         # Chebyshev points of the first kind
        self._barycentric = (-1.0) ** j * np.sin(np.pi * (2 * j + 1) / (2 * nodes))
        self.frequencies = self.scale * (1.0 + self._t) / (1.0 - self._t)
        self._table = [self._screened(u, imaginary=True) for u in self.frequencies]
        self._theta, self._theta_weights = np.polynomial.legendre.leggauss(64)

    def _screened(self, x, imaginary):
        """(W_c on the basis, the intraband head term) at the frequency i x or x."""
        weights = (-4.0 * self.gaps / (self.gaps ** 2 + x ** 2) if imaginary
                   else 4.0 * self.gaps / (x ** 2 - self.gaps ** 2))
        size = self.C.shape[1]
        if not self.head:
            root = np.sqrt(self.v)
            S = root[:, None] * ((self.C.T * weights) @ self.C) * root[None, :]
            inverse = np.linalg.inv(np.eye(size) - S)
            return root[:, None] * (inverse - np.eye(size)) * root[None, :], 0.0
        W, intraband = np.zeros((size, size)), 0.0
        for charges, entry in self.head:
            C = np.hstack([np.asarray(charges, dtype=float)[:, None], self.C])
            root = np.sqrt(np.concatenate([[entry], self.v]))
            S = root[:, None] * ((C.T * weights) @ C) * root[None, :]
            inverse = np.linalg.inv(np.eye(size + 1) - S)
            W += root[1:, None] * (inverse[1:, 1:] - np.eye(size)) * root[None, 1:] / len(self.head)
            intraband += self.constant * (inverse[0, 0] - 1.0) / len(self.head)
        return W, intraband

    def _lorentzian(self, a, F):
        """int_0^inf F_m(u) a_m / (a_m^2 + u^2) du for a_m > 0, F given on the
        Chebyshev frequencies (rows: m)."""
        total = np.zeros(len(a))
        split = np.arctan(3.0 * self.scale / a)
        for lower, upper in ((np.zeros(len(a)), split), (split, np.full(len(a), 0.5 * np.pi))):
            theta = 0.5 * (upper - lower)[:, None] * (self._theta[None, :] + 1.0) + lower[:, None]
            u = a[:, None] * np.tan(theta)
            t = (u - self.scale) / (u + self.scale)
            difference = t[:, :, None] - self._t[None, None, :]
            exact = np.abs(difference) < 1e-14
            difference = np.where(exact, 1.0, difference)
            terms = self._barycentric[None, None, :] / difference
            values = np.einsum("mqj,mj->mq", terms, F) / terms.sum(axis=2)
            hit = exact.any(axis=2)
            if hit.any():
                values[hit] = np.einsum("mqj,mj->mq", exact.astype(float), F)[hit]
            total += 0.5 * (upper - lower) * (values @ self._theta_weights)
        return total

    def correlation(self, n, coefficients, frequency):
        """Sigma^c_nn at the real frequency `frequency`; `coefficients[m]` are
        the coefficients of the load of psi_n psi_m in the basis."""
        G = np.asarray(coefficients, dtype=float)
        F = np.empty((len(self.energies), len(self.frequencies)))
        for k, (W, intraband) in enumerate(self._table):
            F[:, k] = np.einsum("mi,ij,mj->m", G, W, G)
            F[n, k] += intraband
        a = frequency - self.energies
        sign = np.where(a >= 0.0, 1.0, -1.0)                       # a = 0 is approached from above
        value = -np.sum(sign * self._lorentzian(np.maximum(np.abs(a), 1e-12), F)) / np.pi
        for m, level in enumerate(self.energies):
            # a = 0 is approached from above, where an empty level has already been crossed.
            crossing = (m < self.occupied and level > frequency) or (m >= self.occupied and level <= frequency)
            if crossing:
                W, intraband = self._screened(abs(level - frequency), imaginary=False)
                element = G[m] @ W @ G[m] + (intraband if m == n else 0.0)
                value += -element if m < self.occupied else element
        return value


    def quasiparticle(self, n, coefficients, iterations=50, tolerance=1e-9, step=2e-3):
        """Solve E = e_n + Sigma^c_nn(E) by Newton's method to convergence (the
        equation is not linearized), the derivative by the seven-point central
        difference (the self-energy is smooth between the levels).
        Returns (E, renormalization factor Z at the solution)."""
        level = energy = self.energies[n]
        slope = 0.0
        for _ in range(iterations):
            value = self.correlation(n, coefficients, energy)
            slope = sum(weight * (self.correlation(n, coefficients, energy + k * step)
                                  - self.correlation(n, coefficients, energy - k * step))
                        for k, weight in ((1, 0.75), (2, -0.15), (3, 1.0 / 60.0))) / step
            change = (level + value - energy) / (1.0 - slope)
            energy += change
            if abs(change) < tolerance:
                break
        return energy, 1.0 / (1.0 - slope)


def self_consistent_quasiparticles(mean_field, occupied, coupling, integrals, head=None, update_screening=True,
                                    tolerance=1e-6, max_iterations=60, damping=0.7, momentum_terms=None, vertex=None,
                                    scratch=None, log=None):
    """Eigenvalue self-consistency on a Hartree-Fock starting point: the
    orbitals and their Coulomb integrals stay fixed, and the quasiparticle
    levels are fed back into the propagator (`update_screening=False`, the
    scheme usually written GW0: the screened interaction keeps the Hartree-Fock
    levels) or into the propagator and the screened interaction both
    (`update_screening=True`). A Hartree-Fock gap is several times too large, so
    the interaction screened with it is too weak and the one-shot gap too large;
    feeding the levels back into the screening is what closes it.

    `integrals` must cover every mode, and so must the blocks of
    `momentum_terms`, the argument tuple of `RandomPhase.set_momentum_terms`;
    `vertex` is that of `RandomPhase.set_vertex`.
    `head` is the pair (constant, momenta)
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
            if momentum_terms is not None:
                rpa.set_momentum_terms(*momentum_terms, scratch=scratch)
                rpa.screening_shifts = energies - mean_field
                rpa.prefetch = True                              # every mode is solved at every iteration
            if vertex is not None:
                rpa.set_vertex(*vertex)
        if momentum_terms is not None:
            rpa.propagator_shifts = energies - mean_field
        rpa.propagator = energies
        produced = np.array([rpa.quasiparticle(n, reference=mean_field, start=energies[n])[0]
                             for n in range(len(energies))])
        change = float(np.abs(produced - energies).max())
        history.append(change)
        if log:
            log(f"  quasiparticle update {iteration:2d}: largest change {change:.2e} Ry")
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
