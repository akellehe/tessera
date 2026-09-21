# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The quasiparticle equation on a momentum set.

A momentum set is a uniform grid of crystal momenta (flat U(1) connections of
the cell) that contains the zone centre, so that sums and differences of its
members are its members again up to a reciprocal vector. A cell sampled on a
set of N_k momenta is the supercell of N_k cells sampled at its zone centre,
and every statement here is held to that identity by the test suite.
"""
import numpy as np

from tessera.drivers.bands import coulomb
from tessera.drivers.bands.screening import long_range_problem


def uniform_set(divisions):
    """The momenta (i / n_1, j / n_2, l / n_3) of a uniform grid through the zone
    centre, in reciprocal coordinates of the cell; `divisions` is one number or
    one per axis."""
    n = [int(divisions)] * 3 if np.isscalar(divisions) else [int(d) for d in divisions]
    return [(i / n[0], j / n[1], l / n[2]) for i in range(n[0]) for j in range(n[1]) for l in range(n[2])]


class SetScreening:
    """The screened interaction of the random-phase approximation at every
    momentum transfer of a set, and the self-energy of a state summed over the
    transfers,

        Sigma_c(n k; w) = sum_q sum_m sum_t |w^t_{nk, m k+q}|^2 / (w - e_m(k + q) -+ W_t(q)) ,

    the pairs of the transfer q running from a filled section at k to an empty
    one at k + q for every k. Orbitals are normalized in the cell, so every
    Coulomb integral carries 1 / N_k, the normalization in the supercell the
    set is equivalent to. A sum k + q that leaves the first zone is brought
    back by a reciprocal vector, moved into the vertex values
    (`MeshCrystal._shifted`), so that every pair of one transfer carries exactly
    that transfer.

    With `head` the entry of the kernel at zero transfer and G = 0 is restored
    as `RandomPhase.set_head` restores it at the zone centre: the pairs of zero
    transfer carry the charges of `MeshCrystal.vanishing_momentum_charges_set`
    along each of `directions` (the Cartesian axes by default), the entry is
    that of the supercell, strength / (N_k V) per unit q^-2, their modes with
    that entry replace the modes of zero transfer, and the intraband term is
    the zero-momentum constant of the set times their residues.

    `nodes` (`set_nodes`) are the offsets of `zero_momentum_order` k > 1, the
    midpoint grid of k per axis inside the cell of momentum space around every
    transfer of the set. The integrand is then evaluated at every transfer of
    the set moved by every offset, the sections at the moved momenta from
    `MeshCrystal.bands_at_set`, and with A_0 the coefficient of the
    zero-momentum term and F the auxiliary function of the kernel,

        Sigma_c = A_0 <F> + (1 / N_k) sum_{q, i} weight_i [ S_n(q + d_i; w) - A_0 F(q + d_i) ] ,

    as `RandomPhase.set_momentum_terms` does at the zone centre. An odd k has
    the offset zero: there the transfers of the set themselves enter, and zero
    transfer as three small ones along the axes.

    `extended` is the result of `MeshCrystal.extend_bands_set`; `bands` is the
    number of bands kept, one number or one per momentum, or with `threshold`
    every level below that energy at every momentum, moved or not.
    """

    def __init__(self, mesh, extended, bands=None, head=True, directions=None, nodes=(), threshold=None, log=None):
        self.mesh, self.extended = mesh, extended
        self.occupied = int(extended["occupied"])
        self.momenta = [np.asarray(kappa, dtype=float) for kappa in extended["momenta"]]
        self.count = len(self.momenta)
        count_below = lambda levels: int(np.sum(np.asarray(levels) < threshold))
        if threshold is not None:
            self.bands = [count_below(extended["levels"][k]) for k in range(self.count)]
        else:
            bands = len(extended["levels"][0]) if bands is None else bands
            self.bands = [int(bands)] * self.count if np.isscalar(bands) else [int(b) for b in bands]
        self.mean_field = [np.asarray(extended["levels"][k], dtype=float)[:self.bands[k]] for k in range(self.count)]
        self.constant = float(extended["zero_momentum"])
        chosen = np.eye(3) if directions is None else np.atleast_2d(np.asarray(directions, dtype=float))
        self.directions = [direction / np.linalg.norm(direction) for direction in chosen]
        on_set = {"levels": self.mean_field, "vectors": extended["vectors"], "bands": self.bands, "on_set": True}
        self.head = None
        if head:
            self.head = self._entry(1.0, self._zero_class(), np.zeros(3), on_set)
            charges = mesh.vanishing_momentum_charges_set(extended, self.bands)
            self.head["charges"] = [sum(direction[axis] * charges[axis] for axis in range(3))
                                    for direction in self.directions]
        self.nodes = nodes = [(np.asarray(offset, dtype=float), float(weight)) for offset, weight in nodes]
        if nodes and not head:
            raise ValueError("the offsets of the zero-momentum order need the zero-momentum term")
        self.entries = []
        if not nodes:
            self.entries = [self._entry(1.0, c, np.zeros(3), on_set) for c in range(self.count)
                            if not (head and c == self._zero_class())]
            self.effective_constant = self.constant
        else:
            size = np.array([len({round(float(kappa[axis]), 9) for kappa in self.momenta}) for axis in range(3)])
            small = [(offset, weight) for offset, weight in nodes if np.abs(offset * size).max() < 0.02]
            if small:                                            # the offset zero: the transfers of the set themselves
                total = sum(weight for _, weight in small)
                self.entries += [self._entry(total, c, np.zeros(3), on_set) for c in range(self.count)
                                 if c != self._zero_class()]
            for offset, weight in nodes:
                found = mesh.bands_at_set(extended, offset, converge=self.bands, log=log)
                kept = self.bands if threshold is None else [count_below(levels) for levels in found["levels"]]
                moved = {"levels": [np.asarray(found["levels"][k], dtype=float)[:kept[k]] for k in range(self.count)],
                         "vectors": found["vectors"], "bands": kept, "on_set": False}
                is_small = any(offset is other for other, _ in small)
                self.entries += [self._entry(weight, c, offset, moved)
                                 for c in ([self._zero_class()] if is_small else range(self.count))]
            average = mesh.zero_momentum + mesh.kernel.auxiliary_function()                 # <F> over every momentum
            self.effective_constant = average - sum(
                entry["weight"] * mesh.kernel.auxiliary_function(tuple(entry["transfer"])) for entry in self.entries
            ) / self.count
        self.solve()

    def _zero_class(self):
        return next(c for c, kappa in enumerate(self.momenta) if not np.any(kappa))

    def _entry(self, weight, c, offset, targets):
        """The pairs of the transfer momenta[c] + offset: filled sections of the
        set, empty sections of `targets` (the set, or the set moved by the
        offset), their loads, potentials and coupling."""
        mesh, extended, occupied = self.mesh, self.extended, self.occupied
        transfer = self.momenta[c] + offset
        loads, pairs, gaps = [], [], []
        for k in range(self.count):
            k2, shift = self.partner(k, c)
            top = targets["bands"][k2]
            sections = mesh._shifted(np.asarray(targets["vectors"][k2])[:, occupied:top], shift)
            for i in range(occupied):
                z = np.asarray(extended["vectors"][k])[:, i]
                loads.append(coulomb.pair_loads(mesh.cell, z.conj(), sections, tuple(self.momenta[k] + transfer),
                                                tuple(-v for v in self.momenta[k])))
                pairs += [(k, i, k2, a) for a in range(occupied, top)]
                gaps.append(targets["levels"][k2][occupied:top] - self.mean_field[k][i])
        loads = np.hstack(loads)
        potentials = mesh.kernel.potential(loads, tuple(transfer) if np.any(transfer) else None)
        return {"weight": float(weight), "class": c, "transfer": transfer, "targets": targets, "pairs": pairs,
                "gaps": np.concatenate(gaps), "potentials": potentials,
                "coupling": loads.conj().T @ potentials / self.count, "charges": None}

    def partner(self, k, c):
        """(index of momenta[k] + momenta[c] in the set, the reciprocal vector that brings it back)."""
        total = self.momenta[k] + self.momenta[c]
        for index, kappa in enumerate(self.momenta):
            shift = total - kappa
            if np.abs(shift - np.rint(shift)).max() < 1e-9:
                return index, np.rint(shift)
        raise ValueError("the momentum set is not closed under addition")

    def solve(self, levels=None):
        """The modes of the screened interaction at every transfer, with the
        level differences moved by `levels` (one array per momentum; the mean
        field when None): a level of a moved momentum moves with the level of
        the same band of the set."""
        shifts = [np.zeros(len(values)) for values in self.mean_field] if levels is None \
            else [np.asarray(l) - m for l, m in zip(levels, self.mean_field)]
        shift_of = lambda k, n: shifts[k][n] if n < len(shifts[k]) else shifts[k][-1]

        def gaps_of(entry):
            gaps = entry["gaps"] + np.array([shift_of(k2, a) - shifts[k][i] for k, i, k2, a in entry["pairs"]])
            if gaps.min() <= 0.0:
                raise ValueError(f"a level difference at the transfer {tuple(entry['transfer'])} is not positive: the "
                                 "levels fed back have closed a gap there")
            return gaps

        for entry in self.entries:
            gaps = gaps_of(entry)
            omega, modes, _ = long_range_problem(gaps, entry["coupling"], np.zeros(len(gaps)), 0.0)
            entry["modes"] = [(omega, modes)]
        self.residues, self.dielectric_constant = [], None
        if self.head is not None:
            gaps, static = gaps_of(self.head), []
            self.head["modes"] = []
            for direction, charges in zip(self.directions, self.head["charges"]):
                u = self.mesh.kernel.momentum_entry_limit(direction) / self.count
                omega, modes, residues = long_range_problem(gaps, self.head["coupling"], charges, u)
                self.head["modes"].append((omega, modes))
                self.residues.append((omega, self.effective_constant * residues / len(self.directions)))
                static.append(1.0 / (1.0 - np.sum(2.0 * residues / omega)))                 # 1 / eps^-1 along the direction
            self.dielectric_constant = float(np.mean(static))

    def _terms(self, state, levels):
        """(weights, poles) of the self-energy of the state (momentum index, band)."""
        k, n = state
        mesh, occupied = self.mesh, self.occupied
        z = np.asarray(self.extended["vectors"][k])[:, n]
        shifts = [np.asarray(l) - m for l, m in zip(levels, self.mean_field)]
        terms = []
        # Without offsets the pairs of zero transfer enter with the modes that carry the G = 0 entry.
        for entry in self.entries + ([self.head] if self.head is not None and not self.nodes else []):
            weight = entry["weight"]
            k2, shift = self.partner(k, entry["class"])
            targets = entry["targets"]
            top = targets["bands"][k2]
            sections = mesh._shifted(np.asarray(targets["vectors"][k2])[:, :top], shift)
            state_loads = coulomb.pair_loads(mesh.cell, z.conj(), sections, tuple(self.momenta[k] + entry["transfer"]),
                                             tuple(-v for v in self.momenta[k]))
            block = state_loads.conj().T @ entry["potentials"] / self.count
            moved = np.array([shifts[k2][m] if m < len(shifts[k2]) else shifts[k2][-1] for m in range(top)])
            poles_of = targets["levels"][k2][:top] + moved
            filled = (np.arange(top) < occupied)[:, None]
            for omega, modes in entry["modes"]:                  # one set, or one per direction of the vanishing momentum
                weights = weight * 2.0 * np.abs(block @ modes) ** 2 / len(entry["modes"])             # bands x modes
                terms.append((weights, np.where(filled, poles_of[:, None] - omega[None, :],
                                                poles_of[:, None] + omega[None, :])))
        for omega, weights in self.residues:                     # the intraband term of the G = 0 entry
            terms.append((weights, levels[k][n] - omega if n < occupied else levels[k][n] + omega))
        return terms

    def quasiparticle(self, state, levels=None, start=None):
        """(mean-field level, quasiparticle level, renormalization) of a state:
        E = e + Sigma_c(E) by Newton's iteration, the propagator on `levels`."""
        k, n = state
        levels = self.mean_field if levels is None else levels
        terms = self._terms(state, levels)
        level = float(self.mean_field[k][n])
        energy, slope = (level if start is None else float(start)), 0.0
        for _ in range(60):
            value = sum(np.sum(w / (energy - p)) for w, p in terms)
            slope = -sum(np.sum(w / (energy - p) ** 2) for w, p in terms)
            step = (level + value - energy) / (1.0 - slope)
            energy += step
            if abs(step) < 1e-11:
                break
        return level, float(energy), float(1.0 / (1.0 - slope))

    def quasiparticles(self, states):
        return {tuple(state): self.quasiparticle(tuple(state)) for state in states}

    def self_consistent(self, update_screening=True, tolerance=1e-5, max_iterations=60, damping=0.7, log=None):
        """Eigenvalue self-consistency on the set, as
        `screening.self_consistent_quasiparticles` at the zone centre: the
        sections and their Coulomb integrals stay fixed, and the quasiparticle
        levels of every kept state are fed back into the propagator
        (`update_screening=False`, GW0) or into the propagator and the screened
        interaction both. Returns (levels per momentum, history of the largest
        change)."""
        levels = [values.copy() for values in self.mean_field]
        history = []
        for iteration in range(max_iterations):
            if update_screening and iteration:
                self.solve(levels)
            produced = [np.array([self.quasiparticle((k, n), levels, start=levels[k][n])[1]
                                  for n in range(self.bands[k])]) for k in range(self.count)]
            change = float(max(np.abs(p - l).max() for p, l in zip(produced, levels)))
            history.append(change)
            levels = [(1.0 - damping) * l + damping * p for l, p in zip(levels, produced)]
            if log:
                log(f"  momentum set, quasiparticle update {iteration:2d}: largest change {change:.2e} Ry")
            if change < tolerance:
                break
        return levels, history


def set_nodes(approximations, momenta):
    """The offsets of `zero_momentum_order` on a momentum set, as (reciprocal
    coordinates of the cell, weight): `Approximations.momentum_nodes`, the
    midpoint grid of the cell of momentum space around the zone centre, scaled
    to the cell around a transfer of the set."""
    size = np.array([len({round(float(kappa[axis]), 9) for kappa in momenta}) for axis in range(3)], dtype=float)
    return [(tuple(np.asarray(offset) / size), weight) for offset, weight in approximations.momentum_nodes]


class SetSymmetry:
    """The momenta of a set that must be solved, and the rest as their images.

    The operations are those the mesh keeps exactly: time reversal (the
    sections of -k are the conjugates of those of k) and the permutations of
    the axes that carry the periodic Kuhn grid, the lattice, the local potential
    of the ions and their separable terms onto themselves. A permutation p
    carries the vertex of grid coordinates c to the one of coordinates c[p] and
    the momentum kappa to kappa[p]; the operations that the crystal does not
    have are found by applying them, and left out.
    """

    def __init__(self, mesh, momenta, tolerance=1e-9):
        import itertools
        self.mesh = mesh
        self.momenta = [np.asarray(kappa, dtype=float) for kappa in momenta]
        cell = mesh.cell
        divisions = np.array(cell.divisions)
        lookup = {tuple(int(v) for v in c): index for index, c in enumerate(np.asarray(cell.index) % divisions)}
        rng = np.random.default_rng(0)
        probe = rng.standard_normal(cell.size)
        nonlocal_ = lambda r: mesh.P @ (mesh.D @ (mesh.P.T @ r))
        self.operations = []
        for permutation in itertools.permutations(range(3)):
            p = list(permutation)
            if not np.array_equal(divisions[p], divisions) or np.abs(cell.lattice[p][:, p] - cell.lattice).max() > tolerance:
                continue
            carried = np.array([lookup[tuple(int(v) for v in c[p])] for c in np.asarray(cell.index) % divisions])
            moved = np.empty(cell.size)
            moved[carried] = probe
            image = np.empty(cell.size)
            image[carried] = nonlocal_(probe)
            scale = max(1.0, np.abs(mesh.ionic).max())
            if np.abs(mesh.ionic[carried] - mesh.ionic).max() > tolerance * scale:
                continue
            if np.abs(nonlocal_(moved) - image).max() > tolerance * max(1.0, np.abs(image).max()):
                continue
            for reverse in (False, True):
                self.operations.append((p, carried, reverse))
        self.images = [None] * len(self.momenta)                 # per momentum: (representative, operation, reciprocal shift)
        self.representatives = []
        for k, kappa in enumerate(self.momenta):
            if self.images[k] is not None:
                continue
            self.representatives.append(k)
            for operation in self.operations:
                p, _, reverse = operation
                target = -kappa[p] if reverse else kappa[p]
                for m, member in enumerate(self.momenta):
                    shift = target - member
                    if self.images[m] is None and np.abs(shift - np.rint(shift)).max() < 1e-9:
                        self.images[m] = (k, operation, np.rint(shift))

    def image(self, k, vectors):
        """The sections of the momentum k from those of its representative."""
        _, (p, carried, reverse), shift = self.images[k]
        vectors = np.asarray(vectors)
        moved = np.empty(vectors.shape, dtype=complex)
        moved[carried] = vectors.conj() if reverse else vectors
        return self.mesh._shifted(moved, -shift) if np.any(shift) else moved

    def complete(self, solved):
        """One entry per momentum of the set from {representative: sections}."""
        return [solved[k] if k in solved else self.image(k, solved[self.images[k][0]]) for k in range(len(self.momenta))]
