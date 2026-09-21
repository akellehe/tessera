# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Two sheets in the ab initio run: relativistic separable pseudopotentials
with their spin-orbit block, and Hartree-Fock on two-component sections.

The pseudopotentials of Hartwigsen, Goedecker and Hutter are closed forms. In
hartree atomic units the local part is

    V_loc(r) = -Z erf(r / sqrt(2) r_loc) / r
               + exp(-r^2 / 2 r_loc^2) sum_i C_i (r / r_loc)^(2 i - 2) ,

and the nonlocal part is separable on Gaussian projectors,

    p_i^l(r) = sqrt(2) r^(l + 2 (i - 1)) exp(-r^2 / 2 r_l^2)
               / ( r_l^(l + (4 i - 1) / 2) sqrt(Gamma(l + (4 i - 1) / 2)) ) ,

    V_nl = sum_l sum_ij sum_m |p_i^l Y_lm> h^l_ij <p_j^l Y_lm|
         + sum_{l >= 1} sum_ij sum_{m s, m' s'} |p_i^l Y_lm s> k^l_ij (L . S)_{m s, m' s'} <p_j^l Y_lm' s'| ,

with h^l the average over the two total angular momenta j = l +- 1/2 weighted
by their degeneracies and k^l their difference: an electron of total angular
momentum j sees h^l + k^l <L . S>_j with <L . S> = l / 2 for j = l + 1/2 and
-(l + 1) / 2 for j = l - 1/2. The table of the paper lists the diagonal
entries; the off-diagonal ones follow from them by fixed ratios (its equations
20 to 28). `HghPseudopotential` is a `pseudopotential.Pseudopotential` whose
radial functions are these closed forms, in the rydberg units of the drivers,
together with the spin-orbit coefficients `K`. Only the published entries are
held (`HGH_TABLE`): an element is added by copying its rows from the paper.

Spin is carried as two sheets, two copies of the cell that share one geometry
(`SparsePencilComposition.directSum`), and the spin-orbit term is the
attachment block between them: one more term of low rank in the left-hand
matrix of the pencil, on the same load vectors `M beta` as the scalar
projectors, with the complex Hermitian core k (x) L . S. The two-sheet pencil
is diagonalized whole by `SparsePencilSolver` (`solve_two_sheets`), so the
coupling is carried to all orders. `run_hartree_fock_two_sheets` iterates the
mean field on the two-component sections: the density sums both components,
and the exchange operator couples the sheets through the off-diagonal spin
blocks of the density matrix.

Reference: Hartwigsen, Goedecker & Hutter, Physical Review B 58, 3641 (1998),
Table I and equations 1 to 29.
"""
import argparse
import dataclasses
import itertools
import json
from dataclasses import dataclass, field

import numpy as np
import scipy.linalg
import scipy.sparse as sp
from scipy.special import erf, gamma

from tessera import chainhodge as ch
from tessera.drivers.bands import pseudopotential as pp

# Table I of the paper, hartree atomic units: the valence charge, r_loc, the coefficients C_i of the local part, and per
# angular momentum l the radius r_l, the diagonal h_ii and, for l >= 1, the diagonal spin-orbit k_ii.
HGH_TABLE = {
    "Ga": {"valence": 3, "rloc": 0.560000, "C": (),
           "channels": ((0, 0.610791, (2.369325, -0.249015, -0.551796), ()),
                        (1, 0.704596, (0.746305, -0.513132), (0.029607, -0.000873)),
                        (2, 0.982580, (0.075437,), (0.001486,)))},
    "As": {"valence": 5, "rloc": 0.520000, "C": (),
           "channels": ((0, 0.456400, (4.560761, 1.692389, -1.373804), ()),
                        (1, 0.550562, (1.812247, -0.646727), (0.052466, 0.020562)),
                        (2, 0.685283, (0.312373,), (0.004273,)))},
}

# Equations 20 to 28 of the paper: (h_12 / h_22, h_13 / h_33, h_23 / h_33) per angular momentum.
_RATIOS = {0: (-0.5 * np.sqrt(3.0 / 5.0), 0.5 * np.sqrt(5.0 / 21.0), -0.5 * np.sqrt(100.0 / 63.0)),
           1: (-0.5 * np.sqrt(5.0 / 7.0), np.sqrt(35.0 / 11.0) / 6.0, -14.0 / (6.0 * np.sqrt(11.0))),
           2: (-0.5 * np.sqrt(7.0 / 9.0), 0.5 * np.sqrt(63.0 / 143.0), -18.0 / (2.0 * np.sqrt(143.0)))}

HARTREE_IN_RYDBERG = 2.0


def coefficient_matrix(l, diagonal, size):
    """The symmetric `size` x `size` matrix of one angular momentum from its
    published diagonal (missing entries are zero)."""
    d = np.zeros(3)
    d[:len(diagonal)] = diagonal
    full = np.diag(d)
    r12, r13, r23 = _RATIOS[l]
    full[0, 1] = full[1, 0] = r12 * d[1]
    full[0, 2] = full[2, 0] = r13 * d[2]
    full[1, 2] = full[2, 1] = r23 * d[2]
    return full[:size, :size]


def gaussian_projector(l, i, width, radius):
    """p_i^l(r) of the paper (i counts from 1): normalized, int p^2 r^2 dr = 1."""
    radius = np.asarray(radius, dtype=float)
    power = l + 2 * (i - 1)
    order = l + (4 * i - 1) / 2.0
    return np.sqrt(2.0) * radius ** power * np.exp(-0.5 * (radius / width) ** 2) / (width ** order * np.sqrt(gamma(order)))


@dataclass
class HghPseudopotential(pp.Pseudopotential):
    """A `Pseudopotential` with closed-form radial functions and the spin-orbit
    coefficients `K` (Ry, the shape of `D`). `channels` holds (l, i, r_l) per
    projector. The tabulated arrays of the base class are these closed forms on
    a radial mesh, for the radial solver and the plane-wave reference."""
    rloc: float = 0.0
    C: tuple = ()
    channels: list = field(default_factory=list)
    K: np.ndarray = None

    def local_at(self, radius):
        radius = np.asarray(radius, dtype=float)
        safe = np.maximum(radius, 1e-12)
        x = radius / self.rloc
        coulomb = np.where(radius > 1e-8, erf(safe / (np.sqrt(2.0) * self.rloc)) / safe,
                           np.sqrt(2.0 / np.pi) / self.rloc)
        polynomial = sum(c * x ** (2 * i) for i, c in enumerate(self.C)) if self.C else 0.0
        return HARTREE_IN_RYDBERG * (-self.valence * coulomb + np.exp(-0.5 * x ** 2) * polynomial)

    def short_range_at(self, radius, width):
        """V_loc plus the potential of a Gaussian charge Z of that width: a
        difference of two error functions and the Gaussian of the local part."""
        radius = np.asarray(radius, dtype=float)
        safe = np.maximum(radius, 1e-12)
        smooth = np.where(radius > 1e-8, 2.0 * self.valence * erf(safe / (np.sqrt(2.0) * width)) / safe,
                          2.0 * self.valence * np.sqrt(2.0 / np.pi) / width)
        return self.local_at(radius) + smooth

    def projector_at(self, index, radius):
        l, i, width = self.channels[index]
        return gaussian_projector(l, i, width, radius)


def hgh(element, density=None):
    """The published pseudopotential of `element` (`HGH_TABLE`)."""
    if element not in HGH_TABLE:
        raise ValueError(f"no published entry for {element} is held; copy its rows from Table I of the paper "
                         f"into HGH_TABLE (held: {sorted(HGH_TABLE)})")
    return from_parameters(element, HGH_TABLE[element], density)


def from_parameters(element, entry, density=None, radial_step=0.005, radial_extent=30.0):
    """The pseudopotential of one table entry (hartree atomic units, the layout
    of `HGH_TABLE`). `density` is the valence density as (r, 4 pi r^2 rho),
    which the paper does not give: it starts a self-consistent loop and screens
    the pseudo-atom of the resolution test, and no converged number depends on
    it. By default it is the spherical Hartree density of the pseudo-atom
    (`hartree_atom_density`)."""
    r = np.arange(1, int(round(radial_extent / radial_step)) + 1) * radial_step
    channels, blocks_h, blocks_k = [], [], []
    for l, width, h, k in entry["channels"]:
        count = len(h)
        channels += [(l, i + 1, width) for i in range(count)]
        blocks_h.append(HARTREE_IN_RYDBERG * coefficient_matrix(l, h, count))
        blocks_k.append(HARTREE_IN_RYDBERG * coefficient_matrix(l, k, count))
    pseudo = HghPseudopotential(
        element=element, valence=float(entry["valence"]), r=r, local=np.zeros_like(r),
        projectors=[(l, r * gaussian_projector(l, i, width, r)) for l, i, width in channels],
        D=scipy.linalg.block_diag(*blocks_h), density=np.zeros_like(r),
        rloc=entry["rloc"], C=tuple(entry["C"]), channels=channels, K=scipy.linalg.block_diag(*blocks_k))
    pseudo.local = pseudo.local_at(r)
    if density is None:
        pseudo.density = hartree_atom_density(pseudo)
    else:
        pseudo.density = np.interp(r, np.asarray(density[0], dtype=float), np.asarray(density[1], dtype=float), right=0.0)
    return pseudo


def hartree_atom_density(pseudo, radius=16.0, step=0.02, tolerance=1e-8, max_iterations=200):
    """4 pi r^2 rho on `pseudo.r` of the spherical pseudo-atom in the Hartree
    mean field, inside a sphere of `radius`: two electrons in the lowest s
    level and the rest spread over the lowest p level."""
    r = np.arange(step, radius, step)
    n = len(r)
    kinetic = np.diag(np.full(n, 2.0 / step ** 2)) - (np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)) / step ** 2
    fillings = {0: min(2.0, pseudo.valence), 1: max(0.0, pseudo.valence - 2.0)}
    separable = {}
    for l in fillings:
        channel = [i for i, (li, _) in enumerate(pseudo.projectors) if li == l]
        B = np.array([np.interp(r, pseudo.r, pseudo.projectors[i][1], right=0.0) for i in channel]).T * np.sqrt(step)
        separable[l] = B @ pseudo.D[np.ix_(channel, channel)] @ B.T
    from tessera.drivers.bands.abinitio import PulayMixer
    local = pseudo.local_at(r)
    shell = pseudo.valence * r ** 2 * np.exp(-r) / 2.0                      # 4 pi r^2 rho of an exponential, to start
    mixer = PulayMixer(0.5)
    for _ in range(max_iterations):
        inside = np.cumsum(shell) * step
        outside = np.cumsum((shell / r)[::-1])[::-1] * step
        hartree = 2.0 * (inside / r + outside - shell / r * step)
        produced = np.zeros(n)
        for l, electrons in fillings.items():
            if electrons == 0.0:
                continue
            H = kinetic + np.diag(local + hartree + l * (l + 1) / r ** 2) + separable[l]
            _, u = scipy.linalg.eigh(H, subset_by_index=[0, 0])
            produced += electrons * u[:, 0] ** 2 / step
        if np.abs(produced - shell).sum() * step < tolerance:
            break
        shell = mixer.next(shell, produced)
    return np.interp(pseudo.r, r, produced, right=0.0)


# ---------------------------------------------------------------- L . S on the real harmonics

def _harmonic_polynomials(l):
    """The real harmonics of `pseudopotential.real_harmonics`, in its order, as
    homogeneous polynomials: dicts from the exponents (a, b, c) of x^a y^b z^c
    to the coefficient, up to the common normalization of the order."""
    if l == 0:
        return [{(0, 0, 0): 1.0}]
    if l == 1:
        return [{(1, 0, 0): 1.0}, {(0, 1, 0): 1.0}, {(0, 0, 1): 1.0}]
    if l == 2:
        a, b = np.sqrt(15.0), np.sqrt(5.0 / 4.0)
        return [{(1, 1, 0): a}, {(0, 1, 1): a}, {(1, 0, 1): a},
                {(0, 0, 2): 2.0 * b, (2, 0, 0): -b, (0, 2, 0): -b}, {(2, 0, 0): 0.5 * a, (0, 2, 0): -0.5 * a}]
    raise NotImplementedError("real harmonics are implemented up to l = 2")


def angular_momentum(l):
    """The three matrices of L = -i r x grad on the real harmonics of order l,
    exactly: L maps a monomial to monomials of the same degree, and the image
    of a harmonic polynomial is expanded in the harmonic polynomials."""
    basis = _harmonic_polynomials(l)
    monomials = sorted({exponents for polynomial in basis for exponents in polynomial}
                       | {tuple(e) for e in itertools.product(range(l + 1), repeat=3) if sum(e) == l})
    index = {m: i for i, m in enumerate(monomials)}
    columns = np.zeros((len(monomials), len(basis)))
    for j, polynomial in enumerate(basis):
        for exponents, value in polynomial.items():
            columns[index[exponents], j] = value
    matrices = []
    for axis in range(3):
        u, v = (axis + 1) % 3, (axis + 2) % 3                              # (r x grad)_axis = r_u d_v - r_v d_u
        image = np.zeros_like(columns)
        for j, polynomial in enumerate(basis):
            for exponents, value in polynomial.items():
                for raised, lowered, sign in ((u, v, 1.0), (v, u, -1.0)):
                    if exponents[lowered] > 0:
                        moved = list(exponents)
                        moved[lowered] -= 1
                        moved[raised] += 1
                        image[index[tuple(moved)], j] += sign * value * exponents[lowered]
        expansion, residual = np.linalg.lstsq(columns, image, rcond=None)[:2]
        if np.abs(columns @ expansion - image).max() > 1e-12:
            raise ArithmeticError("the harmonics of this order are not closed under L")
        matrices.append(-1j * expansion)
    return matrices


_PAULI = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], dtype=complex)


def l_dot_s(l):
    """L . S on the 2 (2 l + 1) states (spin, real harmonic), spin the slow
    index: eigenvalues l / 2 (2 l + 2 times) and -(l + 1) / 2 (2 l times)."""
    return sum(np.kron(0.5 * _PAULI[a], L) for a, L in enumerate(angular_momentum(l)))


def expectation(l, j):
    """<L . S> at the total angular momentum j = l +- 1/2."""
    if abs(j - (l + 0.5)) < 1e-12:
        return 0.5 * l
    if l > 0 and abs(j - (l - 0.5)) < 1e-12:
        return -0.5 * (l + 1)
    raise ValueError("j is l + 1/2 or l - 1/2")


def radial_levels(pseudo, l, j, count, strength, **kwargs):
    """The levels of angular momentum l and total angular momentum j of the
    screened, confined pseudo-atom: the radial equation of
    `pseudopotential.radial_levels` with the coefficients h + k <L . S>_j."""
    coupled = dataclasses.replace(pseudo, D=pseudo.D + expectation(l, j) * pseudo.K)
    return pp.radial_levels(coupled, l, count, strength, **kwargs)


# ---------------------------------------------------------------- the two-sheet pencil

def spin_orbit_core(ions):
    """The core of the spin-orbit term on the two-sheet projector loads
    diag(P, P), P with one column per (ion, projector, m) in the order of
    `MeshCrystal._projectors`: k_ij (L . S)_{s m, s' m'} between projectors of
    one ion and one angular momentum, zero elsewhere. `ions` are the
    pseudopotentials, one per ion; one without `K` contributes nothing."""
    starts, rank = [], 0
    for pseudo in ions:
        spans = []
        for l, _ in pseudo.projectors:
            spans.append(rank)
            rank += 2 * l + 1
        starts.append(spans)
    core = np.zeros((2 * rank, 2 * rank), dtype=complex)
    for pseudo, spans in zip(ions, starts):
        K = getattr(pseudo, "K", None)
        if K is None:
            continue
        for i, j in zip(*np.nonzero(K)):
            l = pseudo.projectors[i][0]
            width = 2 * l + 1
            coupling = K[i, j] * l_dot_s(l)
            for s, t in itertools.product(range(2), repeat=2):
                rows = slice(s * rank + spans[i], s * rank + spans[i] + width)
                columns = slice(t * rank + spans[j], t * rank + spans[j] + width)
                core[rows, columns] += coupling[s * width:(s + 1) * width, t * width:(t + 1) * width]
    return core


def two_sheet_term(P, D, core):
    """(P2, D2) of the two-sheet pencil: the loads doubled per sheet, sheet the
    slow index, and the core kron(1, D) + core."""
    P = np.asarray(P, dtype=complex)
    n, rank = P.shape
    P2 = np.zeros((2 * n, 2 * rank), dtype=complex)
    P2[:n, :rank] = P
    P2[n:, rank:] = P
    return P2, np.kron(np.eye(2), np.asarray(D, dtype=complex)) + core


def solve_two_sheets(A, M, P2, D2, count, sigma, tolerance=1e-10):
    """The lowest `count` eigenpairs of the two-sheet pencil
    (A (+) A + P2 D2 P2^dagger, M (+) M): the direct sum by
    `SparsePencilComposition.directSum`, the low-rank term through the Woodbury
    identity inside `SparsePencilSolver`, the shift certified by inertia.
    Returns (values, vectors, residual, below); the rows of the vectors are
    (sheet, vertex), sheet the slow index."""
    sheet = ch.SparsePencil(0, sp.csc_matrix(A, dtype=complex), sp.csc_matrix(M, dtype=complex))
    both = ch.SparsePencilComposition.directSum(sheet, sheet)
    read = ch.SparsePencilSolver.lowestWithLowRank(
        sp.csc_matrix(both.A), sp.csc_matrix(both.M), np.asarray(P2, dtype=complex), np.asarray(D2, dtype=complex),
        count, sigma, tolerance=tolerance)
    return (np.array(read.eigenvalues.values).real, np.asarray(read.vectors),
            read.eigenvalues.certificate.residual, bool(read.shiftBelowSpectrum))


def time_reversal_defect(levels, tolerance=1e-7):
    """The largest splitting inside the consecutive pairs of `levels`: zero for
    a time-reversal-invariant pencil on two sheets, whose levels are Kramers
    doublets."""
    levels = np.asarray(levels)
    pairs = levels[:2 * (len(levels) // 2)].reshape(-1, 2)
    return float(np.abs(pairs[:, 1] - pairs[:, 0]).max())


def multiplet_splitting(levels, upper, lower):
    """The mean of the `upper` highest of the `upper + lower` given levels minus
    the mean of the rest: the distance between the centres of two multiplets.
    A field of lower symmetry that the mesh adds (a rank-two tensor) has no
    trace on either multiplet of a p level, so it moves this distance only at
    second order."""
    levels = np.sort(np.asarray(levels))
    if len(levels) != upper + lower:
        raise ValueError("the levels of both multiplets, no more")
    return float(levels[lower:].mean() - levels[:lower].mean())


# ---------------------------------------------------------------- the pseudo-atom on two sheets

def pseudo_atom_levels(pseudo, side, divisions, strength, count, sigma=None, spin_orbit=True):
    """The lowest `count` two-sheet levels (Ry) of the screened, confined
    pseudo-atom at the centre of a cubic cell of `side` on the mesh: the
    resolution test of `pseudopotential` with the spin-orbit block. Returns
    (levels, residual, below)."""
    from tessera.drivers.bands.crystal import CrystalCell
    cell = CrystalCell.cubic(side, divisions, kinetic_scale=1.0)
    offset = cell.fractional - 0.5
    offset -= np.rint(offset)
    offset = offset @ cell.lattice
    radius = np.linalg.norm(offset, axis=1)
    potential = pp.screened_potential(pseudo, radius, strength)
    M = cell.mass.dressed().real.tocsc()
    A = (cell.stiffness.dressed().real + cell.weighted_mass(potential).dressed().real).tocsc()
    columns = [pseudo.projector_at(i, radius) * harmonic
               for i, (l, _) in enumerate(pseudo.projectors) for harmonic in pp.real_harmonics(l, offset)]
    P = M @ np.array(columns).T
    D = np.zeros((P.shape[1], P.shape[1]))
    spans = np.cumsum([0] + [2 * l + 1 for l, _ in pseudo.projectors])
    for i, j in zip(*np.nonzero(pseudo.D)):
        for m in range(2 * pseudo.projectors[i][0] + 1):
            D[spans[i] + m, spans[j] + m] = pseudo.D[i, j]
    core = spin_orbit_core([pseudo]) if spin_orbit else np.zeros((2 * P.shape[1], 2 * P.shape[1]))
    P2, D2 = two_sheet_term(P, D, core)
    sigma = float(potential.min()) - 2.0 - 2.0 * np.abs(np.linalg.eigvalsh(D2)).max() if sigma is None else sigma
    values, _, residual, below = solve_two_sheets(A, M, P2, D2, count, sigma)
    return values, residual, below


# ---------------------------------------------------------------- Hartree-Fock on two sheets

def _spinor_exchange(mesh, filled, targets):
    """The exchange operator of the filled two-component sections applied to
    the columns of `targets`, as load vectors:

        (K phi)^s (r) = - sum_j psi_j^s (r) int sum_t conj(psi_j^t (r')) phi^t (r') v(r - r') dr' ,

    one Poisson solve per pair through the mesh kernel, the pair density
    summed over the sheets. The off-diagonal spin blocks of the density matrix
    couple the sheets."""
    n = mesh.cell.size
    loads = mesh.triple.loads
    W = np.zeros(targets.shape, dtype=complex)
    for j in range(filled.shape[1]):
        up, down = filled[:n, j], filled[n:, j]
        pair = mesh.kernel.potential(loads(up.conj(), targets[:n]) + loads(down.conj(), targets[n:]),
                                     None, mesh.zero_momentum)
        W[:n] -= loads(up, pair)
        W[n:] -= loads(down, pair)
    return W


def _spinor_load(mesh, filled):
    """The load vector of the density of the filled two-component sections."""
    n = mesh.cell.size
    loads = mesh.triple.loads
    return sum((loads(filled[:n, j].conj(), filled[:n, j:j + 1]) + loads(filled[n:, j].conj(), filled[n:, j:j + 1]))[:, 0]
               for j in range(filled.shape[1])).real


def doubled(run, bands=None):
    """The one-sheet orbitals of a run as two-component sections, each once per
    sheet, ordered by level: the start of the two-sheet loop."""
    vectors, levels = np.asarray(run["vectors"]), np.asarray(run["levels"])
    bands = vectors.shape[1] if bands is None else bands
    n = vectors.shape[0]
    spinors = np.zeros((2 * n, 2 * bands), dtype=complex)
    spinors[:n, 0::2] = vectors[:, :bands]
    spinors[n:, 1::2] = vectors[:, :bands]
    return {"vectors": spinors, "levels": np.repeat(levels[:bands], 2)}


def run_hartree_fock_two_sheets(mesh, bands, start, spin_orbit=True, tolerance=1e-5, mixing=0.3, max_outer=60,
                                max_inner=30, log=None):
    """Hartree-Fock of `mesh` (`abinitio.MeshCrystal`) on two sheets with the
    spin-orbit block of its pseudopotentials, at the zone centre. `bands`
    counts one-sheet bands (2 `bands` levels are computed); `start` is a
    one-sheet run (`MeshCrystal.run_hartree_fock`) or an earlier two-sheet one.
    The loops are those of `MeshCrystal.run_hartree_fock`: exchange rebuilt on
    every computed level and compressed, K = -xi xi^dagger, in an outer loop,
    the Hartree potential converged at fixed exchange in an inner one. The
    first entry of "history_levels" is the diagonalization of the two-sheet
    pencil in the mean field of `start`, before any feedback. "energy" is the
    electronic energy, as in the one-sheet loop."""
    from tessera.drivers.bands.abinitio import PulayMixer
    cell, crystal = mesh.cell, mesh.crystal
    n, filled_count, count = cell.size, crystal.electrons, 2 * bands
    weights = mesh.kernel.weights
    start = dict(start)
    if np.asarray(start["vectors"]).shape[0] == n:
        start = doubled(start, bands)
    orbitals, values = np.asarray(start["vectors"])[:, :count], np.asarray(start["levels"])[:count]
    core = spin_orbit_core([pseudo for pseudo, _ in crystal.ions])
    if not spin_orbit:
        core = np.zeros_like(core)
    projectors, D_projectors = two_sheet_term(mesh.P, mesh.D, core)
    rank = projectors.shape[1]

    def nodal(vectors):
        filled = vectors[:, :filled_count]
        return (np.abs(filled[:n]) ** 2 + np.abs(filled[n:]) ** 2).sum(axis=1)

    norm = lambda difference: np.sqrt(weights @ difference ** 2 / crystal.volume) * crystal.volume / crystal.electrons
    history, history_levels, density = [], [], nodal(orbitals)
    residual, below = 0.0, True
    for outer in range(max_outer):
        W = _spinor_exchange(mesh, orbitals[:, :filled_count], orbitals)
        xi = mesh._compress(W, orbitals)
        P2 = np.hstack([projectors, xi])
        D2 = np.zeros((P2.shape[1], P2.shape[1]), dtype=complex)
        D2[:rank, :rank] = D_projectors
        D2[rank:, rank:] = -np.eye(xi.shape[1])
        mixer = PulayMixer(mixing)
        hartree = mesh.kernel.potential(_spinor_load(mesh, orbitals[:, :filled_count])).real
        inner_tolerance = max(0.3 * tolerance, 0.3 * (history[-1] if history else 1e-2))
        inner_density = density
        for inner in range(max_inner):
            A = (mesh.stiffness + cell.weighted_mass(mesh.ionic + hartree).dressed().real).tocsc()
            values, produced, residual, below = solve_two_sheets(A, mesh.mass, P2, D2, count, float(values[0]) - 1.0)
            if outer == 0 and inner == 0:
                history_levels.append(values.copy())
            out_density = nodal(produced)
            inner_change = norm(out_density - inner_density)
            inner_density = out_density
            if inner_change < inner_tolerance:
                break
            hartree = mixer.next(hartree, mesh.kernel.potential(_spinor_load(mesh, produced[:, :filled_count])).real)
        change = norm(out_density - density)
        history.append(change)
        history_levels.append(values.copy())
        if log:
            log(f"  two sheets, exchange update {outer:2d}: density change {change:.2e} ({inner + 1} inner) gap "
                f"{(values[filled_count] - values[filled_count - 1]) * 13.605693:.4f} eV")
        orbitals, density = produced, out_density
        if change < tolerance:
            break
    if not below:
        # The inertia certificate of the solver needs A - sigma M positive definite. Strongly repulsive projectors
        # leave levels of the local part below the lowest level of the whole operator, so the last pencil is solved
        # once more from below the local potential, where the certificate applies.
        certified_values, _, certified_residual, below = solve_two_sheets(
            A, mesh.mass, P2, D2, count, float((mesh.ionic + hartree).min()) - 0.1)
        below = below and np.abs(certified_values - values).max() < 1e-7
        residual = max(residual, certified_residual)
    # E = 1/2 sum over the filled sections of (h_aa + e_a), h the kinetic and ionic part with the spin-orbit block.
    filled = orbitals[:, :filled_count]
    ionic = (mesh.stiffness + cell.weighted_mass(mesh.ionic).dressed().real).tocsc()
    overlap = filled.conj().T @ projectors
    one_particle = sum(np.einsum("vi,vi->i", filled[s * n:(s + 1) * n].conj(), ionic @ filled[s * n:(s + 1) * n])
                       for s in range(2)) + np.einsum("ip,pq,iq->i", overlap, D_projectors, overlap.conj())
    energy = 0.5 * float(np.sum(one_particle.real + values[:filled_count]))
    return {"levels": values, "vectors": orbitals, "residual": residual, "shift_below_spectrum": below,
            "energy": energy,
            "history": history, "history_levels": history_levels, "converged": history[-1] < tolerance,
            "spacing": cell.spacing, "time_reversal_defect": time_reversal_defect(values),
            "certified": bool(below and residual < 1e-8 and history[-1] < tolerance)}


# ---------------------------------------------------------------- gallium arsenide

def certify_one_sheet(mesh, run, bands):
    """The inertia certificate of a converged one-sheet run from a shift below
    the local potential (see `run_hartree_fock_two_sheets`): the operator is
    rebuilt from the orbitals and its lowest levels solved once more. Returns
    (below, the largest difference from the levels of the run)."""
    orbitals = np.asarray(run["vectors"])[:, :bands]
    occupied = mesh.crystal.electrons // 2
    filled = orbitals[:, :occupied]
    load = 2.0 * sum(mesh.triple.loads(filled[:, j], filled[:, j:j + 1])[:, 0] for j in range(occupied))
    potential = mesh.ionic + mesh.kernel.potential(load).real
    A = (mesh.stiffness + mesh.cell.weighted_mass(potential).dressed().real).tocsc()
    P, D = mesh._compressed(mesh._exchange(filled, orbitals), orbitals, mesh.P)
    from tessera.drivers.bands.abinitio import solve_with_projectors
    values, _, _, below = solve_with_projectors(A, mesh.mass, P, D, bands, float(potential.min()) - 0.1)
    return below, float(np.abs(values - np.asarray(run["levels"])[:bands]).max())


def zone_centre_flags(mesh, vectors):
    """Which levels of the conventional zinc-blende cell come from the zone
    centre of the primitive cell: the character of the three face-centring
    translations sums to 3 for those and to -1 for the levels folded in from
    the X points. Two-sheet vectors are summed over the sheets."""
    from tessera.drivers.bands.gaas import translation_characters
    n, half = mesh.cell.size, mesh.cell.divisions[0] // 2
    shifts = [(0, half, half), (half, 0, half), (half, half, 0)]
    vectors = np.asarray(vectors, dtype=complex)
    total = 0.0
    for sheet in range(vectors.shape[0] // n):
        read = type("Read", (), {"kappa": (0.0, 0.0, 0.0), "vectors": vectors[sheet * n:(sheet + 1) * n]})
        total = total + translation_characters(mesh.cell, read, shifts).real.sum(axis=0)
    return total > 1.0


def _checkpointed(path, compute, log):
    """`compute(start)` continued from the file at `path` until it converges."""
    start = None
    if path is not None and path.exists():
        start = dict(np.load(path))
        if bool(start["converged"]):
            return start
        log(f"  continuing from {path.name}")
    result = dict(compute(start))
    # The first diagonalization of a two-sheet loop is that of its first call.
    first = result.pop("history_levels", [[]])[0]
    result["first_levels"] = start["first_levels"] if start is not None and np.size(start["first_levels"]) else first
    if path is not None:
        np.savez(path, **{key: np.asarray(value) for key, value in result.items()})
    return result


def gallium_arsenide_splitting(divisions, bands=24, a=None, approximations=None, checkpoint=None, max_updates=100,
                               log=print):
    """The spin-orbit splitting of the top of the valence band of gallium
    arsenide at the zone centre, with the published relativistic
    pseudopotentials on meshes `divisions` of the conventional cell:
    Hartree-Fock on one sheet, then on two sheets with the spin-orbit block.
    The top of the valence band, a triplet on one sheet, is a quartet above a
    doublet on two; the splitting is the distance between their centres
    (`multiplet_splitting`), read after the first diagonalization of the
    two-sheet pencil and at self-consistency, extrapolated over the meshes
    with `richardson` and reported against the measured one
    (`reference.GALLIUM_ARSENIDE`). `checkpoint` is a directory that holds the
    state of every loop, from which a later call continues; a call makes at
    most `max_updates` exchange updates per loop."""
    import pathlib
    from tessera.drivers.bands import BOHR, RYDBERG
    from tessera.drivers.bands.abinitio import Crystal, MeshCrystal
    from tessera.drivers.bands.crystal import richardson, richardson_amplification
    from tessera.drivers.bands.reference import GALLIUM_ARSENIDE
    lattice_constant = (GALLIUM_ARSENIDE.lattice_constant if a is None else a) / BOHR
    crystal = Crystal.zinc_blende(lattice_constant, hgh("Ga"), hgh("As"), conventional=True)
    occupied = crystal.electrons // 2
    folder = None if checkpoint is None else pathlib.Path(checkpoint)
    if folder is not None:
        folder.mkdir(parents=True, exist_ok=True)
    rows, previous = [], None
    for n in divisions:
        mesh = MeshCrystal(crystal, n, approximations=approximations)

        def one_sheet(start, mesh=mesh, previous=previous):
            if start is None and previous is not None:
                start = mesh.prolonged(*previous)
            return mesh.run_hartree_fock(bands, max_outer=max_updates, start=start, log=log)

        single = _checkpointed(None if folder is None else folder / f"one_sheet_{n}.npz", one_sheet, log)
        row = {"divisions": int(n), "spacing": float(mesh.cell.spacing), "one_sheet_converged": bool(single["converged"])}
        rows.append(row)
        if not row["one_sheet_converged"]:
            break
        previous = (mesh, single)
        flags = zone_centre_flags(mesh, single["vectors"])
        below, difference = certify_one_sheet(mesh, single, bands)
        row.update({"one_sheet_certified": bool(below and difference < 1e-4),
                    "one_sheet_gap": float((single["levels"][occupied] - single["levels"][occupied - 1]) * RYDBERG),
                    "triplet_from_centre": bool(flags[occupied - 3:occupied].all()),
                    "triplet": (np.asarray(single["levels"])[occupied - 3:occupied] * RYDBERG).tolist()})

        def two_sheets(start, mesh=mesh, single=single):
            return run_hartree_fock_two_sheets(mesh, bands, single if start is None else start,
                                               max_outer=max_updates, log=log)

        double = _checkpointed(None if folder is None else folder / f"two_sheets_{n}.npz", two_sheets, log)
        row["two_sheets_converged"] = bool(double["converged"])
        if not row["two_sheets_converged"]:
            break
        top = slice(2 * occupied - 6, 2 * occupied)
        first = double["first_levels"]
        row.update({"two_sheets_certified": bool(double["certified"]),
                    "time_reversal_defect": float(double["time_reversal_defect"]),
                    "sextet_from_centre": bool(zone_centre_flags(mesh, double["vectors"])[top].all()),
                    "sextet": (np.sort(np.asarray(double["levels"])[top]) * RYDBERG).tolist(),
                    "splitting_first_diagonalization": multiplet_splitting(np.asarray(first)[top], 4, 2) * RYDBERG,
                    "splitting": multiplet_splitting(np.asarray(double["levels"])[top], 4, 2) * RYDBERG,
                    "two_sheet_gap": float((double["levels"][2 * occupied] - double["levels"][2 * occupied - 1]) * RYDBERG)})
        log(f"N={n:3d} splitting {row['splitting']:.4f} eV (first diagonalization "
            f"{row['splitting_first_diagonalization']:.4f}), measured {GALLIUM_ARSENIDE.spin_orbit_splitting} eV")
    result = {"runs": rows, "measured_splitting": GALLIUM_ARSENIDE.spin_orbit_splitting,
              "converged": all(row.get("two_sheets_converged", False) for row in rows)}
    done = [row for row in rows if "splitting" in row]
    if len(done) > 1:
        spacings = [row["spacing"] for row in done]
        result["extrapolated"] = float(richardson(spacings, [row["splitting"] for row in done])[0])
        result["amplification"] = float(richardson_amplification(spacings))
    return result


def main(argv=None):
    from tessera.drivers.bands.settings import Approximations
    parser = argparse.ArgumentParser(
        prog="python -m tessera.drivers.bands.spinorbit",
        description="The spin-orbit splitting of the valence band top of GaAs on two sheets, with the relativistic "
                    "pseudopotentials of Hartwigsen, Goedecker and Hutter, against the measured splitting. Published "
                    "inputs cannot be extended: the table of the paper is held for gallium (three electrons) and "
                    "arsenic (five) as printed, and carries no valence density, which only starts the loops.")
    parser.add_argument("--divisions", type=int, nargs="+", default=(12, 16, 20, 24),
                        help="mesh divisions of the conventional cell (even); every mesh beyond the first removes one "
                             "even order of the mesh error")
    parser.add_argument("--bands", type=int, default=24, help="one-sheet bands of the self-consistent loops")
    parser.add_argument("--lattice-constant", type=float, default=None, help="angstrom; the measured one by default")
    parser.add_argument("--checkpoint", default=None, help="a directory for the state of the loops, to continue from")
    parser.add_argument("--max-updates", type=int, default=100, help="exchange updates per loop in this call")
    parser.add_argument("--out", default=None, help="write the result as JSON")
    defaults = Approximations()
    group = parser.add_argument_group("approximations made for the sake of cost")
    group.add_argument("--refinement-terms", type=int, default=defaults.refinement_terms,
                       help="terms of the refinement series of the zero-momentum constant, 1 to 5")
    group.add_argument("--lattice-images", type=int, default=defaults.lattice_images,
                       help="periodic images per axis in the lattice sums (odd)")
    args = parser.parse_args(argv)
    approximations = Approximations(refinement_terms=args.refinement_terms, lattice_images=args.lattice_images)
    result = gallium_arsenide_splitting(args.divisions, args.bands, args.lattice_constant, approximations,
                                        args.checkpoint, args.max_updates)
    print(json.dumps(result, indent=1))
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(result, handle, indent=1)
    return result


if __name__ == "__main__":
    main()
