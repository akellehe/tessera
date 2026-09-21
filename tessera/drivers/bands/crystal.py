# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The periodic cell, its degree-zero pencil at a crystal momentum, and the
sparse solve.

The pencil of a cell is assembled once, at the trivial connection, from the C++
objects: `A0 = d_1 M_1 d_1^T` (the stiffness matrix of piecewise-linear finite
elements) and `M0` (their mass matrix). A crystal momentum is the flat
connection `U_vw = exp(i k . dx_vw)` with `dx_vw` the unwrapped displacement of
the edge. Because it is a pure gauge on every top simplex, dressing by it is a
unitary congruence of every local block, and the assembled matrices are the
entrywise products `A0 * Phi` and `M0 * Phi` with `Phi_vw = U_vw`. That is what
`pencil` forms, so a scan over crystal momenta costs one assembly.
`CrystalCell.certify` holds the entrywise form to the C++ `CovariantChainHodge`
and measures the premises of the Hermitian specialization.
"""
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

import tessera
from tessera import chainhodge as ch
from tessera import cobordism as cob
from tessera.drivers.bands import HBAR2_OVER_2M


@dataclass
class BandRead:
    """The lowest bands at one crystal momentum."""
    kappa: tuple
    energies: np.ndarray          # eV, ascending
    vectors: np.ndarray           # nodal values, M-orthonormal columns
    residual: float
    conditioning: float
    holds: bool                   # the solver's certificate
    shift_below_spectrum: bool
    solves: int

    def certified(self):
        """The read stands: the residual met its tolerance and the shift was
        certified below the spectrum, so these are the lowest bands."""
        return self.holds and self.shift_below_spectrum


@dataclass
class CellCertificate:
    """The measured premises of the Hermitian specialization at one momentum."""
    allowable: bool
    margin: float
    continuation_ambiguous: bool
    unitary: bool
    max_curvature_defect: float
    max_holonomy_defect: float
    regime: object
    hermitian_defect_stiffness: float
    hermitian_defect_mass: float
    mass_positive_definite: bool
    dressing_defect: float
    notes: list = field(default_factory=list)

    def holds(self, tolerance=1e-12):
        return (self.allowable and not self.continuation_ambiguous and self.unitary
                and self.max_curvature_defect < tolerance and self.max_holonomy_defect < tolerance
                and self.regime == cob.CertificateRegime.ComplexSymmetricPencil
                and self.hermitian_defect_stiffness < tolerance
                and self.hermitian_defect_mass < tolerance and self.mass_positive_definite
                and self.dressing_defect < tolerance)


class GridMatrix:
    """A sparse matrix on the vertices of a cell together with the unwrapped grid
    step of every stored entry, which is what the Bloch dressing needs: the entry
    (v, w) is multiplied by exp(2 pi i kappa . step_vw)."""

    def __init__(self, cell, matrix):
        coo = sp.coo_matrix(matrix)
        n = np.array(cell.divisions)
        d = (cell.index[coo.col] - cell.index[coo.row]) % n
        d = np.where(d == n - 1, -1, d)
        if np.any(np.abs(d) > 1):
            raise ValueError("a stored entry joins vertices that are not neighbours on the grid")
        self.row, self.col, self.data = coo.row, coo.col, coo.data.astype(complex)
        self.step = d / n
        self.shape = coo.shape

    def dressed(self, kappa=(0.0, 0.0, 0.0)):
        phase = np.exp(2j * np.pi * (self.step @ np.asarray(kappa, dtype=float)))
        return sp.csc_matrix((self.data * phase, (self.row, self.col)), shape=self.shape)


class CrystalCell:
    """A periodic cell with lattice vectors `lattice` (rows, angstrom) meshed
    with `divisions` grid steps per axis.

    `kinetic_scale` is hbar^2 / 2m in the units of the run: the pencil
    eigenvalue `lambda` (inverse squared length) is the energy
    `kinetic_scale * lambda`. Pass 1.0 for a dimensionless problem.
    """

    def __init__(self, lattice, divisions, kinetic_scale=HBAR2_OVER_2M, crossover=8):
        self.lattice = np.asarray(lattice, dtype=float).reshape(3, 3)
        if np.isscalar(divisions):
            divisions = (int(divisions),) * 3
        self.divisions = tuple(int(n) for n in divisions)
        self.kinetic_scale = float(kinetic_scale)
        gram = self.lattice @ self.lattice.T
        self.grid = tessera.PeriodicKuhnGrid(*self.divisions, gram.tolist())
        self.complex = cob.ChainComplex.fromTopCells(self.grid.cells())
        self.edges = self.complex.kSimplexVertices(1)
        self.squared_lengths = self.grid.squaredLengths(self.edges)
        # The dense crossover is kept small: nothing here asks for a dense kernel.
        self.base = ch.ChainHodge(self.complex, self.squared_lengths, ch.Preset.L2,
                                  ch.Branch.Continuation, crossover)
        trivial = ch.CovariantChainHodge(self.base, ch.Connection.trivial(self.complex), 7, False)
        pencil = trivial.sparsePencil()
        self.size = pencil.A.shape[0]
        n2, n3 = self.divisions[1], self.divisions[2]
        ids = np.arange(self.size)
        self.index = np.column_stack([ids // (n2 * n3), (ids // n3) % n2, ids % n3])
        self.fractional = self.index / np.array(self.divisions, dtype=float)
        self.positions = self.fractional @ self.lattice
        self.volume = abs(np.linalg.det(self.lattice))
        self.reciprocal = 2.0 * np.pi * np.linalg.inv(self.lattice).T   # rows b_a, b_a . a_c = 2 pi delta
        self.stiffness = GridMatrix(self, pencil.A)
        self.mass = GridMatrix(self, pencil.M)

    @classmethod
    def cubic(cls, a, divisions, **kwargs):
        return cls(a * np.eye(3), divisions, **kwargs)

    @property
    def spacing(self):
        """The mesh spacing used for extrapolation: the cube root of the volume per vertex."""
        return (self.volume / self.size) ** (1.0 / 3.0)

    # ------------------------------------------------------------------ assembly

    def weighted_mass(self, values):
        """M_0[V] for vertex values `values` (the bilinear form of multiplication by V)."""
        values = np.asarray(values).astype(complex)
        weighted = GridMatrix(self, ch.WhitneyMass.assembleVertexPotential(
            self.complex, self.squared_lengths, list(values)))
        # The interpolant lies between its vertex values, so the weighted form
        # is bounded below by this times the mass matrix.
        weighted.floor = float(values.real.min())
        return weighted

    def pencil(self, kappa=(0.0, 0.0, 0.0), potential=None):
        """(A, M) at the crystal momentum `kappa` (reciprocal coordinates), in
        energy units: `A = kinetic_scale * A0^U + M0^U[V]`. `potential` is
        either vertex values or a `GridMatrix` from `weighted_mass`."""
        A = self.kinetic_scale * self.stiffness.dressed(kappa)
        M = self.mass.dressed(kappa)
        if potential is not None:
            weighted = potential if isinstance(potential, GridMatrix) else self.weighted_mass(potential)
            A = A + weighted.dressed(kappa)
        return A.tocsc(), M

    def momentum(self, kappa):
        """The Cartesian crystal momentum k = sum_a kappa_a b_a (inverse angstrom)."""
        return np.asarray(kappa, dtype=float) @ self.reciprocal

    # ------------------------------------------------------------------ solve

    def solve(self, kappa=(0.0, 0.0, 0.0), count=8, potential=None, sigma=None, tolerance=1e-10,
              extra=None):
        """The `count` lowest bands at `kappa`. `sigma` defaults to one energy
        unit below the minimum of the potential, which lies below the spectrum
        because the kinetic part is positive semidefinite. `extra` is added to
        the left-hand matrix as it stands (for instance a spin-independent term
        already dressed by the caller)."""
        A, M = self.pencil(kappa, potential)
        if extra is not None:
            A = (A + extra).tocsc()
        if sigma is None:
            floor = 0.0
            if isinstance(potential, GridMatrix):
                floor = getattr(potential, "floor", 0.0)
            elif potential is not None:
                floor = float(np.min(np.real(potential)))
            sigma = min(floor, 0.0) - max(1.0, 1e-3 * self.kinetic_scale)
        return solve_pencil(A, M, count, sigma, tolerance, kappa=tuple(kappa))

    def bands(self, path, count=8, potential=None, **kwargs):
        """`solve` along a list of crystal momenta; a weighted potential is
        assembled once."""
        if potential is not None and not isinstance(potential, GridMatrix):
            potential = self.weighted_mass(potential)
        return [self.solve(kappa, count, potential, **kwargs) for kappa in path]

    # ------------------------------------------------------------------ certificates

    def certify(self, kappa=(0.0, 0.0, 0.0), compare_dressing=True):
        """Measure the premises of the Hermitian specialization at `kappa` on
        the C++ objects, and hold the entrywise dressing of `pencil` to them."""
        links = self.grid.blochLinks(self.edges, list(kappa))
        U = ch.Connection(self.complex, links)
        cov = ch.CovariantChainHodge(self.base, U, 7, False)
        curvature = max((abs(U.curvature(*t) - 1.0) for t in self.complex.kSimplexVertices(2)), default=0.0)
        holonomy = max(abs(U.holonomy(self.grid.fundamentalCycle(a)) - np.exp(2j * np.pi * kappa[a]))
                       for a in range(3))
        reference = cov.sparsePencil()
        A, M = sp.csc_matrix(reference.A), sp.csc_matrix(reference.M)
        defect = lambda X: sp.linalg.norm(X - X.conj().T) / sp.linalg.norm(X)
        dressing = 0.0
        if compare_dressing:
            mine_A = self.stiffness.dressed(kappa)
            mine_M = self.mass.dressed(kappa)
            dressing = max(abs(mine_A - A).max() / abs(A).max(), abs(mine_M - M).max() / abs(M).max())
        # The dense regime certificate is below the crossover only; the sparse
        # Hermitian defects above are the same identities at any size.
        regime = cob.CertificateRegime.ComplexSymmetricPencil
        notes = []
        if self.size < self.base.crossoverDimension():
            regime = cov.regimeCertificate(0).regime
        else:
            notes.append("regime read from the sparse Hermitian defects (above the dense crossover)")
        try:
            # The solver refuses a mass matrix that has no Cholesky factorization.
            positive = ch.SparsePencilSolver.lowest(M, M, 1, -1.0).shiftBelowSpectrum
        except ValueError:
            positive = False
        certificate = self.base.certificate()
        return CellCertificate(
            allowable=certificate.allowable, margin=certificate.margin,
            continuation_ambiguous=certificate.continuationAmbiguous, unitary=U.isUnitary(),
            max_curvature_defect=float(curvature), max_holonomy_defect=float(holonomy), regime=regime,
            hermitian_defect_stiffness=float(defect(A)), hermitian_defect_mass=float(defect(M)),
            mass_positive_definite=bool(positive), dressing_defect=float(dressing), notes=notes)


def solve_pencil(A, M, count, sigma, tolerance=1e-10, kappa=(0.0, 0.0, 0.0), **options):
    """`SparsePencilSolver.lowest` as a `BandRead`."""
    read = ch.SparsePencilSolver.lowest(A, M, count, sigma, tolerance=tolerance, **options)
    certificate = read.eigenvalues.certificate
    return BandRead(kappa=tuple(kappa), energies=np.array(read.eigenvalues.values).real,
                    vectors=np.asarray(read.vectors), residual=certificate.residual,
                    conditioning=certificate.conditioning, holds=certificate.holds(),
                    shift_below_spectrum=read.shiftBelowSpectrum, solves=read.solves)


def richardson(spacings, values, orders=(2, 4)):
    """Extrapolate `values` measured at mesh spacings `spacings` to zero spacing
    with the error model v(h) = v_0 + sum_p c_p h^p over the given `orders`.
    Piecewise-linear elements on a uniform mesh have an even expansion, so three
    meshes remove the h^2 and h^4 terms. `values` may be an array per mesh (one
    column per band). Returns (extrapolated, leading coefficients)."""
    h = np.asarray(spacings, dtype=float)
    values = np.asarray(values, dtype=float)
    if len(h) != len(orders) + 1:
        raise ValueError("richardson needs exactly one more mesh than error orders")
    design = np.column_stack([np.ones_like(h)] + [h ** p for p in orders])
    coefficients = np.linalg.solve(design, values)
    return coefficients[0], coefficients[1]


def k_path(points, segments=8):
    """The piecewise-linear path through `points` (reciprocal coordinates) with
    `segments` steps per leg, endpoints included once."""
    points = [np.asarray(p, dtype=float) for p in points]
    path = [points[0]]
    for start, end in zip(points[:-1], points[1:]):
        for step in range(1, segments + 1):
            path.append(start + (end - start) * step / segments)
    return [tuple(p) for p in path]


def track_bands(cell, reads, overlap_threshold=0.5):
    """Follow bands along a path of crystal momenta by overlap, not by ordering.

    The eigenvectors of the dressed pencil are the cell-periodic parts of the
    Bloch functions, so frames at neighbouring momenta live in one space and are
    compared in the undressed mass matrix: `O = Z_k^dagger M_0 Z_k'`. Each band
    at one momentum is continued to the band at the next with which it shares
    the most weight (an optimal assignment on |O|^2, the squared cosines of the
    principal angles between one-dimensional frames), so crossings are passed
    through instead of being read as avoided.

    Returns (tracked, weights): `tracked[p, b]` is the energy of band `b` (as
    labelled at the first momentum) at path point `p`, and `weights[p, b]` the
    overlap weight it was continued with (1 at the first point). A weight below
    `overlap_threshold` marks a band that left the computed window.
    """
    from scipy.optimize import linear_sum_assignment
    mass = cell.mass.dressed()
    count = reads[0].vectors.shape[1]
    tracked = np.empty((len(reads), count))
    weights = np.ones((len(reads), count))
    order = np.arange(count)
    tracked[0] = reads[0].energies
    for p in range(1, len(reads)):
        previous = reads[p - 1].vectors[:, order]
        current = reads[p].vectors
        # Each frame is orthonormal in its own dressed mass matrix; in the common
        # undressed one the norms differ from 1 at the order of the step in k.
        norm = lambda Z: np.einsum("ij,ij->j", Z.conj(), mass @ Z).real
        overlap = np.abs(previous.conj().T @ (mass @ current)) ** 2 / np.outer(norm(previous), norm(current))
        rows, cols = linear_sum_assignment(-overlap)
        order = cols[np.argsort(rows)]
        tracked[p] = reads[p].energies[order]
        weights[p] = overlap[np.arange(count), order]
    lost = weights < overlap_threshold
    return tracked, np.where(lost, -weights, weights)
