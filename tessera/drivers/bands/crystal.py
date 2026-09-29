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
        self.displacement = self.step @ cell.lattice            # Cartesian, from the row vertex to the column vertex
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
        weighted.values = values
        return weighted

    def bloch_links(self, kappa=None):
        """The links of the flat connection of the crystal momentum `kappa` (the
        zone centre when None), in the canonical edge order; kept per momentum."""
        key = (0.0, 0.0, 0.0) if kappa is None else tuple(float(v) for v in kappa)
        if not hasattr(self, "_links"):
            self._links = {}
        if key not in self._links:
            if len(self._links) > 64:
                self._links.clear()
            self._links[key] = self.grid.blochLinks(self.edges, list(key))
        return self._links[key]

    def bloch_link_array(self, kappa=None):
        """`bloch_links(kappa)` as an array, kept per momentum."""
        key = (0.0, 0.0, 0.0) if kappa is None else tuple(float(v) for v in kappa)
        if not hasattr(self, "_link_arrays"):
            self._link_arrays = {}
        if key not in self._link_arrays:
            if len(self._link_arrays) > 64:
                self._link_arrays.clear()
            self._link_arrays[key] = np.asarray(self.bloch_links(kappa), dtype=complex)
        return self._link_arrays[key]

    @property
    def pair_loader(self):
        """`chainhodge.PairLoads` of the cell: the loads of products of sections."""
        if not hasattr(self, "_pair_loader"):
            self._pair_loader = ch.PairLoads(self.complex, self.squared_lengths)
        return self._pair_loader

    @property
    def edge_displacements(self):
        """The Cartesian displacement of every stored link, source to target."""
        if not hasattr(self, "_edge_displacements"):
            steps = np.array([self.grid.displacement(int(x), int(y)) for x, y in self.edges], dtype=float)
            self._edge_displacements = (steps / np.array(self.divisions)) @ self.lattice
        return self._edge_displacements

    def covariant(self, kappa=(0.0, 0.0, 0.0)):
        """The `CovariantChainHodge` of the cell at the flat connection whose
        links are the Bloch phases of the crystal momentum `kappa`
        (`PeriodicKuhnGrid.blochLinks`)."""
        links = self.grid.blochLinks(self.edges, [float(x) for x in kappa])
        return ch.CovariantChainHodge(self.base, ch.Connection(self.complex, links), 7, False)

    def pencil(self, kappa=(0.0, 0.0, 0.0), potential=None):
        """(A, M) at the crystal momentum `kappa` (reciprocal coordinates), in
        energy units: `A = kinetic_scale * A0^U + M0^U[V]`, assembled by
        `CovariantChainHodge.sparsePencil` and `dressedVertexPotential` at the
        connection of `covariant(kappa)`. `potential` is either vertex values or
        a `GridMatrix` from `weighted_mass`. (`GridMatrix.dressed` is the same
        matrices by an entrywise rule, which `certify` holds to this assembly.)"""
        cov = self.covariant(kappa)
        assembled = cov.sparsePencil()
        A = self.kinetic_scale * sp.csc_matrix(assembled.A)
        if potential is not None:
            values = potential.values if isinstance(potential, GridMatrix) else np.asarray(potential).astype(complex)
            A = A + sp.csc_matrix(cov.dressedVertexPotential(list(values)))
        return A.tocsc(), sp.csc_matrix(assembled.M)

    def spacetime(self, kappa=(0.0, 0.0, 0.0)):
        """The cell as a `Spacetime` whose edges carry the declared fields: the
        lengths set by `PeriodicKuhnGrid.build` (`Edge.setLength`) and the Bloch
        phase of the crystal momentum on each edge's own source-to-target
        orientation (`Edge.setPhase`)."""
        signature = tessera.Signature(3, tessera.Lorentzian)
        spacetime = tessera.Spacetime(tessera.Metric(True, signature), tessera.CDT, 1.0, 1.0,
                                      tessera.PREFERRED, self.grid)
        spacetime.build()
        for edge in spacetime.getEdgeList().toVector():
            edge.setPhase(self.grid.blochPhase(edge.getSource().getId(), edge.getTarget().getId(), list(kappa)))
        return spacetime

    def pencil_from_spacetime(self, kappa=(0.0, 0.0, 0.0)):
        """(A, M) read back from the declared fields of `spacetime(kappa)` through
        `WhitneyMass.complexOf`, `WhitneyMass.squaredLengthsOf` and
        `Connection.fromSpacetime`: the route by which a relaxed geometry would
        reach the solver. It equals `pencil(kappa)` without a potential."""
        spacetime = self.spacetime(kappa)
        K = ch.WhitneyMass.complexOf(spacetime)
        lengths = ch.WhitneyMass.squaredLengthsOf(spacetime, K)
        base = ch.ChainHodge(K, lengths, ch.Preset.L2, ch.Branch.Continuation, self.base.crossoverDimension())
        cov = ch.CovariantChainHodge(base, ch.Connection.fromSpacetime(spacetime, K), 7, False)
        pencil = cov.sparsePencil()
        return self.kinetic_scale * sp.csc_matrix(pencil.A), sp.csc_matrix(pencil.M)

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


def richardson_amplification(spacings, orders=None):
    """The sum of the absolute weights with which `richardson` combines the
    meshes: the factor by which it multiplies whatever in the values does not
    follow the error model (a residual of the self-consistency, a mesh outside
    the asymptotic regime). 5.6 for divisions 16, 24, 32 and two orders; 27 for
    8, 12, 16, 20, 24, 32 and five."""
    h = np.asarray(spacings, dtype=float)
    orders = tuple(2 * (p + 1) for p in range(len(h) - 1)) if orders is None else orders
    design = np.column_stack([np.ones_like(h)] + [h ** p for p in orders])
    return float(np.abs(np.linalg.inv(design)[0]).sum())


def richardson(spacings, values, orders=None):
    """Extrapolate `values` measured at mesh spacings `spacings` to zero spacing
    with the error model v(h) = v_0 + sum_p c_p h^p over the given `orders`.
    Piecewise-linear elements on a uniform mesh have an even expansion, and by
    default every mesh but one removes one even order: six meshes remove h^2 to
    h^10. `values` may be an array per mesh (one column per band). Returns
    (extrapolated, leading coefficients); see `richardson_amplification`."""
    h = np.asarray(spacings, dtype=float)
    values = np.asarray(values, dtype=float)
    orders = tuple(2 * (p + 1) for p in range(len(h) - 1)) if orders is None else orders
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


def band_fibers(cell, read, degeneracy=1e-7):
    """The bands of a `BandRead` as `observables.SpectralFiber` objects, one per
    level, a degenerate level being one fiber of rank equal to its multiplicity
    (levels closer than `degeneracy` times the spread of the read).

    The frame of a fiber is M_0^(1/2) Z on the vertices: the eigenvectors are
    the cell-periodic parts of the Bloch functions, frames at different momenta
    are compared in the undressed mass matrix, and the principal angles the
    library measures with the plain inner product are then the angles in that
    mass matrix. The mass matrix commutes with the grid translations, so its
    square root is a Fourier multiplier. The band certificate carries what the
    sparse solver certified. Returns (fibers, the band indices of each)."""
    from tessera import observables
    shape = cell.divisions
    row = cell.mass.dressed().getrow(0)
    stencil = np.zeros(cell.size)
    stencil[row.indices] = row.data.real
    root = np.sqrt(np.fft.fftn(stencil.reshape(shape)).real)
    field = read.vectors.T.reshape((-1,) + shape)
    frames = np.fft.ifftn(np.fft.fftn(field, axes=(1, 2, 3)) * root, axes=(1, 2, 3)).reshape(-1, cell.size).T
    energies = np.asarray(read.energies)
    spread = max(float(energies[-1] - energies[0]), 1.0)
    groups = [[0]]
    for b in range(1, len(energies)):
        (groups[-1].append(b) if energies[b] - energies[b - 1] < degeneracy * spread else groups.append([b]))
    split = lambda values, key: {key + "_re": [float(x) for x in np.real(values)],
                                 key + "_im": [float(x) for x in np.imag(values)]}
    cells = [[int(v)] for v in range(cell.size)]
    fibers = []
    for group in groups:
        frame = frames[:, group]
        certificate = {
            "accepted": bool(read.certified()), "degree": 0, "rank": len(group), "self_adjoint": True,
            "lower_gap": float("inf"), "upper_gap": float("inf"), "nearest_discarded_separation": float("inf"),
            "frequency_lower": float(energies[group[0]]), "frequency_upper": float(energies[group[-1]]),
            "eigen_residual": float(read.residual), "left_residual": float(read.residual),
            "positive_signature": len(group), "negative_signature": 0, "isotropic": False, "left_frame_refusal": "",
            **{key: float("nan") for key in (
                "localization", "localization_support_fraction", "localization_excess", "projector_residual",
                "gram_defect", "projector_norm", "frame_condition_number", "pairing_determinant_re",
                "pairing_determinant_im", "pairing_condition", "pairing_scale", "metric_symmetry_defect")},
            "certificate": {"conditioning": float(read.conditioning), "dense_reference_error": float("nan"),
                            "domain": "band-window", "grade": "certified-numerical" if read.certified()
                            else "heuristic-discovery", "regime": "positive-semidefinite",
                            "residual": float(read.residual), "tolerance": 1e-9}}
        record = {"schema_version": 2, "record_type": "spectral_fiber", "cells": cells, "rows": cell.size,
                  "rank": len(group), "certificate": certificate,
                  **split(energies[group].astype(complex), "eigenvalues"), **split(frame.ravel(), "right_frame"),
                  **split(frame.ravel(), "left_frame"), **split(np.ones(cell.size, dtype=complex), "weights")}
        fibers.append(observables.SpectralFiber.fromRecord(record))
    return fibers, groups


def track_bands(cell, reads, overlap_threshold=0.5):
    """Follow bands along a path of crystal momenta by overlap, not by ordering,
    with the overlap rule of `SpectralFiberTracker.matchFibers` (principal
    angles of the frames on shared cells) on the fibers of `band_fibers`.

    Between two neighbouring momenta every fiber is matched to its best partner
    in both directions; fibers joined by a match form a group (a level that
    splits or merges joins more than two), and within a group the band labels
    are handed on in order of energy. A crossing puts the two bands in
    different groups, so it is passed through instead of being read as avoided.

    Returns (tracked, weights): `tracked[p, b]` is the energy of band `b` (as
    labelled at the first momentum) at path point `p`, and `weights[p, b]` the
    overlap it was continued with (1 at the first point); a band with no
    partner above `overlap_threshold`, or in a group that does not conserve the
    number of bands, left the computed window and carries a weight of zero or
    below."""
    from tessera import observables
    count = reads[0].vectors.shape[1]
    tracked = np.empty((len(reads), count))
    weights = np.ones((len(reads), count))
    tracked[0] = reads[0].energies
    labels = np.arange(count)                        # labels[i] = the label of band i at the previous point
    previous, previous_groups = band_fibers(cell, reads[0])
    for p in range(1, len(reads)):
        current, current_groups = band_fibers(cell, reads[p])
        forward = observables.SpectralFiberTracker.matchFibers(previous, current, overlap_threshold)
        backward = observables.SpectralFiberTracker.matchFibers(current, previous, overlap_threshold)
        # The library normalizes the overlap by the larger rank; the share of the smaller
        # subspace that lies in the larger one is what links a level to the parts it splits into.
        def contained(match, a, b):
            ranks = len(previous_groups[a]), len(current_groups[b])
            return match.overlap.subspaceOverlap * max(ranks) / min(ranks)
        links = {(m.fromIndex, m.toIndex): contained(m, m.fromIndex, m.toIndex) for m in forward}
        for m in backward:
            key = (m.toIndex, m.fromIndex)
            links[key] = max(contained(m, *key), links.get(key, 0.0))
        links = {key: weight for key, weight in links.items() if weight >= overlap_threshold}
        # Connected groups of the bipartite match graph.
        parent = list(range(len(previous) + len(current)))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for (a, b) in links:
            parent[find(a)] = find(len(previous) + b)
        new_labels = np.full(count, -1)
        lost = np.ones(count, dtype=bool)
        overlap_of = np.zeros(count)
        for rootnode in {find(x) for x in range(len(parent))}:
            sources = [i for a in range(len(previous)) if find(a) == rootnode for i in previous_groups[a]]
            targets = [i for b in range(len(current)) if find(len(previous) + b) == rootnode for i in current_groups[b]]
            weight = min((w for (a, b), w in links.items() if find(a) == rootnode), default=0.0)
            for source, target in zip(sorted(sources), sorted(targets)):
                new_labels[target] = labels[source]
                overlap_of[target] = weight
                lost[target] = len(sources) != len(targets) or weight < overlap_threshold
        free = [label for label in range(count) if label not in set(new_labels[new_labels >= 0])]
        for target in np.nonzero(new_labels < 0)[0]:
            new_labels[target] = free.pop(0)
        tracked[p, new_labels] = reads[p].energies
        weights[p, new_labels] = np.where(lost, -overlap_of, overlap_of)
        labels, previous, previous_groups = new_labels, current, current_groups
    return tracked, weights
