# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The accelerator of the exchange update of Hartree-Fock on the mesh (#1173).

The exchange update of `MeshCrystal.run_hartree_fock` replaces the exchange
energy, which is concave in the covariance, by its tangent at the current
state and minimizes what is left, so every update lowers the energy. The rate
is set by the softest rotation of the filled sections into the empty ones, and
near a saddle of the energy that rotation is soft enough to hold the loop for
tens of updates. Most of the rotation lies inside the span of the bands the
pencil has just produced, and inside a span the mean field is a problem of
small matrices that is solved without truncation: the Coulomb integrals of the
span are computed once (`SpanIntegrals`, `set_integrals`), the state is a pure
Slater covariance on the modes of the span (the library's `CovarianceState`,
`SpanMeanField.certificate`), its Fock operator is the Wick contraction that
`ModeInteraction.fock` defines, and the energy is minimized over the Slater
frames of the span by Newton steps in a trust region (`SpanMeanField.minimize`).
The exchange operator of the next update is built from the minimizing frame.

Nothing is extrapolated. A step inside the span is kept only if the energy
falls, the update that follows lowers it again, and at a stationary state of
the mesh the span holds the filled sections as a stationary frame, which the
minimization returns unchanged: the fixed points are those of the plain update.
The filled frames of earlier updates can join the span
(`Approximations.exchange_history`); they change the path and its cost, never
the fixed point.

The second variation of the energy is positive at a minimum and has a negative
direction at a saddle. Hartree-Fock has more than one stationary state, so a
run returns the lowest eigenvalue of the second variation on its last span
together with its energy, to say which kind of state it stopped at.
"""
import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import LinearOperator, eigsh

from tessera import quantum
from tessera.drivers.bands import coulomb


class SpanIntegrals:
    """The Coulomb integrals (pq|rs) of the real M-orthonormal columns of
    `modes` at the zone centre, through the mesh's kernel, with the pair loads
    and their potentials kept per unordered pair: the exchange operator of any
    frame of the span then costs no further Poisson solve (`exchange`)."""

    def __init__(self, kernel, cell, modes, block=64):
        self.modes = np.asarray(modes, dtype=float)
        size, count = self.modes.shape
        self.count, self.cell = count, cell
        rows, columns = np.triu_indices(count)
        self.index = np.zeros((count, count), dtype=int)
        self.index[rows, columns] = self.index[columns, rows] = np.arange(len(rows))
        self.loads = np.empty((size, len(rows)))
        for p in range(count):
            self.loads[:, self.index[p, p:]] = coulomb.pair_loads(cell, self.modes[:, p], self.modes[:, p:])
        self.potentials = np.empty_like(self.loads)
        for start in range(0, len(rows), block):
            self.potentials[:, start:start + block] = kernel.potential(self.loads[:, start:start + block]).real
        packed = self.loads.T @ self.potentials
        packed = 0.5 * (packed + packed.T)
        self.tensor = packed[self.index[:, :, None, None], self.index[None, None, :, :]]

    def density_load(self, frame):
        """The load vector of the density 2 sum_j z_j^2 of the frame Z = modes C."""
        P = frame @ frame.T
        weights = np.where(np.eye(self.count, dtype=bool), 2.0 * P, 4.0 * P)[np.triu_indices(self.count)]
        return self.loads @ weights

    def exchange(self, frame, zero_momentum):
        """K[Z] applied to every mode of the span, Z = modes C the filled frame,
        as load vectors: what `MeshCrystal._exchange(Z, modes)` returns."""
        filled = self.modes @ frame
        W = np.zeros_like(self.modes)
        pair = np.empty((frame.shape[1],) + self.modes.shape)
        for n in range(self.count):
            pair[:, :, n] = (self.potentials[:, self.index[:, n]] @ frame).T
        # The pair density of the modes p and n has the charge delta_pn, which the kernel's entry at zero
        # momentum turns into a constant potential.
        pair += zero_momentum * frame.T[:, None, :]
        for j in range(frame.shape[1]):
            W -= coulomb.pair_loads(self.cell, filled[:, j], pair[j])
        return W

    def response_tensor(self, zero_momentum):
        """R[m, n, q, p] with G[D]_mn = sum_pq R[m, n, q, p] D[q, p], the part of
        the closed-shell Fock operator that is linear in the covariance D of one
        sheet: 2 (mn|pq) - (qm|pn) - c delta_qm delta_pn."""
        R = 2.0 * self.tensor                                    # (mn|pq) = (mn|qp) for real modes
        R -= self.tensor.transpose(1, 3, 0, 2)
        m, n = np.meshgrid(np.arange(self.count), np.arange(self.count), indexing="ij")
        R[m, n, m, n] -= zero_momentum
        return R


def set_integrals(mesh, momenta, constant, modes):
    """The response tensors of the spans `modes[k]` (M_k-orthonormal sections of
    momenta[k]): R[k][k'][m, n, q, p] with
    G_k[D]_mn = (1 / N_k) sum_k' sum_pq R[k][k'][m, n, q, p] D_k'[q, p]. The
    direct part pairs the periodic pair densities of each momentum through the
    kernel of zero mean; the exchange part pairs the densities
    conj(z_pk') z_nk, of momentum k - k', through the kernel at that momentum,
    whose entry at zero transfer is N_k times the constant of the set
    (`MeshCrystal.run_hartree_fock_set`)."""
    cell, kernel, count = mesh.cell, mesh.kernel, len(momenta)

    def pair_densities(other, k):
        minus = tuple(-v for v in momenta[other])
        return np.stack([coulomb.pair_loads(cell, modes[other][:, p].conj(), modes[k], momenta[k], minus)
                         for p in range(modes[other].shape[1])], axis=1)            # [vertex, p at k', n at k]

    periodic = [pair_densities(k, k) for k in range(count)]
    hartree = [kernel.potential(tau.reshape(cell.size, -1)) for tau in periodic]
    R = [[None] * count for _ in range(count)]
    for k in range(count):
        for other in range(count):
            same = other == k
            transfer = tuple(a - b for a, b in zip(momenta[k], momenta[other]))
            tau = periodic[k] if same else pair_densities(other, k)
            flat = tau.reshape(cell.size, -1)
            potential = kernel.potential(flat, None if same else transfer, count * constant if same else None)
            shape = (tau.shape[1], tau.shape[2], tau.shape[1], tau.shape[2])
            exchange = (flat.conj().T @ potential).reshape(shape)                   # [q, m, p, n]
            direct = (periodic[k].reshape(cell.size, -1).T @ hartree[other]).reshape(
                (periodic[k].shape[1],) * 2 + (periodic[other].shape[1],) * 2)      # [m, n, p, q]
            R[k][other] = 2.0 * direct.transpose(0, 1, 3, 2) - exchange.transpose(1, 3, 0, 2)
    return R


class SpanMeanField:
    """Hartree-Fock on the modes of a span, two sheets filled alike, one block
    of modes per crystal momentum: the state is the pure covariance of the
    frames C_k (modes by filled, orthonormal), P_k = C_k C_k^dagger on each
    sheet, and the Fock operator is F_k = h_k + (1 / N_k) sum_k' R[k][k'] P_k'
    (`SpanIntegrals.response_tensor`, `set_integrals`). The energy per cell is
    (1 / N_k) sum_k tr((h_k + F_k) P_k)."""

    def __init__(self, one_particle, response, filled):
        self.one_particle = [0.5 * (h + h.conj().T) for h in map(np.asarray, one_particle)]
        self.tensors, self.filled = response, int(filled)
        self.blocks = len(self.one_particle)
        self.weight = 1.0 / self.blocks
        self.counts = [len(h) for h in self.one_particle]
        self.complex = any(np.iscomplexobj(h) for h in self.one_particle)

    def response(self, D):
        """The part of the Fock operators that is linear in the covariances."""
        return [self.weight * sum(np.tensordot(self.tensors[k][other], D[other], axes=2)
                                  for other in range(self.blocks)) for k in range(self.blocks)]

    def fock(self, P):
        return [h + g for h, g in zip(self.one_particle, self.response(P))]

    def energy(self, frames):
        P = [C @ C.conj().T for C in frames]
        return float(self.weight * sum(np.trace((h + F) @ p).real for h, F, p in zip(self.one_particle, self.fock(P), P)))

    def certificate(self, frames):
        """The state as the library's `CovarianceState` on every mode of the
        span with its two sheets: the purity defect and the particle number,
        and the largest entry of [F, P], which vanishes at a stationary state."""
        frame = np.zeros((2 * sum(self.counts), 2 * self.filled * self.blocks), dtype=complex)
        row = column = 0
        for _ in range(2):
            for C in frames:
                frame[row:row + C.shape[0], column:column + C.shape[1]] = C
                row, column = row + C.shape[0], column + C.shape[1]
        state = quantum.CovarianceState.fromSlaterFrame(frame)
        P = [C @ C.conj().T for C in frames]
        return {"purity_defect": float(state.purityDefect()), "particles": float(state.particleNumber().real),
                "commutator": max(float(np.abs(F @ p - p @ F).max()) for F, p in zip(self.fock(P), P))}

    # -- the first and second variations

    @staticmethod
    def _complete(frame):
        """A unitary basis of the span whose first columns span the frame."""
        q, _ = np.linalg.qr(np.hstack([frame, np.eye(frame.shape[0])]))
        return q[:, :frame.shape[0]]

    def _split(self, basis, matrix):
        rotated = basis.conj().T @ matrix @ basis
        n = self.filled
        return rotated[:n, :n], rotated[n:, :n], rotated[n:, n:]

    def _at(self, frames):
        bases = [self._complete(C) for C in frames]
        focks = self.fock([C @ C.conj().T for C in frames])
        return bases, [self._split(basis, F) for basis, F in zip(bases, focks)]

    def gradient(self, frames):
        """dE / d kappa_k (empty by filled): 4 F_k[empty, filled] / N_k."""
        return [4.0 * self.weight * split[1] for split in self._at(frames)[1]]

    def hessian_product(self, bases, splits, kappas):
        """The second variation of the energy applied to the rotations
        `kappas`: (4 / N_k) (F_ee kappa - kappa F_ff + G[D]_ef), D the
        first-order change of the covariances, [[0, kappa^dagger], [kappa, 0]]
        in the bases whose first columns are the frames."""
        n, D = self.filled, []
        for basis, kappa in zip(bases, kappas):
            first = np.zeros((basis.shape[0],) * 2, dtype=kappa.dtype)
            first[n:, :n], first[:n, n:] = kappa, kappa.conj().T
            D.append(basis @ first @ basis.conj().T)
        G = self.response(D)
        return [4.0 * self.weight * (F_ee @ kappa - kappa @ F_ff + self._split(basis, g)[1])
                for basis, (F_ff, _, F_ee), kappa, g in zip(bases, splits, kappas, G)]

    # The rotations of every momentum as one real vector, for conjugate gradients and the eigenvalue solver.

    def _pack(self, kappas):
        flat = np.concatenate([kappa.ravel() for kappa in kappas])
        return np.concatenate([flat.real, flat.imag]) if self.complex else flat.real

    def _unpack(self, vector):
        if self.complex:
            half = len(vector) // 2
            vector = vector[:half] + 1j * vector[half:]
        kappas, start = [], 0
        for count in self.counts:
            shape = (count - self.filled, self.filled)
            kappas.append(vector[start:start + shape[0] * shape[1]].reshape(shape))
            start += shape[0] * shape[1]
        return kappas

    def hessian(self, frames):
        """The matrix of the second variation on the packed rotations."""
        bases, splits = self._at(frames)
        size = len(self._pack(self._unpack(np.zeros(2 * sum((c - self.filled) * self.filled for c in self.counts)))))
        columns = [self._pack(self.hessian_product(bases, splits, self._unpack(unit))) for unit in np.eye(size)]
        return np.array(columns).T

    def lowest_curvature(self, frames):
        """The lowest eigenvalue of the second variation at the frames."""
        bases, splits = self._at(frames)
        size = (2 if self.complex else 1) * sum((c - self.filled) * self.filled for c in self.counts)
        if size == 0:
            return np.inf
        if size <= 64:
            return float(np.linalg.eigvalsh(self.hessian(frames))[0])
        operator = LinearOperator((size, size), dtype=float, matvec=lambda v: self._pack(
            self.hessian_product(bases, splits, self._unpack(np.asarray(v).ravel()))))
        return float(eigsh(operator, k=1, which="SA", tol=1e-8, return_eigenvectors=False)[0])

    # -- the minimization

    def minimize(self, frames, tolerance=1e-10, max_steps=200, radius=0.5):
        """Newton steps on the Slater frames inside a trust region, the step of
        each found by conjugate gradients on the second variation (Steihaug's
        truncation at the region's boundary or at a direction of negative
        curvature), the frames moved by the exponential of the rotation. A step
        is kept only if the energy falls; once the predicted fall is below the
        resolution of the energy the step is the Newton step of a positive
        second variation and is kept on that. Returns the frames and a read:
        the energy before and after, the steps taken and the largest entry of
        the gradient left."""
        n = self.filled
        energy = start = self.energy(frames)
        steps, gradient_norm = 0, np.inf
        for steps in range(max_steps + 1):
            bases, splits = self._at(frames)
            gradient = self._pack([4.0 * self.weight * split[1] for split in splits])
            gradient_norm = float(np.abs(gradient).max()) if len(gradient) else 0.0
            if gradient_norm < tolerance or steps == max_steps:
                break
            product = lambda v: self._pack(self.hessian_product(bases, splits, self._unpack(v)))
            while True:
                step, predicted, interior = _steihaug(product, gradient, radius)
                moved = []
                for basis, kappa in zip(bases, self._unpack(step)):
                    X = np.zeros((basis.shape[0],) * 2, dtype=kappa.dtype)
                    X[n:, :n], X[:n, n:] = kappa, -kappa.conj().T
                    moved.append((basis @ expm(X))[:, :n])
                change = self.energy(moved) - energy
                resolved = abs(predicted) > 64.0 * np.finfo(float).eps * max(1.0, abs(energy))
                if change < 0.0 or (not resolved and interior) or radius < 1e-12:
                    break
                radius *= 0.25
            if change >= 0.0 and (resolved or not interior):
                break                                            # no descent left
            frames, energy = moved, min(energy, energy + change)
            if resolved:
                ratio = change / predicted
                if ratio > 0.75 and not interior:
                    radius = min(2.0 * radius, 1.5)
                elif ratio < 0.25:
                    radius *= 0.25
        return frames, {"energy_before": start, "energy": energy, "steps": steps, "gradient": gradient_norm}


def _steihaug(product, gradient, radius, tolerance=1e-3, max_iterations=200):
    """min g.k + k.H.k / 2 within |k| <= radius by conjugate gradients, stopped
    at the boundary or at a direction of negative curvature; returns the step,
    the predicted change of the energy, and whether the step is interior."""
    step = np.zeros_like(gradient)
    residual = gradient.copy()
    direction = -residual
    stop = tolerance * np.linalg.norm(gradient)

    def to_boundary(step, direction):
        a, b, c = direction @ direction, 2.0 * (step @ direction), step @ step - radius ** 2
        return step + (-b + np.sqrt(b * b - 4.0 * a * c)) / (2.0 * a) * direction

    def predicted(step):
        return float(gradient @ step + 0.5 * (step @ product(step)))

    for _ in range(max_iterations):
        curved = product(direction)
        curvature = direction @ curved
        if curvature <= 0.0:
            step = to_boundary(step, direction)
            return step, predicted(step), False
        alpha = (residual @ residual) / curvature
        if np.linalg.norm(step + alpha * direction) >= radius:
            step = to_boundary(step, direction)
            return step, predicted(step), False
        step = step + alpha * direction
        new_residual = residual + alpha * curved
        if np.linalg.norm(new_residual) < stop:
            break
        direction = -new_residual + (new_residual @ new_residual) / (residual @ residual) * direction
        residual = new_residual
    return step, predicted(step), True


def orthonormal(mass, orbitals):
    """The same span with M-orthonormal columns, each a combination of the
    columns before it (a start interpolated from another mesh is not
    orthonormal on this one)."""
    gram = orbitals.conj().T @ (mass @ orbitals)
    return orbitals @ np.linalg.inv(np.linalg.cholesky(0.5 * (gram + gram.conj().T))).conj().T


def extended_span(mass, orbitals, frames, tolerance=1e-14):
    """`orbitals` followed by an M-orthonormal basis of what the earlier filled
    `frames` have outside their span; directions whose squared norm outside the
    span is below `tolerance` are already in it."""
    if not frames:
        return orbitals
    extra = np.hstack(frames)
    for _ in range(2):
        extra = extra - orbitals @ (orbitals.conj().T @ (mass @ extra))
    values, vectors = np.linalg.eigh(extra.conj().T @ (mass @ extra))
    keep = values > tolerance
    if not keep.any():
        return orbitals
    extra = extra @ (vectors[:, keep] / np.sqrt(values[keep]))
    extra = extra - orbitals @ (orbitals.conj().T @ (mass @ extra))
    return np.hstack([orbitals, orthonormal(mass, extra)])


class ExchangeAccelerator:
    """The state of the accelerator over the updates of one run: the filled
    frames of the last `history` updates, which join the span of the next, and
    the reads of every minimization."""

    def __init__(self, mesh, filled, history):
        self.mesh, self.filled, self.history = mesh, int(filled), int(history)
        self.frames, self.reads = [], []
        self.last = None

    def _one_particle(self, pencil, mass_projectors, modes):
        overlap = modes.conj().T @ mass_projectors
        return modes.conj().T @ (pencil @ modes) + overlap @ self.mesh.D @ overlap.conj().T

    def _finish(self, mean_field, frames, read):
        self.reads.append(read)
        self.last = (mean_field, frames)
        self.frames = self.frames[len(self.frames) - self.history:] if self.history else []

    def zone_centre(self, orbitals):
        """The rotated span at the zone centre: (orbitals with the minimizing
        frame first, K[frame] applied to them, the load of the frame's density)."""
        mesh, cell = self.mesh, self.mesh.cell
        span = extended_span(mesh.mass, orthonormal(mesh.mass, orbitals), self.frames)
        integrals = SpanIntegrals(mesh.kernel, mesh.cell, span)
        ionic = (mesh.stiffness + cell.weighted_mass(mesh.ionic).dressed().real).tocsc()
        mean_field = SpanMeanField([self._one_particle(ionic, mesh.P, span)],
                                   [[integrals.response_tensor(mesh.zero_momentum)]], self.filled)
        frames, read = mean_field.minimize([np.eye(span.shape[1])[:, :self.filled]])
        rotation = mean_field._complete(frames[0])
        rotated = span @ rotation
        self.frames.append(rotated[:, :self.filled])
        self._finish(mean_field, frames, read)
        frame = rotation[:, :self.filled]
        return rotated, integrals.exchange(frame, mesh.zero_momentum) @ rotation, integrals.density_load(frame)

    def momentum_set(self, momenta, constant, masses, projectors, orbitals):
        """The rotated spans of a momentum set, the minimizing frames first."""
        mesh, cell, count = self.mesh, self.mesh.cell, len(momenta)
        spans = [extended_span(masses[k], orthonormal(masses[k], orbitals[k]), [frames[k] for frames in self.frames])
                 for k in range(count)]
        local = cell.weighted_mass(mesh.ionic)
        one_particle = [self._one_particle(cell.pencil(momenta[k], local)[0], projectors[k], spans[k])
                        for k in range(count)]
        mean_field = SpanMeanField(one_particle, set_integrals(mesh, momenta, constant, spans), self.filled)
        frames, read = mean_field.minimize([np.eye(span.shape[1], dtype=complex)[:, :self.filled] for span in spans])
        rotated = [span @ mean_field._complete(C) for span, C in zip(spans, frames)]
        self.frames.append([z[:, :self.filled] for z in rotated])
        self._finish(mean_field, frames, read)
        return rotated

    def record(self):
        """What a run returns of the accelerator: the energy of the state each
        exchange operator was built from (it falls from one update to the next),
        the Newton steps each minimization took, and the lowest eigenvalue of
        the second variation on the last span."""
        mean_field, frames = self.last
        return {"energies": [read["energy"] for read in self.reads], "span_steps": [read["steps"] for read in self.reads],
                "lowest_curvature": mean_field.lowest_curvature(frames)}
