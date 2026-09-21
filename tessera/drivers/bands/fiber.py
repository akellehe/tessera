# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""A static potential as the phase of the timelike edges of the history complex.

The history of a cell K over one tick is W = K x [0, 1], triangulated by
`Spacetime.prismCells`. Its fiber edges (the edges with a time component) carry
the Euclidean squared length `tau^2` on top of their spatial part, and the
non-compact part of the connection, Im(phi_e) = tau * V_e with V_e the mean of
the potential over the edge: the link read forward in time, from the lower
level to the upper, is `exp(tau V_e)`, and carries a value at the upper level
back to the lower one. Spatial edges keep phi = 0. A constant potential is then a
pure gauge on W, and a varying one has curvature `exp(tau (V(w) - V(v)))` on
the vertical triangles: the electric field.

All vertices of one slab lie on its two levels, so the degree-zero pencil of W
with a mass term, `F = A_0(W) + m^2 M_0(W)`, is already its own boundary
response, in blocks F_00, F_01, F_10, F_11 between the levels. A stack of
identical slabs is stationary at an interior level when

    F_10 u_{j-1} + (F_00 + F_11) u_j + F_01 u_{j+1} = 0,

and a solution u_j = T^j u_0 that decays forward in time is a mode of the tick
map with |T| < 1. The operator on W is second order in time, a Klein-Gordon
operator, so its modes come in pairs T and 1/T and the decay rate
E = -ln(T) / tau of the decaying branch obeys (E - V)^2 = L + m^2: for a mass
large against the spatial levels L and the potential,

    E = m + (L / 2m + V) + O(tau^2) + O(V^2 / m) + O(L^2 / m^3),

the static Hamiltonian with the kinetic scale 1 / 2m, shifted by the rest mass.
That is the statement `tick_levels` measures against `static_levels`.

One condition is particular to the mesh. On a staircase slab the derivative in
time of a piecewise-linear function is constant on each top simplex and equal
to the jump at the one vertex where the simplex changes level, so the time
stiffness is `D / tau` with `D` the lumped (diagonal) mass matrix of the cell,
while the mass term `m^2 M_0(W)` carries the consistent mass matrix `M`. As
tau -> 0 the tick map therefore solves `(A + m^2 M) u = E^2 D u` exactly, and
`M` differs from `D` by a term of relative size h^2 L on a mode of spatial
level L. Multiplied by m^2 that is a change of the kinetic coefficient from
1 / 2m to (1 - c m^2 h^2) / 2m: the reduction to the static Hamiltonian needs
the mesh to resolve the Compton wavelength, m h << 1, on top of L << m^2. A
cell of six divisions cannot meet both. `mass_term="lumped"` adds the mass term
with the same lumping as the time stiffness, `m^2 D`, for which the tick map
solves `(A + m^2 D) u = E^2 D u` and the reduction holds at any mesh spacing,
against the static route with the lumped mass matrix; `"consistent"` is the
literal `m^2 M_0(W)`.
"""
import numpy as np
import scipy.linalg

import tessera
from tessera import chainhodge as ch
from tessera import cobordism as cob


class HistorySlab:
    """One tick of the history of the periodic cell `cell` (a `CrystalCell`)."""

    def __init__(self, cell, tau, potential=None, mass=1.0, mass_term="lumped"):
        if mass_term not in ("lumped", "consistent"):
            raise ValueError("mass_term is 'lumped' or 'consistent'")
        self.cell, self.tau, self.mass = cell, float(tau), float(mass)
        n = cell.size
        self.cells = tessera.Spacetime.prismCells(cell.grid.cells(), 1)
        self.complex = cob.ChainComplex.fromTopCells(self.cells)
        edges = self.complex.kSimplexVertices(1)
        V = np.zeros(n) if potential is None else np.asarray(potential, dtype=float)
        squared, links = [], []
        for x, y in edges:
            level_x, level_y = x // n, y // n
            a, b = x % n, y % n
            spatial = 0.0 if a == b else cell.grid.squaredLength(a, b)
            timelike = level_x != level_y
            squared.append(complex(spatial + (self.tau ** 2 if timelike else 0.0)))
            # The canonical orientation runs forward in time (the lower level has
            # the smaller id). The sign is fixed by the constant potential, which
            # must raise every decay rate by exactly V.
            links.append(complex(np.exp(self.tau * 0.5 * (V[a] + V[b]))) if timelike else 1.0 + 0.0j)
        self.base = ch.ChainHodge(self.complex, squared, ch.Preset.L2, ch.Branch.Continuation,
                                  max(512, 2 * n + 1))
        self.connection = ch.Connection(self.complex, links)
        self.operator = ch.CovariantChainHodge(self.base, self.connection, 7, False)
        pencil = self.operator.pencil(0)
        if mass_term == "consistent":
            weight = pencil.B
        else:
            # Row sums of the undressed mass matrix: the lumped mass, whose
            # diagonal the connection does not dress.
            weight = np.diag(np.asarray(self.base.Minv(0).sum(axis=1)).ravel())
        F = pencil.A + self.mass ** 2 * weight
        self.blocks = (F[:n, :n], F[:n, n:], F[n:, :n], F[n:, n:])

    def tick_levels(self, count):
        """The `count` smallest decay rates E = -ln(T) / tau of the decaying
        branch of the stacked slab, from the linearized quadratic eigenproblem."""
        F00, F01, F10, F11 = self.blocks
        n = F00.shape[0]
        zero, identity = np.zeros((n, n)), np.eye(n)
        # F10 u + (F00 + F11) T u + F01 T^2 u = 0 as a linear pencil in (u, T u).
        left = np.block([[zero, identity], [-F10, -(F00 + F11)]])
        right = np.block([[identity, zero], [zero, F01]])
        T = scipy.linalg.eigvals(left, right)
        T = T[np.isfinite(T)]
        decaying = T[(np.abs(T) < 1.0) & (np.abs(T) > 0.0)]
        rates = -np.log(decaying) / self.tau
        order = np.argsort(rates.real)
        return rates[order][:count]

    def max_curvature(self):
        """max |F_t - 1| over the triangles: zero iff the potential is a pure gauge."""
        return max(abs(self.connection.curvature(*t) - 1.0) for t in self.complex.kSimplexVertices(2))


def static_levels(cell, potential, mass, count, mass_term="lumped"):
    """The lowest levels of the static route with the kinetic scale 1 / 2m,
    L / 2m + V on the cell, with the lumped or the consistent mass matrix
    (dense; the fixtures here are small)."""
    A = cell.stiffness.dressed().toarray().real / (2.0 * mass)
    M = cell.mass.dressed().toarray().real
    if mass_term == "lumped":
        M = np.diag(M.sum(axis=1))
        if potential is not None:
            A = A + M * np.asarray(potential, dtype=float)[:, None]
    elif potential is not None:
        A = A + cell.weighted_mass(potential).dressed().toarray().real
    return scipy.linalg.eigh(A, M, eigvals_only=True)[:count]


def klein_gordon_levels(cell, mass, count, mass_term="lumped"):
    """sqrt of the levels of (A + m^2 W, D) with D the lumped mass matrix and W
    the lumped or consistent one: the exact tau -> 0 limit of the free tick map."""
    A = cell.stiffness.dressed().toarray().real
    M = cell.mass.dressed().toarray().real
    D = np.diag(M.sum(axis=1))
    W = D if mass_term == "lumped" else M
    return np.sqrt(scipy.linalg.eigh(A + mass ** 2 * W, D, eigvals_only=True))[:count]
