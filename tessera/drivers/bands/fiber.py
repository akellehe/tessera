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
tau -> 0 the tick map therefore solves

    (A + m^2 M) u = E^2 D u

exactly (`klein_gordon_levels`), and that, not the continuum dispersion, is what
the framework computes. `M` differs from `D` by a term of relative size h^2 L on
a mode of spatial level L; multiplied by m^2 it changes the kinetic coefficient
from 1 / 2m to (1 - c m^2 h^2) / 2m, so the reduction to the static Hamiltonian
needs the mesh to resolve the Compton wavelength, m h << 1, on top of L << m^2.

With a potential the limit is (S0 + E S1 + E^2 S2) u = 0 in closed form
(`tick_limit`): the static relativistic problem (A + m^2 M) u = D (E - V)^2 u
plus the curvature of the connection on the vertical triangles, a term per
spatial simplex that depends on its volume, on the potential at its vertices
and on the order of their ids, and on nothing else (`staircase_blocks`).
"""
import numpy as np
import scipy.linalg

import tessera
from tessera import chainhodge as ch
from tessera import cobordism as cob


class HistorySlab:
    """One tick of the history of the periodic cell `cell` (a `CrystalCell`)."""

    def __init__(self, cell, tau, potential=None, mass=1.0):
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
        F = pencil.A + self.mass ** 2 * pencil.B
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


def static_levels(cell, potential, mass, count):
    """The lowest positive levels of the static relativistic problem on the
    cell,

        (A + m^2 M) u = D (E - V)^2 u ,

    a quadratic eigenproblem in E: nothing is expanded in V or in 1 / m. M is
    the Whitney mass matrix, and D the lumped one, which is what the time
    stiffness of a staircase slab produces; the potential sits on the vertices
    as it does on the vertical edges of the slab. Without a potential these are
    the square roots of the levels of (A + m^2 M, D), the exact tau -> 0 limit
    of the free tick map. Dense; the fixtures here are small."""
    A = cell.stiffness.dressed().toarray().real
    M = cell.mass.dressed().toarray().real
    D = M.sum(axis=1)
    if potential is None:
        return np.sqrt(scipy.linalg.eigh(A + mass ** 2 * M, np.diag(D), eigvals_only=True))[:count]
    V = np.asarray(potential, dtype=float)
    n = len(D)
    constant = A + mass ** 2 * M - np.diag(D * V ** 2)
    left = np.block([[np.zeros((n, n)), np.eye(n)], [-constant, -np.diag(2.0 * D * V)]])
    right = np.block([[np.eye(n), np.zeros((n, n))], [np.zeros((n, n)), -np.diag(D)]])
    levels = scipy.linalg.eigvals(left, right)
    levels = levels[np.isfinite(levels)]
    return np.sort(levels[levels.real > 0.0].real)[:count]


def staircase_blocks(potentials):
    """The curvature of the connection on one staircase prism, in closed form.

    `potentials[..., i]` is the potential on the vertices of a spatial simplex
    of dimension d in ascending vertex id, i = 0 .. d: the order in which
    `Spacetime.prismCells` climbs the staircase, and the order that names the
    base vertex b(sigma) = min(sigma) through which `CovariantChainHodge`
    transports. Returns (c0, c1), the blocks over those vertices of what the
    prism adds, per unit volume of the spatial simplex, to S0 and to S1 of
    `tick_limit` beyond -D V^2 and 2 D V. With the differences
    D_a = V_a - V_i and N = 4 (d + 1)(d + 2)(d + 3),

        N c1_ii = 4 (2d + 3 - i) sum_{a<i} D_a ,
        N c1_ij = -2 [ sum_{a<i} (V_a - V_j) + (d + 3 - i)(V_i - V_j) ]             (i < j) ,
        N c0_ii = -V_i N c1_ii - (2d + 3 - 2i) sum_{a<i} D_a^2 - (sum_{a<i} D_a)^2 ,
        N c0_ij = -V_i N c1_ij                                                      (i < j) ,

    both symmetric. They are the first and second order in the tick of the
    time part of the dressed stiffness, sum over the d + 1 simplices of the
    staircase of

        sum_{q != v, r != w} U_{v b} U_{b b'} U_{b' w} int w_vq . w_wr ,   b = min(v, q), b' = min(w, r) ,

    with w_vq = lambda_q d lambda_v - lambda_v d lambda_q the Whitney forms of
    degree one, of which only the time components of d lambda enter: they are
    -1 / tau and 1 / tau on the two ends of the vertical edge of the simplex
    and zero elsewhere, so nothing of the spatial geometry but the volume
    appears. Every entry is a sum of differences of the potential, the
    electric field through the vertical triangles, and vanishes for a constant
    potential; the arithmetic is integer until the last division, so exact
    rational potentials give exact blocks (which is how the test suite holds
    this to the expansion of the Whitney mass matrix)."""
    W = np.asarray(potentials)
    d = W.shape[-1] - 1
    before = np.cumsum(W, axis=-1) - W                           # sum of V_a over a < i
    squares = np.cumsum(W * W, axis=-1) - W * W
    c0 = np.zeros(W.shape + (d + 1,), dtype=W.dtype)
    c1 = np.zeros(W.shape + (d + 1,), dtype=W.dtype)
    for i in range(d + 1):
        Vi = W[..., i]
        linear = before[..., i] - i * Vi                         # sum of D_a
        quadratic = squares[..., i] - 2 * Vi * before[..., i] + i * Vi * Vi         # sum of D_a^2
        c1[..., i, i] = 4 * (2 * d + 3 - i) * linear
        c0[..., i, i] = -Vi * c1[..., i, i] - (2 * d + 3 - 2 * i) * quadratic - linear * linear
        for j in range(i + 1, d + 1):
            Vj = W[..., j]
            c1[..., i, j] = c1[..., j, i] = -2 * ((before[..., i] - i * Vj) + (d + 3 - i) * (Vi - Vj))
            c0[..., i, j] = c0[..., j, i] = -Vi * c1[..., i, j]
    scale = 4 * (d + 1) * (d + 2) * (d + 3)
    return c0 / scale, c1 / scale


def tick_curvature(cell, potential):
    """(C0, C1): `staircase_blocks` assembled over the top simplices of the
    cell with their volumes, the terms by which the limit of the tick map
    differs from the static relativistic problem."""
    V = np.asarray(potential, dtype=float)
    simplices = np.sort(np.asarray(cell.complex.orientedTopSimplices(), dtype=np.int64), axis=1)
    volumes = np.asarray(cell.base.certificate().volumes).real
    rows, columns = simplices[:, :, None], simplices[:, None, :]
    rows, columns = np.broadcast_arrays(rows, columns)
    assembled = []
    for block in staircase_blocks(V[simplices]):
        matrix = np.zeros((cell.size, cell.size))
        np.add.at(matrix, (rows, columns), volumes[:, None, None] * block)
        assembled.append(matrix)
    return tuple(assembled)


def tick_limit(cell, potential, mass):
    """The exact limit of the tick map of `HistorySlab` as the tick tends to
    zero, with the potential on the timelike edges, in closed form: with
    T = exp(-tau E) the quadratic eigenproblem F10 + (F00 + F11) T + F01 T^2 = 0
    becomes

        (S0 + E S1 + E^2 S2) u = 0 ,
        S2 = -D ,   S1 = 2 D V + C1 ,   S0 = A + m^2 M - D V^2 + C0 ,

    with A the stiffness matrix, M the Whitney mass matrix, D the lumped one
    and (C0, C1) the curvature of the connection on the vertical triangles
    (`tick_curvature`), which is all that separates this limit from
    `static_levels` at a finite mesh. The spatial part of the dressed stiffness
    and the mass term reach A and m^2 M with no trace of the potential, because
    their blocks are already of the order of the tick. Returns (S0, S1, S2)."""
    A = cell.stiffness.dressed().toarray().real
    M = cell.mass.dressed().toarray().real
    D = M.sum(axis=1)
    V = np.zeros(cell.size) if potential is None else np.asarray(potential, dtype=float)
    C0, C1 = tick_curvature(cell, V)
    return A + mass ** 2 * M - np.diag(D * V ** 2) + C0, 2.0 * np.diag(D * V) + C1, -np.diag(D)


def tick_limit_from_blocks(cell, potential, mass, tau=1e-2):
    """`tick_limit` read off the blocks of the slab at a small tick, the check
    of the closed form against the framework's own assembly:

        S0 = lim (F00 + F01 + F10 + F11) / tau ,  S1 = lim (F10 - F01) ,  S2 = lim tau (F10 + F01) / 2 .

    Reversing the tick inverts every link, which transposes the pencil
    (property (ii) of `CovariantChainHodge`), so the symmetric parts of the
    three expressions are even in the tick and the antisymmetric parts odd.
    The limit is symmetric; the symmetric parts over `tau` and `tau / 2` with
    one Richardson step leave an error of fourth order in the tick."""
    def limits(step):
        F00, F01, F10, F11 = [np.asarray(block) for block in HistorySlab(cell, step, potential, mass).blocks]
        reads = (F00 + F01 + F10 + F11) / step, F10 - F01, 0.5 * step * (F10 + F01)
        return [0.5 * (read + read.T).real for read in reads]
    coarse, fine = limits(tau), limits(0.5 * tau)
    return tuple((4.0 * y - x) / 3.0 for x, y in zip(coarse, fine))


def tick_limit_levels(cell, potential, mass, count):
    """The lowest positive levels of `tick_limit`."""
    S0, S1, S2 = tick_limit(cell, potential, mass)
    n = len(S0)
    left = np.block([[np.zeros((n, n)), np.eye(n)], [-S0, -S1]])
    right = np.block([[np.eye(n), np.zeros((n, n))], [np.zeros((n, n)), S2]])
    levels = scipy.linalg.eigvals(left, right)
    levels = levels[np.isfinite(levels)]
    return np.sort(levels[levels.real > 0.0].real)[:count]


def klein_gordon_levels(cell, mass, count):
    """`static_levels` without a potential."""
    return static_levels(cell, None, mass, count)


def history_spacetime(cell, tau, potential=None, layers=1):
    """The history K x [0, layers] as a `Spacetime` whose edges carry the
    declared fields: the Euclidean squared lengths through `Edge.setLength` and
    the potential as the non-compact part of the phase through `Edge.setPhase`.
    The stored phase is the connection on the edge's own source-to-target
    orientation, so an edge stored forward in time gets phi = -i tau V_e (link
    exp(tau V_e), as in `HistorySlab`) and one stored backward gets +i tau V_e."""
    n = cell.size
    V = np.zeros(n) if potential is None else np.asarray(potential, dtype=float)
    cells = tessera.Spacetime.prismCells(cell.grid.cells(), layers)
    spacetime = tessera.Spacetime.fromVertexTuples(4, cells, 1.0, 0.0)
    for edge in spacetime.getEdgeList().toVector():
        x, y = edge.getSource().getId(), edge.getTarget().getId()
        a, b = x % n, y % n
        ticks = y // n - x // n
        spatial = 0.0 if a == b else cell.grid.squaredLength(a, b)
        edge.setLength(complex(np.sqrt(spatial + (tau * ticks) ** 2)))
        edge.setPhase(-1j * tau * ticks * 0.5 * (V[a] + V[b]))
    return spacetime


def layered_response(cell, tau, potential, mass, layers=2):
    """The boundary response of the history over `layers` ticks onto its first
    and last levels, by the layer API: `PencilLayer.assemble` on the spacetime
    of `history_spacetime`, and `PencilLayer.boundary_response` at
    lambda = -m^2, which is the pencil A_0(W) + m^2 M_0(W) with the interior
    levels eliminated. Returns (F, assembly residual, solve residual) with F in
    blocks over (first level, last level)."""
    n = cell.size
    spacetime = history_spacetime(cell, tau, potential, layers)
    assembled = cob.PencilLayer.assemble([spacetime])
    outer = list(range(n)) + list(range(layers * n, (layers + 1) * n))
    interface = cob.PencilLayer.cells_within(assembled, 0, outer)
    result = cob.PencilLayer.boundary_response(assembled, 0, interface, -mass ** 2)
    order = np.argsort([int(assembled.complex.kSimplexVertices(0)[i][0]) for i in result.interface])
    F = np.asarray(result.response)[np.ix_(order, order)]
    return F, cob.PencilLayer.assembly_residual(assembled, 0), result.solveResidual


def fiber_edge_stiffness(cell, tau, fiber_phase):
    """The stiffness of the connection on the history slab, evaluated on a
    fiber-edge phase pattern: phi^T (d_2 M_2 d_2^T) phi, the squared norm of the
    curvature d(phi) in the Whitney metric of W.

    `fiber_phase` gives one phase per vertex, the integral of the timelike
    component A_0 dt over the vertical edge above it. Every edge with a time
    component carries the line integral of A_0 dt along it (the mean of its two
    endpoint values, since A_0 is interpolated linearly), and spatial edges
    carry nothing. The curvature of that pattern is grad(A_0) ^ dt, constant on
    every top simplex, so the Whitney interpolation reproduces it exactly and

        phi^T (d_2 M_2 d_2^T) phi = (1 / tau) fiber_phase^T A_0(K) fiber_phase ,

    with A_0(K) = d_1 M_1 d_1^T the spatial stiffness matrix: the stiffness of
    the timelike connection is the spatial stiffness up to the fiber measure,
    which is why eliminating it leaves the Coulomb kernel A_0(K)^+."""
    slab = HistorySlab(cell, tau)
    n = cell.size
    phase = np.asarray(fiber_phase, dtype=float)
    pattern = np.array([0.5 * (phase[x % n] + phase[y % n]) if x // n != y // n else 0.0
                        for x, y in slab.complex.kSimplexVertices(1)])
    boundary = slab.base.boundary(2)
    curvature = boundary.T @ pattern
    return float((curvature @ (slab.base.Minv(2) @ curvature)).real)
