# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Machinery shared by the scaling verification plan's test modules (#1208).

The plan's test families and generators (F1-F7), its tolerance policy
(tau = kappa n eps_m cond(G_1), kappa = 10), its reporting format (one JSON
record per test instance), and the dense Python oracles the C++ path is
measured against.

The generators here are the plan's own, at the parameters its tables were
produced with: the flat jittered torus F1 (seed 1), the flat cylinder F2
(seed 2) and the conformally flat torus F3 (seed 1) live in `_fixtures`, and
this module adds the period-ratio-2 torus of the plan's section 7 (the G5 and
G6 fixture), the spectral-statistics ensembles F4 and F5, and the F7 staircase
prisms. Every published angle of the plan's tables G1, G2, G3, G5 and G6 is
reproduced by these generators to the digits it is printed with.
"""
import itertools
import json
import math
import os
import resource
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.linalg import subspace_angles

from tessera import chainhodge as ch
from tessera import cobordism as cob
from tests.chainhodge._fixtures import edges, torus_cells

KS = ch.Branch.KontsevichSegal

# Tolerance policy (plan, "Test catalogue"): relative residuals are compared
# against tau = kappa n eps_m cond(G_1) with kappa = 10.
KAPPA = 10.0
EPS_M = float(np.finfo(float).eps)


def tolerance(n, cond_G1, kappa=KAPPA):
    """tau = kappa n eps_m cond(G_1), the plan's relative residual tolerance."""
    return kappa * n * EPS_M * cond_G1


def condition_number(hodge, k=1):
    """cond(G_k) in the 2-norm. G_k = M_k^{-1} under the Whitney preset, so its
    condition number is cond(M_k); under GRASSMANN_ALL the sparse object is
    G_k itself."""
    try:
        A = hodge.Minv(k).toarray()
    except Exception:                       # GRASSMANN_ALL: the metric is sparse
        A = hodge.chainMetricSparse(k).toarray()
    sv = np.linalg.svd(A, compute_uv=False)
    return float(sv[0] / sv[-1])


_BETTI = {}


def betti_numbers(hodge):
    """The Betti numbers of an instance, memoized on the combinatorics. They
    come from exact integer ranks and cost 12 s on the largest mesh of the
    plan's tables, where every row of a table is a different geometry on the
    same complex."""
    K = hodge.complex()
    key = (K.dimension(), tuple(tuple(int(v) for v in t) for t in K.orientedTopSimplices()))
    if key not in _BETTI:
        _BETTI[key] = list(hodge.betti())
    return list(_BETTI[key])


def rank_report(hodge, k=1, limit=301):
    """The rank conditions (R1)-(R4) of an instance for its record, measured
    only up to `limit` cells: above that the measurement costs more than the
    record is worth here (7.5 s at 432 edges, 13 s at 1728), and the rank
    conditions at scale and their trend with N are the subject of #1204 and
    #1206. Returns None when it was not measured."""
    return hodge.rankConditions(k) if hodge.size(k) <= limit else None


def angles_deg(A, B):
    """Principal angles in degrees between span(A) and span(B), descending."""
    return np.sort(np.degrees(subspace_angles(np.asarray(A), np.asarray(B))))[::-1]


def convergence(sizes, values):
    """The estimated order of convergence of `values` against mesh sizes
    `sizes`: the least-squares slope of log(value) against log(N) over the
    whole sequence ("fit"), the order from each successive doubling
    (log2 of the ratio), and whether the sequence decreases monotonically."""
    sizes = [int(N) for N in sizes]
    values = [float(v) for v in values]
    index = {N: i for i, N in enumerate(sizes)}
    doublings = {f"{N}->{2 * N}": float(np.log2(values[i] / values[index[2 * N]]))
                 for i, N in enumerate(sizes) if 2 * N in index}
    fit = float(-np.polyfit(np.log(sizes), np.log(values), 1)[0])
    return {"fit": fit, "doublings": doublings,
            "monotone": all(a > b for a, b in zip(values, values[1:]))}


# --------------------------------------------------------------------------
# Generators (the plan's "Test families and generators")
# --------------------------------------------------------------------------
def ratio_torus(N, epsilon=None, amp=0.3, jitter=0.15, seed=1, Lt=1.0, Lx=2.0):
    """F3 on a torus with period ratio L_x / L_t = 2 (the plan's section 7
    fixture, used by G5 and G6): vertices of the N x N staircase lattice
    jittered on [0, L_t) x [0, L_x), conformal factor
    phi = amp sin(2 pi t / L_t) cos(2 pi x / L_x) at the edge midpoint, and

        s_e = e^{2 phi} (dx^2 + dt^2)                     (epsilon is None)
        s_e = e^{2 phi} (dx^2 - e^{-2 i epsilon} dt^2)    (otherwise),

    the Kontsevich-Segal rotation of the timelike part of every squared length.
    With unequal periods no lattice diagonal is near null; with equal periods
    the plan measured |s_e| ~ 1e-6 on the diagonals. `amp` may be complex (G6).
    Returns (K, s, W, coords, cells) with W the continuum harmonic edge
    integrals of (dt, dx), `coords` the vertex coordinates and `cells` the
    counter-clockwise top cells (both for the deficit angles of G4)."""
    rng = np.random.default_rng(seed)
    cells, vid = torus_cells(N)
    K = cob.ChainComplex.fromTopCells(cells)
    period = np.array([Lt, Lx])
    coords = {vid(i, j): np.array([(i + jitter * rng.uniform(-1, 1)) * Lt / N,
                                   (j + jitter * rng.uniform(-1, 1)) * Lx / N])
              for i in range(N) for j in range(N)}

    def phi(p):
        return amp * np.sin(2 * np.pi * p[0] / Lt) * np.cos(2 * np.pi * p[1] / Lx)

    rot = None if epsilon is None else np.exp(-2j * epsilon)
    s, W = [], []
    for (a, b) in edges(K):
        d = coords[b] - coords[a]
        d -= period * np.round(d / period)
        q = (d[0] ** 2 + d[1] ** 2) if rot is None else (d[1] ** 2 - rot * d[0] ** 2)
        s.append(complex(np.exp(2 * phi(coords[a] + 0.5 * d)) * q))
        W.append(d.copy())
    return K, s, np.array(W, dtype=complex), coords, _counter_clockwise(N)


def _counter_clockwise(N):
    """The N x N staircase torus cells oriented counter-clockwise in the
    (t, x) plane (the orientation the holonomy of `deficit_angles` walks)."""
    def vid(i, j):
        return (i % N) * N + (j % N)

    out = []
    for i in range(N):
        for j in range(N):
            out.append((vid(i, j), vid(i + 1, j), vid(i + 1, j + 1)))
            out.append((vid(i, j), vid(i + 1, j + 1), vid(i, j + 1)))
    return out


def cylinder_cells(N, L):
    """The counter-clockwise cells of the flat cylinder F2 of `_fixtures`."""
    def vid(i, j):
        return i * N + (j % N)

    out = []
    for i in range(L):
        for j in range(N):
            out.append((vid(i, j), vid(i + 1, j), vid(i + 1, j + 1)))
            out.append((vid(i, j), vid(i + 1, j + 1), vid(i, j + 1)))
    return out


def cdt_like_torus(N, rng):
    """F4: regular N x N torus combinatorics with slice edges
    s ~ U(0.5, 1.5), transverse (timelike) edges s ~ -U(0.1, 0.9) and diagonals
    s ~ U(0.2, 1.5). The vertex index i is the time step, j the slice position,
    so an edge with equal i is a slice edge, one with equal j is transverse and
    the rest are diagonals. No continuum target: used for spectral statistics."""
    cells, _ = torus_cells(N)
    K = cob.ChainComplex.fromTopCells(cells)
    s = []
    for (a, b) in edges(K):
        di, dj = (b // N - a // N) % N, (b % N - a % N) % N
        if di == 0:
            s.append(complex(rng.uniform(0.5, 1.5)))
        elif dj == 0:
            s.append(complex(-rng.uniform(0.1, 0.9)))
        else:
            s.append(complex(rng.uniform(0.2, 1.5)))
    return K, s


def random_sign_torus(N, rng):
    """F5: the same combinatorics with s_e ~ N(0, 1) i.i.d. -- incoherent
    causal structure."""
    cells, _ = torus_cells(N)
    K = cob.ChainComplex.fromTopCells(cells)
    return K, [complex(rng.normal()) for _ in edges(K)]


def prism_complex(base_cells, layers):
    """The tessera staircase prism of a base complex (F7): the dimension-generic
    product triangulation of base x [0, layers], with layer l holding vertex
    v + l * stride."""
    import tessera
    cells = tessera.Spacetime.prismCells([list(c) for c in base_cells], layers)
    return cob.ChainComplex.fromTopCells([list(c) for c in cells]), len(cells)


def prism_geometry(K, base_coords, N, layers, stride, lorentz=False, jitter=0.15,
                   seed=3, height=None, periods=(1.0, 2.0)):
    """Squared lengths and continuum harmonic integrals of an F7 prism whose
    base carries `base_coords` (a dict vertex id -> (t, x)) with periods
    `periods` (None for a direction that does not wrap). Layer l sits at
    z = (l + jitter) * height with the two boundary layers unjittered, and

        s_e = dx^2 + dz^2 + dt^2      (Euclidean)
        s_e = dx^2 + dz^2 - dt^2      (Lorentzian).

    Returns (s, W1, W2): W1 the edge integrals of (dt, dx) and W2 the face
    integrals of dt ^ dx, the flat product's absolute harmonic 1- and 2-forms."""
    rng = np.random.default_rng(seed)
    height = ((periods[0] or 1.0) / N) if height is None else height
    coords = {}
    for l in range(layers + 1):
        z = (l + (jitter * rng.uniform(-1, 1) if 0 < l < layers else 0.0)) * height
        for v, p in base_coords.items():
            coords[v + l * stride] = np.array([p[0], p[1], z])
    period = np.array([periods[0] or math.inf, periods[1] or math.inf, math.inf])

    def delta(a, b):
        d = coords[b] - coords[a]
        with np.errstate(invalid="ignore"):
            d = np.where(np.isfinite(period), d - period * np.round(d / period), d)
        return d

    s, W1 = [], []
    for (a, b) in edges(K):
        d = delta(a, b)
        q = d[1] ** 2 + d[2] ** 2 + (-d[0] ** 2 if lorentz else d[0] ** 2)
        s.append(complex(q))
        W1.append(d[:2])
    W2 = []
    for f in K.kSimplexVertices(2):
        p, q, r = (int(v) for v in f)
        u, w = delta(p, q), delta(p, r)
        W2.append([0.5 * (u[0] * w[1] - u[1] * w[0])])
    return s, np.array(W1, dtype=complex), np.array(W2, dtype=complex)


# --------------------------------------------------------------------------
# G4: deficit angles from the squared lengths alone
# --------------------------------------------------------------------------
def deficit_angles(cells, table, interior_only=True):
    """The Regge deficit at every interior vertex of a 2-complex, from the
    squared lengths alone and in any signature.

    At a vertex v the development of its star around v is the product of the
    complexified rotations taking each edge direction to the next,

        H_v = prod_T (g_uw + i sqrt(det g_T)) / prod_{e in star(v)} s_e,

    with g_uw = (s_vu + s_vw - s_uw)/2 the Gram entry of the wedge (u, w) of
    the triangle T = (v, u, w) taken in the complex's own (counter-clockwise)
    orientation and sqrt(det g_T) the top simplex's one root on the
    Kontsevich-Segal branch. Every edge of the star appears in exactly two
    wedges, so H_v is a ratio of polynomials in s: no root of an edge length
    and no branch choice of an angle enters it. For a flat star H_v = 1
    exactly (in null coordinates each factor telescopes), and the deficit is
    epsilon_v = i log H_v, which for Euclidean data is the ordinary angle
    defect 2 pi - sum theta whenever |epsilon_v| < pi.

    `cells` are counter-clockwise vertex triples (the orientation whose area
    form is dt ^ dx, so that sqrt(det g_T) on the Kontsevich-Segal branch is
    the wedge's area element) and `table` maps a sorted vertex pair to its
    squared length. Returns {vertex: epsilon}."""
    wedges = {}
    star = {}
    for (p, q, r) in cells:
        for (v, u, w) in ((p, q, r), (q, r, p), (r, p, q)):
            wedges.setdefault(v, []).append((u, w))
            star.setdefault(v, set()).update({u, w})

    def s(a, b):
        return complex(table[(min(a, b), max(a, b))])

    out = {}
    for v, pairs in wedges.items():
        if interior_only and len(pairs) != len(star[v]):     # boundary vertex
            continue
        holonomy = 1.0 + 0j
        for (u, w) in pairs:
            g_uu, g_ww, g_uw = s(v, u), s(v, w), 0.5 * (s(v, u) + s(v, w) - s(u, w))
            gram = np.array([[g_uu, g_uw], [g_uw, g_ww]], dtype=complex)
            holonomy *= (g_uw + 2j * volume_on_branch(gram, 2))   # g_uw + i sqrt(det g_T)
        for u in star[v]:
            holonomy /= s(v, u)
        out[v] = complex(1j * np.log(holonomy))
    return out


def euclidean_deficit_angles(cells, table):
    """The ordinary angle defect 2 pi - sum(theta) at every interior vertex,
    by the law of cosines. Real Euclidean data only; the independent reading
    of `deficit_angles`."""
    wedges, star = {}, {}
    for (p, q, r) in cells:
        for (v, u, w) in ((p, q, r), (q, r, p), (r, p, q)):
            wedges.setdefault(v, []).append((u, w))
            star.setdefault(v, set()).update({u, w})
    out = {}
    for v, pairs in wedges.items():
        if len(pairs) != len(star[v]):
            continue
        total = 0.0
        for (u, w) in pairs:
            a, b = table[(min(v, u), max(v, u))].real, table[(min(v, w), max(v, w))].real
            c = table[(min(u, w), max(u, w))].real
            total += math.acos((a + b - c) / (2.0 * math.sqrt(a * b)))
        out[v] = 2.0 * math.pi - total
    return out


# --------------------------------------------------------------------------
# Dense oracles (the plan's X: Python oracle against C++ on identical inputs)
# --------------------------------------------------------------------------
def blade_dot(table, e, f):
    """<u_e, u_f> for edge vectors u_(a,b) = x_b - x_a, by polarization."""
    def S(a, b):
        return 0.0 if a == b else table[(min(a, b), max(a, b))]
    a, b = e
    c, d = f
    return 0.5 * (S(b, c) + S(a, d) - S(b, d) - S(a, c))


def volume_on_branch(gram, d):
    """|T| = sqrt(det g)/d! on the Kontsevich-Segal branch: the product of the
    principal roots of the eigenvalues of g, with the cut at arg = pi resolved
    to +i (so real Lorentzian data gives i sqrt(|det g|)/d!)."""
    root = 1.0 + 0j
    for lam in np.linalg.eigvals(np.asarray(gram, dtype=complex)):
        arg = float(np.angle(lam))
        if arg <= -math.pi + 1e-9:
            arg += 2.0 * math.pi
        root *= math.sqrt(abs(lam)) * np.exp(0.5j * arg)
    return root / math.factorial(d)


def gram_of(K, s, T):
    """The Gram matrix (g_T)_ij = <u_{v0 vi}, u_{v0 vj}> of a top simplex."""
    table = dict(zip(edges(K), s))
    T = tuple(sorted(int(v) for v in T))
    d = len(T) - 1
    return np.array([[blade_dot(table, (T[0], T[i]), (T[0], T[j])) for j in range(1, d + 1)]
                     for i in range(1, d + 1)], dtype=complex)


def whitney_reference(K, s, k):
    """Dense Whitney mass matrix M_k of the specification's oracle, in any
    dimension: per top simplex T,

        (M_k)_{sigma tau} = (k!)^2 sum_{i,j} (-1)^{i+j} I^T_{a_i b_j}
                            det[Gamma_{c_p e_q}],

    with I^T_{ab} = |T| (1 + delta_ab)/((d+1)(d+2)) the integral of
    lambda_a lambda_b, Gamma the inverse Gram extended to lambda_0, and
    c = sigma minus a_i, e = tau minus b_j. The one root per top simplex is on
    the Kontsevich-Segal branch."""
    table = dict(zip(edges(K), s))
    d = K.dimension()
    cells = [tuple(int(v) for v in c) for c in K.kSimplexVertices(k)]
    idx = {c: i for i, c in enumerate(cells)}
    M = np.zeros((len(cells), len(cells)), dtype=complex)
    faces = list(itertools.combinations(range(d + 1), k + 1))
    for T in K.orientedTopSimplices():
        T = tuple(sorted(int(v) for v in T))
        g = np.array([[blade_dot(table, (T[0], T[i]), (T[0], T[j])) for j in range(1, d + 1)]
                      for i in range(1, d + 1)], dtype=complex)
        vol = volume_on_branch(g, d)
        ginv = np.linalg.inv(g)
        Gam = np.zeros((d + 1, d + 1), dtype=complex)
        Gam[1:, 1:] = ginv
        Gam[0, 1:] = -ginv.sum(axis=0)
        Gam[1:, 0] = -ginv.sum(axis=1)
        Gam[0, 0] = ginv.sum()
        unit = vol / ((d + 1) * (d + 2))
        for sigma in faces:
            for tau in faces:
                value = 0j
                for i, ai in enumerate(sigma):
                    for j, bj in enumerate(tau):
                        c = [x for x in sigma if x != ai]
                        e = [x for x in tau if x != bj]
                        minor = np.array([[Gam[p, q] for q in e] for p in c], dtype=complex)
                        value += ((-1) ** (i + j) * unit * (1 + (ai == bj))
                                  * (np.linalg.det(minor) if k else 1.0))
                M[idx[tuple(T[x] for x in sigma)], idx[tuple(T[x] for x in tau)]] += \
                    (math.factorial(k) ** 2) * value
    return M


def quadrature(d):
    """A quadrature rule on the reference d-simplex, exact for quadratic
    integrands (the Whitney inner product is one): the edge midpoints for
    d = 2, and the four-point Keast rule for d = 3. Returns barycentric points
    and weights summing to one."""
    if d == 2:
        return np.array([[0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]]), np.full(3, 1 / 3)
    if d == 3:
        a, b = (5.0 + 3.0 * math.sqrt(5.0)) / 20.0, (5.0 - math.sqrt(5.0)) / 20.0
        points = np.full((4, 4), b)
        np.fill_diagonal(points, a)
        return points, np.full(4, 0.25)
    raise ValueError(f"no quadrature rule for d = {d}")


def simplex_mass_from_coordinates(points, metric, k):
    """The Whitney mass matrix of one d-simplex embedded in flat coordinates,
    integrated directly: the barycentric gradients are the rows of the inverse
    edge matrix, the inner product of k-forms is the determinant of the
    pairwise inner products under the inverse ambient metric, and the integral
    is a quadrature rule exact for the quadratic integrand. Nothing here uses a
    Gram matrix of squared lengths, a branch or the closed forms of the
    library: it is the definition of the Whitney mass matrix, read off the
    geometry.

    The value is the real (coordinate) integral, with the coordinate volume
    |det E|/d!; on a Lorentzian embedding the library's |T| = sqrt(det g)/d! is
    i times that, so the comparison carries the factor explicitly.

    `points` are the d+1 vertices in ambient coordinates (the local order the
    k-faces are indexed lexicographically in) and `metric` the ambient metric
    (the identity, or diag(-1, 1, ..., 1))."""
    points = np.asarray(points, dtype=float)
    metric = np.asarray(metric, dtype=float)
    d = points.shape[0] - 1
    E = points[1:] - points[0]
    Einv = np.linalg.inv(E)
    dlam = [None] * (d + 1)
    for i in range(1, d + 1):
        dlam[i] = Einv[:, i - 1]
    dlam[0] = -sum(dlam[1:])
    inverse = np.linalg.inv(metric)
    inner = np.array([[float(a @ inverse @ b) for b in dlam] for a in dlam])
    volume = abs(np.linalg.det(E)) / math.factorial(d)
    nodes, weights = quadrature(d)
    faces = list(itertools.combinations(range(d + 1), k + 1))
    M = np.zeros((len(faces), len(faces)))
    for fi, sigma in enumerate(faces):
        for fj, tau in enumerate(faces):
            total = 0.0
            for node, weight in zip(nodes, weights):
                value = 0.0
                for i, ai in enumerate(sigma):
                    for j, bj in enumerate(tau):
                        c = [x for x in sigma if x != ai]
                        e = [x for x in tau if x != bj]
                        minor = inner[np.ix_(c, e)] if k else np.zeros((0, 0))
                        value += ((-1) ** (i + j) * node[ai] * node[bj]
                                  * float(np.linalg.det(minor)))
                total += weight * value
            M[fi, fj] = (math.factorial(k) ** 2) * volume * total
    return M, faces


def grassmann_reference(K, s, k):
    """Dense port of the specification oracle's metric(): the number of
    simplices containing both faces times their blade pairing."""
    table = dict(zip(edges(K), s))
    cells = [tuple(int(v) for v in c) for c in K.kSimplexVertices(k)]
    idx = {c: i for i, c in enumerate(cells)}
    n = len(cells)
    Gam = np.zeros((n, n), dtype=complex)
    mult = np.zeros((n, n))
    seen = set()

    def blade(sig, tau):
        if k == 0:
            return 1.0 + 0j
        A = np.array([[blade_dot(table, (sig[0], sig[i]), (tau[0], tau[j]))
                       for j in range(1, k + 1)] for i in range(1, k + 1)], dtype=complex)
        return np.linalg.det(A) / (math.factorial(k) ** 2)

    for kk in range(k, K.dimension() + 1):
        for rho in K.kSimplexVertices(kk):
            rho = tuple(int(v) for v in rho)
            for a in itertools.combinations(rho, k + 1):
                for b in itertools.combinations(rho, k + 1):
                    i, j = idx[a], idx[b]
                    mult[i, j] += 1.0
                    if (i, j) not in seen:
                        Gam[i, j] = blade(a, b)
                        seen.add((i, j))
    return mult * Gam


def boundary_matrix(K, k):
    """The oracle's own d_k as a dense complex matrix (`integer_boundary`)."""
    return np.array(integer_boundary(K, k), dtype=complex)


def whitney_pencil_reference(K, s, k=1):
    """The dense oracle's auxiliary pencil (A_k~, M_k) on geometric images:
    A_k~ = M_k d_k^T M_{k-1}^{-1} d_k M_k + d_{k+1} M_{k+1} d_{k+1}^T."""
    d = K.dimension()
    M = [whitney_reference(K, s, j) for j in range(d + 1)]
    A = np.zeros_like(M[k])
    if k >= 1:
        B = boundary_matrix(K, k)
        A += M[k] @ B.T @ np.linalg.solve(M[k - 1], B @ M[k])
    if k + 1 <= d:
        B = boundary_matrix(K, k + 1)
        A += B @ M[k + 1] @ B.T
    return A, M[k]


def grassmann_pencil_reference(K, s, k=1):
    """The dense oracle's Grassmann pencil (A_k, G_k) on chains:
    A_k = d_k^T G_{k-1} d_k + G_k d_{k+1} G_{k+1}^{-1} d_{k+1}^T G_k."""
    d = K.dimension()
    G = [grassmann_reference(K, s, j) for j in range(d + 1)]
    A = np.zeros_like(G[k])
    if k >= 1:
        B = boundary_matrix(K, k)
        A += B.T @ G[k - 1] @ B
    if k + 1 <= d:
        B = boundary_matrix(K, k + 1)
        A += G[k] @ B @ np.linalg.solve(G[k + 1], B.T @ G[k])
    return A, G[k]


# --------------------------------------------------------------------------
# Exact rational arithmetic (the plan's X, N <= 6 on rational s_e)
# --------------------------------------------------------------------------
def dyadic_torus(N, lorentz=False, jitter=2, seed=1, scale=8):
    """A flat jittered torus F1 whose squared lengths are exact dyadic
    rationals: the lattice has unit spacing and every vertex is displaced by
    an integer multiple of 1/scale, so s_e = dx^2 -+ dt^2 is a rational with a
    power-of-two denominator and its binary64 value is exact. Returns
    (K, s_float, s_exact) with s_exact a list of `Fraction`."""
    rng = np.random.default_rng(seed)
    cells, vid = torus_cells(N)
    K = cob.ChainComplex.fromTopCells(cells)
    coords = {vid(i, j): (Fraction(int(i * scale + rng.integers(-jitter, jitter + 1)), scale),
                          Fraction(int(j * scale + rng.integers(-jitter, jitter + 1)), scale))
              for i in range(N) for j in range(N)}
    exact = []
    for (a, b) in edges(K):
        d = [coords[b][c] - coords[a][c] for c in (0, 1)]
        d = [x - N * round(Fraction(x, N)) for x in d]
        exact.append(d[1] ** 2 - d[0] ** 2 if lorentz else d[1] ** 2 + d[0] ** 2)
    return K, [complex(float(v)) for v in exact], exact


def rational_grassmann(K, s, k):
    """The Grassmann chain metric G_k in exact rational arithmetic (it is a
    polynomial in the s_e, so no root and no branch enters it)."""
    table = dict(zip(edges(K), s))

    def dot(e, f):
        def S(a, b):
            return Fraction(0) if a == b else table[(min(a, b), max(a, b))]
        a, b = e
        c, d = f
        return Fraction(1, 2) * (S(b, c) + S(a, d) - S(b, d) - S(a, c))

    cells = [tuple(int(v) for v in c) for c in K.kSimplexVertices(k)]
    idx = {c: i for i, c in enumerate(cells)}
    n = len(cells)
    G = [[Fraction(0) for _ in range(n)] for _ in range(n)]
    mult = [[0 for _ in range(n)] for _ in range(n)]
    blades = {}
    for kk in range(k, K.dimension() + 1):
        for rho in K.kSimplexVertices(kk):
            rho = tuple(int(v) for v in rho)
            for a in itertools.combinations(rho, k + 1):
                for b in itertools.combinations(rho, k + 1):
                    i, j = idx[a], idx[b]
                    mult[i][j] += 1
                    if (i, j) not in blades:
                        A = [[dot((a[0], a[p]), (b[0], b[q])) for q in range(1, k + 1)]
                             for p in range(1, k + 1)]
                        blades[(i, j)] = rational_det(A) / math.factorial(k) ** 2
                        G[i][j] = blades[(i, j)]
    return [[G[i][j] * mult[i][j] for j in range(n)] for i in range(n)]


def rational_det(A):
    """Exact determinant of a small matrix of Fractions, by elimination."""
    A = [row[:] for row in A]
    n = len(A)
    det = Fraction(1)
    for c in range(n):
        p = next((r for r in range(c, n) if A[r][c] != 0), None)
        if p is None:
            return Fraction(0)
        if p != c:
            A[c], A[p] = A[p], A[c]
            det = -det
        det *= A[c][c]
        inv = Fraction(1) / A[c][c]
        for r in range(c + 1, n):
            if A[r][c] != 0:
                f = A[r][c] * inv
                for cc in range(c, n):
                    A[r][cc] -= f * A[c][cc]
    return det


def rational_rank(rows):
    """Exact rank over Q of a matrix given as rows of Fractions (or ints)."""
    A = [[Fraction(v) for v in row] for row in rows]
    m, n = len(A), (len(A[0]) if A else 0)
    rank = 0
    for c in range(n):
        p = next((r for r in range(rank, m) if A[r][c] != 0), None)
        if p is None:
            continue
        A[rank], A[p] = A[p], A[rank]
        inv = Fraction(1) / A[rank][c]
        for r in range(rank + 1, m):
            if A[r][c] != 0:
                f = A[r][c] * inv
                for cc in range(c, n):
                    A[r][cc] -= f * A[rank][cc]
        rank += 1
        if rank == m:
            break
    return rank


def integer_boundary(K, k):
    """The integer incidence matrix d_k (n_{k-1} x n_k) of the reference
    orientation (ascending vertex id), built here from the cell lists alone:
    the coefficient of the face that drops the i-th vertex is (-1)^i."""
    cells = [tuple(int(v) for v in c) for c in K.kSimplexVertices(k)]
    faces = {tuple(int(v) for v in f): i for i, f in enumerate(K.kSimplexVertices(k - 1))}
    B = [[0] * len(cells) for _ in range(len(faces))]
    for j, c in enumerate(cells):
        for i in range(len(c)):
            B[faces[c[:i] + c[i + 1:]]][j] = (-1) ** i
    return B


def integer_betti(K):
    """Betti numbers over Q from the exact integer ranks of the incidence
    maps: b_k = n_k - rank(d_k) - rank(d_{k+1}), no floating point anywhere."""
    d = K.dimension()
    ranks = {0: 0, d + 1: 0}
    for k in range(1, d + 1):
        ranks[k] = rational_rank(integer_boundary(K, k))
    return [K.numSimplices(k) - ranks[k] - ranks[k + 1] for k in range(d + 1)]


# --------------------------------------------------------------------------
# Spectral readings (the plan's S series)
# --------------------------------------------------------------------------
def spectrum_summary(eigenvalues, relative=1e-9):
    """Counts of negative, zero, positive and non-real eigenvalues, and
    whether the non-real ones close under conjugation (complex-conjugate
    pairs), at a relative threshold on the largest modulus."""
    ev = np.asarray(eigenvalues, dtype=complex)
    scale = float(np.max(np.abs(ev))) if ev.size else 1.0
    cut = relative * scale
    real = np.abs(ev.imag) <= cut
    complexes = ev[~real]
    paired = all(np.min(np.abs(complexes - np.conj(z))) <= cut for z in complexes)
    return {"negative": int(np.sum(real & (ev.real < -cut))),
            "zero": int(np.sum(real & (np.abs(ev.real) <= cut))),
            "positive": int(np.sum(real & (ev.real > cut))),
            "complex": int(complexes.size),
            "conjugate_pairs": bool(complexes.size and paired),
            "max_abs": scale}


def ep_indicators(spectrum, B, relative=1e-8):
    """The exceptional-point indicator over the spectrum: for each cluster of
    eigenvalues (coalesced within `relative` of the largest modulus) with an
    orthonormal basis Q of its right eigenvectors, sigma_min(Q^T B Q)/||B||.
    It is the normalized complex bilinear restriction of the integration
    specification: zero exactly at an isotropic (exceptional) band, and
    invariant under a unitary change of basis inside the cluster."""
    ev = np.asarray(spectrum.eigenvalues, dtype=complex)
    V = np.asarray(spectrum.vectors, dtype=complex)
    scale = float(np.max(np.abs(ev))) if ev.size else 1.0
    norm = float(np.linalg.norm(B, 2))
    order = np.argsort(ev.real + 1j * ev.imag)
    clusters, current = [], [int(order[0])]
    for a, b in zip(order[:-1], order[1:]):
        if abs(ev[b] - ev[a]) <= relative * scale:
            current.append(int(b))
        else:
            clusters.append(current)
            current = [int(b)]
    clusters.append(current)
    out = []
    for cluster in clusters:
        Q, _ = np.linalg.qr(V[:, cluster])
        sv = np.linalg.svd(Q.T @ B @ Q, compute_uv=False)
        out.extend([float(sv[-1] / norm)] * len(cluster))
    return out


def real_basis(X, tolerance=1e-9):
    """A real orthonormal basis of the span of complex columns X when that span
    is the complexification of a real subspace (real data, so the harmonic
    conditions are real): the leading left singular vectors of [Re X, Im X]."""
    X = np.asarray(X)
    R = np.hstack([X.real, X.imag])
    U, sv, _ = np.linalg.svd(R, full_matrices=False)
    rank = int(np.sum(sv > tolerance * sv[0]))
    return U[:, :rank]


def signature(A, phase=1.0, tolerance=1e-9):
    """The inertia (positive, zero, negative) of a complex symmetric matrix
    that is `phase` times a real symmetric one -- as every M_k is on real
    Lorentzian data, where the global factor of the branch rule is
    phase = i (integration specification section 4.2). The signature of a
    complex symmetric matrix has a meaning only relative to such a factor, so
    the caller states it."""
    A = np.asarray(A, dtype=complex) / phase
    scale = float(np.max(np.abs(A)))
    assert float(np.max(np.abs(A.imag))) <= tolerance * scale, "not a real matrix up to the phase"
    w = np.linalg.eigvalsh(A.real)
    cut = tolerance * float(np.max(np.abs(w)))
    return int(np.sum(w > cut)), int(np.sum(np.abs(w) <= cut)), int(np.sum(w < -cut))


# --------------------------------------------------------------------------
# Reporting format: one JSON record per test instance
# --------------------------------------------------------------------------
class Recorder:
    """The plan's reporting format. One JSON record per test instance, written
    as a line of `records.jsonl` under the session's directory: the pytest
    temporary directory, or `TESSERA_SVP_RECORDS` when that environment
    variable names one. Nothing is written into the repository, and pass/fail
    is always derived by the test from the numbers in the record."""

    FIELDS = ("test", "family", "params", "preset", "n0", "n1", "n2", "betti", "nullity",
              "gap", "rank_conditions", "cond_G1", "tau", "residuals", "angles_deg",
              "spectrum_summary", "ep_indicator_min", "time_s", "memory_MB",
              "oracle_agreement", "published", "notes")

    def __init__(self, directory):
        self.directory = Path(os.environ.get("TESSERA_SVP_RECORDS") or directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "records.jsonl"
        self.records = []
        print(f"[svp] scaling verification plan records: {self.path}")

    def write(self, **fields):
        record = {name: fields.pop(name, None) for name in self.FIELDS}
        record.update(fields)                       # anything else the test measured
        record["memory_MB"] = record["memory_MB"] or round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
        with self.path.open("a") as handle:
            handle.write(json.dumps(record, default=_json_default) + "\n")
        self.records.append(record)
        return record

    def instance(self, test, family, params, hodge, read=None, k=1, started=None,
                 rank_report=None, **extra):
        """A record of one instance: its sizes, Betti numbers, harmonic read,
        rank conditions, cond(G_1) and the tolerance tau it implies."""
        n = [hodge.size(j) if j <= hodge.dimension() else 0 for j in range(3)]
        cond = condition_number(hodge, k)
        fields = {"test": test, "family": family, "params": params,
                  "preset": "L2" if hodge.preset() == ch.Preset.L2 else "GRASSMANN_ALL",
                  "n0": n[0], "n1": n[1], "n2": n[2], "betti": betti_numbers(hodge),
                  "cond_G1": cond, "tau": tolerance(hodge.size(k), cond)}
        if read is not None:
            fields.update({"nullity": read.nullity, "gap": read.gap, "dense": read.dense})
        if rank_report is not None:
            fields["rank_conditions"] = {
                "measured": list(rank_report.measured), "expected": list(rank_report.expected),
                "holds": list(rank_report.holds), "kernel_is_harmonic": rank_report.kernelIsHarmonic,
                "decomposition_holds": rank_report.decompositionHolds}
        if started is not None:
            fields["time_s"] = round(time.time() - started, 3)
        fields.update(extra)
        return self.write(**fields)


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, complex):
        return {"re": value.real, "im": value.imag}
    if isinstance(value, Fraction):
        return [value.numerator, value.denominator]
    return str(value)
