# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""tessera.drivers.root_lattice_spacetime -- the information network of a
qubit network, the root lattice its interactions translate, and the causal
set the order of the interactions makes of it, with one path on all three.

Definitions
-----------
* The network is the one of ``tessera.drivers.entanglement_complex``: n
  qubits named A, B, C, ... (qubit q is the (q + 1)-th letter) in a global
  state, evolved by one SWAP^alpha interaction per time slice t = 1..T on a
  pair drawn by the schedule. S(X) is the von Neumann entropy in nats of the
  one-qubit reduced state of X and I(X:Y) = S(X) + S(Y) - S(XY) the mutual
  information of the pair (X, Y); both are read from the slices the
  simulation records.
* The length of a pair is l(X,Y) = a ln(1 + I_0 / I(X:Y)) when I(X:Y) is
  above the floor, the mode "log1p" of ``entanglement_complex.target_lengths``
  taken unnormalised: a is the scale and I_0 the reference mutual
  information, l falls as I grows and diverges as I -> 0. A pair at or below
  the floor has no edge and no finite length.
* Weights and roots. The local Cartan subalgebra of the network is spanned by
  the Pauli Z of every qubit. Its dual has one coordinate per qubit, and the
  basis vector e_X is the weight of one unit of Z-charge sitting on X. The
  off-diagonal part of the generator of SWAP^alpha on (X, Y) is the pair of
  ladder operators of the single root e_Y - e_X: the interaction translates
  one unit of charge from X to Y or back. The n(n - 1) vectors e_Y - e_X are
  the root system A_{n-1} on the qubit labels, and the lattice they generate,
  the integer vectors m with sum_X m_X = 0, is the root lattice Q.
* Coordinates. Q spans the (n - 1)-dimensional hyperplane of zero sum. The
  figure draws it in the coordinates of the Fourier modes of the cyclic
  order of the qubits: mode k is the pair of vectors cos(2 pi k X / n) and
  sin(2 pi k X / n) over the qubits X, each normalised, and the modes
  k = 1, 2, ... are orthonormal in that hyperplane. The first two are the
  Coxeter plane, in which the cyclic permutation of the qubits is the
  rotation by 2 pi / n and the weights are a regular n-gon; the third is the
  alternating vector (-1)^X when n is even. Two coordinates are exact for
  three qubits and three are exact for four, where Q is the face-centred
  cubic lattice of su(4) and the twelve roots are the vertices of a
  cuboctahedron; for more qubits the drawing is the orthogonal projection
  onto those modes, fixed by the cyclic order of the names and not by any
  fit. For n >= 5 the projected lattice is dense, so only the points within
  two root steps of the origin are drawn, and beyond eight qubits only the
  first shell, the tips of the roots: the second shell has of the order of
  n^4 points.
* Engines. The network runs on the mixed engine of ``entanglement_complex``
  (the global density matrix, up to 14 qubits) or, with ``--state pure``, on
  its pure engine (a state vector, up to 24 qubits, with pure input pairs);
  every entropy and mutual information used here is exact in both.
* Causal order. Every interaction is an event (X, Y, t). An event precedes a
  later one when their pairs share a qubit, and the order is the transitive
  closure of that relation, the causal set of the circuit. Two events on
  disjoint pairs are unrelated: their generators commute. Time is this order
  and nothing else: no edge of the lattice carries a duration. The depth of
  an event is the length of the longest chain of events below it, the
  order's own time coordinate (in causal set theory the proper time between
  two related events scales with the longest chain joining them); the
  schedule's slice index is one linear extension of the order and only lays
  the events out from left to right. The order is stored as a
  ``tessera.quantum.Poset`` whose cover edges are the Hasse diagram.
* Path. With the walk, one unit of charge starts on a chosen qubit at t = 0
  and is carried across every event whose pair contains its current qubit;
  its worldline is the sequence of qubits visited, every step the root
  e_Y - e_X of its event, with the length l(X,Y) of the pair in the state
  after the event. The events it rides form a chain of the causal set. With
  the geodesic, the path is the shortest path between two qubits through the
  edges of the final slice, the metric d of the entanglement complex, and
  its lengths are the final slice's.

Figure
------
Three panels, with the same path on all three: every step carries the same
number and the same colour on each. (1) The information network in a slice:
every qubit a disc of radius S(X) at its Coxeter-plane weight, on a circle
whose radius keeps neighbouring discs apart; an edge for every pair with I
above the floor, coloured by l(X,Y), solid when the pair has interacted and
dashed otherwise; the path's edges in bold. (2) The root lattice in three
coordinates (two with ``--lattice-dims 2``): the roots as arrows from the
origin, the lattice points within two root steps, the positions the charge
can occupy, and the path as root translations composed tip to tail. (3) The
causal set: the Hasse diagram of the events, each at the horizontal position
of its slice and at height equal to its depth, the covers as lines, the
events the path rides numbered and coloured as its steps. With
``--animate`` one frame per slice t = 0..T shows the network in the state
of slice t, the path up to t, and the events up to t.
"""

import argparse
import itertools
import math
from fractions import Fraction

import numpy as np

from tessera.drivers import entanglement_complex as ec

I_MAX = ec.I_MAX
LENGTH_MODE = "log1p"
PATH_MODES = ("walk", "geodesic")
LATTICE_DIMS = (2, 3)
LATTICE_STEPS = 2
#: Beyond this many qubits only the first shell of the lattice is drawn: the
#: second shell of A_{n-1} has of the order of n^4 points.
LATTICE_SHELL_LIMIT = 8


def lattice_steps(n):
    """How many root steps of the lattice the figure shows for n qubits."""
    return LATTICE_STEPS if n <= LATTICE_SHELL_LIMIT else 1


def qubit_index(token, n):
    """The index of a qubit named by its letter (A, B, ...) or its index."""
    token = str(token).strip()
    if token.isdigit():
        q = int(token)
    elif len(token) == 1 and token.isalpha():
        q = ord(token.upper()) - 65
    else:
        raise ValueError("a qubit is a letter A, B, ... or an index 0, 1, ...: %r" % token)
    if not 0 <= q < n:
        raise ValueError("qubit %r is not among the %d qubits" % (token, n))
    return q


# ============================================================ weights and roots

def fourier_basis(n, dims):
    """Orthonormal vectors of the zero-sum hyperplane from the Fourier modes
    of the cyclic order of the n qubits: for k = 1, 2, ... the normalised
    cos(2 pi k X / n) and sin(2 pi k X / n), skipping a vanishing sine.
    Returns the n x dims matrix whose row X holds the coordinates of the
    weight e_X; when the hyperplane has fewer than `dims` dimensions the
    remaining columns are zero."""
    vectors = []
    for k in range(1, n // 2 + 1):
        for f in (np.cos, np.sin):
            v = f(2.0 * math.pi * k * np.arange(n) / n)
            norm = np.linalg.norm(v)
            if norm > 1e-9:
                vectors.append(v / norm)
    W = np.zeros((n, dims))
    for j, v in enumerate(vectors[:dims]):
        W[:, j] = v
    return W


def weights(n, dims=3):
    """The coordinates of the weights e_0, ..., e_{n-1} in the first `dims`
    Fourier modes; exact for n = dims + 1, a projection beyond."""
    return fourier_basis(n, dims)


def coxeter_weights(n):
    """The weights in the Coxeter plane, the first two Fourier modes: the
    regular n-gon of radius sqrt(2 / n)."""
    return weights(n, 2)


def roots(n):
    """The n(n - 1) roots e_Y - e_X as integer vectors (entry +1 at Y, -1 at
    X), each with its ordered pair (X, Y)."""
    out = []
    for X, Y in itertools.permutations(range(n), 2):
        m = np.zeros(n, dtype=int)
        m[Y] += 1
        m[X] -= 1
        out.append(((X, Y), m))
    return out


def lattice_points(n, steps=LATTICE_STEPS):
    """The points of the root lattice within `steps` root steps of the origin:
    distinct integer vectors with zero sum, the origin first."""
    origin = tuple([0] * n)
    shell, frontier = {origin}, {origin}
    vectors = [m for _, m in roots(n)]
    for _ in range(steps):
        frontier = {tuple(np.array(p) + m) for p in frontier for m in vectors}
        shell |= frontier
    return np.array([origin] + sorted(shell - {origin}), dtype=int)


def project(vectors, W):
    """The coordinates sum_X m_X e_X of integer vectors m, with W the weights."""
    return np.asarray(vectors, dtype=float) @ W


# ============================================================ causal order

def events(slices):
    """The interaction events (t, (X, Y)) of the recorded slices, in order."""
    return [(int(sl["t"]), tuple(int(q) for q in sl["pair"]))
            for sl in slices if sl["pair"] is not None]


def causal_order(evts):
    """The strict partial order of the events, the transitive closure of
    "later and sharing a qubit": the boolean matrix P with P[i, j] when event
    i precedes event j, and the covers (i, j), related pairs with no third
    event between them, which form the Hasse diagram."""
    m = len(evts)
    P = np.zeros((m, m), dtype=bool)
    for i, (ti, pi) in enumerate(evts):
        for j in range(i + 1, m):
            tj, pj = evts[j]
            if tj > ti and set(pi) & set(pj):
                P[i, j] = True
    for k in range(m):
        P |= np.outer(P[:, k], P[k, :])
    covers = [(int(i), int(j)) for i, j in zip(*np.nonzero(P))
              if not (P[i, :] & P[:, j]).any()]
    return P, covers


def depths(P):
    """The depth of every event: the length of the longest chain of events
    below it, 0 for an event with nothing in its past."""
    m = len(P)
    d = np.zeros(m, dtype=int)
    for j in range(m):
        below = np.nonzero(P[:, j])[0]
        if len(below):
            d[j] = d[below].max() + 1
    return d


def poset(evts):
    """The causal set as a ``tessera.quantum.Poset``, one node per event and
    one cover edge per link of the Hasse diagram; also P and the covers."""
    from tessera import quantum
    P, covers = causal_order(evts)
    ps = quantum.Poset(len(evts))
    for i, j in covers:
        ps.addCover(i, j)
    return ps, P, covers


# ============================================================ paths

def walk(evts, start):
    """The worldline of one unit of charge that starts on qubit `start` and is
    carried across every event whose pair contains its current qubit: the
    steps (t, X, Y), the charge moving from X to Y at slice t."""
    steps, current = [], int(start)
    for t, (X, Y) in evts:
        if current == X:
            steps.append((t, X, Y))
            current = Y
        elif current == Y:
            steps.append((t, Y, X))
            current = X
    return steps


def geodesic(D, W, start, end):
    """The shortest path from `start` to `end` through the edges W (1 where
    a pair has an edge) with lengths D, as the qubits visited; empty when
    the two are not joined."""
    from scipy.sparse.csgraph import csgraph_from_dense, shortest_path
    graph = csgraph_from_dense(np.where(W > 0, D, np.inf), null_value=np.inf)
    dist, pred = shortest_path(graph, directed=False, indices=int(start),
                               return_predecessors=True)
    if start == end:
        return [int(start)]
    if not np.isfinite(dist[end]):
        return []
    path = [int(end)]
    while path[-1] != start:
        path.append(int(pred[path[-1]]))
    return path[::-1]


def lengths(MI, floor, scale, reference):
    """l(X,Y) = a ln(1 + I_0 / I(X:Y)) for every pair above the floor,
    unnormalised, with the edge indicator W; a pair without an edge has
    length 0 in D and 0 in W."""
    D, W = ec.target_lengths(MI, LENGTH_MODE, floor, scale=scale, reference=reference,
                             normalise=False)
    if D is None:
        D = np.zeros_like(np.asarray(MI, dtype=float))
    return D, W


def displacement(n, steps):
    """The integer lattice vector the steps add up to: e_end - e_start."""
    m = np.zeros(n, dtype=int)
    for _, X, Y in steps:
        m[Y] += 1
        m[X] -= 1
    return m


# ============================================================ analysis

def analyse(result, floor, scale, reference, path_mode, start, end=None, dims=3):
    """Everything the report and the figure need, from the simulation's
    result (``entanglement_complex.simulate`` with time slices)."""
    slices = result["slices"]
    if slices is None:
        raise ValueError("the simulation must record time slices (timesteps > 0)")
    if dims not in LATTICE_DIMS:
        raise ValueError("the lattice is drawn in 2 or 3 coordinates, not %r" % (dims,))
    n, names = len(result["names"]), result["names"]
    T = slices[-1]["t"]
    evts = events(slices)
    ps, P, covers = poset(evts)
    D, W = lengths(slices[-1]["MI"], floor, scale, reference)
    interacted = np.zeros((n, n), dtype=bool)
    event_lengths = []
    for t, (X, Y) in evts:
        interacted[X, Y] = interacted[Y, X] = True
        Dt, Wt = lengths(slices[t]["MI"], floor, scale, reference)
        event_lengths.append(float(Dt[X, Y]) if Wt[X, Y] > 0 else math.inf)
    if path_mode == "walk":
        steps = walk(evts, start)
        times = [e[0] for e in evts]
        step_lengths = [event_lengths[times.index(t)] for t, _, _ in steps]
        visited = [start] + [Y for _, _, Y in steps]
    elif path_mode == "geodesic":
        if end is None:
            end = n - 1
        visited = geodesic(D, W, start, end)
        steps = [(T, X, Y) for X, Y in zip(visited[:-1], visited[1:])]
        step_lengths = [float(D[X, Y]) for _, X, Y in steps]
    else:
        raise ValueError("the path mode must be one of %s" % (PATH_MODES,))
    finish = visited[-1] if visited else start
    direct = geodesic(D, W, start, finish)
    direct_length = sum(float(D[X, Y]) for X, Y in zip(direct[:-1], direct[1:]))
    return {"n": n, "names": names, "T": T, "dims": dims, "weights": weights(n, dims),
            "lattice": lattice_points(n, lattice_steps(n)), "lattice_steps": lattice_steps(n),
            "slices": slices, "state": result.get("state", "mixed"),
            "S": slices[-1]["S"], "MI": slices[-1]["MI"], "D": D, "W": W,
            "interacted": interacted, "events": evts, "event_lengths": event_lengths,
            "poset": ps, "precedes": P, "covers": covers, "depth": depths(P),
            "path_mode": path_mode, "start": start, "end": end, "steps": steps,
            "step_lengths": step_lengths, "visited": visited,
            "displacement": displacement(n, steps), "direct": direct,
            "direct_length": direct_length, "scale": scale, "reference": reference,
            "floor": floor}


def print_report(rep, log=print):
    names, n = rep["names"], rep["n"]
    label = "l(X,Y) = %g ln(1 + %g / I(X:Y))" % (rep["scale"], rep["reference"])
    log("INFORMATION NETWORK (final slice t = %d; entropies in nats; %s; %s global state)"
        % (rep["T"], label, rep["state"]))
    for q in range(n):
        log("  S(%s) = %.6f" % (names[q], rep["S"][q]))
    for i, j in itertools.combinations(range(n), 2):
        if rep["W"][i, j] > 0:
            log("  %s%s: I = %.6f   l = %.6f%s" % (names[i], names[j], rep["MI"][i, j],
                                                   rep["D"][i, j],
                                                   "" if rep["interacted"][i, j] else
                                                   "   (never interacted)"))
    exact = "exact" if n == rep["dims"] + 1 else "a projection"
    log("ROOT LATTICE A_%d: %d roots e_Y - e_X, %d lattice points within %d root step(s); "
        "drawn in %d Fourier coordinates (%s for %d qubits)"
        % (n - 1, n * (n - 1), len(rep["lattice"]), rep["lattice_steps"], rep["dims"], exact,
           n))
    m = len(rep["events"])
    log("CAUSAL ORDER: %d events, %d related pairs of %d, %d covers (Hasse links), "
        "depth up to %d" % (m, int(rep["precedes"].sum()), m * (m - 1) // 2,
                            len(rep["covers"]), int(rep["depth"].max()) if m else 0))
    for k, ((t, (X, Y)), l) in enumerate(zip(rep["events"], rep["event_lengths"])):
        log("  event %3d  t = %3d  (%s,%s)   l after = %s   depth %d   in the future of "
            "%d event(s)" % (k, t, names[X], names[Y],
                             "%.6f" % l if math.isfinite(l) else "inf",
                             rep["depth"][k], int(rep["precedes"][:k, k].sum())))
    if rep["path_mode"] == "walk":
        log("WORLDLINE of a unit of charge from %s" % names[rep["start"]])
    else:
        log("GEODESIC from %s to %s in the final slice" % (names[rep["start"]],
                                                           names[rep["end"]]))
    total = 0.0
    for k, ((t, X, Y), l) in enumerate(zip(rep["steps"], rep["step_lengths"]), 1):
        total += l
        log("  step %2d  t = %3d  %s -> %s   root e_%s - e_%s   l = %s" % (
            k, t, names[X], names[Y], names[Y], names[X],
            "%.6f" % l if math.isfinite(l) else "inf"))
    log("  visited: %s   total length %s" % (
        " -> ".join(names[q] for q in rep["visited"]),
        "%.6f" % total if math.isfinite(total) else "inf"))
    log("  displacement: %s" % (" + ".join(
        "%+d e_%s" % (c, names[q]) for q, c in enumerate(rep["displacement"]) if c) or "0"))
    if rep["direct"] and len(rep["visited"]) > 1:
        log("  shortest path %s -> %s in the final slice: %s   d = %.6f"
            % (names[rep["start"]], names[rep["visited"][-1]],
               " -> ".join(names[q] for q in rep["direct"]), rep["direct_length"]))


# ============================================================ figure

def step_colours(count):
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("plasma")
    return [cmap(x) for x in np.linspace(0.08, 0.85, max(count, 1))]


def _label(k, l):
    return "%d: %s" % (k, "%.2f" % l if math.isfinite(l) else "inf")


def draw(rep, title, save=None, upto=None):
    """The three panels at slice `upto` (the last by default); returns the
    figure."""
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from matplotlib.patches import Circle, Polygon
    from tessera.drivers import baryon_poles as bp

    n, names, W3, dims = rep["n"], rep["names"], rep["weights"], rep["dims"]
    T = rep["T"] if upto is None else int(upto)
    sl = rep["slices"][T]
    S = sl["S"]
    D, Wedge = lengths(sl["MI"], rep["floor"], rep["scale"], rep["reference"])
    shown = [(k, s, l) for k, (s, l) in enumerate(zip(rep["steps"], rep["step_lengths"]), 1)
             if s[0] <= T]
    colours = step_colours(len(rep["steps"]))
    interacted = np.zeros((n, n), dtype=bool)
    for t, (X, Y) in rep["events"]:
        if t <= T:
            interacted[X, Y] = interacted[Y, X] = True
    finite = [l for l in rep["event_lengths"] if math.isfinite(l)] + list(rep["D"][rep["W"] > 0])
    norm = colors.Normalize(vmin=min(finite) if finite else 0.0,
                            vmax=max(finite) if finite else 1.0)
    cmap = plt.get_cmap("viridis")
    palette = ec.CATEGORICAL
    w2 = W3[:, :2]

    fig = plt.figure(figsize=(19, 6.6), facecolor=bp.SURFACE)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.1, 1.0], wspace=0.24,
                            left=0.03, right=0.98, top=0.86, bottom=0.08)

    # (1) the information network
    ax = fig.add_subplot(grid[0])
    bp.style_axis(ax)
    ax.set_aspect("equal")
    ring_radius = max(1.0, 1.5 * max(float(S.max()), 1e-3)
                      / (math.sin(math.pi / n) * math.sqrt(2.0 / n)))
    xy = ring_radius * w2
    for i, j in itertools.combinations(range(n), 2):
        if Wedge[i, j] > 0:
            ax.plot(*xy[[i, j]].T, color=cmap(norm(D[i, j])), linewidth=1.4,
                    linestyle="-" if interacted[i, j] else "--", zorder=1)
    for k, (t, X, Y), l in shown:
        ax.plot(*xy[[X, Y]].T, color=colours[k - 1], linewidth=3.2, zorder=2)
        ax.annotate(_label(k, l), xy[[X, Y]].mean(axis=0), ha="center", va="center",
                    fontsize=8, color=bp.INK,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor=bp.SURFACE,
                              edgecolor=colours[k - 1]), zorder=4)
    for q in range(n):
        ax.add_patch(Circle(xy[q], radius=float(S[q]), facecolor=ec.FILL,
                            edgecolor=palette[q % len(palette)], linewidth=1.5, zorder=3))
        ax.annotate(names[q], xy[q], ha="center", va="center", color=bp.INK, zorder=5)
    lim = float(np.abs(xy).max()) + float(S.max()) + 0.3
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_title("information network, slice %d\ndisc radius S(X) in nats, edge colour "
                 "l(X,Y), path steps numbered" % T, loc="left", color=bp.INK, fontsize=10)
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046,
                       pad=0.02)
    bar.set_label("l(X,Y) = a ln(1 + I_0 / I(X:Y))", color=bp.INK_MUTED)

    # (2) the root lattice
    start = rep["start"]
    pts = project(rep["lattice"], W3)
    R = project([m for _, m in roots(n)], W3)
    exact = "exact" if n == dims + 1 else "projected"
    if dims == 3:
        reach = float(np.abs(R).max()) * 1.15 + 0.1
        inside = pts[(np.abs(pts) <= reach).all(axis=1)]
        ax = fig.add_subplot(grid[1], projection="3d")
        ax.set_facecolor(bp.SURFACE)
        ax.set_proj_type("ortho")
        ax.view_init(elev=22, azim=-55)
        ax.scatter(inside[:, 0], inside[:, 1], inside[:, 2], s=9, color=bp.INK_MUTED,
                   alpha=0.5)
        many = n > LATTICE_SHELL_LIMIT
        ax.quiver(np.zeros(len(R)), np.zeros(len(R)), np.zeros(len(R)), R[:, 0], R[:, 1],
                  R[:, 2], color=bp.INK_MUTED, alpha=0.2 if many else 0.5,
                  linewidth=0.5 if many else 1.1, arrow_length_ratio=0.1)
        ring = np.vstack([W3 - W3[start], (W3 - W3[start])[:1]])
        ax.plot(ring[:, 0], ring[:, 1], ring[:, 2], color=bp.INK_MUTED, linestyle="--",
                linewidth=0.8, alpha=0.7)
        for q in range(n):
            p = W3[q] - W3[start]
            ax.text(p[0], p[1], p[2], names[q], color=bp.INK_MUTED, fontsize=8)
        position = np.zeros(3)
        for k, (t, X, Y), l in shown:
            step = W3[Y] - W3[X]
            ax.quiver(position[0], position[1], position[2], step[0], step[1], step[2],
                      color=colours[k - 1], linewidth=2.6, arrow_length_ratio=0.18)
            mid = position + (0.3 + 0.2 * ((k - 1) % 3)) * step
            ax.text(mid[0], mid[1], mid[2], _label(k, l), color=bp.INK, fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.12", facecolor=bp.SURFACE,
                              edgecolor=colours[k - 1]))
            position = position + step
        ax.scatter([0.0], [0.0], [0.0], s=50, color=bp.INK)
        ax.scatter([position[0]], [position[1]], [position[2]], s=50, facecolor=bp.SURFACE,
                   edgecolor=bp.INK, linewidth=1.5)
        ax.set_xlim(-reach, reach)
        ax.set_ylim(-reach, reach)
        ax.set_zlim(-reach, reach)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    else:
        reach = float(np.abs(pts).max()) + 0.3
        ax = fig.add_subplot(grid[1])
        bp.style_axis(ax)
        ax.set_aspect("equal")
        ax.scatter(pts[:, 0], pts[:, 1], s=9, color=bp.INK_MUTED, alpha=0.6, zorder=1)
        many = n > LATTICE_SHELL_LIMIT
        for _, m in roots(n):
            v = project(m, W3)
            ax.annotate("", xy=v, xytext=(0.0, 0.0),
                        arrowprops=dict(arrowstyle="->", color=bp.INK_MUTED,
                                        alpha=0.15 if many else 0.35,
                                        linewidth=0.5 if many else 0.7), zorder=1)
        ring = W3 - W3[start]
        ax.add_patch(Polygon(ring, closed=True, fill=False, edgecolor=bp.INK_MUTED,
                             linestyle="--", linewidth=0.8, zorder=1))
        for q in range(n):
            ax.annotate(names[q], ring[q] + 0.12 * W3[q], ha="center", va="center",
                        fontsize=8, color=bp.INK_MUTED)
        position = np.zeros(2)
        for k, (t, X, Y), l in shown:
            step = W3[Y] - W3[X]
            ax.annotate("", xy=position + step, xytext=position,
                        arrowprops=dict(arrowstyle="-|>", color=colours[k - 1],
                                        linewidth=2.4, shrinkA=0, shrinkB=0), zorder=3)
            ax.annotate(_label(k, l), position + (0.3 + 0.2 * ((k - 1) % 3)) * step,
                        ha="center", va="center",
                        fontsize=8, color=bp.INK,
                        bbox=dict(boxstyle="round,pad=0.15", facecolor=bp.SURFACE,
                                  edgecolor=colours[k - 1]), zorder=4)
            position = position + step
        ax.scatter([0.0], [0.0], s=60, color=bp.INK, zorder=5)
        ax.scatter([position[0]], [position[1]], s=60, facecolor=bp.SURFACE,
                   edgecolor=bp.INK, linewidth=1.5, zorder=5)
        ax.set_xlim(-reach, reach)
        ax.set_ylim(-reach, reach)
    ax.set_title("root lattice A_%d in %d Fourier coordinates (%s)\nroots e_Y - e_X from "
                 "the origin, the path as translations" % (n - 1, dims, exact),
                 loc="left", color=bp.INK, fontsize=10)

    # (3) the causal set
    ax = fig.add_subplot(grid[2])
    bp.style_axis(ax)
    evts = [(k, e) for k, e in enumerate(rep["events"]) if e[0] <= T]
    depth = rep["depth"]
    for i, j in rep["covers"]:
        if rep["events"][j][0] <= T:
            ax.plot([rep["events"][i][0], rep["events"][j][0]], [depth[i], depth[j]],
                    color=bp.INK_MUTED, linewidth=0.9, alpha=0.7, zorder=1)
    ridden = {}
    for k, (t, X, Y), l in shown:
        if rep["path_mode"] == "walk":
            ridden[t] = k
    for k, (t, (X, Y)) in evts:
        if t in ridden:
            c = colours[ridden[t] - 1]
            ax.scatter([t], [depth[k]], s=170, color=c, edgecolor=bp.INK, linewidth=0.8,
                       zorder=3)
            ax.annotate(str(ridden[t]), (t, depth[k]), ha="center", va="center", fontsize=8,
                        color="white", zorder=4)
        else:
            ax.scatter([t], [depth[k]], s=60, color=bp.SURFACE, edgecolor=bp.INK_MUTED,
                       linewidth=1.0, zorder=3)
        ax.annotate("%s%s" % (names[X], names[Y]), (t, depth[k] + 0.28), ha="center",
                    va="bottom", fontsize=7, color=bp.INK_MUTED, zorder=4)
    ax.set_xlim(0, rep["T"] + 1)
    ax.set_ylim(-0.6, max(int(depth.max()) if len(depth) else 0, 1) + 1.0)
    ax.set_xlabel("slice t of the event (one linear extension of the order)",
                  color=bp.INK_MUTED)
    ax.set_ylabel("depth: longest chain below the event", color=bp.INK_MUTED)
    ax.set_title("causal set: the Hasse diagram of the %d events, %d covers\n"
                 "the path's events are a chain, numbered as its steps"
                 % (len(rep["events"]), len(rep["covers"])), loc="left", color=bp.INK,
                 fontsize=10)
    fig.suptitle(title, x=0.02, ha="left", color=bp.INK)
    if save:
        fig.savefig(save, dpi=150, facecolor=bp.SURFACE)
    return fig


def animate(rep, title, path, fps=2, dpi=72):
    """One frame per slice t = 0..T, written as an animated GIF to `path`;
    returns the number of frames."""
    import matplotlib.pyplot as plt
    from PIL import Image
    frames = []
    for t in range(rep["T"] + 1):
        fig = draw(rep, "%s   slice %d of %d" % (title, t, rep["T"]), None, upto=t)
        fig.set_dpi(dpi)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
        plt.close(fig)
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=int(round(1000.0 / fps)), loop=0)
    return len(frames)


# ============================================================ command line

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="The information network of a qubit network, the root lattice its "
                    "interactions translate, and the causal set the order of the "
                    "interactions makes of it, with one path on all three.")
    parser.add_argument("--qubits", type=int, default=4,
                        help="number of qubits (default 4; at most %d with the mixed engine, "
                             "%d with the pure one)" % (ec.MAX_QUBITS, ec.MAX_QUBITS_PURE))
    parser.add_argument("--state", choices=ec.STATE_MODES, default="mixed",
                        help="the engine of entanglement_complex: the global density matrix, "
                             "or a pure global state with pure input pairs, which reaches "
                             "%d qubits (default mixed)" % ec.MAX_QUBITS_PURE)
    parser.add_argument("--timesteps", type=int, default=16,
                        help="interactions, one per time slice (default 16)")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed of the input states and of the schedule (default 0)")
    parser.add_argument("--swap-power", default="1/2",
                        help="alpha of the SWAP^alpha interaction, a fraction (default 1/2)")
    parser.add_argument("--pairs", choices=("all", "chain"), default="all",
                        help="pairs the schedule draws from: all pairs, or the nearest "
                             "neighbours of an open chain (default all)")
    parser.add_argument("--path", choices=PATH_MODES, default="walk",
                        help="walk: the worldline of a unit of charge from --start; "
                             "geodesic: the shortest path from --start to --end in the "
                             "final slice (default walk)")
    parser.add_argument("--start", default="A", help="the path's first qubit (default A)")
    parser.add_argument("--end", default=None,
                        help="the geodesic's last qubit (default the last qubit)")
    parser.add_argument("--lattice-dims", type=int, choices=LATTICE_DIMS, default=3,
                        help="Fourier coordinates the lattice is drawn in: 3 (exact for four "
                             "qubits) or 2, the Coxeter plane (default 3)")
    parser.add_argument("--length-scale", type=float, default=1.0,
                        help="a in l = a ln(1 + I_0/I) (default 1)")
    parser.add_argument("--length-reference", type=float, default=I_MAX,
                        help="I_0 in l = a ln(1 + I_0/I), in nats (default I_max = 2 ln 2)")
    parser.add_argument("--mi-floor", type=float, default=1e-12,
                        help="a pair with I at or below this has no edge (default 1e-12)")
    parser.add_argument("--verbose", action="store_true",
                        help="print the simulation's own slice-by-slice report too")
    parser.add_argument("--save", default=None, help="write the figure to this file")
    parser.add_argument("--animate", default=None,
                        help="write an animated GIF with one frame per slice to this file")
    parser.add_argument("--no-show", action="store_true", help="do not open a window")
    args = parser.parse_args(argv)

    if args.timesteps < 1:
        parser.error("--timesteps must be at least 1")
    if args.length_scale <= 0 or args.length_reference <= 0:
        parser.error("--length-scale and --length-reference must be positive")
    try:
        Fraction(args.swap_power)
    except (ValueError, ZeroDivisionError):
        parser.error("--swap-power must be a number such as 1/2 or 0.25")
    try:
        start = qubit_index(args.start, args.qubits)
        end = None if args.end is None else qubit_index(args.end, args.qubits)
    except ValueError as exc:
        parser.error(str(exc))

    result = ec.simulate(args.qubits, None, args.swap_power, args.seed, "global",
                         args.timesteps, args.pairs, False, print if args.verbose else ec._silent,
                         args.state)
    rep = analyse(result, args.mi_floor, args.length_scale, args.length_reference,
                  args.path, start, end, args.lattice_dims)
    print_report(rep)

    title = ("%d qubits, %d interactions of %s: information network, root lattice, causal set"
             % (args.qubits, args.timesteps, result["interaction"]))
    if args.save or args.animate or not args.no_show:
        import matplotlib
        if args.no_show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if args.save or not args.no_show:
            draw(rep, title, args.save)
        if args.animate:
            count = animate(rep, title, args.animate)
            print("ANIMATION: %d frames written to %s" % (count, args.animate))
        if not args.no_show:
            plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
