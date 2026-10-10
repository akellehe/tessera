# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""tessera.drivers.root_lattice_spacetime -- the information network of a
qubit network, the root lattice its interactions translate, and the spacetime
the order of the interactions gives that lattice.

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
* Coxeter plane. Q spans the (n - 1)-dimensional hyperplane of zero sum. The
  figure projects it onto the Coxeter plane, the plane in which the Coxeter
  element, the cyclic permutation of the qubits in alphabetical order, acts
  as the rotation by 2 pi / n. There the weight e_X is the point at angle
  2 pi X / n on the unit circle, the root e_Y - e_X is the chord from X to
  Y, and the lattice is the set of integer combinations of the chords. The
  projection involves no fit: it is fixed by the cyclic order of the names.
  For n >= 5 the projected lattice is dense in the plane, so the figure
  shows only the points within two root steps of the origin.
* Causal order. Every interaction is an event (X, Y, t). An event precedes a
  later one when their pairs share a qubit, and the order is the transitive
  closure of that relation, the causal set of the circuit. Two events on
  disjoint pairs are unrelated: their generators commute. Time is this order
  and nothing else; no edge of the lattice carries a duration. The order is
  stored as a ``tessera.quantum.Poset`` whose cover edges are the Hasse diagram.
* Path. With the walk, one unit of charge starts on a chosen qubit at t = 0
  and is carried across every event whose pair contains its current qubit;
  its worldline is the sequence of qubits visited, every step the root
  e_Y - e_X of its event, with the length l(X,Y) of the pair in the state
  after the event. With the geodesic, the path is the shortest path between
  two qubits through the edges of the final slice, the metric d of the
  entanglement complex, and its lengths are the final slice's.

Figure
------
Three panels. (1) The information network in the final slice: every qubit a
disc of radius S(X) at its Coxeter-plane weight, on a circle whose radius
keeps neighbouring discs apart; an edge for every pair with I above the
floor, coloured by l(X,Y), solid when the pair interacted at least once and
dashed otherwise; the path's edges in bold with their lengths. (2) The root
lattice in the Coxeter plane: the roots as arrows from the origin, the
lattice points within two root steps, the positions the charge can occupy,
and the path as root translations composed tip to tail from the origin. (3)
The spacetime: the Coxeter plane against the slice index, every qubit a
vertical worldline, every event the chord of its pair at its slice coloured
by l(X,Y) after the event, and the path's worldline in bold.
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
LATTICE_STEPS = 2


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

def coxeter_weights(n):
    """The weights e_0, ..., e_{n-1} in the Coxeter plane: the point at angle
    2 pi q / n on the unit circle for qubit q. They sum to zero, the image of
    the identity direction every root is orthogonal to."""
    angles = 2.0 * math.pi * np.arange(n) / n
    return np.column_stack([np.cos(angles), np.sin(angles)])


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


def project(vectors, weights):
    """The Coxeter-plane coordinates sum_X m_X e_X of integer vectors m."""
    return np.asarray(vectors, dtype=float) @ weights


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

def analyse(result, floor, scale, reference, path_mode, start, end=None):
    """Everything the report and the figure need, from the simulation's
    result (``entanglement_complex.simulate`` with time slices)."""
    slices = result["slices"]
    if slices is None:
        raise ValueError("the simulation must record time slices (timesteps > 0)")
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
        step_lengths = [event_lengths[[e[0] for e in evts].index(t)] for t, _, _ in steps]
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
    return {"n": n, "names": names, "T": T, "weights": coxeter_weights(n),
            "lattice": lattice_points(n), "S": slices[-1]["S"], "MI": slices[-1]["MI"],
            "D": D, "W": W, "interacted": interacted, "events": evts,
            "event_lengths": event_lengths, "poset": ps, "precedes": P, "covers": covers,
            "path_mode": path_mode, "start": start, "end": end, "steps": steps,
            "step_lengths": step_lengths, "visited": visited,
            "displacement": displacement(n, steps), "direct": direct,
            "direct_length": direct_length, "scale": scale, "reference": reference,
            "floor": floor}


def print_report(rep, log=print):
    names, n = rep["names"], rep["n"]
    label = "l(X,Y) = %g ln(1 + %g / I(X:Y))" % (rep["scale"], rep["reference"])
    log("INFORMATION NETWORK (final slice t = %d; entropies in nats; %s)" % (rep["T"], label))
    for q in range(n):
        log("  S(%s) = %.6f" % (names[q], rep["S"][q]))
    for i, j in itertools.combinations(range(n), 2):
        if rep["W"][i, j] > 0:
            log("  %s%s: I = %.6f   l = %.6f%s" % (names[i], names[j], rep["MI"][i, j],
                                                   rep["D"][i, j],
                                                   "" if rep["interacted"][i, j] else
                                                   "   (never interacted)"))
    log("ROOT LATTICE A_%d: %d roots e_Y - e_X, %d lattice points within %d root steps"
        % (n - 1, n * (n - 1), len(rep["lattice"]), LATTICE_STEPS))
    log("CAUSAL ORDER: %d events, %d related pairs of %d, %d covers (Hasse links)"
        % (len(rep["events"]), int(rep["precedes"].sum()),
           len(rep["events"]) * (len(rep["events"]) - 1) // 2, len(rep["covers"])))
    for k, ((t, (X, Y)), l) in enumerate(zip(rep["events"], rep["event_lengths"])):
        above = [i for i in range(k) if rep["precedes"][i, k]]
        log("  event %3d  t = %3d  (%s,%s)   l after = %s   in the future of %d event(s)"
            % (k, t, names[X], names[Y], "%.6f" % l if math.isfinite(l) else "inf",
               len(above)))
    if rep["path_mode"] == "walk":
        log("WORLDLINE of a unit of charge from %s" % names[rep["start"]])
    else:
        log("GEODESIC from %s to %s in the final slice" % (names[rep["start"]],
                                                           names[rep["end"]]))
    total = 0.0
    for (t, X, Y), l in zip(rep["steps"], rep["step_lengths"]):
        total += l
        log("  t = %3d  %s -> %s   root e_%s - e_%s   l = %s" % (
            t, names[X], names[Y], names[Y], names[X],
            "%.6f" % l if math.isfinite(l) else "inf"))
    log("  visited: %s   total length %s" % (
        " -> ".join(names[q] for q in rep["visited"]),
        "%.6f" % total if math.isfinite(total) else "inf"))
    log("  displacement: %s" % (" + ".join(
        "%+d e_%s" % (m, names[q]) for q, m in enumerate(rep["displacement"]) if m) or "0"))
    if rep["direct"] and len(rep["visited"]) > 1:
        log("  shortest path %s -> %s in the final slice: %s   d = %.6f"
            % (names[rep["start"]], names[rep["visited"][-1]],
               " -> ".join(names[q] for q in rep["direct"]), rep["direct_length"]))


# ============================================================ figure

def draw(rep, title, save=None):
    """The three panels; returns the figure."""
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from matplotlib.patches import Circle, Polygon
    from tessera.drivers import baryon_poles as bp

    n, names, w = rep["n"], rep["names"], rep["weights"]
    S, D, W, T = rep["S"], rep["D"], rep["W"], rep["T"]
    finite = [l for l in rep["event_lengths"] if math.isfinite(l)] + list(D[W > 0])
    norm = colors.Normalize(vmin=min(finite) if finite else 0.0,
                            vmax=max(finite) if finite else 1.0)
    cmap = plt.get_cmap("viridis")
    palette = ec.CATEGORICAL

    fig = plt.figure(figsize=(19, 6.6), facecolor=bp.SURFACE)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.15], wspace=0.22,
                            left=0.03, right=0.98, top=0.86, bottom=0.06)

    # (1) the information network
    ax = fig.add_subplot(grid[0])
    bp.style_axis(ax)
    ax.set_aspect("equal")
    radius = max(1.0, 1.5 * max(float(S.max()), 1e-3) / math.sin(math.pi / n))
    xy = radius * w
    for i, j in itertools.combinations(range(n), 2):
        if W[i, j] > 0:
            ax.plot(*xy[[i, j]].T, color=cmap(norm(D[i, j])), linewidth=1.4,
                    linestyle="-" if rep["interacted"][i, j] else "--", zorder=1)
    for (t, X, Y), l in zip(rep["steps"], rep["step_lengths"]):
        ax.plot(*xy[[X, Y]].T, color=bp.INK, linewidth=3.0, zorder=2)
        mid = xy[[X, Y]].mean(axis=0)
        ax.annotate("%.2f" % l if math.isfinite(l) else "inf", mid, ha="center",
                    va="center", fontsize=8, color=bp.INK,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor=bp.SURFACE,
                              edgecolor="none"), zorder=4)
    for q in range(n):
        ax.add_patch(Circle(xy[q], radius=float(S[q]), facecolor=ec.FILL,
                            edgecolor=palette[q % len(palette)], linewidth=1.5, zorder=3))
        ax.annotate(names[q], xy[q], ha="center", va="center", color=bp.INK, zorder=5)
    lim = radius + float(S.max()) + 0.3
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_title("information network, slice %d\ndisc radius S(X) in nats, edge colour l(X,Y)"
                 % T, loc="left", color=bp.INK, fontsize=10)
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046,
                       pad=0.02)
    bar.set_label("l(X,Y) = a ln(1 + I_0 / I(X:Y))", color=bp.INK_MUTED)

    # (2) the root lattice
    ax = fig.add_subplot(grid[1])
    bp.style_axis(ax)
    ax.set_aspect("equal")
    pts = project(rep["lattice"], w)
    ax.scatter(pts[:, 0], pts[:, 1], s=9, color=bp.INK_MUTED, alpha=0.6, zorder=1)
    for _, m in roots(n):
        v = project(m, w)
        ax.annotate("", xy=v, xytext=(0.0, 0.0),
                    arrowprops=dict(arrowstyle="->", color=bp.INK_MUTED, alpha=0.35,
                                    linewidth=0.7), zorder=1)
    start = rep["start"]
    ring = w - w[start]
    ax.add_patch(Polygon(ring, closed=True, fill=False, edgecolor=bp.INK_MUTED,
                         linestyle="--", linewidth=0.8, zorder=1))
    for q in range(n):
        ax.annotate(names[q], ring[q] * 1.0 + 0.12 * w[q], ha="center", va="center",
                    fontsize=8, color=bp.INK_MUTED)
    position = np.zeros(2)
    for k, ((t, X, Y), l) in enumerate(zip(rep["steps"], rep["step_lengths"])):
        step = w[Y] - w[X]
        colour = cmap(norm(l)) if math.isfinite(l) else bp.INK_MUTED
        ax.annotate("", xy=position + step, xytext=position,
                    arrowprops=dict(arrowstyle="-|>", color=colour, linewidth=2.4,
                                    shrinkA=0, shrinkB=0), zorder=3)
        ax.annotate("%.2f" % l if math.isfinite(l) else "inf", position + 0.5 * step,
                    ha="center", va="center", fontsize=8, color=bp.INK,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor=bp.SURFACE,
                              edgecolor="none"), zorder=4)
        position = position + step
    ax.scatter([0.0], [0.0], s=60, color=bp.INK, zorder=5)
    ax.scatter([position[0]], [position[1]], s=60, facecolor=bp.SURFACE, edgecolor=bp.INK,
               linewidth=1.5, zorder=5)
    reach = float(np.abs(pts).max()) + 0.3
    ax.set_xlim(-reach, reach)
    ax.set_ylim(-reach, reach)
    ax.set_title("root lattice A_%d in the Coxeter plane\nroots e_Y - e_X from the origin, "
                 "the path as translations" % (n - 1), loc="left", color=bp.INK, fontsize=10)

    # (3) the spacetime
    ax = fig.add_subplot(grid[2], projection="3d")
    ax.set_facecolor(bp.SURFACE)
    for q in range(n):
        ax.plot([w[q, 0]] * 2, [w[q, 1]] * 2, [0, T], color=palette[q % len(palette)],
                linewidth=1.0, alpha=0.8)
        ax.text(1.15 * w[q, 0], 1.15 * w[q, 1], 0.0, names[q], color=bp.INK, fontsize=9)
    for (t, (X, Y)), l in zip(rep["events"], rep["event_lengths"]):
        ax.plot(w[[X, Y], 0], w[[X, Y], 1], [t, t],
                color=cmap(norm(l)) if math.isfinite(l) else bp.INK_MUTED, linewidth=1.6)
    line, current = [(w[start], 0)], start
    if rep["path_mode"] == "walk":
        for t, X, Y in rep["steps"]:
            line += [(w[X], t), (w[Y], t)]
            current = Y
        line.append((w[current], T))
    else:
        line = [(w[X], T) for X in rep["visited"]]
    if len(line) > 1:
        ax.plot([p[0][0] for p in line], [p[0][1] for p in line], [p[1] for p in line],
                color=bp.INK, linewidth=3.0)
    ax.set_zlim(0, max(T, 1))
    ax.set_zlabel("slice t: the order of the interactions", color=bp.INK_MUTED)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("spacetime: the Coxeter plane against the interaction order\n"
                 "%d events, %d causal relations, %d covers"
                 % (len(rep["events"]), int(rep["precedes"].sum()), len(rep["covers"])),
                 loc="left", color=bp.INK, fontsize=10)
    fig.suptitle(title, x=0.02, ha="left", color=bp.INK)
    if save:
        fig.savefig(save, dpi=150, facecolor=bp.SURFACE)
    return fig


# ============================================================ command line

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="The information network of a qubit network, the root lattice its "
                    "interactions translate, and the spacetime the order of the "
                    "interactions gives it, with one path traversed on all three.")
    parser.add_argument("--qubits", type=int, default=6, help="number of qubits (default 6)")
    parser.add_argument("--timesteps", type=int, default=24,
                        help="interactions, one per time slice (default 24)")
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
    parser.add_argument("--length-scale", type=float, default=1.0,
                        help="a in l = a ln(1 + I_0/I) (default 1)")
    parser.add_argument("--length-reference", type=float, default=I_MAX,
                        help="I_0 in l = a ln(1 + I_0/I), in nats (default I_max = 2 ln 2)")
    parser.add_argument("--mi-floor", type=float, default=1e-12,
                        help="a pair with I at or below this has no edge (default 1e-12)")
    parser.add_argument("--verbose", action="store_true",
                        help="print the simulation's own slice-by-slice report too")
    parser.add_argument("--save", default=None, help="write the figure to this file")
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
                         args.timesteps, args.pairs, False, print if args.verbose else ec._silent)
    rep = analyse(result, args.mi_floor, args.length_scale, args.length_reference,
                  args.path, start, end)
    print_report(rep)

    if args.save or not args.no_show:
        import matplotlib
        if args.no_show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        draw(rep, "%d qubits, %d interactions of %s: information network, root lattice, "
                  "spacetime" % (args.qubits, args.timesteps, result["interaction"]),
             args.save)
        if not args.no_show:
            plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
