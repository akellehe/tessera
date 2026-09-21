# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The skeleton diagrams of the self-energy beyond Sigma = i G W (#1161, #1168),
each held to an independent reference: the closed form at first order, the
textbook second-order exchange for instantaneous lines, the plain frequency
integrals of the Feynman rules on the imaginary axis for retarded lines, the
heavy-boson limit for instantaneous lines, the plain sum over the orderings of
the vertex times for the recursion that replaces it, and, for the rules as a
whole (the enumeration, the signs, the loops), the exact diagonalization of
electrons coupled to bosons, order by order in the coupling."""
import itertools

import numpy as np
import pytest

from tessera.drivers.bands.diagrams import Diagram, SkeletonSelfEnergy, irreducible_diagrams, skeleton_diagrams

NB, OCC, NS = 4, 2, 2
XI, BOSONS = np.array([-1.3, -0.7, 0.6, 1.1]), np.array([0.9, 1.7])


def _model():
    rng = np.random.default_rng(7)
    g = rng.normal(size=(NS, NB, NB)) * 0.3
    L = rng.normal(size=(3, NB, NB)) * 0.4
    g, L = 0.5 * (g + g.transpose(0, 2, 1)), 0.5 * (L + L.transpose(0, 2, 1))
    return g, L, np.einsum("xpq,xrs->pqrs", L, L)


def test_the_counts_of_skeleton_diagrams():
    """1, 1, 6, 49, 542: the expansion of the self-energy in G and W counted in
    zero dimensions (Molinari and Manini, Eur. Phys. J. B 51, 331 (2006))."""
    assert [len(skeleton_diagrams(order)) for order in (1, 2, 3, 4, 5)] == [1, 1, 6, 49, 542]
    assert sum(d.loops for d in skeleton_diagrams(3)) == 2
    for order in (4, 5):
        for diagram in skeleton_diagrams(order):
            assert sorted(v for chord in diagram.chords for v in chord) == list(range(2 * order))
            assert len(diagram.edges) == 2 * order - 1
    with pytest.raises(ValueError):
        skeleton_diagrams(0)


def _name(diagram):
    """A diagram up to the labelling of its vertices: the pairing of the open
    line's vertices, counted from the entry, and of every loop's, counted from
    each of its vertices in turn."""
    following = dict(diagram.edges)
    line, vertex = [diagram.entry], diagram.entry
    while vertex != diagram.exit:
        vertex = following[vertex]
        line.append(vertex)
    loops, placed = [], set(line)
    for start in range(diagram.vertices):
        if start not in placed:
            loop, vertex = [start], following[start]
            while vertex != start:
                loop.append(vertex)
                vertex = following[vertex]
            loops.append(loop)
            placed.update(loop)
    names = []
    for order in itertools.permutations(range(len(loops))):
        for turns in itertools.product(*[range(len(loop)) for loop in loops]):
            label = {vertex: (0, j) for j, vertex in enumerate(line)}
            for position, which in enumerate(order):
                loop = loops[which][turns[which]:] + loops[which][:turns[which]]
                label.update({vertex: (1 + position, len(loop), j) for j, vertex in enumerate(loop)})
            names.append(tuple(sorted(tuple(sorted((label[u], label[v]))) for u, v in diagram.chords)))
    return min(names)


def test_the_enumeration_gives_the_diagrams_written_by_hand():
    line = lambda chords: Diagram(2 * len(chords), tuple((j, j + 1) for j in range(2 * len(chords) - 1)), 0,
                                  2 * len(chords) - 1, tuple(chords))
    chords = ((0, 3), (1, 4), (2, 5))
    by_hand = {1: [line([(0, 1)])], 2: [line([(0, 2), (1, 3)])],
               3: [line(c) for c in ([(0, 2), (1, 4), (3, 5)], [(0, 3), (1, 4), (2, 5)],
                                     [(0, 3), (1, 5), (2, 4)], [(0, 4), (1, 3), (2, 5)])]
               + [Diagram(6, ((0, 1), (1, 2)) + loop, 0, 2, chords, loops=1)
                  for loop in (((3, 4), (4, 5), (5, 3)), ((3, 5), (5, 4), (4, 3)))]}
    for order, diagrams in by_hand.items():
        assert sorted(_name(d) for d in diagrams) == sorted(_name(d) for d in skeleton_diagrams(order))
    for order in (4, 5):                                          # no diagram twice
        assert len({_name(d) for d in skeleton_diagrams(order)}) == len(skeleton_diagrams(order))


def test_first_order_is_the_pole_sum():
    g, _, U = _model()
    n, w = 2, 0.13
    direct = sum(g[s, n, m] ** 2 / (w - XI[m] + (BOSONS[s] if m < OCC else -BOSONS[s]))
                 for s in range(NS) for m in range(NB))
    engine = SkeletonSelfEnergy(XI, OCC, BOSONS, g, U)
    assert engine.evaluate(n, w, 1).real == pytest.approx(direct, abs=1e-14)
    value, derivative = engine.value_and_derivative(n, w, 1)
    numerical = (engine.evaluate(n, w + 1e-5, 1) - engine.evaluate(n, w - 1e-5, 1)).real / 2e-5
    assert value == pytest.approx(direct, abs=1e-10) and derivative == pytest.approx(numerical, rel=1e-6)


def test_second_order_with_instantaneous_lines_is_the_second_order_exchange():
    _, _, U = _model()
    n, w = 2, 0.13
    F, E = range(OCC), range(OCC, NB)
    exchange = -sum(U[n, a, i, b] * U[n, b, i, a] / (w + XI[i] - XI[a] - XI[b]) for i in F for a in E for b in E) \
        - sum(U[n, i, a, j] * U[n, j, a, i] / (w + XI[a] - XI[i] - XI[j]) for i in F for j in F for a in E)
    static = SkeletonSelfEnergy(XI, OCC, np.zeros(0), np.zeros((0, NB, NB)), U)
    assert static.evaluate(n, w, 2).real == pytest.approx(exchange, abs=1e-14)


def _frequency_integral(diagram, g, n, omega, points):
    """(-1)^k (-2)^loops sum over labels of the couplings times the integral
    over the k independent frequencies of the propagators on the imaginary
    axis, retarded lines only. The frequencies are routed over a spanning tree
    of the diagram (the open line, every loop but one of its lines, then
    interaction lines); the lines left out carry the integration variables. An
    interaction line's is integrated by Gauss-Legendre quadrature in the angle
    of nu = tan(angle), a loop's in closed form: the integral over f of prod_j
    1 / (i (f + a_j) - xi_j) is the sum over the filled j of prod_(m != j)
    1 / (xi_j - xi_m + i (a_m - a_j)). The rules of the axes are even and
    unequal, so that no a_m - a_j vanishes on the grid."""
    k, lines = len(diagram.chords), list(diagram.edges) + [(v, u) for u, v in diagram.chords]
    following, entering = dict(diagram.edges), {v: e for e, (u, v) in enumerate(diagram.edges)}
    chord_of = {vertex: c for c, pair in enumerate(diagram.chords) for vertex in pair}
    line = [diagram.entry]
    while line[-1] != diagram.exit:
        line.append(following[line[-1]])
    loops, placed = [], set(line)
    for start in range(diagram.vertices):
        if start not in placed:
            loop = [start]
            while following[loop[-1]] != start:
                loop.append(following[loop[-1]])
            loops.append(loop)
            placed.update(loop)
    parent, tree, left_out = list(range(diagram.vertices)), [], []

    def find(a):
        while parent[a] != a:
            a = parent[a]
        return a

    for index, (a, b) in enumerate(lines):
        if find(a) != find(b):
            parent[find(a)] = find(b)
            tree.append(index)
        else:
            left_out.append(index)

    def path(start, goal):
        stack, seen = [(start, [])], {start}
        while stack:
            node, walked = stack.pop()
            if node == goal:
                return walked
            for index in tree:
                for here, there, sign in ((*lines[index], 1), (*lines[index][::-1], -1)):
                    if here == node and there not in seen:
                        seen.add(there)
                        stack.append((there, walked + [(index, sign)]))

    variables = [index for index in left_out if index >= len(diagram.edges)]
    assert len(variables) == k - len(loops)
    rules = [np.polynomial.legendre.leggauss(points + 2 * axis) for axis in range(len(variables))]
    grids = np.meshgrid(*[np.tan(0.5 * np.pi * theta) for theta, _ in rules], indexing="ij")
    measure = np.ones_like(grids[0])
    for axis, (theta, weight) in enumerate(rules):
        shape = [1] * len(rules)
        shape[axis] = -1
        measure = measure * (0.25 * weight / np.cos(0.5 * np.pi * theta) ** 2).reshape(shape)       # d nu / (2 pi)
    measure = measure.ravel()
    frequency = [np.zeros_like(measure) for _ in lines]
    for index, sign in path(diagram.entry, diagram.exit):
        frequency[index] += sign * omega
    for j, index in enumerate(variables):
        for other, sign in [(index, 1)] + path(lines[index][1], lines[index][0]):
            frequency[other] = frequency[other] + sign * grids[j].ravel()
    closed = []
    for loop in loops:
        labels = list(itertools.product(range(NB), repeat=len(loop)))
        into = [frequency[entering[vertex]] for vertex in loop]
        values = np.zeros((len(labels), len(measure)), dtype=complex)
        for row, p in enumerate(labels):
            for j in (j for j in range(len(loop)) if XI[p[j]] < 0):
                values[row] += np.prod([1.0 / (XI[p[j]] - XI[p[m]] + 1j * (into[m] - into[j]))
                                        for m in range(len(loop)) if m != j], axis=0)
        closed.append((loop, labels, values))
    total = 0.0
    for s in itertools.product(range(NS), repeat=k):
        value = measure.astype(complex)
        for c in range(k):
            value = value * (-2.0 * BOSONS[s[c]] / (frequency[len(diagram.edges) + c] ** 2 + BOSONS[s[c]] ** 2))
        vector = np.broadcast_to(g[s[chord_of[line[0]]]][:, n], (len(measure), NB))
        for vertex in line[1:]:
            vector = (vector / (1j * frequency[entering[vertex]][:, None] - XI[None, :])) @ g[s[chord_of[vertex]]].T
        value = value * vector[:, n]
        for loop, labels, values in closed:                       # the vertex j of a loop joins the labels p_j and p_(j + 1)
            couplings = np.array([np.prod([g[s[chord_of[vertex]]][p[(j + 1) % len(loop)], p[j]]
                                           for j, vertex in enumerate(loop)]) for p in labels])
            value = value * (couplings @ values)
        total += value.sum()
    return (-1) ** k * (-2.0) ** len(loops) * total


def test_the_crossed_diagram_is_its_double_frequency_integral():
    g, _, U = _model()
    engine = SkeletonSelfEnergy(XI, OCC, BOSONS, g, U)
    diagram = skeleton_diagrams(2)[0]
    mine = engine._diagram(diagram, ("dynamic", "dynamic"), 2, 0.4j)
    assert mine == pytest.approx(_frequency_integral(diagram, g, 2, 0.4, 120), abs=5e-6)


def test_the_third_order_diagrams_are_their_triple_frequency_integrals():
    """The four crossings and the two loops, whose own frequency is integrated
    in closed form."""
    g, _, U = _model()
    engine = SkeletonSelfEnergy(XI, OCC, BOSONS, g, U)
    for diagram in skeleton_diagrams(3):
        mine = engine._diagram(diagram, ("dynamic",) * 3, 2, 0.4j)
        assert mine == pytest.approx(_frequency_integral(diagram, g, 2, 0.4, 40), abs=1e-5)


@pytest.mark.slow
def test_the_fourth_order_diagrams_are_their_frequency_integrals():
    """All 49. The quadrature converges as a power of the number of points (a
    line that carries a sum of frequencies is a ridge that the product grid
    resolves only near the origin): the error falls from 12 to 20 points a
    axis, and is below 2e-5 where the diagrams are as large as 5e-4."""
    g, _, U = _model()
    engine = SkeletonSelfEnergy(XI, OCC, BOSONS, g, U)
    coarse = fine = 0.0
    for diagram in skeleton_diagrams(4):
        mine = engine._diagram(diagram, ("dynamic",) * 4, 2, 0.4j)
        coarse = max(coarse, abs(_frequency_integral(diagram, g, 2, 0.4, 12) - mine))
        fine = max(fine, abs(_frequency_integral(diagram, g, 2, 0.4, 20) - mine))
    assert fine < 2e-5 and fine < 0.5 * coarse


def test_instantaneous_lines_are_the_limit_of_heavy_bosons_at_every_order():
    """U = -sum_x 2 g_x g_x / W with g_x = i sqrt(W / 2) L_x: every assignment of
    instantaneous and retarded lines, the triangle loops included, against the
    same diagrams with retarded lines only and three more, heavy, bosons."""
    g, L, U = _model()
    engine = SkeletonSelfEnergy(XI, OCC, BOSONS, g, U)
    heavy = 1e5
    emulated = SkeletonSelfEnergy(XI, OCC, np.concatenate([BOSONS, np.full(3, heavy)]),
                                  np.concatenate([g.astype(complex), 1j * np.sqrt(heavy / 2.0) * L]), None)
    for order in (2, 3):
        assert engine.evaluate(2, 0.13, order) == pytest.approx(emulated.evaluate(2, 0.13, order), abs=5e-6)
    # At fourth order, one diagram of each number of loops, every assignment.
    chosen = {d.loops: d for d in skeleton_diagrams(4)}
    assert sorted(chosen) == [0, 1]
    for diagram in chosen.values():
        mine = sum(engine._diagram(diagram, kinds, 2, 0.13) for kinds in itertools.product(("dynamic", "static"), repeat=4))
        assert mine == pytest.approx(emulated._diagram(diagram, ("dynamic",) * 4, 2, 0.13), abs=5e-6)


def test_the_recursion_over_sets_of_times_is_the_sum_over_the_orderings():
    """Every diagram of the orders 1 to 3, every assignment of instantaneous and
    retarded lines, real and complex frequencies, filled and empty external
    modes; and the same with the label of a retarded line held fixed and
    summed outside, which is how a diagram too wide for the memory is done."""
    g, _, U = _model()
    engine = SkeletonSelfEnergy(XI, OCC, BOSONS, g, U)
    for order in (1, 2, 3):
        for diagram in skeleton_diagrams(order):
            for kinds in itertools.product(("dynamic", "static"), repeat=order):
                for n, w in ((2, 0.13), (1, 0.4j), (0, -0.2 + 0.1j)):
                    reference = engine._orderings(diagram, kinds, n, w)
                    assert engine._diagram(diagram, kinds, n, w) == pytest.approx(reference, abs=1e-14)
                    for c in (c for c in range(order) if kinds[c] == "dynamic"):
                        pieces = sum(engine._diagram(diagram, kinds, n, w, ((c, s),)) for s in range(NS))
                        assert pieces == pytest.approx(reference, abs=1e-14)


@pytest.mark.slow
def test_the_recursion_is_the_sum_over_the_orderings_at_fourth_order():
    """8! orderings a diagram: two crossings and two diagrams with a loop, and
    one assignment with instantaneous lines for each."""
    g, _, U = _model()
    engine = SkeletonSelfEnergy(XI, OCC, BOSONS, g, U)
    diagrams = skeleton_diagrams(4)
    chosen = [d for d in diagrams if d.loops == 0][::20][:2] + [d for d in diagrams if d.loops == 1][::7][:2]
    for diagram in chosen:
        for kinds in (("dynamic",) * 4, ("dynamic", "static", "dynamic", "static")):
            assert engine._diagram(diagram, kinds, 2, 0.13) == pytest.approx(engine._orderings(diagram, kinds, 2, 0.13), abs=1e-13)


@pytest.mark.slow
def test_the_recursion_is_the_sum_over_the_orderings_at_fifth_order():
    """One diagram in ten of the 542, three of its five lines instantaneous,
    which leaves 7! orderings."""
    g, _, U = _model()
    engine = SkeletonSelfEnergy(XI, OCC, BOSONS, g, U)
    kinds = ("static", "dynamic", "static", "dynamic", "static")
    compared = 0
    for diagram in skeleton_diagrams(5)[::10]:
        reference = engine._orderings(diagram, kinds, 2, 0.13)
        assert engine._diagram(diagram, kinds, 2, 0.13) == pytest.approx(reference, abs=1e-13)
        compared += reference != 0.0
    assert compared > 25


def test_a_memory_too_small_for_a_diagram_fixes_labels_and_keeps_the_value():
    g, _, U = _model()
    whole = SkeletonSelfEnergy(XI, OCC, BOSONS, g, U)
    tight = SkeletonSelfEnergy(XI, OCC, BOSONS, g, U, memory=16 * 700)
    assert any(fixed for _, _, _, fixed in tight.work(3)) and not any(fixed for _, _, _, fixed in whole.work(3))
    assert tight.evaluate(2, 0.13, 3, parallel=False) == pytest.approx(whole.evaluate(2, 0.13, 3, parallel=False), abs=1e-14)
    with pytest.raises(MemoryError, match="GiB"):
        SkeletonSelfEnergy(XI, OCC, BOSONS, g, U, memory=16 * 100).work(3)


# -- the rules as a whole against an exact diagonalization

def _exact_orders(xi, occupied, bosons, g, omega, orders, radius=0.15, points=24):
    """The coefficients of lambda^(2 k), k = 0 .. orders, of the self-energy
    Sigma(omega) = G0^-1 - G^-1 of H = sum xi c+ c + sum W b+ b + lambda sum_s
    g^s_pq c+_p c_q (b_s + b+_s), two spins, from the exact Green function of
    the ground state. The boson space holds up to `orders` quanta, which is
    every state that order of the coupling reaches; the coefficients are read
    off a circle of complex lambda^2, where H is complex symmetric and bras are
    transposes."""
    import scipy.linalg
    levels, modes, orbitals = len(xi), len(bosons), 2 * len(xi)          # spin orbital p + levels * spin

    def lower(state, o):
        return (state ^ (1 << o), (-1) ** bin(state & ((1 << o) - 1)).count("1")) if state >> o & 1 else None

    quanta = [q for q in itertools.product(range(orders + 1), repeat=modes) if sum(q) <= orders]
    X = np.zeros((modes, len(quanta), len(quanta)))
    for a, q in enumerate(quanta):
        for s in range(modes):
            up = tuple(v + (j == s) for j, v in enumerate(q))
            if up in quanta:
                X[s, quanta.index(up), a] = X[s, a, quanta.index(up)] = np.sqrt(q[s] + 1)
    electrons = 2 * occupied
    basis = {count: [sum(1 << o for o in chosen) for chosen in itertools.combinations(range(orbitals), count)]
             for count in (electrons - 1, electrons, electrons + 1)}
    index = {count: {state: i for i, state in enumerate(states)} for count, states in basis.items()}

    def annihilator(o, count):                                    # from `count` electrons to one fewer
        matrix = np.zeros((len(basis[count - 1]), len(basis[count])))
        for i, state in enumerate(basis[count]):
            lowered = lower(state, o)
            if lowered:
                matrix[index[count - 1][lowered[0]], i] = lowered[1]
        return matrix

    def hamiltonian(count):
        states = basis[count]
        free = np.array([sum(xi[o % levels] for o in range(orbitals) if state >> o & 1) for state in states])
        density = np.zeros((modes, len(states), len(states)))
        for i, state in enumerate(states):
            for q in range(orbitals):
                lowered = lower(state, q)
                for p in range(q - q % levels, q - q % levels + levels) if lowered else ():
                    if not lowered[0] >> p & 1:
                        sign = lowered[1] * (-1) ** bin(lowered[0] & ((1 << p) - 1)).count("1")
                        density[:, index[count][lowered[0] | 1 << p], i] += sign * g[:, p % levels, q % levels]
        return (free[:, None] + np.array([np.dot(q, bosons) for q in quanta])[None, :]).ravel(), \
            sum(np.kron(density[s], X[s]) for s in range(modes))

    parts = {count: hamiltonian(count) for count in basis}
    filled = index[electrons][sum(1 << (p + levels * spin) for p in range(occupied) for spin in (0, 1))] * len(quanta)
    start = np.zeros(len(parts[electrons][0]), dtype=complex)
    start[filled] = 1.0
    add = [np.kron(annihilator(p, electrons + 1).T, np.eye(len(quanta))) for p in range(levels)]
    remove = [np.kron(annihilator(p, electrons), np.eye(len(quanta))) for p in range(levels)]
    coefficients = [np.zeros((levels, levels), dtype=complex) for _ in range(orders + 1)]
    for j in range(points):
        x = radius * np.exp(2j * np.pi * (j + 0.5) / points)
        coupling = np.sqrt(x)
        diagonal, V = parts[electrons]
        H = np.diag(diagonal) + coupling * V
        factors = scipy.linalg.lu_factor(H - (diagonal[filled] - 0.3) * np.eye(len(diagonal)))
        psi = start
        for _ in range(60):                                       # the ground state, continued from the filled levels
            psi = scipy.linalg.lu_solve(factors, psi)
            psi = psi / np.sqrt(psi @ psi)
        energy = psi @ H @ psi
        assert np.linalg.norm(H @ psi - energy * psi) < 1e-10
        diagonal, V = parts[electrons + 1]
        kets = np.stack([a @ psi for a in add], axis=1)
        G = kets.T @ np.linalg.solve((omega + energy) * np.eye(len(diagonal)) - np.diag(diagonal) - coupling * V, kets)
        diagonal, V = parts[electrons - 1]
        kets = np.stack([r @ psi for r in remove], axis=1)
        G = G + (kets.T @ np.linalg.solve((omega - energy) * np.eye(len(diagonal)) + np.diag(diagonal) + coupling * V, kets)).T
        sigma = np.diag(omega - xi) - np.linalg.inv(G)
        for k in range(orders + 1):
            coefficients[k] += sigma * x ** -k / points
    return coefficients


class _EveryDiagram(SkeletonSelfEnergy):
    """With the bare propagator on the lines, the self-energy of an order is
    every irreducible diagram of that order, insertions and tadpoles included."""

    def diagrams(self, order):
        return irreducible_diagrams(order)


def _coupled(levels, occupied, seed=11):
    """Couplings whose trace over the filled levels vanishes: the density of the
    mean field then sources no boson, and the fermion lines that close on their
    own vertex, which `irreducible_diagrams` leaves to the mean field, are zero."""
    rng = np.random.default_rng(seed)
    g = rng.normal(size=(NS, levels, levels)) * 0.3
    g = 0.5 * (g + g.transpose(0, 2, 1))
    g[:, occupied - 1, occupied - 1] = -np.trace(g[:, :occupied - 1, :occupied - 1], axis1=1, axis2=2)
    return g


def test_the_counts_of_irreducible_diagrams():
    assert [len(irreducible_diagrams(order)) for order in (1, 2, 3, 4)] == [1, 4, 26, 238]
    for order in (2, 3, 4):
        skeletons = {_name(d) for d in skeleton_diagrams(order)}
        assert skeletons < {_name(d) for d in irreducible_diagrams(order)}


def test_the_diagrams_through_third_order_are_the_exact_self_energy():
    g = _coupled(NB, OCC)
    exact = _exact_orders(XI, OCC, BOSONS, g, 0.13, 3)
    engine = _EveryDiagram(XI, OCC, BOSONS, g, None)
    assert np.abs(exact[0]).max() < 1e-13
    for order in (1, 2, 3):
        for n in (1, 2):
            assert engine.evaluate(n, 0.13, order, parallel=False) == pytest.approx(exact[order][n, n], abs=1e-12)


@pytest.mark.slow
def test_the_diagrams_of_fourth_order_are_the_exact_self_energy():
    """All 238 irreducible diagrams, the 49 skeletons among them, four levels
    and two bosons."""
    g = _coupled(NB, OCC)
    exact = _exact_orders(XI, OCC, BOSONS, g, 0.13, 4)
    engine = _EveryDiagram(XI, OCC, BOSONS, g, None)
    for n in (1, 2):
        assert engine.evaluate(n, 0.13, 4, parallel=False) == pytest.approx(exact[4][n, n], abs=1e-11)


@pytest.mark.slow
def test_the_diagrams_of_fifth_order_are_the_exact_self_energy():
    """All 2732 irreducible diagrams, the 542 skeletons among them, on three
    levels (one filled) and two bosons, which keeps the 10 240 contractions a
    diagram small."""
    assert len(irreducible_diagrams(5)) == 2732
    xi = XI[1:]
    g = _coupled(3, 1)
    exact = _exact_orders(xi, 1, BOSONS, g, 0.13, 5)
    engine = _EveryDiagram(xi, 1, BOSONS, g, None)
    assert engine.evaluate(1, 0.13, 5) == pytest.approx(exact[5][1, 1], abs=1e-11)
