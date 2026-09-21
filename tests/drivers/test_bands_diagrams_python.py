# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The skeleton diagrams of the self-energy beyond Sigma = i G W (#1161), each
held to an independent reference: the closed form at first order, the textbook
second-order exchange for instantaneous lines, the plain frequency integrals of
the Feynman rules on the imaginary axis for retarded lines, and the heavy-boson
limit for instantaneous lines at third order."""
import itertools

import numpy as np
import pytest

from tessera.drivers.bands.diagrams import SkeletonSelfEnergy, skeleton_diagrams

NB, OCC, NS = 4, 2, 2
XI, BOSONS = np.array([-1.3, -0.7, 0.6, 1.1]), np.array([0.9, 1.7])


def _model():
    rng = np.random.default_rng(7)
    g = rng.normal(size=(NS, NB, NB)) * 0.3
    L = rng.normal(size=(3, NB, NB)) * 0.4
    g, L = 0.5 * (g + g.transpose(0, 2, 1)), 0.5 * (L + L.transpose(0, 2, 1))
    return g, L, np.einsum("xpq,xrs->pqrs", L, L)


def test_the_counts_of_skeleton_diagrams():
    assert [len(skeleton_diagrams(order)) for order in (1, 2, 3)] == [1, 1, 6]
    assert sum(d.loops for d in skeleton_diagrams(3)) == 2
    with pytest.raises(NotImplementedError):
        skeleton_diagrams(4)


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


def _quadrature(diagram, g, n, omega, points):
    """(-1)^k sum over labels of the couplings times the integral over the boson
    frequencies of the propagators, retarded lines only, one open fermion line."""
    chords, k = diagram.chords, len(diagram.chords)
    theta, weight = np.polynomial.legendre.leggauss(points)
    nu, dnu = np.tan(0.5 * np.pi * theta), 0.5 * np.pi * weight / np.cos(0.5 * np.pi * theta) ** 2 / (2 * np.pi)
    grids = np.meshgrid(*[nu] * k, indexing="ij")
    measure = np.ones_like(grids[0])
    for axis in range(k):
        shape = [1] * k
        shape[axis] = points
        measure = measure * dnu.reshape(shape)
    labels, total = len(diagram.edges), 0.0
    state = lambda a, j: a[j] if 0 <= j < labels else n                    # the line between vertices j and j + 1
    for a in itertools.product(range(NB), repeat=labels):
        G = np.ones_like(grids[0], dtype=complex)
        for j in range(labels):
            G = G / (1j * omega + 1j * sum(grids[c] for c, (u, v) in enumerate(chords) if u <= j < v) - XI[a[j]])
        for s in itertools.product(range(NS), repeat=k):
            coupling, D = 1.0, np.ones_like(grids[0])
            for c, (u, v) in enumerate(chords):
                coupling *= g[s[c], state(a, u), state(a, u - 1)] * g[s[c], state(a, v), state(a, v - 1)]
                D = D * (-2.0 * BOSONS[s[c]] / (grids[c] ** 2 + BOSONS[s[c]] ** 2))
            total += coupling * np.sum(G * D * measure)
    return (-1) ** k * total


def test_the_crossed_diagram_is_its_double_frequency_integral():
    g, _, U = _model()
    engine = SkeletonSelfEnergy(XI, OCC, BOSONS, g, U)
    diagram = skeleton_diagrams(2)[0]
    mine = engine._diagram(diagram, ("dynamic", "dynamic"), 2, 0.4j)
    assert mine == pytest.approx(_quadrature(diagram, g, 2, 0.4, 120), abs=5e-6)


@pytest.mark.slow
def test_the_third_order_crossings_are_their_triple_frequency_integrals():
    g, _, U = _model()
    engine = SkeletonSelfEnergy(XI, OCC, BOSONS, g, U)
    for diagram in [d for d in skeleton_diagrams(3) if d.loops == 0]:
        mine = engine._diagram(diagram, ("dynamic",) * 3, 2, 0.4j)
        assert mine == pytest.approx(_quadrature(diagram, g, 2, 0.4, 40), abs=1e-5)


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
